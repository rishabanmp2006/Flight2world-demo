"""Video -> clean keyframes ready for reconstruction.

  1. letterbox crop, hard-cut detection, sharpest frame per ~10% image shift (scripts/prepare_dataset.py)
  2. lens undistortion from a calibration (removes barrel distortion; SfM then only refines a pinhole model)
  3. exposure normalisation across keyframes (variable illumination between views)
  4. light edge-preserving filtering against video compression blocking

Each keyframe is written as JPEG with its 35 mm-equivalent focal length in EXIF; GPS is added later.

Keyframes come out in chronological order and keep the overlap the selection asked for.  The video's **shots**
(runs of frames between hard cuts) are grouped into **logical flights** through the telemetry (see
sih3d/flight.py): one continuous flight whose footage happens to contain cuts stays ONE flight, so a cut is
never mistaken for a second pass, while a video that really contains two flights is reported as two.  Pass
`segment="single-pass"` to keep the keyframes of the largest logical flight only.
"""
from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_dataset import analyse_video, detect_letterbox, geodetic_to_enu, log, select_keyframes, \
    split_segments
from sih3d import flight


def load_calibration(path):
    """.npz with intrinsic_matrix + distCoeff (Zurich AGZ style) or .json {"K": 3x3, "dist": [...], "size": [w, h]}."""
    path = Path(path)
    if path.suffix == ".npz":
        c = np.load(path)
        return {"K": c["intrinsic_matrix"].astype(float), "dist": c["distCoeff"].ravel().astype(float), "size": None}
    import json
    c = json.loads(path.read_text())
    return {"K": np.array(c["K"], float), "dist": np.array(c["dist"], float), "size": c.get("size")}


def _undistort_maps(calib, frame_w, frame_h, crop):
    k = calib["K"].copy()
    if calib.get("size"):
        sx, sy = frame_w / calib["size"][0], frame_h / calib["size"][1]
        k[0] *= sx
        k[1] *= sy
    x0, y0, x1, y1 = crop
    k[0, 2] -= x0
    k[1, 2] -= y0
    w, h = x1 - x0, y1 - y0
    new_k, roi = cv2.getOptimalNewCameraMatrix(k, calib["dist"], (w, h), 0, (w, h))
    maps = cv2.initUndistortRectifyMap(k, calib["dist"], None, new_k, (w, h), cv2.CV_16SC2)
    return maps, new_k, roi


def _normalise_exposure(paths, strength=0.7):
    """Pull each keyframe's brightness and contrast (LAB lightness) part-way to the median of all keyframes."""
    stats = []
    for p in paths:
        small = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_4)
        lum = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        stats.append((lum.mean(), lum.std()))
    stats = np.array(stats)
    target_mean, target_std = np.median(stats, axis=0)
    for p, (m, s) in zip(paths, stats):
        img = cv2.imread(str(p))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        gain = 1 + strength * (target_std / max(s, 1e-3) - 1)
        shift = strength * (target_mean - m)
        lab[:, :, 0] = np.clip((lab[:, :, 0] - m) * gain + m + shift, 0, 255)
        cv2.imwrite(str(p), cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"target_lightness": round(float(target_mean), 1), "lightness_range_before": [round(float(stats[:, 0].min()), 1),
            round(float(stats[:, 0].max()), 1)]}


def _candidate_positions(telemetry, cands, frame_times=None):
    """Local ENU metres of every candidate frame from the telemetry, NaN outside its time range.

    `frame_times` maps a video frame index to the time on the telemetry clock (used by the Zurich AGZ input,
    whose video frames carry their own log timestamps); without it the candidate time (idx/fps) is used.
    """
    times = np.array([frame_times[c["frame"]] if frame_times is not None else c["t"] for c in cands], float)
    lat = np.interp(times, telemetry.t, telemetry.lat) if len(telemetry.t) else np.full(len(times), np.nan)
    lon = np.interp(times, telemetry.t, telemetry.lon) if len(telemetry.t) else np.full(len(times), np.nan)
    alt = np.interp(times, telemetry.t, telemetry.alt) if len(telemetry.t) else np.full(len(times), np.nan)
    inside = (times >= telemetry.t[0] - 0.5) & (times <= telemetry.t[-1] + 0.5) if len(telemetry.t) else \
        np.zeros(len(times), bool)
    lat, lon, alt = np.where(inside, lat, np.nan), np.where(inside, lon, np.nan), np.where(inside, alt, np.nan)
    if not np.isfinite(lat).any():
        return times, np.full((len(times), 3), np.nan)
    origin = (float(np.nanmedian(lat)), float(np.nanmedian(lon)), float(np.nanmedian(alt)))
    enu = np.full((len(times), 3), np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(alt)
    enu[ok] = geodetic_to_enu(lat[ok], lon[ok], alt[ok], origin)
    return times, enu


def _shot_spans(cands, segs, positions):
    """Per-shot summary rows for flight.logical_flights: times, frame count, start/end position."""
    shots = []
    for a, b in segs:
        first, last = cands[a], cands[b - 1]
        p0 = positions[a] if positions is not None and len(positions) else None
        p1 = positions[b - 1] if positions is not None and len(positions) else None
        shots.append({"start_s": float(first["t"]), "end_s": float(last["t"]), "frames": int(b - a),
                      "start_enu": None if p0 is None or not np.isfinite(p0).all() else p0,
                      "end_enu": None if p1 is None or not np.isfinite(p1).all() else p1,
                      "distance_m": 0.0})
    return shots


def extract_keyframes(video, out_dir, calib=None, sample_fps=5.0, min_shift=0.10, max_gap=2.0, blur_ratio=0.35,
                      cut_ratio=3.0, segment="all", normalise=True, deblock=True, max_size=0, telemetry=None,
                      frame_times=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"{video}: cannot open video")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop = detect_letterbox(video, n_frames)
    log(f"Analysing {n_frames} frames ({width}x{height}), crop {crop}")
    fps, cands, cuts = analyse_video(video, crop, sample_fps, cut_ratio)
    segs = split_segments(cands, cuts)

    # Video shots (the edit) vs logical flights (the physical trajectory): cuts inside one continuous flight
    # must not turn it into several flights.  Grouping uses the telemetry when it is available; without it all
    # shots are conservatively treated as one logical flight (never as several).
    positions = None
    if telemetry is not None and len(getattr(telemetry, "t", [])) > 2:
        _, positions = _candidate_positions(telemetry, cands, frame_times)
    flights = flight.logical_flights(_shot_spans(cands, segs, positions))
    flights_source = "telemetry" if positions is not None else "assumed_single"

    if segment == "all":
        chosen = list(range(len(segs)))
    elif segment == "longest":
        chosen = [max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])]
    elif segment == "single-pass":
        biggest = flight.largest_flight(flights)
        chosen = list(flights[biggest]["shots"]) if biggest is not None else list(range(len(segs)))
    else:
        chosen = [int(segment)]
    no_gps = np.full((len(cands), 3), np.nan)
    kept, blurry = [], []
    for k in chosen:
        kk, bb, _ = select_keyframes(cands, segs[k], no_gps, min_shift, None, max_gap, blur_ratio)
        kept += kk
        blurry += bb
    wanted = {cands[i]["frame"]: cands[i] for i in kept}

    maps, new_k = None, None
    x0, y0, x1, y1 = crop
    out_w, out_h = x1 - x0, y1 - y0
    if calib:
        maps, new_k, _ = _undistort_maps(calib, width, height, crop)
    cap = cv2.VideoCapture(str(video))
    records, idx, last = [], 0, max(wanted)
    while idx <= last:
        if idx not in wanted:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        frame = frame[y0:y1, x0:x1]
        if maps is not None:
            frame = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
        if deblock:
            frame = cv2.bilateralFilter(frame, 5, 12, 3)
        if max_size and max(frame.shape[:2]) > max_size:
            s = max_size / max(frame.shape[:2])
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        name = f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        c = wanted[idx]
        records.append({"name": name, "frame": idx, "t": round(c["t"], 4), "sharpness": round(c["sharpness"], 1)})
        idx += 1
    cap.release()

    exposure = _normalise_exposure([out_dir / r["name"] for r in records]) if normalise and records else None
    scale = (max_size / max(out_w, out_h)) if max_size and max(out_w, out_h) > max_size else 1.0
    focal35 = float(new_k[0, 0] * scale / (max(out_w, out_h) * scale) * 36) if new_k is not None else None
    order = [r["frame"] for r in records]
    report = {"video": str(video), "resolution": [width, height], "fps": round(fps, 3), "frames": n_frames,
              "crop_x0_y0_x1_y1": list(crop), "cuts_s": [round(f / fps, 2) for f in cuts], "shots": len(segs),
              "shots_used": chosen, "keyframes": len(records), "dropped_blurry": len(blurry),
              "undistorted": calib is not None, "focal35": round(focal35, 2) if focal35 else None,
              "exposure": exposure, "deblock": deblock,
              # single-pass bookkeeping: shots are the edit, flights are the trajectory
              "segment": segment, "logical_flights": len(flights), "flights_source": flights_source,
              "flights": [{"shots": f["shots"], "frames": f["frames"], "start_s": round(f["start_s"], 2),
                           "end_s": round(f["end_s"], 2)} for f in flights],
              "keyframes_chronological": bool(all(a <= b for a, b in zip(order, order[1:])))}
    if segment == "single-pass" and len(flights) > 1:
        report["segment_note"] = (f"'single-pass' kept the largest of {len(flights)} logical flights "
                                  f"(shot indices {chosen})")
    return records, report
