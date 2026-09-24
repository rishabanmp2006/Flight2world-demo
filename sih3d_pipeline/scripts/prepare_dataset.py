#!/usr/bin/env python3
"""Prepare single-pass drone data for 3D reconstruction (SIH26158).

  video : drone video (+ DJI .SRT or CSV telemetry) -> sharp, well-spaced keyframes tagged with GPS
  images: photo sequence with EXIF GPS              -> sharp, time-ordered frames with GPS

Output (usable directly by OpenDroneMap and COLMAP):
  OUT/images/            keyframes (JPEG, GPS written into EXIF)
  OUT/geo.txt            OpenDroneMap geolocation file (EPSG:4326)
  OUT/gps_priors.txt     COLMAP model_aligner reference: name lat lon alt  (use --ref_is_gps 1)
  OUT/telemetry.csv      per-keyframe time, GPS, local ENU metres, sharpness
  OUT/contact_sheet.jpg  thumbnail grid for a quick visual check
  OUT/report.json        what was kept or dropped, and why

Examples:
  python scripts/prepare_dataset.py video  --video DJI_0001.MP4 --telemetry DJI_0001.SRT --out data/processed/flight1
  python scripts/prepare_dataset.py images --images data/odm_aukerman/images --out data/processed/aukerman
"""
import argparse
import csv
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image

THUMB_W = 640
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
CUT_MIN_DIFF = 12.0  # mean grey-level change (0-255) below which consecutive frames are never a cut


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- geodesy

WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3


def geodetic_to_ecef(lat, lon, alt):
    lat, lon, alt = np.radians(lat), np.radians(lon), np.asarray(alt, float)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(n + alt) * np.cos(lat) * np.cos(lon),
                     (n + alt) * np.cos(lat) * np.sin(lon),
                     (n * (1 - WGS84_E2) + alt) * np.sin(lat)], axis=-1)


def geodetic_to_enu(lat, lon, alt, origin):
    """Local East-North-Up metres relative to origin (lat, lon, alt)."""
    lat0, lon0, alt0 = origin
    d = geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(lat0, lon0, alt0)
    la, lo = math.radians(lat0), math.radians(lon0)
    r = np.array([[-math.sin(lo), math.cos(lo), 0.0],
                  [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                  [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
    return d @ r.T


# ---------------------------------------------------------------- telemetry

NUM = r"(-?\d+(?:\.\d+)?)"


def _num(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def parse_srt(path):
    """DJI subtitle telemetry. Handles the newer '[latitude: ..] [longitude: ..] [abs_alt: ..]' style
    and the older 'GPS(lon,lat,..) BAROMETER:..' style. Returns rows of (t_seconds, lat, lon, alt)."""
    rows = []
    text = Path(path).read_text(errors="ignore").replace("\r", "")
    for block in re.split(r"\n\s*\n", text):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", block)
        if not m:
            continue
        h, mi, s, frac = m.groups()
        t = int(h) * 3600 + int(mi) * 60 + int(s) + int(frac) / 10 ** len(frac)
        lat = _num(r"latitude\s*[:=]\s*" + NUM, block)
        lon = _num(r"longitude\s*[:=]\s*" + NUM, block)
        alt = _num(r"abs_alt\s*[:=]\s*" + NUM, block)
        if lat is None or lon is None:
            g = re.search(r"GPS\s*\(\s*" + NUM + r"\s*,\s*" + NUM, block)
            if g:
                a, b = map(float, g.groups())
                # Older DJI firmware writes GPS(lon, lat, ...); trust magnitudes when they disagree.
                lat, lon = (a, b) if abs(b) > 90 else (b, a)
        if alt is None:
            alt = _num(r"BAROMETER\s*[:=]?\s*" + NUM, block)
        if alt is None:
            alt = _num(r"rel_alt\s*[:=]\s*" + NUM, block)
        if lat is None or lon is None or (lat == 0 and lon == 0):
            continue
        rows.append((t, lat, lon, alt if alt is not None else 0.0))
    return rows


def _to_seconds(value, scale):
    try:
        return float(value) * scale
    except (TypeError, ValueError):
        return datetime.fromisoformat(str(value).strip()).timestamp()


def parse_csv(path):
    """Flight-log CSV. Column names are matched loosely (time / latitude / longitude / altitude)."""
    with open(path, newline="", errors="ignore") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}

        def pick(*keys):
            for k in keys:
                for low, orig in cols.items():
                    if low == k or low.startswith(k):
                        return orig
            return None

        tc = pick("time", "timestamp", "seconds")
        la, lo = pick("latitude", "lat"), pick("longitude", "lon", "lng")
        al = pick("altitude", "abs_alt", "alt", "height")
        if not (tc and la and lo):
            sys.exit(f"{path}: need time, latitude and longitude columns; found {reader.fieldnames}")
        scale = 0.001 if re.search(r"ms|milli", tc, re.IGNORECASE) else 1.0
        rows = []
        for r in reader:
            try:
                t = _to_seconds(r[tc], scale)
                lat, lon = float(r[la]), float(r[lo])
                alt = float(r[al]) if al and r[al] not in ("", None) else 0.0
            except (TypeError, ValueError):
                continue
            if lat == 0 and lon == 0:
                continue
            rows.append((t, lat, lon, alt))
    if rows:
        t0 = min(r[0] for r in rows)
        rows = [(t - t0, lat, lon, alt) for t, lat, lon, alt in rows]
    return rows


def load_telemetry(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".srt":
        rows = parse_srt(path)
    elif suffix == ".csv":
        rows = parse_csv(path)
    else:
        sys.exit(f"{path}: telemetry must be a DJI .SRT or a .csv flight log")
    if len(rows) < 2:
        sys.exit(f"{path}: fewer than 2 GPS readings could be parsed")
    return rows


def apply_gps_table(items, path, match):
    """GPS for photos without EXIF GPS, from a CSV keyed by what the file name holds: a capture time
    (UAVScenes '<ros_stamp>.jpg' + rtk_positions_raw.csv, interpolated) or an image id
    (Zurich AGZ '<imgid>.jpg' + OnboardGPS.csv, exact)."""
    with open(path, newline="", errors="ignore") as f:
        rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    if not rows:
        sys.exit(f"{path}: empty CSV")
    cols = list(rows[0])

    def col(*names):
        return next((n for n in names if n in cols), None)

    lat, lon, alt = col("lat", "latitude"), col("lon", "lng", "longitude"), col("alt", "altitude", "abs_alt")
    key = col("imgid", "image_id", "id", "frame") if match == "id" else \
        col("headerstamp", "timestamp", "timpstemp", "time", "t")
    if not (lat and lon and key):
        sys.exit(f"{path}: need a {match} column plus lat and lon; found {cols}")
    table = []
    for r in rows:
        try:
            table.append((float(r[key]), float(r[lat]), float(r[lon]), float(r[alt]) if alt and r[alt] else 0.0))
        except ValueError:
            continue
    table.sort()
    keys, vals = np.array([t[0] for t in table]), np.array([t[1:] for t in table])

    out, matched = [], 0
    for p, _, when in items:
        try:
            k = float(p.stem)
        except ValueError:
            sys.exit(f"{p.name}: with --gps-csv the file name must be the {match} (e.g. 1658131847.149.jpg, 59001.jpg)")
        coords = None
        if match == "id":
            j = int(np.searchsorted(keys, k))
            if j < len(keys) and keys[j] == k:
                coords = tuple(float(v) for v in vals[j])
        else:
            if keys[0] <= k <= keys[-1]:
                coords = tuple(float(np.interp(k, keys, vals[:, c])) for c in range(3))
            when = datetime.fromtimestamp(k)
        matched += coords is not None
        out.append((p, coords, when))
    out.sort(key=lambda it: float(it[0].stem))
    log(f"GPS from {path}: {matched} of {len(out)} images matched")
    return out


def interpolate_gps(rows, times, offset):
    """GPS (lat, lon, alt) at each time; NaN outside the telemetry range."""
    tt = np.array([r[0] for r in rows]) + offset
    order = np.argsort(tt)
    tt, vals = tt[order], np.array([r[1:] for r in rows], float)[order]
    out = np.full((len(times), 3), np.nan)
    inside = (times >= tt[0] - 0.5) & (times <= tt[-1] + 0.5)
    for k in range(3):
        out[inside, k] = np.interp(times[inside], tt, vals[:, k])
    return out


# ---------------------------------------------------------------- image measures

def shrink(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (THUMB_W, max(1, round(h * THUMB_W / w))), interpolation=cv2.INTER_AREA)


def sharpness(gray):
    """Variance of the Laplacian: drops sharply on motion blur and defocus."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def flow_shift(prev, cur):
    """Median feature displacement between two frames as a fraction of image width.
    None when tracking fails (too few features survive): a cut, very fast motion, or no texture."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 20:
        return None
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 20:
        return None
    d = np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)
    return float(np.median(d)) / prev.shape[1]


def detect_letterbox(path, n_frames, samples=24, dark=20):
    """Black bars burned into the video (cinematic letterbox, pillarbox). Returns crop x0, y0, x1, y1."""
    cap = cv2.VideoCapture(str(path))
    rows = cols = None
    for idx in np.linspace(0, max(n_frames - 1, 0), samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        r, c = g.mean(axis=1), g.mean(axis=0)
        rows = r if rows is None else np.maximum(rows, r)
        cols = c if cols is None else np.maximum(cols, c)
    cap.release()
    if rows is None:
        sys.exit(f"{path}: could not decode any frames")

    def span(profile):
        size = len(profile)
        lit = np.where(profile > dark)[0]
        if not len(lit):
            return 0, size
        a, b = int(lit[0]), int(lit[-1]) + 1
        if a < 0.02 * size and size - b < 0.02 * size:
            return 0, size
        margin = 2  # bar edges are soft after compression
        return min(a + margin, size), max(b - margin, 0)

    y0, y1 = span(rows)
    x0, x1 = span(cols)
    return x0, y0, x1, y1


# ---------------------------------------------------------------- EXIF

def _dms(value):
    value = abs(value)
    d = int(value)
    m = int((value - d) * 60)
    s = (value - d - m / 60) * 3600
    return (d, 1), (m, 1), (int(round(s * 10000)), 10000)


def write_gps_exif(path, lat, lon, alt, focal35=None):
    gps = {
        piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
        piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
        piexif.GPSIFD.GPSLatitude: _dms(lat),
        piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
        piexif.GPSIFD.GPSLongitude: _dms(lon),
        piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
        piexif.GPSIFD.GPSAltitude: (int(round(abs(alt) * 1000)), 1000),
    }
    exif = {"0th": {}, "Exif": {}, "GPS": gps, "1st": {}}
    if focal35:
        exif["Exif"][piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(focal35))
    piexif.insert(piexif.dump(exif), str(path))


def read_exif_gps(path):
    """(lat, lon, alt) or None, and capture datetime or None."""
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            gps = exif.get_ifd(0x8825)
            when = exif.get_ifd(0x8769).get(36867) or exif.get(306)
    except Exception:
        return None, None

    def deg(v, ref):
        d, m, s = (float(x) for x in v)
        val = d + m / 60 + s / 3600
        return -val if str(ref).strip("\x00b'") in ("S", "W") else val

    coords = None
    if 2 in gps and 4 in gps:
        alt = float(gps.get(6, 0.0))
        if gps.get(5) in (1, b"\x01"):
            alt = -alt
        coords = (deg(gps[2], gps.get(1, "N")), deg(gps[4], gps.get(3, "E")), alt)
    try:
        when = datetime.strptime(str(when).strip("\x00 "), "%Y:%m:%d %H:%M:%S") if when else None
    except ValueError:
        when = None
    return coords, when


def order_images(folder):
    """GPS-tagged photos sorted by capture time (file name when any photo lacks a timestamp)."""
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXT)
    if not files:
        sys.exit(f"{folder}: no images found")
    items = [(p, *read_exif_gps(p)) for p in files]
    if all(when for _, _, when in items):
        items.sort(key=lambda it: (it[2], it[0].name))
    return items


# ---------------------------------------------------------------- outputs

def contact_sheet(img_dir, names, out_path, cols=6, width=320):
    pick = names if len(names) <= 36 else [names[i] for i in np.linspace(0, len(names) - 1, 36).astype(int)]
    tiles = []
    for n in pick:
        im = cv2.imread(str(img_dir / n), cv2.IMREAD_REDUCED_COLOR_4)
        if im is None:
            continue
        h = int(im.shape[0] * width / im.shape[1])
        im = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(im, n, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, thick, cv2.LINE_AA)
        tiles.append(im)
    if not tiles:
        return
    th = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])


def write_outputs(out, records, report):
    """records: dicts with name, source, t, lat, lon, alt, sharpness (lat/lon NaN when unknown)."""
    geo = [r for r in records if not math.isnan(r["lat"])]
    if geo:
        origin = (geo[0]["lat"], geo[0]["lon"], geo[0]["alt"])
        enu = geodetic_to_enu(np.array([r["lat"] for r in geo]), np.array([r["lon"] for r in geo]),
                              np.array([r["alt"] for r in geo]), origin)
        for r, e in zip(geo, enu):
            r["east"], r["north"], r["up"] = (float(v) for v in e)
        steps = np.linalg.norm(np.diff(enu, axis=0), axis=1) if len(enu) > 1 else np.array([0.0])
        report["gps"] = {
            "tagged_frames": len(geo),
            "origin_lat_lon_alt": origin,
            "track_length_m": round(float(steps.sum()), 2),
            "median_keyframe_spacing_m": round(float(np.median(steps)), 2),
            "altitude_range_m": [round(float(enu[:, 2].min()), 2), round(float(enu[:, 2].max()), 2)],
        }
        with open(out / "geo.txt", "w") as f:
            f.write("EPSG:4326\n")
            f.writelines(f"{r['name']} {r['lon']:.8f} {r['lat']:.8f} {r['alt']:.3f}\n" for r in geo)
        with open(out / "gps_priors.txt", "w") as f:
            f.writelines(f"{r['name']} {r['lat']:.8f} {r['lon']:.8f} {r['alt']:.3f}\n" for r in geo)
    else:
        report["warnings"].append("No GPS: the model cannot be georeferenced or metrically scaled "
                                  "(SIH26158 requires GPS). Pass --telemetry with the .SRT/CSV.")

    fields = ["name", "source", "t", "lat", "lon", "alt", "east", "north", "up", "sharpness"]
    with open(out / "telemetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: ("" if isinstance(r.get(k), float) and math.isnan(r[k]) else r.get(k, "")) for k in fields})

    contact_sheet(out / "images", [r["name"] for r in records], out / "contact_sheet.jpg")
    report["keyframes"] = len(records)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log(json.dumps({k: report[k] for k in ("keyframes", "gps", "warnings") if k in report}, indent=2, default=str))
    log(f"Done -> {out}")


def prepare_out(out, overwrite):
    out = Path(out)
    images = out / "images"
    if images.exists() and any(images.iterdir()):
        if not overwrite:
            sys.exit(f"{images} is not empty; pass --overwrite to replace it")
        shutil.rmtree(images)
    images.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------- video mode

def same_scene(a, b):
    """True when two frames share enough geometrically consistent ORB matches to be one continuous shot."""
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return False
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 25:
        return False
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 5.0)
    return mask is not None and int(mask.sum()) >= max(25, 0.3 * len(good))


def analyse_video(path, crop, sample_fps, cut_ratio):
    """Pass 1. Every frame: a thumbnail difference, whose spikes propose hard cuts that ORB matching
    then confirms (so fast motion is not mistaken for a cut). Sampled frames: sharpness and image shift."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / sample_fps))
    window = max(5, round(fps / 2))
    x0, y0, x1, y1 = crop
    cands, cuts, diffs = [], [], []
    prev_small = prev_tiny = prev_sampled = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(shrink(frame[y0:y1, x0:x1]), cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(small, (160, 90), interpolation=cv2.INTER_AREA)
        if cut_ratio > 0 and prev_tiny is not None:
            d = float(cv2.absdiff(tiny, prev_tiny).mean())
            recent = float(np.median(diffs[-window:])) if diffs else 0.0
            if d > max(CUT_MIN_DIFF, cut_ratio * recent) and not same_scene(prev_small, small):
                cuts.append(idx)
            diffs.append(d)
        if idx % step == 0:
            c = {"frame": idx, "t": idx / fps, "sharpness": sharpness(small), "shift": 0.0}
            if prev_sampled is not None:
                c["shift"] = flow_shift(prev_sampled, small)
            cands.append(c)
            prev_sampled = small
            if len(cands) % 100 == 0:
                log(f"  analysed {idx}/{total} frames")
        prev_small, prev_tiny = small, tiny
        idx += 1
    cap.release()
    return fps, cands, cuts


def split_segments(cands, cuts):
    """Candidate index ranges for each continuous shot between hard cuts."""
    segs, start, pending = [], 0, sorted(cuts)
    for i, c in enumerate(cands):
        crossed = False
        while pending and c["frame"] >= pending[0]:
            pending.pop(0)
            crossed = True
        if crossed and i > start:
            segs.append((start, i))
            start = i
    segs.append((start, len(cands)))
    return segs


def select_keyframes(cands, seg, gps, min_shift, min_move, max_gap, blur_ratio):
    """Bin the shot by accumulated motion (image shift, or metres of GPS travel when --min-move is set)
    and keep the sharpest frame per bin, so spacing follows overlap instead of time. A bin also closes
    after max_gap seconds, so slow orbits and rotations still get enough views."""
    sub = list(range(*seg))
    med = float(np.median([cands[i]["sharpness"] for i in sub]))
    motion = [0.0]
    for prev, cur in zip(sub, sub[1:]):
        if min_move and not (np.isnan(gps[prev]).any() or np.isnan(gps[cur]).any()):
            d = geodetic_to_enu(gps[cur, 0], gps[cur, 1], gps[cur, 2], tuple(gps[prev]))
            step = float(np.linalg.norm(d)) / min_move
        else:
            s = cands[cur]["shift"]
            step = 1.0 if s is None else s / min_shift
        motion.append(max(step, (cands[cur]["t"] - cands[prev]["t"]) / max_gap))
    bins = np.floor(np.cumsum(motion) + 1e-9).astype(int)
    kept, blurry = [], []
    for b in np.unique(bins):
        members = [sub[j] for j in np.flatnonzero(bins == b)]
        best = max(members, key=lambda i: cands[i]["sharpness"])
        (blurry if cands[best]["sharpness"] < blur_ratio * med else kept).append(best)
    gaps = int(np.sum(np.diff(bins[np.isin(sub, kept)]) > 2)) if len(kept) > 1 else 0
    return kept, blurry, gaps


def write_video_frames(path, frame_ids, crop, img_dir, max_size):
    x0, y0, x1, y1 = crop
    cap = cv2.VideoCapture(str(path))
    wanted, last, idx, names = set(frame_ids), max(frame_ids), 0, {}
    while idx <= last:
        if idx not in wanted:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame[y0:y1, x0:x1]
            if max_size and max(frame.shape[:2]) > max_size:
                s = max_size / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            names[idx] = f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(img_dir / names[idx]), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    return names


def run_video(args):
    out = prepare_out(args.out, args.overwrite)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"{args.video}: cannot open video")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    crop = (0, 0, width, height) if args.no_crop else detect_letterbox(args.video, n_frames)
    report = {"mode": "video", "input": str(args.video), "telemetry": str(args.telemetry or ""),
              "resolution": [width, height], "crop_x0_y0_x1_y1": crop, "warnings": []}
    if crop != (0, 0, width, height):
        report["warnings"].append(f"Black bars detected and cropped: {crop}")

    log(f"Pass 1: analysing {args.video} ({width}x{height}, {n_frames} frames)")
    fps, cands, cuts = analyse_video(args.video, crop, args.sample_fps, args.cut_ratio)
    report.update(fps=round(fps, 3), duration_s=round(n_frames / fps, 2), sampled_frames=len(cands))
    times = np.array([c["t"] for c in cands])

    if args.telemetry:
        rows = load_telemetry(args.telemetry)
        gps = interpolate_gps(rows, times, args.telemetry_offset)
        report["telemetry_readings"] = len(rows)
        missing = int(np.isnan(gps[:, 0]).sum())
        if missing:
            report["warnings"].append(f"{missing} sampled frames fall outside the telemetry time range")
    else:
        gps = np.full((len(cands), 3), np.nan)

    segs = split_segments(cands, cuts)
    report["cuts_s"] = [round(f / fps, 2) for f in cuts]
    report["shots"] = [{"shot": k, "start_s": round(times[a], 2), "end_s": round(times[b - 1], 2),
                        "duration_s": round(times[b - 1] - times[a], 2)} for k, (a, b) in enumerate(segs)]
    if len(segs) > 1:
        report["warnings"].append(f"{len(segs) - 1} hard cuts found: this is an edited video, not one "
                                  "continuous pass. Reconstruct shots separately (--segment).")
    if args.segment == "all":
        chosen = list(range(len(segs)))
    elif args.segment == "longest":
        chosen = [max(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])]
    else:
        chosen = [int(args.segment)]
    report["shots_used"] = chosen

    kept, dropped_blurry, gaps = [], [], 0
    for k in chosen:
        kk, bb, gg = select_keyframes(cands, segs[k], gps, args.min_shift, args.min_move, args.max_gap,
                                      args.blur_ratio)
        kept += kk
        dropped_blurry += bb
        gaps += gg
    if not kept:
        sys.exit("No keyframes survived; lower --blur-ratio or --min-shift")
    report["dropped_blurry_frames"] = [cands[i]["frame"] for i in dropped_blurry]
    if gaps:
        report["warnings"].append(f"{gaps} coverage gaps (motion > 2 steps between keyframes); "
                                  "raise --sample-fps")

    log(f"Pass 2: writing {len(kept)} keyframes")
    names = write_video_frames(args.video, [cands[i]["frame"] for i in kept], crop, out / "images", args.max_size)
    records = []
    for i in kept:
        c = cands[i]
        if c["frame"] not in names:
            continue
        lat, lon, alt = gps[i]
        rec = {"name": names[c["frame"]], "source": c["frame"], "t": round(c["t"], 3),
               "lat": float(lat), "lon": float(lon), "alt": float(alt), "sharpness": round(c["sharpness"], 1)}
        if not math.isnan(lat):
            write_gps_exif(out / "images" / rec["name"], lat, lon, alt, args.focal35)
        records.append(rec)
    write_outputs(out, records, report)


# ---------------------------------------------------------------- images mode

def split_strips(coords, turn_deg=45.0):
    """Flight-line id per photo: a new strip starts where the heading turns by more than turn_deg."""
    if len(coords) < 3 or any(c is None for c in coords):
        return [0] * len(coords)
    enu = geodetic_to_enu(*np.array(coords).T, coords[0])
    heading = np.degrees(np.arctan2(np.diff(enu[:, 0]), np.diff(enu[:, 1])))
    ids, strip = [0, 0], 0
    for k in range(1, len(heading)):
        if abs((heading[k] - heading[k - 1] + 180) % 360 - 180) > turn_deg:
            strip += 1
        ids.append(strip)
    return ids


def run_images(args):
    out = prepare_out(args.out, args.overwrite)
    items = order_images(args.images)
    if args.gps_csv:
        items = apply_gps_table(items, args.gps_csv, args.match)
    log(f"Scoring {len(items)} images from {args.images}")
    scored = []
    for p, coords, when in items:
        g = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        scored.append({"path": p, "coords": coords, "when": when, "sharpness": sharpness(shrink(g)) if g is not None else 0.0})
    med = float(np.median([s["sharpness"] for s in scored]))
    report = {"mode": "images", "input": str(args.images), "images_found": len(scored), "warnings": []}
    strips = split_strips([s["coords"] for s in scored])
    report["strips"] = {str(k): strips.count(k) for k in sorted(set(strips))}
    if args.strip != "all":
        pick = max(set(strips), key=strips.count) if args.strip == "longest" else int(args.strip)
        scored = [s for s, k in zip(scored, strips) if k == pick]
        if not scored:
            sys.exit(f"strip {pick} does not exist; strips are {report['strips']}")
        report["strip_used"] = pick

    kept, blurry, too_close, last = [], [], [], None
    for s in scored:
        if s["sharpness"] < args.blur_ratio * med:
            blurry.append(s["path"].name)
            continue
        if args.min_move and last and s["coords"] and last["coords"]:
            if float(np.linalg.norm(geodetic_to_enu(*s["coords"], last["coords"]))) < args.min_move:
                too_close.append(s["path"].name)
                continue
        kept.append(s)
        last = s
    report.update(dropped_blurry=blurry, dropped_too_close=too_close)
    no_gps = sum(1 for s in kept if not s["coords"])
    if no_gps:
        report["warnings"].append(f"{no_gps} kept images have no EXIF GPS")

    t0 = kept[0]["when"] if kept and kept[0]["when"] else None
    records = []
    for i, s in enumerate(kept):
        shutil.copy2(s["path"], out / "images" / s["path"].name)
        if args.gps_csv and s["coords"]:
            write_gps_exif(out / "images" / s["path"].name, *s["coords"], args.focal35)
        lat, lon, alt = s["coords"] or (math.nan, math.nan, math.nan)
        t = (s["when"] - t0).total_seconds() if t0 and s["when"] else float(i)
        records.append({"name": s["path"].name, "source": str(s["path"]), "t": t, "lat": lat, "lon": lon,
                        "alt": alt, "sharpness": round(s["sharpness"], 1)})
    write_outputs(out, records, report)


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    v = sub.add_parser("video", help="drone video (+ telemetry) -> keyframes")
    v.add_argument("--video", required=True, type=Path)
    v.add_argument("--telemetry", type=Path, help="DJI .SRT or flight-log .csv")
    v.add_argument("--telemetry-offset", type=float, default=0.0, help="seconds added to telemetry times to sync")
    v.add_argument("--sample-fps", type=float, default=5.0, help="frames per second analysed (default 5)")
    v.add_argument("--min-shift", type=float, default=0.10,
                   help="image shift between keyframes as a fraction of width (0.10 ~ 90%% overlap)")
    v.add_argument("--min-move", type=float, help="use GPS travel in metres between keyframes instead of image shift")
    v.add_argument("--max-gap", type=float, default=2.0, help="at most this many seconds between keyframes")
    v.add_argument("--cut-ratio", type=float, default=3.0,
                   help="a frame difference this many times the recent median proposes a hard cut; "
                        "0 disables cut detection (known single-pass input, e.g. images_to_video.py output)")
    v.add_argument("--segment", default="longest", help="'longest', 'all', or a shot number from report.json")
    v.add_argument("--no-crop", action="store_true", help="keep black bars")
    v.add_argument("--max-size", type=int, default=0, help="downscale long edge to this many pixels (0 = keep)")
    v.add_argument("--focal35", type=float, help="35mm-equivalent focal length, written to EXIF")

    i = sub.add_parser("images", help="GPS-tagged photo sequence -> filtered frames")
    i.add_argument("--images", required=True, type=Path)
    i.add_argument("--min-move", type=float, default=0.0, help="drop photos closer than this many metres to the last kept")
    i.add_argument("--strip", default="all",
                   help="'all', 'longest', or a strip number: keep one flight line of a grid survey (single pass)")
    i.add_argument("--gps-csv", type=Path, help="GPS table for photos without EXIF GPS (e.g. UAVScenes, Zurich AGZ)")
    i.add_argument("--match", choices=["time", "id"], default="time",
                   help="with --gps-csv: file name is a capture time (interpolated) or an image id (exact)")
    i.add_argument("--focal35", type=float, help="35mm-equivalent focal length written to EXIF with --gps-csv")

    for p in (v, i):
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--blur-ratio", type=float, default=0.35,
                       help="drop frames sharper than less than this fraction of the median (default 0.35)")
        p.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    run_video(args) if args.mode == "video" else run_images(args)


if __name__ == "__main__":
    main()
