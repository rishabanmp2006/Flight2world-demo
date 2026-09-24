"""Accuracy checks for a finished run.

Camera positions: where the reconstruction placed each keyframe vs a reference (e.g. the Zurich AGZ Pix4D poses),
alongside the raw and fused GPS for the same frames, so the effect of each stage is visible.
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
from pyproj import CRS, Transformer


def project_crs(project):
    first = (Path(project) / "odm_georeferencing" / "coords.txt").read_text().splitlines()[0]
    m = re.match(r"WGS84 UTM (\d+)([NS])", first)
    return CRS.from_epsg((32600 if m.group(2) == "N" else 32700) + int(m.group(1)))


def reconstructed_positions(project):
    """name -> camera centre in the project's UTM zone (metres)."""
    project = Path(project)
    rec = json.loads((project / "opensfm" / "reconstruction.json").read_text())[0]
    off = [float(v) for v in (project / "odm_georeferencing" / "coords.txt").read_text().splitlines()[1].split()[:2]]
    out = {}
    for name, shot in rec["shots"].items():
        rot, _ = cv2.Rodrigues(np.array(shot["rotation"], float))
        c = -rot.T @ np.array(shot["translation"], float)
        out[name] = np.array([c[0] + off[0], c[1] + off[1], c[2]])
    return out


def agz_reference(log_dir):
    """imgid -> Pix4D camera position (UTM 32N) from GroundTruthAGL.csv."""
    from sih3d.telemetry import _read_columns
    gt = _read_columns(Path(log_dir) / "GroundTruthAGL.csv")
    return {int(i): np.array([x, y, z]) for i, x, y, z in zip(gt["imgid"], gt["x_gt"], gt["y_gt"], gt["z_gt"])
            if not np.isnan(i)}


def position_errors(estimate, reference):
    """estimate, reference: aligned (n, 3) arrays in metres."""
    d = estimate - reference
    h = np.hypot(d[:, 0], d[:, 1])
    rel = d - d.mean(axis=0)  # shape error after removing a constant offset
    hr = np.hypot(rel[:, 0], rel[:, 1])
    r = lambda a: round(float(a), 2)
    return {"n": int(len(d)), "horizontal_median_m": r(np.median(h)), "horizontal_p90_m": r(np.percentile(h, 90)),
            "horizontal_max_m": r(h.max()), "vertical_median_m": r(np.median(d[:, 2])),
            "vertical_spread_p10_p90_m": r(np.percentile(d[:, 2], 90) - np.percentile(d[:, 2], 10)),
            "offset_removed_horizontal_median_m": r(np.median(hr)), "offset_removed_horizontal_p90_m": r(np.percentile(hr, 90))}


def _umeyama(src, dst, with_scale):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d0.T @ s0 / len(src))
    fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        fix[2, 2] = -1
    rot = u @ fix @ vt
    scale = float((sig * np.diag(fix)).sum() / s0.var(0).sum()) if with_scale else 1.0
    return scale, rot, mu_d - scale * rot @ mu_s


def _icp(src, tree, ref, with_scale, thresholds=(8.0, 4.0, 2.0, 1.0, 0.5), iters=8):
    total_s, total_r, total_t = 1.0, np.eye(3), np.zeros(3)
    cur = src.copy()
    for thr in thresholds:
        for _ in range(iters):
            dist, idx = tree.query(cur, distance_upper_bound=thr)
            ok = np.isfinite(dist)
            if ok.sum() < 1000:
                break
            s, r, t = _umeyama(cur[ok], ref[idx[ok]], with_scale)
            cur = s * cur @ r.T + t
            total_s, total_r, total_t = s * total_s, r @ total_r, s * r @ total_t + t
    return cur, total_s, total_r, total_t


def cloud_vs_lidar(project, lidar_paths, lidar_crs="EPSG:2056", lidar_z_offset=-0.111, max_points=300_000, seed=0,
                   cloud_path=None):
    """Reconstructed cloud vs reference airborne LiDAR.

    lidar_z_offset converts reference heights to the cloud's height system (swissSURFACE3D LN02 -> EGM96 at Zurich
    is -0.111 m; drone GPS altitude above mean sea level ~ EGM96).
    Reports distances as georeferenced, after rigid alignment (shape error) and the scale from a similarity fit."""
    import laspy
    from scipy.spatial import cKDTree

    project = Path(project)
    crs = project_crs(project)
    las = laspy.read(cloud_path or project / "odm_georeferencing" / "odm_georeferenced_model.laz")
    pts = np.column_stack([las.x, las.y, las.z])
    if "inferred" in set(las.point_format.dimension_names):
        pts = pts[np.asarray(las.inferred) == 0]  # accuracy of measured geometry only
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    lo, hi = pts.min(0) - 20, pts.max(0) + 20

    to = Transformer.from_crs(lidar_crs, crs, always_xy=True)
    refs = []
    for path in lidar_paths:
        ref = laspy.read(path)
        x, y = to.transform(np.asarray(ref.x), np.asarray(ref.y))
        z = np.asarray(ref.z) + lidar_z_offset
        keep = (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
        refs.append(np.column_stack([x[keep], y[keep], z[keep]]))
    ref = np.vstack(refs)
    if len(ref) < 1000:
        return {"note": "reference LiDAR does not overlap the reconstruction"}
    tree = cKDTree(ref)

    def dist_stats(p):
        d, _ = tree.query(p)
        r = lambda a: round(float(a), 3)
        return {"median_m": r(np.median(d)), "p90_m": r(np.percentile(d, 90)),
                "within_0.5m": r(np.mean(d < 0.5)), "within_1m": r(np.mean(d < 1.0))}

    # Solve in a local frame centred on the cloud: rotations about a UTM origin hundreds of km away are ill-conditioned.
    c = pts.mean(axis=0)
    local_tree = cKDTree(ref - c)
    rigid_local, _, rot, t_local = _icp(pts - c, local_tree, ref - c, with_scale=False)
    rigid = rigid_local + c
    _, scale, _, _ = _icp(rigid_local, local_tree, ref - c, with_scale=True, thresholds=(1.0, 0.5))
    angle = float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1))))
    heading = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    t_global = t_local + c - rot @ c  # same transform expressed for absolute coordinates: p -> rot @ p + t_global
    return {"reference_points_used": int(len(ref)), "cloud_points_sampled": int(len(pts)),
            "as_georeferenced": dist_stats(pts),
            "after_rigid_alignment": dist_stats(rigid),
            "georeference_offset_m": [round(float(v), 2) for v in (rot @ c + t_global - c)],
            "rigid_rotation_deg": round(angle, 3), "heading_error_deg": round(heading, 3),
            "rigid_alignment": {"rotation": rot.tolist(), "translation": t_global.tolist()},
            "similarity_scale": round(scale, 5), "scale_error_pct": round(100 * (scale - 1), 2)}


def compare_agz(project, records, log_dir, first_imgid):
    """records carry frame index, raw_lat/raw_lon/raw_alt and lat/lon/alt (fused) per keyframe."""
    crs = project_crs(project)
    to = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    ref = agz_reference(log_dir)
    recon = reconstructed_positions(project)
    ids = np.array(sorted(ref))
    xyz = np.array([ref[i] for i in ids])
    # References exist every 30th image (1 s, ~1.4 m of travel): interpolate them to each keyframe.
    rows = [r for r in records if ids[0] <= first_imgid + r["frame"] <= ids[-1]]
    if not rows:
        return {"note": "keyframes fall outside the reference poses"}
    img = np.array([first_imgid + r["frame"] for r in rows], float)
    refs = np.column_stack([np.interp(img, ids, xyz[:, k]) for k in range(3)])
    out = {"reference": "Pix4D camera poses (GroundTruthAGL.csv, every 30th image, interpolated to keyframes); "
                        "seeded by the same GPS, so not an independent survey"}
    for label, keys in (("raw_gps", ("raw_lon", "raw_lat", "raw_alt")), ("fused_gps", ("lon", "lat", "alt"))):
        x, y = to.transform([r[keys[0]] for r in rows], [r[keys[1]] for r in rows])
        out[label] = position_errors(np.column_stack([x, y, [r[keys[2]] for r in rows]]), refs)
    have = [i for i, r in enumerate(rows) if r["name"] in recon]
    if have:
        est = np.array([recon[rows[i]["name"]] for i in have])
        out["reconstruction"] = position_errors(est, refs[have])
    return out
