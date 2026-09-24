"""Roll correction about the flight line — the one rotation a single straight pass leaves unconstrained.

GPS positions along a near-straight track fix heading and along-track slope, but not the rotation about the track
itself: on Zurich AGZ three identical runs came out tilted 2, 12 and 19.6 degrees against LiDAR while heading was
always within ~1.5 degrees. Building walls fix exactly that axis: facades along a street face across it, so a roll
about the track tips their normals up or down.

Method:
  1. track direction t = principal axis of the reconstructed camera centres
  2. surface normals on planar patches of AI-classified building points (local PCA)
  3. find the single angle theta about t that makes wall normals horizontal, with a robust (clipped) cost
  4. rotate the cloud about its centre by theta; heading and along-track slope are left to GPS

An earlier version solved a full 3D 'up' vector from walls plus road normals; along-track pitch was then set by a few
hundred noisy road normals and made a good run worse (2.0 -> 5.1 deg), so only the degenerate axis is corrected now.

Measured on Zurich AGZ (roll about the track still needed to match official swissBUILDINGS3D walls + LiDAR ground,
multi-start rigid alignment):
  run          before                  correction applied    after
  no AI        19.3 deg                -12.1 deg             7.9 deg
  AI preview    8.0 deg                -12.5 deg            -3.6 deg
  AI full      >=19.1 deg (see note)   -36.1 deg             4.0 deg
For the two preview runs the check is additive to within ~1 deg (before - applied ~= after), so it is a real measurement.
For the full run it is not: the alignment starts from translations only and cannot converge from a ~36-40 deg tilt,
so 19.1 is a lower bound; the wall-consensus count (780 -> 78,104 vertical patches) and road heights (38.5 deg) point to
~36-40 deg. The 'after' values start near level and are reliable. Heading and along-track slope stay within ~1-2 deg.
On by default; --no-level skips it.
"""
import json
from pathlib import Path

import cv2
import numpy as np


def _normals(points, k=16):
    from scipy.spatial import cKDTree
    _, idx = cKDTree(points).query(points, k=k)
    nb = points[idx] - points[idx].mean(axis=1, keepdims=True)
    vals, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb) / k)
    return vecs[:, :, 0], vals[:, 0] / np.maximum(vals.sum(axis=1), 1e-12)


def _axis_rotation(axis, theta):
    return cv2.Rodrigues(np.asarray(axis, float) * theta)[0]


def track_axis(project):
    """Unit direction of the camera track (camera centres in the cloud's projected coordinates)."""
    rec = json.loads((Path(project) / "opensfm" / "reconstruction.json").read_text())[0]
    centres = []
    for shot in rec["shots"].values():
        rot, _ = cv2.Rodrigues(np.array(shot["rotation"], float))
        centres.append(-rot.T @ np.array(shot["translation"], float))
    c = np.array(centres)
    _, s, vt = np.linalg.svd(c - c.mean(axis=0), full_matrices=False)
    straightness = float(s[0] / max(s.sum(), 1e-9))
    return vt[0] / np.linalg.norm(vt[0]), straightness


def estimate_roll(points, classes, axis, building_code=6, max_points=300_000, max_flatness=0.02, search_deg=60.0,
                  inlier_deg=5.0, seed=0):
    """Angle about `axis` that makes the most planar building patches exactly vertical.

    Counting near-vertical normals (a consensus vote) instead of fitting all normals keeps pitched roofs from
    winning: street facades are far more numerous. The vote works for any tilt within +-search_deg, where a fixed
    'is it roughly vertical' pre-filter would drop the walls of a strongly tilted model (AGZ full run: >25 deg)."""
    m = np.flatnonzero(classes == building_code)
    if len(m) > max_points:
        m = np.random.default_rng(seed).choice(m, max_points, replace=False)
    if len(m) < 2000:
        return None, {"note": "too few building points to estimate roll"}
    n, flat = _normals(points[m])
    n = n[(flat < max_flatness) & (np.abs(n @ axis) < 0.5)]  # planar patches not facing along the street
    if len(n) < 500:
        return None, {"note": "too few planar patches to estimate roll", "planar_normals": int(len(n))}
    tol = np.sin(np.radians(inlier_deg))

    def votes(theta):
        return int(np.sum(np.abs((n @ _axis_rotation(axis, theta).T)[:, 2]) < tol))

    grid = np.radians(np.arange(-search_deg, search_deg + 0.01, 0.5))
    counts = np.array([votes(t) for t in grid])
    best = grid[int(np.argmax(counts))]
    fine = np.radians(np.arange(-1.0, 1.01, 0.05)) + best
    best = float(fine[int(np.argmax([votes(t) for t in fine]))])
    # refine on the consensus walls with a least-squares fit of their residual tilt
    walls = n[np.abs((n @ _axis_rotation(axis, best).T)[:, 2]) < tol]
    local = np.radians(np.arange(-inlier_deg, inlier_deg + 0.01, 0.02)) + best
    best = float(local[int(np.argmin([np.mean((walls @ _axis_rotation(axis, t).T)[:, 2] ** 2) for t in local]))])
    second = np.sort(counts)[-10] if len(counts) >= 10 else 0
    return best, {"planar_normals": int(len(n)), "vertical_walls_before": votes(0.0), "vertical_walls_after": votes(best),
                  "roll_correction_deg": round(float(np.degrees(best)), 2),
                  "consensus_strength": round(float(counts.max() / max(np.median(counts), 1)), 2)}


def level_cloud(in_path, out_path, project, max_roll_deg=60.0, min_straightness=0.8):
    """Write a roll-corrected copy of a classified cloud (rotation about the cloud centre)."""
    import laspy
    axis, straightness = track_axis(project)
    info = {"track_straightness": round(straightness, 3), "track_axis": [round(float(v), 4) for v in axis]}
    if straightness < min_straightness:
        return {**info, "applied": False, "note": "flight path is not a single straight pass; GPS constrains all axes"}
    las = laspy.read(in_path)
    pts = np.column_stack([las.x, las.y, las.z])
    centre = pts.mean(axis=0)
    theta, est = estimate_roll(pts - centre, np.asarray(las.classification), axis)
    info.update(est)
    if theta is None:
        return {**info, "applied": False}
    if abs(np.degrees(theta)) > max_roll_deg:
        return {**info, "applied": False, "note": f"roll above {max_roll_deg} deg looks unreliable; not applied"}
    rot = _axis_rotation(axis, theta)
    new = (pts - centre) @ rot.T + centre
    las.x, las.y, las.z = new[:, 0], new[:, 1], new[:, 2]
    las.write(out_path)
    return {**info, "applied": True, "rotation": rot.tolist(), "centre": centre.tolist(), "output": str(out_path)}
