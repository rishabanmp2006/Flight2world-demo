"""AI gap filling for surfaces a single pass saw poorly (occluded or grazing-angle surfaces).

Photogrammetry leaves holes where a surface was seen from too few angles: facades behind trees, ground under
parked cars, walls viewed edge-on. For each keyframe:
  1. Depth Anything V2 predicts relative (inverse) depth for every pixel.
  2. The reconstructed cloud is projected into the frame. Where it exists it gives true metric depth, and a robust
     linear fit maps the AI inverse depth onto it (per-frame scale + shift). Frames whose fit is poor are skipped.
  3. Pixels with no reconstructed point nearby, that are not sky or a moving object, are back-projected with the
     fitted depth, only inside the depth range the fit was supported by.
  4. Filled points are kept only if they extend measured geometry (within `max_extend` metres of it).
Every added point has extra dimension inferred=1, so estimated surfaces are never confused with measured ones.
"""
import json
from pathlib import Path

import cv2
import laspy
import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def _robust_line(x, y, iters=5, keep=0.8):
    m = np.ones(len(x), bool)
    a, b = 0.0, float(np.median(y))
    for _ in range(iters):
        a, b = np.polyfit(x[m], y[m], 1)
        r = np.abs(a * x + b - y)
        m = r <= np.quantile(r, keep)
    return float(a), float(b)


FILLABLE = ("ground", "low vegetation", "high vegetation", "building", "road")


def fill_gaps(project, frames_dir, labels_dir, out_path, cloud_path=None, device=None, cache_dir="models/hf",
              cell=2, stride=4, cover_px=8, max_fit_err=0.10, voxel=0.10, min_gap=0.15, max_extend=1.5,
              max_share=0.15):
    """Defaults tuned on Zurich AGZ: with max_extend=3 m and fit error up to 15%, AI points were 60% of the cloud and
    ~1.5x less accurate than measured ones on facades (1.26 m vs 0.82 m median). Tighter limits fill real holes only."""
    import torch
    from scipy.spatial import cKDTree
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    from sih3d.semantics import CLASSES, NAMES, _project

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    project = Path(project)
    rec = json.loads((project / "opensfm" / "reconstruction.json").read_text())[0]
    off_x, off_y = [float(v) for v in (project / "odm_georeferencing" / "coords.txt").read_text().splitlines()[1].split()[:2]]
    cloud_path = Path(cloud_path or project / "sih3d_classified.laz")
    las = laspy.read(cloud_path)
    pts = np.column_stack([np.asarray(las.x) - off_x, np.asarray(las.y) - off_y, np.asarray(las.z)])
    proc = AutoImageProcessor.from_pretrained(MODEL, cache_dir=cache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL, cache_dir=cache_dir).to(device).eval()
    codes = np.array([c[1] for c in CLASSES], np.uint8)

    add_xyz, add_rgb, add_cls, fit_errors, used = [], [], [], [], 0
    step = max(1, stride // cell)
    for name, shot in rec["shots"].items():
        img = cv2.imread(str(Path(frames_dir) / name))
        lab_full = cv2.imread(str(Path(labels_dir) / f"{Path(name).stem}_labels.png"), cv2.IMREAD_UNCHANGED)
        if img is None or lab_full is None:
            continue
        cam = rec["cameras"][shot["camera"]]
        w, h = cam["width"], cam["height"]
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        bw, bh = w // cell + 1, h // cell + 1

        u, v, z = _project(pts, shot, cam)
        ok = (z > 0.3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[ok], v[ok], z[ok]
        idx = (v // cell).astype(np.int64) * bw + (u // cell).astype(np.int64)
        far_first = np.argsort(-z)
        zbuf = np.full(bw * bh, np.nan)
        zbuf[idx[far_first]] = z[far_first]
        zbuf = zbuf.reshape(bh, bw)
        covered = ~np.isnan(zbuf)
        yy, xx = np.nonzero(covered)
        if len(yy) < 500:
            continue

        with torch.no_grad():
            inputs = proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
            pred = model(**inputs).predicted_depth.unsqueeze(1)
            ai = torch.nn.functional.interpolate(pred, size=(bh, bw), mode="bilinear")[0, 0].float().cpu().numpy()
        true_z = zbuf[yy, xx]
        a, b = _robust_line(ai[yy, xx], 1.0 / true_z)
        with np.errstate(divide="ignore", invalid="ignore"):
            fit_z = 1.0 / (a * ai[yy, xx] + b)
        good = np.isfinite(fit_z) & (fit_z > 0)
        err = float(np.median(np.abs(fit_z[good] - true_z[good]) / true_z[good])) if good.any() else 1.0
        fit_errors.append(err)
        if err > max_fit_err:
            continue
        zmin, zmax = np.percentile(true_z, [2, 98])

        lab = cv2.resize(lab_full, (bw, bh), interpolation=cv2.INTER_NEAREST)
        k = 2 * max(1, cover_px // cell) + 1
        near = cv2.dilate(covered.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        hole = ~near & np.isin(lab, [NAMES.index(n) for n in FILLABLE])
        grid = np.zeros_like(hole)
        grid[::step, ::step] = True
        hy, hx = np.nonzero(hole & grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            zf = 1.0 / (a * ai[hy, hx] + b)
        keep = np.isfinite(zf) & (zf >= zmin) & (zf <= zmax)
        hy, hx, zf = hy[keep], hx[keep], zf[keep]
        if not len(zf):
            used += 1
            continue

        uu, vv = (hx + 0.5) * cell, (hy + 0.5) * cell
        size = max(w, h)
        fx = cam.get("focal_x", cam.get("focal"))
        fy = cam.get("focal_y", fx)
        xn = ((uu + 0.5 - w / 2) / size - cam.get("c_x", 0.0)) / fx
        yn = ((vv + 0.5 - h / 2) / size - cam.get("c_y", 0.0)) / fy
        rot, _ = cv2.Rodrigues(np.array(shot["rotation"], float))
        cam_pts = np.column_stack([xn * zf, yn * zf, zf])
        add_xyz.append((cam_pts - np.array(shot["translation"], float)) @ rot)
        add_rgb.append(img[np.minimum(vv.astype(int), h - 1), np.minimum(uu.astype(int), w - 1)][:, ::-1])
        add_cls.append(codes[lab[hy, hx]])
        used += 1

    stats = {"model": MODEL, "frames_used": used, "median_depth_fit_error": round(float(np.median(fit_errors)), 3)
             if fit_errors else None}
    if not add_xyz:
        las.write(out_path)
        return {**stats, "inferred_points_added": 0, "output": str(out_path)}

    add = np.vstack(add_xyz)
    rgb = np.vstack(add_rgb)
    cls = np.concatenate(add_cls)
    _, first = np.unique(np.floor(add / voxel).astype(np.int64), axis=0, return_index=True)
    add, rgb, cls = add[first], rgb[first], cls[first]
    d, _ = cKDTree(pts).query(add, distance_upper_bound=max_extend)
    keep = np.isfinite(d) & (d > min_gap)  # extend measured surfaces, never float free
    add, rgb, cls, d = add[keep], rgb[keep], cls[keep], d[keep]
    cap = int(max_share * len(pts))
    capped = len(add) > cap
    if capped:  # sparse (preview) clouds look like one big hole: keep the points closest to measured geometry
        nearest = np.argsort(d)[:cap]
        add, rgb, cls = add[nearest], rgb[nearest], cls[nearest]
    stats["capped_at_share"] = max_share if capped else None

    if "inferred" not in set(las.point_format.dimension_names):
        las.add_extra_dim(laspy.ExtraBytesParams(name="inferred", type=np.uint8, description="1 = AI-estimated point"))
    n0, m = len(las.points), len(add)
    out = laspy.LasData(las.header)
    out.points = laspy.ScaleAwarePointRecord.zeros(n0 + m, header=las.header)
    out.x = np.concatenate([np.asarray(las.x), add[:, 0] + off_x])
    out.y = np.concatenate([np.asarray(las.y), add[:, 1] + off_y])
    out.z = np.concatenate([np.asarray(las.z), add[:, 2]])
    for i, band in enumerate(("red", "green", "blue")):
        out[band] = np.concatenate([np.asarray(las[band]), rgb[:, i].astype(np.uint16) * 257])
    out.classification = np.concatenate([np.asarray(las.classification), cls])
    dims = set(las.point_format.dimension_names)
    if "views" in dims:
        out.views = np.concatenate([np.asarray(las.views), np.zeros(m, np.uint16)])
    out.inferred = np.concatenate([np.zeros(n0, np.uint8), np.ones(m, np.uint8)])
    out.write(out_path)
    return {**stats, "measured_points": int(n0), "inferred_points_added": int(m),
            "inferred_share": round(m / (n0 + m), 3), "output": str(out_path)}
