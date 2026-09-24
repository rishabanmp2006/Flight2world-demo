#!/usr/bin/env python3
"""Compare buildings in a reconstructed point cloud with official LoD2 building models (SemCityLockeD).

The LoD2 OBJ files use an undocumented local frame, so the reconstruction is aligned to them with a
rigid 2D transform only (rotation + translation, optional mirror for axis conventions), found by
template-matching building masks. Scale is never fitted: whatever size difference remains is the
reconstruction's own metric error.

Reported per matched building: footprint length, width, area and height, model vs LoD2.
Reported overall: scale from distances between building centres, which is insensitive to fuzzy roof edges.

  python scripts/compare_lod.py \
      --cloud data/odm_projects/semcity/odm_georeferencing/odm_georeferenced_model.laz \
      --roof data/semcitylocked/LoD_model/semantic_obj/TUM_CC_lod2_RoofSurface.obj \
      --ground data/semcitylocked/LoD_model/semantic_obj/TUM_CC_lod2_GroundSurface.obj \
      --out data/results/semcity
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import laspy
import numpy as np

MIN_BUILDING_M2 = 50.0
MIN_HEIGHT_M = 3.0


# ---------------------------------------------------------------- LoD2 raster

def read_obj(path):
    verts, faces = [], []
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.array(verts), faces


def rasterize_lod(roof_path, ground_path, cell):
    """Roof and ground height rasters (north-up if the frame is), plus pixel->metre origin."""
    rv, rf = read_obj(roof_path)
    gv, gf = read_obj(ground_path)
    both = np.vstack([rv, gv])
    up = int(np.argmin(both.max(0) - both.min(0)))  # smallest extent = vertical
    sign = 1.0 if rv[:, up].mean() > gv[:, up].mean() else -1.0
    hx, hy = [a for a in range(3) if a != up]
    x0, y1 = both[:, hx].min(), both[:, hy].max()
    w = int(math.ceil((both[:, hx].max() - x0) / cell)) + 1
    h = int(math.ceil((y1 - both[:, hy].min()) / cell)) + 1

    def draw(verts, faces):
        img = np.full((h, w), np.nan, np.float32)
        heights = np.array([sign * verts[fc, up].mean() for fc in faces])
        for k in np.argsort(heights):  # ascending, so the highest surface wins
            pts = np.column_stack([(verts[faces[k], hx] - x0) / cell, (y1 - verts[faces[k], hy]) / cell])
            cv2.fillPoly(img, [np.round(pts).astype(np.int32)], float(heights[k]))
        return img

    return draw(rv, rf), draw(gv, gf), (x0, y1)


# ---------------------------------------------------------------- model raster

def rasterize_cloud(path, cell):
    las = laspy.read(path)
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z).astype(np.float32)
    x0, y1 = x.min(), y.max()
    w, h = int((x.max() - x0) / cell) + 1, int((y1 - y.min()) / cell) + 1
    lin = ((y1 - y) / cell).astype(np.int64) * w + ((x - x0) / cell).astype(np.int64)
    order = np.argsort(z, kind="stable")
    dsm = np.full(h * w, np.nan, np.float32)
    dsm[lin[order]] = z[order]  # ascending z: the last (highest) write per cell remains
    dsm = dsm.reshape(h, w)
    valid = ~np.isnan(dsm)
    # Ground: morphological opening wider than any building on the campus.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(150 / cell) | 1, int(150 / cell) | 1))
    ground = cv2.dilate(cv2.erode(np.where(valid, dsm, 1e6).astype(np.float32), k), k)
    ndsm = np.where(valid, dsm - ground, np.nan)
    coverage = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return ndsm, coverage, (x0, y1), len(z)


# ---------------------------------------------------------------- buildings

def components(mask, cell):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] * cell ** 2 >= MIN_BUILDING_M2]
    return labels, keep


def footprint(labels, k, cell, origin):
    rows, cols = np.nonzero(labels == k)
    (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32))
    x0, y1 = origin
    return {"length_m": (max(a, b) + 1) * cell, "width_m": (min(a, b) + 1) * cell,
            "area_m2": len(rows) * cell ** 2,
            "centre_m": (x0 + (cols.mean() + 0.5) * cell, y1 - (rows.mean() + 0.5) * cell)}


# ---------------------------------------------------------------- alignment

def rotate(img, angle):
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * s + w * c) + 1, int(h * c + w * s) + 1
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_NEAREST), m


def search(lod, bld, cov, angles, mirrors):
    best = None
    for mirror in mirrors:
        b0, c0 = (bld[:, ::-1], cov[:, ::-1]) if mirror else (bld, cov)
        for a in angles:
            tb, m = rotate(np.ascontiguousarray(b0), a)
            tc, _ = rotate(np.ascontiguousarray(c0), a)
            pad = max(tb.shape)
            img = cv2.copyMakeBorder(lod, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            res = np.nan_to_num(cv2.matchTemplate(img, tb, cv2.TM_CCORR_NORMED, mask=tc), nan=0, posinf=0, neginf=0)
            _, score, _, (tx, ty) = cv2.minMaxLoc(res)
            if best is None or score > best["score"]:
                best = {"score": float(score), "angle": float(a), "mirror": mirror,
                        "tx": tx - pad, "ty": ty - pad, "m": m, "width": bld.shape[1]}
    return best


def affine(best):
    """3x3 pixel transform: model raster -> LoD raster (same cell size)."""
    f = np.array([[-1, 0, best["width"] - 1], [0, 1, 0], [0, 0, 1]], float) if best["mirror"] else np.eye(3)
    r = np.vstack([best["m"], [0, 0, 1]])
    t = np.array([[1, 0, best["tx"]], [0, 1, best["ty"]], [0, 0, 1]], float)
    return t @ r @ f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True, type=Path)
    ap.add_argument("--roof", required=True, type=Path)
    ap.add_argument("--ground", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--min-iou", type=float, default=0.5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cell = args.cell

    print("Rasterizing LoD2 ...")
    roof_h, ground_h, lod_origin = rasterize_lod(args.roof, args.ground, cell)
    lod_mask = ~np.isnan(roof_h)
    print("Rasterizing point cloud ...")
    ndsm, coverage, model_origin, n_points = rasterize_cloud(args.cloud, cell)
    model_mask = cv2.morphologyEx((np.nan_to_num(ndsm) >= MIN_HEIGHT_M).astype(np.uint8), cv2.MORPH_OPEN,
                                  np.ones((5, 5), np.uint8))

    print("Aligning (rotation + translation only, no scale) ...")
    half = lambda im: (cv2.resize(im.astype(np.float32), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) > 0.25).astype(np.float32)
    coarse = search(half(lod_mask), half(model_mask), half(coverage), np.arange(0, 360, 2), [False, True])
    fine = search(lod_mask.astype(np.float32), model_mask.astype(np.float32), coverage.astype(np.float32),
                  np.arange(coarse["angle"] - 2, coarse["angle"] + 2.01, 0.25), [coarse["mirror"]])
    a = affine(fine)
    print(f"  best angle {fine['angle']:.2f} deg, mirror {fine['mirror']}, match score {fine['score']:.3f}")

    model_labels, model_keep = components(model_mask, cell)
    lod_labels, lod_keep = components(lod_mask, cell)
    warped = cv2.warpAffine(model_labels.astype(np.float32), a[:2], lod_mask.shape[::-1],
                            flags=cv2.INTER_NEAREST, borderValue=0).astype(np.int32)
    edge = cv2.dilate((coverage == 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge[[0, -1], :] = edge[:, [0, -1]] = True

    pairs = []
    for mk in model_keep:
        wm = warped == mk
        if not wm.any() or (edge & (model_labels == mk)).any():
            continue  # partly outside the flight: its size would be cut off
        hits = np.bincount(lod_labels[wm], minlength=lod_labels.max() + 1)
        hits[0] = 0
        lk = int(hits.argmax())
        if lk not in lod_keep:
            continue
        inter = hits[lk]
        iou = inter / (wm.sum() + (lod_labels == lk).sum() - inter)
        if iou >= args.min_iou:
            pairs.append((iou, mk, lk))
    pairs.sort(reverse=True)
    used_m, used_l, rows = set(), set(), []
    for iou, mk, lk in pairs:
        if mk in used_m or lk in used_l:
            continue
        used_m.add(mk)
        used_l.add(lk)
        m = footprint(model_labels, mk, cell, model_origin)
        l = footprint(lod_labels, lk, cell, lod_origin)
        m["height_m"] = float(np.nanpercentile(ndsm[model_labels == mk], 95))
        lroof, lground = roof_h[lod_labels == lk], ground_h[lod_labels == lk]
        l["height_m"] = float(np.nanmax(lroof) - np.nanmean(lground)) if (~np.isnan(lground)).any() else float("nan")
        rows.append({"iou": round(float(iou), 2), "model": m, "lod2": l})

    if len(rows) < 2:
        raise SystemExit(f"Only {len(rows)} buildings matched; check the alignment picture and inputs")

    # Scale from centre-to-centre distances (fuzzy roof edges cancel out).
    ratios = [math.dist(p["model"]["centre_m"], q["model"]["centre_m"]) / math.dist(p["lod2"]["centre_m"], q["lod2"]["centre_m"])
              for p, q in itertools.combinations(rows, 2)
              if math.dist(p["lod2"]["centre_m"], q["lod2"]["centre_m"]) > 20]
    dist_err = [abs(math.dist(p["model"]["centre_m"], q["model"]["centre_m"]) - math.dist(p["lod2"]["centre_m"], q["lod2"]["centre_m"]))
                for p, q in itertools.combinations(rows, 2)]

    def err(key):
        d = np.array([r["model"][key] - r["lod2"][key] for r in rows])
        pct = np.array([100 * (r["model"][key] - r["lod2"][key]) / r["lod2"][key] for r in rows])
        d, pct = d[~np.isnan(d)], pct[~np.isnan(pct)]
        return {"median_error": round(float(np.median(d)), 2), "median_abs_error": round(float(np.median(np.abs(d))), 2),
                "median_abs_pct": round(float(np.median(np.abs(pct))), 1)}

    summary = {
        "buildings_matched": len(rows),
        "alignment": {k: fine[k] for k in ("angle", "mirror", "score")},
        "scale_from_centre_distances": {"median_ratio": round(float(np.median(ratios)), 4) if ratios else None,
                                        "scale_error_pct": round(100 * (float(np.median(ratios)) - 1), 2) if ratios else None,
                                        "pairs": len(ratios),
                                        "median_abs_distance_error_m": round(float(np.median(dist_err)), 2)},
        "length": err("length_m"), "width": err("width_m"), "height": err("height_m"), "area": err("area_m2"),
        "point_cloud_points": n_points,
        "notes": "LoD2 is generalised (decimetre-level). Footprint length/width include fuzzy roof edges on the "
                 "model side; centre-distance scale does not. Alignment uses rotation+translation only.",
    }
    for r in rows:
        for side in ("model", "lod2"):
            r[side] = {k: (round(v, 2) if isinstance(v, float) else [round(c, 2) for c in v]) for k, v in r[side].items()}
    (args.out / "lod2_comparison.json").write_text(json.dumps({"summary": summary, "buildings": rows}, indent=2))

    # Picture: LoD2 buildings grey, aligned model buildings green, matched ones numbered.
    vis = np.full(lod_mask.shape + (3,), 30, np.uint8)
    vis[lod_mask] = (150, 150, 150)
    wmask = warped > 0
    vis[wmask] = (0.4 * vis[wmask] + 0.6 * np.array([60, 200, 60])).astype(np.uint8)
    for i, (iou, mk, lk) in enumerate([p for p in pairs if p[1] in used_m and p[2] in used_l]):
        ys, xs = np.nonzero(lod_labels == lk)
        cv2.putText(vis, str(i + 1), (int(xs.mean()) - 8, int(ys.mean()) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "lod2_alignment.png"), vis)

    print(json.dumps(summary, indent=2))
    print(f"\n{'#':>2} {'IoU':>4} | {'length model/LoD2':>18} | {'width model/LoD2':>17} | {'height model/LoD2':>18}")
    for i, r in enumerate(rows, 1):
        m, l = r["model"], r["lod2"]
        print(f"{i:>2} {r['iou']:>4} | {m['length_m']:>7} / {l['length_m']:<8} | {m['width_m']:>7} / {l['width_m']:<7} | "
              f"{m['height_m']:>7} / {l['height_m']:<8}")
    print(f"Wrote {args.out}/lod2_comparison.json and lod2_alignment.png")


if __name__ == "__main__":
    main()
