#!/usr/bin/env python3
"""Measure a building on a georeferenced point cloud and compare it with a reference footprint.

Reference = an OpenStreetMap building outline (Overpass JSON). OSM outlines are traced from aerial
imagery, so treat them as roughly 0.5-1 m accurate: good for a first check, not survey-grade truth.

Two different errors are reported, because SIH26158 needs both:
  dimension error  roof length / width on the model vs the reference   -> metric scale accuracy
  centre offset    model roof centre vs reference centre               -> absolute georeferencing accuracy

  python scripts/measure_building.py \
      --cloud data/odm_projects/aukerman/odm_georeferencing/odm_georeferenced_model.laz \
      --reference data/ground_truth/aukerman_osm_buildings.json --way 749959908 --out data/results/aukerman
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import laspy
import numpy as np
from pyproj import CRS, Transformer


def load_reference(path, way_id):
    for e in json.loads(Path(path).read_text())["elements"]:
        if e["id"] == way_id and e.get("geometry"):
            return [(p["lat"], p["lon"]) for p in e["geometry"]], e.get("tags", {})
    sys.exit(f"way {way_id} not found in {path}")


def cloud_crs(las, cloud_path):
    try:
        crs = las.header.parse_crs()
        if crs:
            return crs
    except Exception:
        pass
    coords = Path(cloud_path).parent / "coords.txt"  # OpenDroneMap writes "WGS84 UTM 17N" on line 1
    if coords.exists():
        m = re.match(r"WGS84 UTM (\d+)([NS])", coords.read_text().splitlines()[0])
        if m:
            return CRS.from_epsg((32600 if m.group(2) == "N" else 32700) + int(m.group(1)))
    sys.exit("Cannot determine the point cloud's coordinate system")


def rect(xy):
    """Minimum-area rectangle: centre, long side, short side, angle."""
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.asarray(xy, np.float32))
    return (cx, cy), max(w, h), min(w, h), angle


def polygon_area(xy):
    x, y = np.asarray(xy).T
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud", required=True, type=Path, help="georeferenced .laz/.las point cloud")
    ap.add_argument("--reference", required=True, type=Path, help="Overpass JSON with building outlines")
    ap.add_argument("--way", required=True, type=int, help="OSM way id of the building")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--search", type=float, default=15.0, help="metres searched around the reference (GPS offset)")
    ap.add_argument("--min-height", type=float, default=2.0, help="roof points are this far above ground")
    ap.add_argument("--cell", type=float, default=0.25, help="raster cell size in metres")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    las = laspy.read(args.cloud)
    crs = cloud_crs(las, args.cloud)
    to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    poly_ll, tags = load_reference(args.reference, args.way)
    poly = np.array([to_xy.transform(lon, lat) for lat, lon in poly_ll])
    (rcx, rcy), ref_len, ref_wid, _ = rect(poly)

    # Window around the reference, big enough to absorb GPS-only georeferencing offsets.
    r = max(ref_len, ref_wid) / 2 + args.search
    x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    near = (np.abs(x - rcx) < r) & (np.abs(y - rcy) < r)
    x, y, z = x[near], y[near], z[near]
    if len(z) < 1000:
        sys.exit(f"Only {len(z)} points near the reference building; is the cloud covering it?")
    xmin, ymax, n = rcx - r, rcy + r, int(math.ceil(2 * r / args.cell))
    ix = np.clip(((x - xmin) / args.cell).astype(int), 0, n - 1)
    iy = np.clip(((ymax - y) / args.cell).astype(int), 0, n - 1)

    ground = float(np.percentile(z, 10))  # open park: most of the window is ground
    high = z > ground + args.min_height
    occupancy = np.zeros((n, n), np.uint8)
    occupancy[iy[high], ix[high]] = 1
    occupancy = cv2.morphologyEx(occupancy, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(occupancy, connectivity=8)

    def cell_xy(mask):
        rows, cols = np.nonzero(mask)
        return np.column_stack([xmin + (cols + 0.5) * args.cell, ymax - (rows + 0.5) * args.cell])

    # Pick the raised object nearest the reference that is building-shaped (fills its bounding rectangle);
    # tree canopies are ragged and score low.
    candidates = []
    for k in range(1, count):
        area = stats[k, cv2.CC_STAT_AREA] * args.cell ** 2
        if area < 20:
            continue
        xy = cell_xy(labels == k)
        (cx, cy), length, width, _ = rect(xy)
        fill = area / max((length + args.cell) * (width + args.cell), 1e-6)
        candidates.append({"label": k, "area_m2": area, "fill": fill,
                           "dist_m": math.hypot(cx - rcx, cy - rcy)})
    if not candidates:
        sys.exit("No raised object found near the reference building; try a lower --min-height")
    shaped = [c for c in candidates if c["fill"] >= 0.7] or candidates
    best = min(shaped, key=lambda c: c["dist_m"])
    mask = labels == best["label"]
    xy = cell_xy(mask)
    (mcx, mcy), m_len, m_wid, m_ang = rect(xy)
    m_len, m_wid = m_len + args.cell, m_wid + args.cell  # cell centres sit half a cell inside the edge
    roof_pts = mask[iy, ix] & high
    roof_z = z[roof_pts]
    height = float(np.percentile(roof_z, 95) - ground) if len(roof_z) else float("nan")

    mlon, mlat = to_ll.transform(mcx, mcy)
    rlon, rlat = to_ll.transform(rcx, rcy)
    report = {
        "reference": {"source": f"OpenStreetMap way {args.way}", "tags": tags,
                      "length_m": round(ref_len, 2), "width_m": round(ref_wid, 2),
                      "area_m2": round(float(polygon_area(poly)), 1), "centre_lat_lon": [rlat, rlon]},
        "model": {"length_m": round(m_len, 2), "width_m": round(m_wid, 2), "height_m": round(height, 2),
                  "area_m2": round(best["area_m2"], 1), "roof_points": int(roof_pts.sum()),
                  "rectangle_fill": round(best["fill"], 2), "centre_lat_lon": [mlat, mlon]},
        "errors": {"length_m": round(m_len - ref_len, 2), "length_pct": round(100 * (m_len - ref_len) / ref_len, 1),
                   "width_m": round(m_wid - ref_wid, 2), "width_pct": round(100 * (m_wid - ref_wid) / ref_wid, 1),
                   "centre_offset_m": round(math.hypot(mcx - rcx, mcy - rcy), 2)},
        "settings": {"crs": crs.to_string(), "cell_m": args.cell, "min_height_m": args.min_height,
                     "ground_z": round(ground, 2), "search_m": args.search},
        "other_raised_objects": [{k: round(v, 2) for k, v in c.items() if k != "label"}
                                 for c in candidates if c is not best],
        "notes": "OSM outlines are traced from imagery (~0.5-1 m). Dimension error reflects metric scale; "
                 "centre offset reflects GPS-only georeferencing (no ground control points).",
    }
    (args.out / "building_measurement.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Top-down picture: height map, model rectangle (green), OSM outline in place (red) and moved onto
    # the model centre (yellow) so shape and position errors can be seen separately.
    zmax = np.full((n, n), np.nan)
    np.fmax.at(zmax, (iy, ix), z)
    img = np.nan_to_num((zmax - ground) / max(height, 1.0), nan=0.0)
    img = cv2.cvtColor((np.clip(img, 0, 1.2) / 1.2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    scale = 4
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(pts):
        pts = np.asarray(pts)
        return np.column_stack([(pts[:, 0] - xmin) / args.cell * scale,
                                (ymax - pts[:, 1]) / args.cell * scale]).astype(np.int32)

    box = cv2.boxPoints(cv2.minAreaRect(xy.astype(np.float32)))
    cv2.polylines(img, [px(poly)], True, (60, 60, 255), 2, cv2.LINE_AA)
    cv2.polylines(img, [px(poly - [rcx, rcy] + [mcx, mcy])], True, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.polylines(img, [px(box)], True, (80, 220, 80), 2, cv2.LINE_AA)
    lines = [f"model  {m_len:.2f} x {m_wid:.2f} m, h {height:.1f} m",
             f"OSM    {ref_len:.2f} x {ref_wid:.2f} m",
             f"error  {report['errors']['length_pct']:+.1f}% / {report['errors']['width_pct']:+.1f}%, "
             f"offset {report['errors']['centre_offset_m']:.2f} m"]
    for k, text in enumerate(lines):
        for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
            cv2.putText(img, text, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick, cv2.LINE_AA)
    cv2.imwrite(str(args.out / "building_measurement.png"), img)

    print(json.dumps({k: report[k] for k in ("reference", "model", "errors")}, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out / 'building_measurement.json'} and building_measurement.png")


if __name__ == "__main__":
    main()
