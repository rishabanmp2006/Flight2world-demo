"""Facade and roof accuracy against official CityGML LoD2 buildings (e.g. swisstopo swissBUILDINGS3D 3.0).

The GML is streamed straight out of its zip (it can be several GB), keeping only buildings near the reconstruction.
Wall, roof and ground polygons are sampled into dense reference points; every reconstructed point near a building
gets the distance to the nearest official surface of each type. Wall distances are the facade accuracy.

Pass the rigid alignment found against LiDAR (evaluate.cloud_vs_lidar) to measure shape error only, without the
GPS georeferencing offset.
"""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pyproj import Transformer

GML = "{http://www.opengis.net/gml}"
SURFACES = {"WallSurface": "wall", "RoofSurface": "roof", "GroundSurface": "ground"}


def load_buildings(zip_path, bbox, bbox_crs, gml_crs="EPSG:2056"):
    """bbox (minx, miny, maxx, maxy) in bbox_crs. Returns [{id, wall: [polys], roof: [...], ground: [...]}] in gml_crs."""
    to_gml = Transformer.from_crs(bbox_crs, gml_crs, always_xy=True)
    xs, ys = to_gml.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    lo, hi = (min(xs), min(ys)), (max(xs), max(ys))
    buildings = []
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith((".gml", ".xml")))
        with zf.open(member) as f:
            for _, el in ET.iterparse(f, events=("end",)):
                if not el.tag.endswith("}Building"):
                    continue
                b = {"id": el.get(f"{GML}id"), "wall": [], "roof": [], "ground": []}
                inside = False
                for surf in el.iter():
                    kind = SURFACES.get(surf.tag.rsplit("}", 1)[-1])
                    if not kind:
                        continue
                    for pos in surf.iter(f"{GML}posList"):
                        c = np.array(pos.text.split(), float).reshape(-1, 3)
                        b[kind].append(c)
                        if not inside and np.any((c[:, 0] >= lo[0]) & (c[:, 0] <= hi[0]) & (c[:, 1] >= lo[1]) & (c[:, 1] <= hi[1])):
                            inside = True
                if inside:
                    buildings.append(b)
                el.clear()
    return buildings


def _sample_polygon(poly, spacing):
    """Points on a planar 3D polygon (fan triangulation, area-weighted)."""
    p = poly[:-1] if np.allclose(poly[0], poly[-1]) else poly
    out = [p]
    for i in range(1, len(p) - 1):
        a, b, c = p[0], p[i], p[i + 1]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        n = int(area / spacing ** 2)
        if n == 0:
            continue
        r = np.random.default_rng(i).random((n, 2))
        flip = r.sum(1) > 1
        r[flip] = 1 - r[flip]
        out.append(a + r[:, :1] * (b - a) + r[:, 1:] * (c - a))
    return np.vstack(out)


def cloud_vs_buildings(cloud_xyz, cloud_crs, buildings, gml_crs="EPSG:2056", z_offset=-0.111, spacing=0.2,
                       max_distance=3.0, alignment=None):
    """cloud_xyz (n,3) in cloud_crs. alignment: optional (rotation 3x3, translation 3) applied to the cloud first."""
    from scipy.spatial import cKDTree

    to = Transformer.from_crs(gml_crs, cloud_crs, always_xy=True)
    pts = np.asarray(cloud_xyz, float)
    if alignment is not None:
        rot, t = np.asarray(alignment[0]), np.asarray(alignment[1])
        pts = pts @ rot.T + t
    refs = {}
    for kind in ("wall", "roof"):
        samples = [_sample_polygon(poly, spacing) for b in buildings for poly in b[kind]]
        if not samples:
            continue
        s = np.vstack(samples)
        x, y = to.transform(s[:, 0], s[:, 1])
        refs[kind] = cKDTree(np.column_stack([x, y, s[:, 2] + z_offset]))
    if not refs:
        return {"note": "no official building surfaces near the reconstruction"}
    dists = {k: tree.query(pts, distance_upper_bound=max_distance)[0] for k, tree in refs.items()}
    nearest_kind = np.array(list(dists)).take(np.argmin(np.vstack([dists[k] for k in dists]), axis=0))
    out = {"buildings": len(buildings)}
    r = lambda a: round(float(a), 3)
    for k in dists:
        d = dists[k][(nearest_kind == k) & np.isfinite(dists[k])]
        if len(d):
            out[k] = {"points_compared": int(len(d)), "median_m": r(np.median(d)), "p90_m": r(np.percentile(d, 90)),
                      "within_0.3m": r(np.mean(d < 0.3)), "within_0.5m": r(np.mean(d < 0.5))}
    return out


def load_cloud(path, max_points=300_000, seed=0):
    import laspy
    las = laspy.read(path)
    pts = np.column_stack([las.x, las.y, las.z])
    if "inferred" in set(las.point_format.dimension_names):
        pts = pts[np.asarray(las.inferred) == 0]  # accuracy of measured geometry only
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    return pts


def bbox_of(pts, pad=10.0):
    return (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)


def check(project_cloud, cloud_crs, citygml_zip, alignment=None):
    pts = load_cloud(project_cloud)
    buildings = load_buildings(citygml_zip, bbox_of(pts), cloud_crs)
    return cloud_vs_buildings(pts, cloud_crs, buildings, alignment=alignment)
