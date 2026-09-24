"""Web-ready exports for viewer/index.html.

viewer/data/<name>/
  positions.f32   float32 xyz, metres, centred on `centre` (Z up)
  colors.u8       uint8 rgb from the photos
  classes.u8      uint8 LAS class code (AI classification when available)
  views.u8        uint8 number of cameras that saw the point (confidence), 0 when unknown
  meta.json       count, centre (projected CRS), CRS, legend
viewer/models.json  list of available models (mesh and/or points) for the viewer's model picker
"""
import json
from pathlib import Path

import laspy
import numpy as np

from sih3d.semantics import CLASSES


def export_points(name, cloud_path, crs, out_root="viewer", max_points=2_000_000, title=None, note="", mesh_dir=None):
    out = Path(out_root) / "data" / name
    out.mkdir(parents=True, exist_ok=True)
    las = laspy.read(cloud_path)
    n = len(las.points)
    keep = np.sort(np.random.default_rng(0).choice(n, max_points, replace=False)) if n > max_points else slice(None)
    xyz = np.column_stack([np.asarray(las.x)[keep], np.asarray(las.y)[keep], np.asarray(las.z)[keep]])
    centre = xyz.mean(axis=0)
    (xyz - centre).astype(np.float32).tofile(out / "positions.f32")
    rgb = np.column_stack([np.asarray(las.red)[keep], np.asarray(las.green)[keep], np.asarray(las.blue)[keep]])
    if rgb.max() > 255:
        rgb = rgb >> 8
    rgb.astype(np.uint8).tofile(out / "colors.u8")
    np.asarray(las.classification)[keep].astype(np.uint8).tofile(out / "classes.u8")
    dims = set(las.point_format.dimension_names)
    views = np.asarray(las.views)[keep] if "views" in dims else np.zeros(len(xyz))
    np.clip(views, 0, 255).astype(np.uint8).tofile(out / "views.u8")
    inferred = np.asarray(las.inferred)[keep] if "inferred" in dims else np.zeros(len(xyz))
    inferred.astype(np.uint8).tofile(out / "inferred.u8")

    legend = {}
    for cname, code, bgr, _ in CLASSES:
        if cname != "sky" and code not in legend:
            legend[code] = {"name": cname, "rgb": [bgr[2], bgr[1], bgr[0]]}
    legend.setdefault(1, {"name": "unclassified", "rgb": [128, 128, 128]})
    meta = {"name": name, "count": int(len(xyz)), "centre": centre.tolist(), "crs": str(crs),
            "has_views": "views" in dims, "inferred_points": int(inferred.sum()),
            "legend": {str(k): v for k, v in legend.items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    index_path = Path(out_root) / "models.json"
    models = json.loads(index_path.read_text()) if index_path.exists() else []
    models = [m for m in models if m["key"] != name]
    models.append({"key": name, "title": title or name, "note": note, "points": f"data/{name}/",
                   "mesh": f"../{mesh_dir}/" if mesh_dir else None})
    index_path.write_text(json.dumps(models, indent=1))
    return {"points_exported": int(len(xyz)), "of": int(n), "folder": str(out)}
