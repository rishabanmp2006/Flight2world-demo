"""AI scene labels -> classified, confidence-scored point cloud.

1. SegFormer (ADE20K, 150 classes) labels every keyframe; its classes are merged into survey classes below.
2. Every point of OpenDroneMap's georeferenced cloud is projected into every keyframe with OpenDroneMap's own
   camera poses and lens model; a per-image depth buffer keeps only points that camera can actually see.
3. Each visible point takes one vote from the label under it. The majority wins when at least `min_votes`
   cameras agree; otherwise OpenDroneMap's geometric class (ground / not ground) is kept.
The number of cameras that saw each point is stored as extra dimension "views" (a confidence measure).
"""
import json
from pathlib import Path

import cv2
import laspy
import numpy as np

MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# name, LAS class code (LAS 1.4), BGR preview colour, ADE20K labels merged into it
CLASSES = [
    ("unlabelled", 1, (128, 128, 128), []),
    ("ground", 2, (70, 120, 160), ["earth", "sand", "field", "land", "mountain", "hill", "rock", "dirt track"]),
    ("low vegetation", 3, (100, 210, 130), ["grass", "flower"]),
    ("high vegetation", 5, (40, 140, 40), ["tree", "plant", "palm"]),
    ("building", 6, (60, 60, 220), ["building", "house", "wall", "skyscraper", "hovel", "tower", "door", "windowpane",
                                    "awning", "roof", "column", "canopy", "booth"]),
    ("water", 9, (200, 130, 40), ["water", "sea", "river", "lake", "pool", "swimming pool", "fountain"]),
    ("road", 11, (150, 150, 150), ["road", "sidewalk", "path", "runway", "floor", "stairs", "step", "stairway"]),
    ("vehicle", 64, (0, 210, 255), ["car", "truck", "bus", "van", "bicycle", "minibike", "boat", "ship", "airplane"]),
    ("person", 65, (255, 0, 255), ["person", "animal"]),
    ("structure", 66, (0, 150, 255), ["fence", "pole", "streetlight", "signboard", "railing", "bridge", "bench",
                                      "traffic light", "bannister", "pier", "flag"]),
    ("sky", 1, (235, 206, 135), ["sky"]),
]
NAMES = [c[0] for c in CLASSES]
SKY, VEHICLE, PERSON = NAMES.index("sky"), NAMES.index("vehicle"), NAMES.index("person")


def label_frames(frames_dir, names, out_dir, device=None, cache_dir="models/hf", scale=0.5):
    """Write <stem>_labels.png (class index per pixel, at `scale` resolution) for every keyframe."""
    import torch
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = AutoImageProcessor.from_pretrained(MODEL, cache_dir=cache_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL, cache_dir=cache_dir).to(device).eval()
    lut = np.zeros(len(model.config.id2label), np.uint8)
    for i, label in model.config.id2label.items():
        for k, (_, _, _, ade) in enumerate(CLASSES):
            if label.strip().lower() in ade:
                lut[int(i)] = k
    totals = np.zeros(len(CLASSES))
    for name in names:
        img = cv2.imread(str(Path(frames_dir) / name))
        h, w = img.shape[:2]
        with torch.no_grad():
            inputs = proc(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), return_tensors="pt").to(device)
            logits = model(**inputs).logits
            size = (int(h * scale), int(w * scale))
            ade = torch.nn.functional.interpolate(logits, size=size, mode="bilinear").argmax(1)[0].cpu().numpy()
        lab = lut[ade]
        cv2.imwrite(str(out_dir / f"{Path(name).stem}_labels.png"), lab)
        totals += np.bincount(lab.ravel(), minlength=len(CLASSES))
    share = totals / max(totals.sum(), 1)
    return {"model": MODEL, "frames": len(names), "pixel_share": {n: round(float(s), 3) for n, s in zip(NAMES, share) if s > 0.001}}


def _project(points, shot, cam):
    """OpenSfM projection -> pixel u, v, depth, for points in the reconstruction frame."""
    rot, _ = cv2.Rodrigues(np.array(shot["rotation"], float))
    pc = points @ rot.T + np.array(shot["translation"], float)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xn, yn = pc[:, 0] / z, pc[:, 1] / z
    w, h = cam["width"], cam["height"]
    size = max(w, h)
    if cam["projection_type"] == "brown":
        r2 = xn * xn + yn * yn
        radial = 1 + cam["k1"] * r2 + cam["k2"] * r2 ** 2 + cam["k3"] * r2 ** 3
        xd = xn * radial + 2 * cam["p1"] * xn * yn + cam["p2"] * (r2 + 2 * xn * xn)
        yd = yn * radial + cam["p1"] * (r2 + 2 * yn * yn) + 2 * cam["p2"] * xn * yn
        u = (cam["focal_x"] * xd + cam["c_x"]) * size - 0.5 + w / 2
        v = (cam["focal_y"] * yd + cam["c_y"]) * size - 0.5 + h / 2
    else:  # perspective
        r2 = xn * xn + yn * yn
        radial = 1 + cam.get("k1", 0) * r2 + cam.get("k2", 0) * r2 ** 2
        u = cam["focal"] * xn * radial * size - 0.5 + w / 2
        v = cam["focal"] * yn * radial * size - 0.5 + h / 2
    return u, v, z


def classify_cloud(project, labels_dir, out_path, min_votes=2, zbuf_scale=0.25, depth_tol=0.01, depth_abs=0.3):
    project, labels_dir = Path(project), Path(labels_dir)
    rec = json.loads((project / "opensfm" / "reconstruction.json").read_text())[0]
    off_x, off_y = [float(v) for v in (project / "odm_georeferencing" / "coords.txt").read_text().splitlines()[1].split()[:2]]
    las = laspy.convert(laspy.read(project / "odm_georeferencing" / "odm_georeferenced_model.laz"),
                        point_format_id=7, file_version="1.4")
    pts = np.column_stack([np.asarray(las.x) - off_x, np.asarray(las.y) - off_y, np.asarray(las.z)])
    n = len(pts)
    votes = np.zeros((n, len(CLASSES)), np.uint16)
    views = np.zeros(n, np.uint16)
    used = 0
    for name, shot in rec["shots"].items():
        lab_path = labels_dir / f"{Path(name).stem}_labels.png"
        if not lab_path.exists():
            continue
        lab = cv2.imread(str(lab_path), cv2.IMREAD_UNCHANGED)
        cam = rec["cameras"][shot["camera"]]
        w, h = cam["width"], cam["height"]
        u, v, z = _project(pts, shot, cam)
        ok = (z > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(ok)
        u, v, z = u[ok], v[ok], z[ok]
        bw, bh = int(w * zbuf_scale) + 1, int(h * zbuf_scale) + 1
        cell = (v * zbuf_scale).astype(np.int64) * bw + (u * zbuf_scale).astype(np.int64)
        zbuf = np.full(bw * bh, np.inf)
        far_first = np.argsort(-z)
        zbuf[cell[far_first]] = z[far_first]  # nearest point per cell is written last
        vis = z <= zbuf[cell] * (1 + depth_tol) + depth_abs
        lh, lw = lab.shape
        li = lab[np.minimum((v[vis] * lh / h).astype(int), lh - 1), np.minimum((u[vis] * lw / w).astype(int), lw - 1)]
        votes[idx[vis], li] += 1  # each point appears once per image, so fancy += is exact
        views[idx[vis]] += 1
        used += 1

    cand = votes.copy()
    cand[:, 0] = 0
    cand[:, SKY] = 0  # sky labels on points are boundary bleed, not a class
    best = cand.argmax(1)
    top = cand[np.arange(n), best]
    codes = np.array([c[1] for c in CLASSES], np.uint8)
    cls = np.asarray(las.classification).copy()
    decided = top >= min_votes
    cls[decided] = codes[best[decided]]
    las.classification = cls
    las.add_extra_dim(laspy.ExtraBytesParams(name="views", type=np.uint16, description="cameras seeing point"))
    las.views = views
    las.write(out_path)

    share = {NAMES[k]: round(float(np.mean(best[decided] == k)), 3) for k in range(len(CLASSES))
             if k not in (0, SKY) and np.any(best[decided] == k)}
    return {"points": int(n), "keyframes_used": used, "labelled_by_ai": round(float(decided.mean()), 3),
            "class_share_of_labelled": share, "median_views": int(np.median(views)),
            "seen_by_2plus_cameras": round(float(np.mean(views >= 2)), 3), "output": str(out_path)}
