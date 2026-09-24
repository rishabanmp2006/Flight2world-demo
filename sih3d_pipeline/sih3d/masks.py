"""AI masks for moving objects and sky.

OpenDroneMap hands <image>_mask.png files to OpenMVS (--ignore-mask-label 0): pixels with value 0 produce no depth,
so cars, people, animals and sky leave no ghosts or floating points in the dense cloud and mesh.
(OpenDroneMap's OpenSfM fork deliberately keeps feature points under masks, so camera solving is unaffected.)

Two sources are combined:
  YOLO11-seg instance masks   sharp outlines of COCO moving-object classes
  SegFormer scene labels      sky, vehicles and people the detector missed
"""
from pathlib import Path

import cv2
import numpy as np

MOVING = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "boat", "airplane",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def make_masks(frames_dir, names, out_dir, labels_dir=None, model="models/yolo11n-seg.pt", device=None, conf=0.3,
               dilate_frac=0.008, max_masked=0.85):
    import torch
    from ultralytics import YOLO

    from sih3d.semantics import PERSON, SKY, VEHICLE

    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(model)
    counts, fractions, skipped = {}, [], []
    for name in names:
        img = cv2.imread(str(Path(frames_dir) / name))
        h, w = img.shape[:2]
        ignore = np.zeros((h, w), bool)
        result = yolo(img, device=device, conf=conf, verbose=False, retina_masks=True)[0]
        if result.masks is not None:
            for m, c in zip(result.masks.data.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                label = result.names[int(c)]
                if label in MOVING:
                    ignore |= cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                    counts[label] = counts.get(label, 0) + 1
        if labels_dir:
            lab = cv2.imread(str(Path(labels_dir) / f"{Path(name).stem}_labels.png"), cv2.IMREAD_UNCHANGED)
            if lab is not None:
                lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
                ignore |= np.isin(lab, [SKY, VEHICLE, PERSON])
        k = max(3, int(dilate_frac * max(w, h)) | 1)
        ignore = cv2.dilate(ignore.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        frac = float(ignore.mean())
        if frac > max_masked:  # an almost fully masked view would contribute nothing; leave it unmasked
            skipped.append(name)
            continue
        fractions.append(frac)
        cv2.imwrite(str(out_dir / f"{Path(name).stem}_mask.png"), np.where(ignore, 0, 255).astype(np.uint8))
    return {"detector": Path(model).name, "frames": len(names), "masks_written": len(fractions),
            "mean_masked_share": round(float(np.mean(fractions)), 3) if fractions else 0.0,
            "moving_objects_detected": counts, "left_unmasked_mostly_sky_or_objects": skipped}
