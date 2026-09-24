"""
Deterministic unit tests for odm.setup_project() hardlink-with-copy-fallback.

Inspected implementation:
  - sih3d_pipeline/sih3d/odm.py  (_hardlink_or_copy helper, setup_project)

Covered behaviour:
  * frame images are hard-linked into data/odm_projects/<name>/images/
  * linked files share inode + data with the source; sources stay readable in place
  * optional masks (<stem>_mask.png) are hard-linked the same way
  * OSError from Path.hardlink_to() (e.g. cross-device) falls back to shutil.copy2
    for both frames and masks
  * geo.txt format/content is byte-identical to the pre-change implementation
  * repeated setup removes stale linked images/masks; link sources survive
  * no unrelated files or sibling projects are modified; no symlinks are used

All tests are fast, deterministic, CPU-only and avoid:
  GPU, Docker, ODM, COLMAP/OpenMVS, real reconstruction runs, network.
  Only temporary directories are used.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# make pipeline importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sih3d.odm import _hardlink_or_copy, setup_project

JPEG_BYTES = b"\xff\xd8\xff\xe0-fake-jpeg-payload-"
MASK_BYTES = b"\x89PNG-fake-mask-payload-"


def _fused(n):
    """GPS fusion result aligned with records: lat/lon/alt/h_std/v_std sequences."""
    return {
        "lat": [12.9716 + i * 1e-5 for i in range(n)],
        "lon": [77.5946 + i * 1e-5 for i in range(n)],
        "alt": [100.0 + i * 0.5 for i in range(n)],
        "h_std": [0.10] * n,
        "v_std": [0.20] * n,
    }


def _write(dirpath, name, payload):
    p = Path(dirpath) / name
    p.write_bytes(payload)
    return p


def _links_to(src, dst):
    """True if dst is a real hard link to src (same device + inode, not a symlink)."""
    src, dst = Path(src), Path(dst)
    if not (src.is_file() and dst.is_file()) or dst.is_symlink():
        return False
    s, d = os.stat(src), os.stat(dst)
    return (s.st_dev, s.st_ino) == (d.st_dev, d.st_ino)


def _make_frames(frames_dir, names):
    return [_write(frames_dir, n, JPEG_BYTES + n.encode()) for n in names]


class TestHardlinkOrCopyHelper(unittest.TestCase):
    def test_hardlinks_on_same_filesystem(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _write(td, "src.jpg", JPEG_BYTES)
            dst = td / "dst.jpg"
            _hardlink_or_copy(src, dst)
            self.assertTrue(_links_to(src, dst))
            self.assertGreaterEqual(os.stat(src).st_nlink, 2)
            self.assertEqual(dst.read_bytes(), JPEG_BYTES)

    def test_falls_back_to_copy_on_oserror(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = _write(td, "src.jpg", JPEG_BYTES)
            dst = td / "dst.jpg"
            err = OSError(18, "Invalid cross-device link")
            with mock.patch.object(Path, "hardlink_to", side_effect=err):
                _hardlink_or_copy(src, dst)
            # a real, independent copy: same data, different inode
            self.assertTrue(dst.is_file())
            self.assertEqual(dst.read_bytes(), JPEG_BYTES)
            self.assertNotEqual(os.stat(src).st_ino, os.stat(dst).st_ino)
            self.assertFalse(_links_to(src, dst))


class TestSetupProjectHardlinks(unittest.TestCase):
    def test_frames_hardlinked_into_images_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            names = ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
            srcs = _make_frames(frames, names)
            records = [{"name": n} for n in names]
            project = setup_project(td / "projects", "run1", frames, records, _fused(len(names)))
            self.assertEqual(Path(project), td / "projects" / "run1")
            images = Path(project) / "images"
            self.assertEqual(sorted(p.name for p in images.iterdir()), sorted(names))
            for src, n in zip(srcs, names):
                dst = images / n
                self.assertTrue(_links_to(src, dst), f"{dst} is not a hard link to {src}")
                self.assertFalse(dst.is_symlink(), "symlinks must never be used")
                self.assertEqual(dst.read_bytes(), src.read_bytes())

    def test_linked_frames_share_inode_and_data(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            srcs = _make_frames(frames, ["a.jpg", "b.jpg"])
            project = setup_project(td / "projects", "run1", frames, [{"name": s.name} for s in srcs], _fused(2))
            images = Path(project) / "images"
            for src in srcs:
                s, d = os.stat(src), os.stat(images / src.name)
                self.assertEqual((s.st_dev, s.st_ino), (d.st_dev, d.st_ino))
                self.assertGreaterEqual(s.st_nlink, 2)
                self.assertEqual(s.st_size, d.st_size)
                self.assertEqual((images / src.name).read_bytes(), src.read_bytes())

    def test_source_remains_readable_after_linking(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            srcs = _make_frames(frames, ["a.jpg", "b.jpg"])
            project = setup_project(td / "projects", "run1", frames, [{"name": s.name} for s in srcs], _fused(2))
            for src in srcs:
                self.assertTrue(src.is_file(), "source must stay in place")
                self.assertGreaterEqual(os.stat(src).st_nlink, 2)
                self.assertEqual(src.read_bytes(), JPEG_BYTES + src.name.encode())
                with open(src, "rb") as fh:
                    self.assertEqual(fh.read(4), JPEG_BYTES[:4])
            # removing the links must never touch the sources
            import shutil
            shutil.rmtree(Path(project) / "images")
            for src in srcs:
                self.assertTrue(src.is_file())
                self.assertEqual(src.read_bytes(), JPEG_BYTES + src.name.encode())

    def test_masks_hardlinked(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            masks = td / "masks"
            masks.mkdir()
            names = ["frame_0000.jpg", "frame_0001.jpg", "frame_0002.jpg"]
            _make_frames(frames, names)
            mask_names = ["frame_0000_mask.png", "frame_0001_mask.png"]
            msrcs = [_write(masks, m, MASK_BYTES) for m in mask_names]  # frame_0002 has no mask
            project = setup_project(td / "projects", "run1", frames, [{"name": n} for n in names],
                                    _fused(len(names)), masks_dir=masks)
            images = Path(project) / "images"
            self.assertEqual(sorted(p.name for p in images.iterdir()),
                             sorted(names + mask_names))
            for src, m in zip(msrcs, mask_names):
                dst = images / m
                self.assertTrue(_links_to(src, dst), f"mask {dst} is not a hard link to {src}")
                self.assertFalse(dst.is_symlink())
                self.assertEqual(dst.read_bytes(), MASK_BYTES)
            self.assertFalse((images / "frame_0002_mask.png").exists())

    def test_fallback_for_frames_when_hardlink_raises(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            names = ["frame_0000.jpg", "frame_0001.jpg"]
            srcs = _make_frames(frames, names)
            err = OSError(18, "Invalid cross-device link")
            with mock.patch.object(Path, "hardlink_to", side_effect=err):
                project = setup_project(td / "projects", "run1", frames, [{"name": n} for n in names], _fused(2))
            images = Path(project) / "images"
            self.assertEqual(sorted(p.name for p in images.iterdir()), sorted(names))
            for src, n in zip(srcs, names):
                dst = images / n
                # fallback copy: identical data, independent inode, source untouched
                self.assertTrue(dst.is_file())
                self.assertEqual(dst.read_bytes(), src.read_bytes())
                self.assertNotEqual(os.stat(src).st_ino, os.stat(dst).st_ino)
                self.assertEqual(src.read_bytes(), JPEG_BYTES + n.encode())

    def test_fallback_for_masks_when_hardlink_raises(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            masks = td / "masks"
            masks.mkdir()
            names = ["frame_0000.jpg", "frame_0001.jpg"]
            _make_frames(frames, names)
            mask_names = ["frame_0000_mask.png", "frame_0001_mask.png"]
            msrcs = [_write(masks, m, MASK_BYTES) for m in mask_names]
            err = OSError(18, "Invalid cross-device link")
            with mock.patch.object(Path, "hardlink_to", side_effect=err):
                project = setup_project(td / "projects", "run1", frames, [{"name": n} for n in names],
                                        _fused(2), masks_dir=masks)
            images = Path(project) / "images"
            for src, m in zip(msrcs, mask_names):
                dst = images / m
                self.assertTrue(dst.is_file())
                self.assertEqual(dst.read_bytes(), MASK_BYTES)
                self.assertNotEqual(os.stat(src).st_ino, os.stat(dst).st_ino)
                self.assertFalse(_links_to(src, dst))

    def test_geo_txt_format_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            _make_frames(frames, ["a.jpg", "b.jpg"])
            fused = {
                "lat": [12.9716, 12.97161],
                "lon": [77.5946, 77.59461],
                "alt": [100.0, 99.25],
                "h_std": [0.10, 0.01],  # second value below the 0.05 clamp
                "v_std": [0.20, 0.02],
            }
            project = setup_project(td / "projects", "run1", frames, [{"name": "a.jpg"}, {"name": "b.jpg"}], fused)
            expected = (
                "EPSG:4326\n"
                "a.jpg 77.594600000 12.971600000 100.000 nan nan nan 0.100 0.200\n"
                "b.jpg 77.594610000 12.971610000 99.250 nan nan nan 0.050 0.050\n"
            )
            self.assertEqual((Path(project) / "geo.txt").read_text(), expected)

    def test_repeated_setup_removes_stale_images_and_masks(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            masks = td / "masks"
            masks.mkdir()
            names = ["f_a.jpg", "f_b.jpg"]
            _make_frames(frames, names)
            _write(masks, "f_a_mask.png", MASK_BYTES)
            _write(masks, "f_b_mask.png", MASK_BYTES)
            setup_project(td / "projects", "run1", frames, [{"name": n} for n in names], _fused(2), masks_dir=masks)
            # second run: f_a dropped, f_c added -> stale image AND stale mask must go
            second = ["f_b.jpg", "f_c.jpg"]
            _make_frames(frames, ["f_c.jpg"])
            project = setup_project(td / "projects", "run1", frames, [{"name": n} for n in second],
                                    _fused(2), masks_dir=masks)
            images = Path(project) / "images"
            self.assertEqual(sorted(p.name for p in images.iterdir()),
                             sorted(["f_b.jpg", "f_b_mask.png", "f_c.jpg"]))
            self.assertFalse((images / "f_a.jpg").exists())
            self.assertFalse((images / "f_a_mask.png").exists())
            # removing the stale link must leave the frame source intact
            self.assertTrue((frames / "f_a.jpg").is_file())
            self.assertEqual((frames / "f_a.jpg").read_bytes(), JPEG_BYTES + b"f_a.jpg")
            # surviving frame is still a live hard link, geo.txt rewritten for the new set
            self.assertTrue(_links_to(frames / "f_b.jpg", images / "f_b.jpg"))
            text = (Path(project) / "geo.txt").read_text()
            self.assertIn("f_b.jpg", text)
            self.assertIn("f_c.jpg", text)
            self.assertNotIn("f_a.jpg", text)

    def test_unrelated_files_and_projects_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frames = td / "frames"
            frames.mkdir()
            masks = td / "masks"
            masks.mkdir()
            names = ["frame_0000.jpg", "frame_0001.jpg"]
            _make_frames(frames, names)
            _write(masks, "frame_0000_mask.png", MASK_BYTES)
            # bystanders that setup_project must never touch
            bystanders = [_write(frames, "not_a_keyframe.jpg", b"keep-me"),
                          _write(frames, "notes.txt", b"keep-me-too"),
                          _write(masks, "misc_mask.png", b"keep-the-mask")]
            before = {p: os.stat(p) for p in bystanders}
            # sibling project that must stay intact
            sibling = td / "projects" / "other"
            sibling.mkdir(parents=True)
            (sibling / "sentinel.txt").write_text("do-not-touch")
            project = setup_project(td / "projects", "run1", frames, [{"name": n} for n in names],
                                    _fused(2), masks_dir=masks)
            for p in bystanders:
                after = os.stat(p)
                self.assertEqual((before[p].st_dev, before[p].st_ino, before[p].st_size, before[p].st_mtime_ns),
                                 (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                                 f"{p} was modified")
            # project contains exactly images/ + geo.txt; images exactly the linked files
            self.assertEqual(sorted(p.name for p in Path(project).iterdir()), ["geo.txt", "images"])
            self.assertEqual(sorted(p.name for p in (Path(project) / "images").iterdir()),
                             sorted(["frame_0000.jpg", "frame_0000_mask.png", "frame_0001.jpg"]))
            self.assertEqual((sibling / "sentinel.txt").read_text(), "do-not-touch")


if __name__ == "__main__":
    unittest.main()
