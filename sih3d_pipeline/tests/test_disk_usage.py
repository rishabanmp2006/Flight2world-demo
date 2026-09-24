"""Tests for disk-usage report (sih3d disk-usage).

Uses temporary directories only – never touches real data/runs or
data/odm_projects. Verifies size calculation, sorting, missing handling,
and that no files are modified/deleted.
"""

import json
import sys
import tempfile
from pathlib import Path
import unittest

# Robustly locate repo root and pipeline directory for both
# tests/test_disk_usage.py and sih3d_pipeline/tests/test_disk_usage.py locations
_this = Path(__file__).resolve()
REPO_ROOT = None
PIPE = None
for parent in _this.parents:
    if (parent / "sih3d_pipeline" / "sih3d" / "__main__.py").exists():
        REPO_ROOT = parent
        PIPE = parent / "sih3d_pipeline"
        break
    if (parent / "sih3d" / "__main__.py").exists():
        PIPE = parent
        REPO_ROOT = parent.parent
        break
if REPO_ROOT is None or PIPE is None:
    # Fallback to original logic
    REPO_ROOT = _this.parents[1]
    PIPE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sih3d.disk_usage import _dir_size, _human_size, collect_disk_usage, format_report


def _write_bytes(path: Path, size: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


class TestDiskUsageHelpers(unittest.TestCase):
    def test_human_size(self):
        self.assertEqual(_human_size(0), "0 B")
        self.assertEqual(_human_size(512), "512 B")
        self.assertEqual(_human_size(1024), "1.0 KB")
        self.assertEqual(_human_size(1536), "1.5 KB")
        self.assertEqual(_human_size(1024*1024), "1.0 MB")
        self.assertEqual(_human_size(1024*1024*1024), "1.00 GB")

    def test_dir_size(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "a" / "b").mkdir(parents=True)
            _write_bytes(td / "a" / "file1.bin", 100)
            _write_bytes(td / "a" / "b" / "file2.bin", 200)
            self.assertEqual(_dir_size(td / "a"), 300)
            self.assertEqual(_dir_size(td / "nonexistent"), 0)

    def test_dir_size_missing_graceful(self):
        # Should not raise for missing path
        self.assertEqual(_dir_size(Path("/tmp/nonexistent_disk_usage_test_123456")), 0)


class TestCollectDiskUsage(unittest.TestCase):
    def test_all_major_dirs_and_laz_sorted(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            name = "runA"
            # Create each major dir with distinct sizes
            sizes = {
                "frames": 5000,
                "labels": 3000,
                "masks": 1000,
                "images": 4000,
                "opensfm": 8000,
                "openmvs": 2000,
                "odm_filterpoints": 1500,
                "odm_meshing": 1200,
                "odm_georeferencing": 6000,
                "odm_texturing": 3500,
            }
            # work dirs
            for sub in ["frames", "labels", "masks"]:
                _write_bytes(work / name / sub / "f.bin", sizes[sub])
            # project dirs
            for sub in ["images", "opensfm", "openmvs", "odm_filterpoints", "odm_meshing", "odm_georeferencing", "odm_texturing"]:
                _write_bytes(projects / name / sub / "f.bin", sizes[sub])
            # final LAZ files at project root
            _write_bytes(projects / name / "sih3d_final.laz", 9000)
            _write_bytes(projects / name / "sih3d_classified.laz", 7000)
            # viewer
            _write_bytes(viewer / "data" / name / "positions.f32", 2500)

            items, total = collect_disk_usage(work, projects, viewer, name)
            # Check all expected labels present
            labels = [lbl for lbl, _, _ in items]
            for expected in ["data/runs/runA/frames", "data/odm_projects/runA/opensfm", "viewer/data/runA"]:
                # labels contain run name
                self.assertTrue(any(expected.split("/")[-1] in lbl or lbl == expected for lbl in labels),
                                f"missing {expected} in {labels}")

            # Check sorted largest to smallest
            sizes_sorted = [sz for _, _, sz in items]
            self.assertEqual(sizes_sorted, sorted(sizes_sorted, reverse=True))

            # Total should be sum of all listed sizes (including LAZ files)
            # Note: odm_georeferencing dir size includes only its own file (6000), not LAZ at root
            expected_total = sum(sizes.values()) + 9000 + 7000 + 2500
            self.assertEqual(total, expected_total)

            # Ensure final LAZ files are listed
            laz_labels = [lbl for lbl, _, _ in items if "laz" in lbl.lower()]
            self.assertTrue(any("sih3d_final.laz" in lbl for lbl in laz_labels))
            self.assertTrue(any("sih3d_classified.laz" in lbl for lbl in laz_labels))

    def test_missing_directories_graceful(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            name = "partial"
            # Only create a subset
            _write_bytes(work / name / "frames" / "a.jpg", 1234)
            _write_bytes(projects / name / "odm_georeferencing" / "f.bin", 4321)
            items, total = collect_disk_usage(work, projects, viewer, name)
            # Should return 2 items, not raise
            self.assertEqual(len(items), 2)
            self.assertEqual(total, 1234 + 4321)
            # Sorted descending
            self.assertGreaterEqual(items[0][2], items[1][2])

    def test_no_data_graceful(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            work.mkdir()
            projects.mkdir()
            viewer.mkdir()
            items, total = collect_disk_usage(work, projects, viewer, "nonexistent")
            self.assertEqual(items, [])
            self.assertEqual(total, 0)
            lines = format_report("nonexistent", items, total)
            self.assertTrue(any("No data found" in l for l in lines))
            self.assertTrue(any("Total:" in l for l in lines))

    def test_does_not_modify_files(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            name = "immutable"
            p = work / name / "frames"
            p.mkdir(parents=True)
            f = p / "frame.jpg"
            _write_bytes(f, 1000)
            mtime_before = f.stat().st_mtime
            size_before = f.stat().st_size
            items, total = collect_disk_usage(work, projects, viewer, name)
            # Ensure file still exists and unchanged
            self.assertTrue(f.exists())
            self.assertEqual(f.stat().st_size, size_before)
            self.assertEqual(f.stat().st_mtime, mtime_before)
            # Also test format_report doesn't modify
            format_report(name, items, total)
            self.assertTrue(f.exists())

    def test_viewer_alternative_path(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            # viewer root passed as viewer/data (alternative)
            viewer_alt = td / "viewer" / "data"
            name = "runB"
            _write_bytes(viewer_alt / name / "positions.f32", 777)
            items, total = collect_disk_usage(work, projects, viewer_alt, name)
            self.assertTrue(any("runB" in lbl for lbl, _, _ in items))
            self.assertEqual(total, 777)


class TestDiskUsageCLI(unittest.TestCase):
    def test_cli_disk_usage_output(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            name = "cli_test"
            _write_bytes(work / name / "frames" / "a.bin", 2048)
            _write_bytes(projects / name / "opensfm" / "b.bin", 4096)
            _write_bytes(viewer / "data" / name / "c.bin", 1024)
            # Run CLI via sih3d_pipeline as cwd
            result = subprocess.run(
                [sys.executable, "-m", "sih3d", "disk-usage", "--name", name,
                 "--work", str(work), "--projects", str(projects), "--viewer", str(viewer)],
                capture_output=True, text=True, cwd=str(PIPE)
            )
            self.assertEqual(result.returncode, 0)
            # log goes to stderr
            out = result.stderr + result.stdout
            self.assertIn("Disk usage for run 'cli_test'", out)
            self.assertIn("Total:", out)
            # Should be sorted: opensfm (4096) before frames (2048) before viewer (1024)
            # Check order by finding indices
            idx_opensfm = out.find("opensfm")
            idx_frames = out.find("frames")
            idx_viewer = out.find("viewer")
            self.assertTrue(idx_opensfm != -1 and idx_frames != -1 and idx_viewer != -1)
            self.assertTrue(idx_opensfm < idx_frames < idx_viewer)

    def test_cli_missing_graceful(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            work.mkdir()
            projects.mkdir()
            viewer.mkdir()
            result = subprocess.run(
                [sys.executable, "-m", "sih3d", "disk-usage", "--name", "ghost",
                 "--work", str(work), "--projects", str(projects), "--viewer", str(viewer)],
                capture_output=True, text=True, cwd=str(PIPE)
            )
            self.assertEqual(result.returncode, 0)
            out = result.stderr + result.stdout
            self.assertIn("No data found", out)
            self.assertIn("Total:", out)

    def test_cli_does_not_delete(self):
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            viewer = td / "viewer"
            name = "nodelete"
            p = work / name / "frames"
            p.mkdir(parents=True)
            f = p / "keep.jpg"
            _write_bytes(f, 500)
            # run disk-usage
            subprocess.run(
                [sys.executable, "-m", "sih3d", "disk-usage", "--name", name,
                 "--work", str(work), "--projects", str(projects), "--viewer", str(viewer)],
                capture_output=True, text=True, cwd=str(PIPE)
            )
            self.assertTrue(f.exists())
