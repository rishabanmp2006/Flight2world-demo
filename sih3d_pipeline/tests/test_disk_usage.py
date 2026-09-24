"""Tests for disk-usage report (sih3d disk-usage).

Uses temporary directories only – never touches real data/runs or
data/odm_projects. Verifies size calculation, sorting, missing handling,
no double-counting of the ODM georeferencing model, hardlink (inode)
deduplication in the unique total, and that no files are modified/deleted.
"""

import os
import shutil
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
    return path


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

            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            # Check all expected labels present
            labels = [lbl for lbl, _, _ in items]
            for expected in ["data/runs/runA/frames", "data/odm_projects/runA/opensfm", "viewer/data/runA"]:
                # labels contain run name
                self.assertTrue(any(expected.split("/")[-1] in lbl or lbl == expected for lbl in labels),
                                f"missing {expected} in {labels}")

            # Check sorted largest to smallest
            sizes_sorted = [sz for _, _, sz in items]
            self.assertEqual(sizes_sorted, sorted(sizes_sorted, reverse=True))

            # Logical total is the sum of all listed sizes (including LAZ files)
            # Note: odm_georeferencing dir size includes only its own file (6000), not LAZ at root
            expected_total = sum(sizes.values()) + 9000 + 7000 + 2500
            self.assertEqual(logical, expected_total)
            # No hardlinks in this fixture, so unique total equals logical total
            self.assertEqual(unique, expected_total)

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
            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            # Should return 2 items, not raise
            self.assertEqual(len(items), 2)
            self.assertEqual(logical, 1234 + 4321)
            self.assertEqual(unique, 1234 + 4321)
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
            logical, unique = total
            self.assertEqual(logical, 0)
            self.assertEqual(unique, 0)
            lines = format_report("nonexistent", items, total)
            self.assertTrue(any("No data found" in l for l in lines))
            self.assertTrue(any("Logical total:" in l for l in lines))
            self.assertTrue(any("Unique total:" in l for l in lines))

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
            items, (logical, unique) = collect_disk_usage(work, projects, viewer_alt, name)
            self.assertTrue(any("runB" in lbl for lbl, _, _ in items))
            self.assertEqual(logical, 777)
            self.assertEqual(unique, 777)


class TestNoDoubleCounting(unittest.TestCase):
    """The ODM georeferencing model must be counted exactly once, through
    the recursive odm_georeferencing/ directory target only."""

    def test_odm_georeferenced_model_laz_counted_once_via_directory(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, projects, viewer = td / "runs", td / "projects", td / "viewer"
            name = "geoLaz"
            _write_bytes(projects / name / "odm_georeferencing" / "odm_georeferenced_model.laz", 8000)
            _write_bytes(projects / name / "odm_georeferencing" / "odm_metadata.csv", 500)
            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            labels = [lbl for lbl, _, _ in items]
            # No standalone row for the LAZ file...
            self.assertNotIn(f"data/odm_projects/{name}/odm_georeferencing/odm_georeferenced_model.laz", labels)
            # ...its bytes are fully contained in the directory row
            dir_rows = [sz for lbl, _, sz in items if lbl == f"data/odm_projects/{name}/odm_georeferencing"]
            self.assertEqual(dir_rows, [8000 + 500])
            self.assertEqual(logical, 8000 + 500)
            self.assertEqual(unique, 8000 + 500)


class TestHardlinkDedup(unittest.TestCase):
    """The unique total must count each (st_dev, st_ino) exactly once."""

    def test_hardlinked_file_in_two_counted_targets(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, projects, viewer = td / "runs", td / "projects", td / "viewer"
            name = "hardlinked"
            src = _write_bytes(work / name / "frames" / "k1.jpg", 3000)
            (projects / name / "images").mkdir(parents=True)
            os.link(src, projects / name / "images" / "k1.jpg")
            _write_bytes(projects / name / "opensfm" / "x.txt", 100)
            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            rows = {lbl: sz for lbl, _, sz in items}
            # Both rows remain visible at their full logical size...
            self.assertEqual(rows[f"data/runs/{name}/frames"], 3000)
            self.assertEqual(rows[f"data/odm_projects/{name}/images"], 3000)
            # ...the logical total counts the shared bytes once per row...
            self.assertEqual(logical, 3000 + 3000 + 100)
            # ...but the unique total counts the inode only once.
            self.assertEqual(unique, 3000 + 100)

    def test_copies_with_different_inodes_counted_separately(self):
        # _hardlink_or_copy() falls back to shutil.copy2(): identical
        # content, different inodes -> both must count in the unique total.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, projects, viewer = td / "runs", td / "projects", td / "viewer"
            name = "copied"
            src = _write_bytes(work / name / "frames" / "k1.jpg", 1500)
            (projects / name / "images").mkdir(parents=True)
            shutil.copy2(src, projects / name / "images" / "k1.jpg")
            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            self.assertEqual(logical, 1500 + 1500)
            self.assertEqual(unique, 1500 + 1500)

    def test_hardlinked_classified_filled_pair(self):
        # fill_gaps() no-op path hardlinks sih3d_classified.laz to
        # sih3d_filled.laz (holes.py); both rows must remain visible.
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, projects, viewer = td / "runs", td / "projects", td / "viewer"
            name = "fillnoop"
            src = _write_bytes(projects / name / "sih3d_classified.laz", 4000)
            os.link(src, projects / name / "sih3d_filled.laz")
            _write_bytes(projects / name / "sih3d_final.laz", 2500)
            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            rows = {lbl: sz for lbl, _, sz in items}
            # Both rows remain visible at their full logical size...
            self.assertEqual(rows[f"data/odm_projects/{name}/sih3d_classified.laz"], 4000)
            self.assertEqual(rows[f"data/odm_projects/{name}/sih3d_filled.laz"], 4000)
            self.assertEqual(logical, 4000 + 4000 + 2500)
            # ...the shared inode contributes only once to the unique total.
            self.assertEqual(unique, 4000 + 2500)

    def test_logical_total_is_sum_of_displayed_rows(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, projects, viewer = td / "runs", td / "projects", td / "viewer"
            name = "sumcheck"
            src = _write_bytes(work / name / "frames" / "a.jpg", 2000)
            (projects / name / "images").mkdir(parents=True)
            os.link(src, projects / name / "images" / "a.jpg")
            _write_bytes(viewer / "data" / name / "p.f32", 700)
            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            # Logical total stays the plain sum of the displayed rows...
            self.assertEqual(logical, sum(sz for _, _, sz in items))
            # ...while the unique total is smaller because of the hardlink.
            self.assertLess(unique, logical)
            self.assertEqual(unique, 2000 + 700)

    def test_report_lists_logical_and_unique_totals(self):
        items = [
            ("data/runs/x/frames", Path("/nonexistent/frames"), 3000),
            ("data/odm_projects/x/images", Path("/nonexistent/images"), 3000),
            ("data/odm_projects/x/opensfm", Path("/nonexistent/opensfm"), 100),
        ]
        lines = format_report("x", items, (6100, 3100))
        self.assertTrue(any("Logical total: 6.0 KB (6100 bytes) in 3 entries" in l for l in lines))
        self.assertTrue(any("Unique total:" in l and "3100 bytes" in l and "inode" in l for l in lines))
        # Hardlinked rows are not hidden from the report
        self.assertTrue(any("frames" in l for l in lines))
        self.assertTrue(any("images" in l for l in lines))


class TestCompletedRunBaseline(unittest.TestCase):
    """Deterministic baseline for the measurement path: a miniature of the
    layout a completed `python -m sih3d run` produces (see cmd_run in
    sih3d/__main__.py), with the real hardlink topology (frames/masks ->
    images, fill_gaps no-op classified -> filled).  All sizes are fixed byte
    counts, so the logical/unique totals are exactly reproducible."""

    def test_full_run_layout_logical_vs_unique(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, projects, viewer = td / "runs", td / "projects", td / "viewer"
            name = "baseline"
            w, pr, v = work / name, projects / name, viewer / "data" / name

            # data/runs/<name>: keyframes + AI outputs (images hardlinks these)
            f1 = _write_bytes(w / "frames" / "k1.jpg", 3000)
            f2 = _write_bytes(w / "frames" / "k2.jpg", 2000)
            _write_bytes(w / "labels" / "l1.png", 300)
            _write_bytes(w / "labels" / "l2.png", 300)
            m1 = _write_bytes(w / "masks" / "k1_mask.png", 400)
            m2 = _write_bytes(w / "masks" / "k2_mask.png", 400)
            _write_bytes(w / "keyframes.json", 50)
            _write_bytes(w / "report.json", 60)
            # data/odm_projects/<name>: hardlinked images + ODM stages + AI clouds
            (pr / "images").mkdir(parents=True)
            for src in (f1, f2, m1, m2):
                os.link(src, pr / "images" / src.name)
            _write_bytes(pr / "geo.txt", 80)
            _write_bytes(pr / "opensfm" / "reconstruction.json", 1000)
            _write_bytes(pr / "openmvs" / "mvs.obj", 1200)
            _write_bytes(pr / "odm_filterpoints" / "filtered.laz", 900)
            _write_bytes(pr / "odm_meshing" / "mesh.obj", 700)
            _write_bytes(pr / "odm_georeferencing" / "odm_georeferenced_model.laz", 5000)
            _write_bytes(pr / "odm_texturing" / "odm_textured_model_geo.obj", 2500)
            _write_bytes(pr / "odm_texturing" / "textures" / "0.png", 3000)
            cls = _write_bytes(pr / "sih3d_classified.laz", 4000)
            os.link(cls, pr / "sih3d_filled.laz")  # fill_gaps no-op (holes.py)
            _write_bytes(pr / "sih3d_final.laz", 4500)
            # viewer/data/<name>
            _write_bytes(v / "positions.f32", 800)
            _write_bytes(v / "meta.json", 90)

            items, (logical, unique) = collect_disk_usage(work, projects, viewer, name)
            rows = {lbl: sz for lbl, _, sz in items}
            # All 14 target rows present (10 dirs + 3 root LAZs + viewer)...
            self.assertEqual(len(items), 14)
            self.assertEqual(rows[f"data/runs/{name}/frames"], 5000)
            self.assertEqual(rows[f"data/odm_projects/{name}/images"], 5800)
            # ...odm_georeferenced_model.laz only via its directory row (no double count)...
            self.assertNotIn(f"data/odm_projects/{name}/odm_georeferencing/odm_georeferenced_model.laz", rows)
            self.assertEqual(rows[f"data/odm_projects/{name}/odm_georeferencing"], 5000)
            # ...and hardlinked rows remain visible at their full logical size...
            self.assertEqual(rows[f"data/odm_projects/{name}/sih3d_classified.laz"], 4000)
            self.assertEqual(rows[f"data/odm_projects/{name}/sih3d_filled.laz"], 4000)
            sizes = [sz for _, _, sz in items]
            self.assertEqual(sizes, sorted(sizes, reverse=True))
            # Baseline totals: logical sums the rows (shared bytes counted per row),
            # unique counts each inode once.
            self.assertEqual(logical, 39890)
            self.assertEqual(unique, 30090)
            # Hardlink savings: frames+masks hardlinked into images (5800) +
            # the no-op classified/filled pair (4000).
            self.assertEqual(logical - unique, 9800)
            lines = format_report(name, items, (logical, unique))
            self.assertTrue(any("Logical total: 39.0 KB (39890 bytes) in 14 entries" in l for l in lines))
            self.assertTrue(any("Unique total:" in l and "30090 bytes" in l for l in lines))


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
            self.assertIn("Logical total:", out)
            self.assertIn("Unique total:", out)
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
            self.assertIn("Logical total:", out)
            self.assertIn("Unique total:", out)

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
