"""Tests for explicit cleanup command (sih3d cleanup).

Uses temporary directories only – never touches real data/runs or
data/odm_projects.  Verifies audit-identified intermediates only.
"""

import json
import sys
import tempfile
from pathlib import Path

import unittest

# make pipeline importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sih3d.cleanup import DELETABLE_SUBDIRS, get_deletable_paths, is_run_successful, run_cleanup, _dir_size


def _make_successful_run(work_root: Path, projects_root: Path, name: str = "testrun"):
    work = work_root / name
    proj = projects_root / name
    work.mkdir(parents=True, exist_ok=True)
    proj.mkdir(parents=True, exist_ok=True)
    # report marks successful
    (work / "report.json").write_text(json.dumps({"reconstruction": {"ok": True}, "outputs": {"point_cloud": "fake.laz"}}))
    # final cloud marker for is_run_successful fallback
    (proj / "odm_georeferencing").mkdir(parents=True, exist_ok=True)
    (proj / "odm_georeferencing" / "odm_georeferenced_model.laz").write_text("laz")
    (proj / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 32N\n600000 5000000")
    return work, proj


class TestCleanupDeletableSet(unittest.TestCase):
    def test_only_four_deletable(self):
        self.assertEqual(DELETABLE_SUBDIRS, ["opensfm", "openmvs", "odm_filterpoints", "odm_meshing"])

    def test_get_deletable_returns_only_existing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proj = td / "proj"
            proj.mkdir()
            (proj / "opensfm").mkdir()
            (proj / "openmvs").mkdir()
            # leave odm_filterpoints and odm_meshing missing
            found = get_deletable_paths(proj)
            self.assertEqual(set(p.name for p in found), {"opensfm", "openmvs"})

    def test_get_deletable_ignores_protected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            proj = td / "proj"
            proj.mkdir()
            # create protected dirs that must NOT be considered deletable
            for protected in ["images", "odm_georeferencing", "odm_texturing", "odm_dem"]:
                (proj / protected).mkdir()
            (proj / "opensfm").mkdir()
            found = get_deletable_paths(proj)
            self.assertEqual([p.name for p in found], ["opensfm"])


class TestCleanupDryRun(unittest.TestCase):
    def test_dry_run_does_not_delete(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            work_dir, proj_dir = _make_successful_run(work, projects, "run1")
            for sub in DELETABLE_SUBDIRS:
                p = proj_dir / sub
                p.mkdir(parents=True, exist_ok=True)
                (p / "file.bin").write_bytes(b"x" * 1024)
            res = run_cleanup(work, projects, "run1", confirm=False)
            self.assertTrue(res["dry_run"])
            self.assertEqual(len(res["deletable"]), 4)
            for sub in DELETABLE_SUBDIRS:
                self.assertTrue((proj_dir / sub).exists(), f"{sub} should still exist after dry-run")

    def test_confirm_deletes_only_deletable(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            work_dir, proj_dir = _make_successful_run(work, projects, "run2")
            # deletable
            for sub in DELETABLE_SUBDIRS:
                p = proj_dir / sub
                p.mkdir(parents=True, exist_ok=True)
                (p / "a.txt").write_text("hello")
            # protected – must survive
            (work_dir / "frames").mkdir()
            (work_dir / "frames" / "frame_000001.jpg").write_text("keep")
            (proj_dir / "odm_texturing").mkdir(parents=True, exist_ok=True)
            (proj_dir / "odm_texturing" / "odm_textured_model_geo.obj").write_text("mesh")
            (proj_dir / "sih3d_final.laz").write_text("final")
            (work_dir / "report.json").write_text(json.dumps({"reconstruction": {"ok": True}}))
            # keep coords
            # need coords for is_run_successful fallback
            res = run_cleanup(work, projects, "run2", confirm=True)
            self.assertFalse(res["dry_run"])
            self.assertEqual(len(res["deleted"]), 4)
            for sub in DELETABLE_SUBDIRS:
                self.assertFalse((proj_dir / sub).exists(), f"{sub} should be deleted")
            # protected survive
            self.assertTrue((work_dir / "frames" / "frame_000001.jpg").exists())
            self.assertTrue((proj_dir / "odm_texturing" / "odm_textured_model_geo.obj").exists())
            self.assertTrue((proj_dir / "sih3d_final.laz").exists())
            self.assertTrue((proj_dir / "odm_georeferencing" / "coords.txt").exists())
            self.assertTrue((work_dir / "report.json").exists())

    def test_already_missing_graceful(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            work_dir, proj_dir = _make_successful_run(work, projects, "run3")
            # only one deletable exists
            (proj_dir / "opensfm").mkdir(parents=True, exist_ok=True)
            (proj_dir / "opensfm" / "x").write_text("y")
            # others missing – should not raise
            res = run_cleanup(work, projects, "run3", confirm=True)
            self.assertEqual(len(res["deleted"]), 1)
            self.assertFalse((proj_dir / "opensfm").exists())
            # second run – already missing
            res2 = run_cleanup(work, projects, "run3", confirm=True)
            self.assertEqual(res2["deleted"], [])
            self.assertEqual(res2["deletable"], [])

    def test_nothing_to_clean_still_success(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            _make_successful_run(work, projects, "emptyrun")
            # no deletable dirs created
            res = run_cleanup(work, projects, "emptyrun", confirm=False)
            self.assertEqual(res["deletable"], [])
            self.assertEqual(res["total_bytes"], 0)
            res2 = run_cleanup(work, projects, "emptyrun", confirm=True)
            self.assertEqual(res2["deleted"], [])


class TestCleanupVerification(unittest.TestCase):
    def test_refuses_if_not_successful(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            work_dir = work / "bad"
            proj_dir = projects / "bad"
            work_dir.mkdir(parents=True)
            proj_dir.mkdir(parents=True)
            (work_dir / "report.json").write_text(json.dumps({"reconstruction": {"ok": False}}))
            (proj_dir / "opensfm").mkdir()
            with self.assertRaises(SystemExit) as cm:
                run_cleanup(work, projects, "bad", confirm=True)
            self.assertIn("not marked successful", str(cm.exception))
            # ensure not deleted
            self.assertTrue((proj_dir / "opensfm").exists())

    def test_refuses_if_missing_project(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            work.mkdir(parents=True)
            projects.mkdir(parents=True)
            with self.assertRaises(SystemExit):
                run_cleanup(work, projects, "nonexistent", confirm=False)

    def test_success_via_final_cloud_without_report(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            work_dir = work / "norport"
            proj_dir = projects / "norport"
            work_dir.mkdir(parents=True)
            proj_dir.mkdir(parents=True)
            # no report.json, but final cloud exists
            (proj_dir / "odm_georeferencing").mkdir(parents=True)
            (proj_dir / "odm_georeferencing" / "odm_georeferenced_model.laz").write_text("laz")
            (proj_dir / "opensfm").mkdir()
            (proj_dir / "opensfm" / "a").write_text("x")
            # should be considered successful via fallback
            ok, _ = is_run_successful(work_dir, proj_dir)
            self.assertTrue(ok)
            res = run_cleanup(work, projects, "norport", confirm=True)
            self.assertEqual(len(res["deleted"]), 1)

    def test_is_run_successful_checks_report_ok(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "w"
            proj = td / "p"
            work.mkdir()
            proj.mkdir()
            (work / "report.json").write_text(json.dumps({"reconstruction": {"ok": True}}))
            ok, _ = is_run_successful(work, proj)
            self.assertTrue(ok)
            (work / "report.json").write_text(json.dumps({"reconstruction": {"ok": False}}))
            # no final cloud, so should be false
            ok2, _ = is_run_successful(work, proj)
            self.assertFalse(ok2)


class TestCleanupSize(unittest.TestCase):
    def test_size_reporting(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs"
            projects = td / "projects"
            _, proj_dir = _make_successful_run(work, projects, "sizerun")
            p = proj_dir / "opensfm"
            p.mkdir()
            data = b"a" * 2048
            (p / "file1.bin").write_bytes(data)
            (p / "sub").mkdir()
            (p / "sub" / "file2.bin").write_bytes(data)
            sz = _dir_size(p)
            self.assertEqual(sz, 4096)
            res = run_cleanup(work, projects, "sizerun", confirm=False)
            self.assertEqual(res["total_bytes"], 4096)
            self.assertEqual(len(res["sized"]), 1)
            self.assertEqual(res["sized"][0][1], 4096)


if __name__ == "__main__":
    unittest.main()
