"""Tests for explicit cleanup command (sih3d cleanup).

Uses temporary directories only – never touches real data/runs or
data/odm_projects.  Verifies audit-identified intermediates only, plus the
superseded AI clouds (sih3d_classified.laz / sih3d_filled.laz), which are
deletable only when the run is successful AND levelling produced
sih3d_final.laz.
"""

import json
import os
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

from sih3d.cleanup import (DELETABLE_SUBDIRS, SUPERSEDED_LAZS, collect_cleanup_info,
                           get_deletable_paths, is_run_successful, run_cleanup, _dir_size)


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
    def test_deletable_set_includes_odm_outputs(self):
        self.assertEqual(DELETABLE_SUBDIRS,
                         ["opensfm", "openmvs", "odm_filterpoints", "odm_meshing",
                          "odm_texturing_25d", "odm_orthophoto", "odm_dem", "odm_report"])

    def test_odm_texturing_25d_in_deletable_set(self):
        # the 2.5D textured mesh intermediate joins the audit-approved set;
        # odm_texturing/ (the delivered 3D mesh) must stay out of it
        self.assertIn("odm_texturing_25d", DELETABLE_SUBDIRS)
        self.assertNotIn("odm_texturing", DELETABLE_SUBDIRS)

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
            for protected in ["images", "odm_georeferencing", "odm_texturing"]:
                (proj / protected).mkdir()
            # odm_dem is an ODM map output: deletable like the other intermediates
            (proj / "odm_dem").mkdir()
            (proj / "opensfm").mkdir()
            found = get_deletable_paths(proj)
            self.assertEqual([p.name for p in found], ["opensfm", "odm_dem"])


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
            self.assertEqual(len(res["deletable"]), len(DELETABLE_SUBDIRS))
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
            self.assertEqual(len(res["deleted"]), len(DELETABLE_SUBDIRS))
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


def _make_leveled_run(work_root: Path, projects_root: Path, name: str = "leveled"):
    """Successful run fixture with the full LAZ chain: classified, filled and
    sih3d_final.laz (i.e. levelling was applied)."""
    work, proj = _make_successful_run(work_root, projects_root, name)
    (proj / "sih3d_classified.laz").write_bytes(b"classified-cloud")
    (proj / "sih3d_filled.laz").write_bytes(b"filled-cloud")
    (proj / "sih3d_final.laz").write_bytes(b"final-cloud")
    return work, proj


def _make_dirs(proj: Path):
    for sub in DELETABLE_SUBDIRS:
        p = proj / sub
        p.mkdir(parents=True, exist_ok=True)
        (p / "a.txt").write_text("hello")


class TestSupersededLazCleanup(unittest.TestCase):
    """sih3d_classified.laz / sih3d_filled.laz become deletable only when BOTH
    gates hold: is_run_successful() and sih3d_final.laz exists."""

    def test_superseded_constant(self):
        self.assertEqual(SUPERSEDED_LAZS, ["sih3d_classified.laz", "sih3d_filled.laz"])

    # -- 1a. successful run + final exists: reported as deletable -------------

    def test_eligible_when_successful_and_final_exists(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "ok1")
            names = {p.name for p in get_deletable_paths(proj)}
            self.assertIn("sih3d_classified.laz", names)
            self.assertIn("sih3d_filled.laz", names)
            listed, total, sized = collect_cleanup_info(proj)
            listed_names = {p.name for p in listed}
            self.assertIn("sih3d_classified.laz", listed_names)
            self.assertIn("sih3d_filled.laz", listed_names)
            # real file sizes are reported, not 0
            by_name = {p.name: s for p, s in sized}
            self.assertEqual(by_name["sih3d_classified.laz"], len(b"classified-cloud"))
            self.assertEqual(by_name["sih3d_filled.laz"], len(b"filled-cloud"))
            self.assertGreater(total, 0)

    # -- 1b/7. dry-run lists them but deletes nothing -------------------------

    def test_dry_run_lists_but_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "ok2")
            _make_dirs(proj)
            res = run_cleanup(work, td / "projects", "ok2", confirm=False)
            self.assertTrue(res["dry_run"])
            deletable_names = {Path(p).name for p in res["deletable"]}
            self.assertIn("sih3d_classified.laz", deletable_names)
            self.assertIn("sih3d_filled.laz", deletable_names)
            self.assertEqual(len(res["deletable"]), len(DELETABLE_SUBDIRS) + 2)
            # nothing deleted
            self.assertTrue((proj / "sih3d_classified.laz").exists())
            self.assertTrue((proj / "sih3d_filled.laz").exists())
            self.assertTrue((proj / "sih3d_final.laz").exists())
            for sub in DELETABLE_SUBDIRS:
                self.assertTrue((proj / sub).exists())

    # -- 1c. --yes deletes exactly the two intermediates ----------------------

    def test_confirm_deletes_both_and_protects_deliverables(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "ok3")
            _make_dirs(proj)
            (proj / "odm_texturing").mkdir(parents=True, exist_ok=True)
            (proj / "odm_texturing" / "odm_textured_model_geo.obj").write_text("mesh")
            res = run_cleanup(work, td / "projects", "ok3", confirm=True)
            self.assertFalse(res["dry_run"])
            deleted_names = {Path(p).name for p in res["deleted"]}
            self.assertIn("sih3d_classified.laz", deleted_names)
            self.assertIn("sih3d_filled.laz", deleted_names)
            # existing dir targets deleted exactly as before
            for sub in DELETABLE_SUBDIRS:
                self.assertIn(sub, deleted_names)
                self.assertFalse((proj / sub).exists())
            # the two intermediates are gone
            self.assertFalse((proj / "sih3d_classified.laz").exists())
            self.assertFalse((proj / "sih3d_filled.laz").exists())
            # deliverables and protected outputs survive
            self.assertTrue((proj / "sih3d_final.laz").exists())
            self.assertTrue((proj / "odm_georeferencing" / "odm_georeferenced_model.laz").exists())
            self.assertTrue((proj / "odm_georeferencing" / "coords.txt").exists())
            self.assertTrue((proj / "odm_texturing" / "odm_textured_model_geo.obj").exists())
            self.assertTrue((work / "report.json").exists())

    # -- 2. successful run but final.laz missing (levelling not applied) ------

    def test_not_eligible_when_final_missing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "nolevel")
            # levelling did not run/apply: no sih3d_final.laz -> filled (or
            # classified) is the delivered cloud and must be kept
            (proj / "sih3d_classified.laz").write_bytes(b"classified-cloud")
            (proj / "sih3d_filled.laz").write_bytes(b"filled-cloud")
            self.assertFalse((proj / "sih3d_final.laz").exists())
            names = {p.name for p in get_deletable_paths(proj)}
            self.assertNotIn("sih3d_classified.laz", names)
            self.assertNotIn("sih3d_filled.laz", names)
            res = run_cleanup(td / "runs", td / "projects", "nolevel", confirm=True)
            deleted_names = {Path(p).name for p in res["deleted"]}
            self.assertNotIn("sih3d_classified.laz", deleted_names)
            self.assertNotIn("sih3d_filled.laz", deleted_names)
            self.assertTrue((proj / "sih3d_classified.laz").exists())
            self.assertTrue((proj / "sih3d_filled.laz").exists())

    # -- 3. final.laz exists but run unsuccessful ------------------------------

    def test_not_eligible_when_run_unsuccessful(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "bad1")
            # corrupt report -> is_run_successful False even though sih3d_final.laz exists
            # (every real LAZ is a success marker, so an unreadable report is the
            # deterministic way to make the run unsuccessful here)
            (work / "report.json").write_text("{not valid json")
            ok, reason = is_run_successful(work, proj)
            self.assertFalse(ok)
            with self.assertRaises(SystemExit) as cm:
                run_cleanup(td / "runs", td / "projects", "bad1", confirm=True)
            self.assertIn("not marked successful", str(cm.exception))
            # neither file touched
            self.assertTrue((proj / "sih3d_classified.laz").exists())
            self.assertTrue((proj / "sih3d_filled.laz").exists())
            self.assertTrue((proj / "sih3d_final.laz").exists())

    # -- 6. hardlinked intermediates (no-op gap filling) -----------------------

    def test_hardlinked_intermediates_removed_without_touching_odm_laz(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "link1")
            # no-op fill_gaps hardlinks filled onto classified: same inode
            (proj / "sih3d_classified.laz").unlink()
            (proj / "sih3d_filled.laz").unlink()
            payload = b"same-inode-cloud"
            (proj / "sih3d_classified.laz").write_bytes(payload)
            os.link(proj / "sih3d_classified.laz", proj / "sih3d_filled.laz")
            s_c, s_f = os.stat(proj / "sih3d_classified.laz"), os.stat(proj / "sih3d_filled.laz")
            self.assertEqual((s_c.st_dev, s_c.st_ino), (s_f.st_dev, s_f.st_ino))
            odm_laz = proj / "odm_georeferencing" / "odm_georeferenced_model.laz"
            odm_laz.write_bytes(b"odm-original-cloud-bytes")
            res = run_cleanup(td / "runs", td / "projects", "link1", confirm=True)
            deleted_names = {Path(p).name for p in res["deleted"]}
            self.assertIn("sih3d_classified.laz", deleted_names)
            self.assertIn("sih3d_filled.laz", deleted_names)
            # both output paths removed (inode freed)...
            self.assertFalse((proj / "sih3d_classified.laz").exists())
            self.assertFalse((proj / "sih3d_filled.laz").exists())
            # ...and the ODM LAZ is untouched
            self.assertTrue(odm_laz.exists())
            self.assertEqual(odm_laz.read_bytes(), b"odm-original-cloud-bytes")
            self.assertTrue((proj / "sih3d_final.laz").exists())


class TestOdmOutputDirs(unittest.TestCase):
    """odm_orthophoto/, odm_dem/, odm_report/ – ODM map outputs the pipeline
    never reads after the run.  They must behave exactly like the other
    audit-approved intermediates: listed as candidates, cleanable only under
    the existing successful-run gate, protected artifacts untouched."""

    def test_odm_output_dirs_are_deletable_candidates(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "odmouts")
            for sub in ["odm_orthophoto", "odm_dem", "odm_report"]:
                (proj / sub).mkdir()
            names = {p.name for p in get_deletable_paths(proj)}
            for sub in ["odm_orthophoto", "odm_dem", "odm_report"]:
                self.assertIn(sub, names)
            # protected outputs stay out of the candidate set
            for protected in ["images", "odm_georeferencing", "odm_texturing"]:
                self.assertNotIn(protected, names)
            # missing ODM outputs are tolerated (no candidates for them)
            names_only = {p.name for p in get_deletable_paths(proj) if p.name == "odm_report"}
            self.assertEqual(names_only, {"odm_report"})

    def test_confirm_deletes_odm_output_dirs_keeps_protected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "odmdel")
            for sub in ["odm_orthophoto", "odm_dem", "odm_report"]:
                (proj / sub).mkdir()
                (proj / sub / "out.bin").write_bytes(b"x" * 128)
            # protected – must survive
            (proj / "odm_texturing").mkdir()
            (proj / "odm_texturing" / "odm_textured_model_geo.obj").write_text("mesh")
            (proj / "sih3d_final.laz").write_text("final")
            res = run_cleanup(td / "runs", td / "projects", "odmdel", confirm=True)
            deleted_names = {Path(p).name for p in res["deleted"]}
            for sub in ["odm_orthophoto", "odm_dem", "odm_report"]:
                self.assertIn(sub, deleted_names)
                self.assertFalse((proj / sub).exists())
            # deliverables and protected outputs survive
            self.assertTrue((proj / "odm_georeferencing" / "odm_georeferenced_model.laz").exists())
            self.assertTrue((proj / "odm_georeferencing" / "coords.txt").exists())
            self.assertTrue((proj / "odm_texturing" / "odm_textured_model_geo.obj").exists())
            self.assertTrue((proj / "sih3d_final.laz").exists())

    def test_unsuccessful_run_keeps_odm_output_dirs(self):
        # the new dirs get no extra gate: they are deleted only when the
        # existing successful-run gate passes (same as the other intermediates)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs" / "badodm"
            proj = td / "projects" / "badodm"
            work.mkdir(parents=True)
            proj.mkdir(parents=True)
            (work / "report.json").write_text(json.dumps({"reconstruction": {"ok": False}}))
            (proj / "odm_orthophoto").mkdir()
            (proj / "odm_dem").mkdir()
            with self.assertRaises(SystemExit):
                run_cleanup(td / "runs", td / "projects", "badodm", confirm=True)
            self.assertTrue((proj / "odm_orthophoto").exists())
            self.assertTrue((proj / "odm_dem").exists())


class TestOdmTexturing25dCleanup(unittest.TestCase):
    """odm_texturing_25d/ – the 2.5D textured mesh written by ODM's texturing
    stage, consumed only by odm_orthophoto.  It must join the audit-approved
    intermediates: deleted only through the existing successful-run gate, while
    odm_texturing/, odm_georeferencing/ and the final LAZ stay protected."""

    def test_odm_texturing_25d_is_deletable_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "t25dcand")
            (proj / "odm_texturing_25d").mkdir()
            (proj / "odm_texturing_25d" / "odm_textured_model_geo.obj").write_bytes(b"x" * 1600)
            # odm_texturing_25d listed, its delivered 3D twin never
            names = {p.name for p in get_deletable_paths(proj)}
            self.assertIn("odm_texturing_25d", names)
            self.assertNotIn("odm_texturing", names)
            # its size is reported
            listed, total, sized = collect_cleanup_info(proj)
            by_name = {p.name: s for p, s in sized}
            self.assertEqual(by_name["odm_texturing_25d"], 1600)
            self.assertEqual(total, 1600)

    def test_dry_run_keeps_odm_texturing_25d(self):
        # incomplete cleanup (dry-run, no --yes) must preserve it
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "t25ddry")
            (proj / "odm_texturing_25d").mkdir()
            (proj / "odm_texturing_25d" / "odm_textured_model_geo.obj").write_text("mesh25d")
            res = run_cleanup(td / "runs", td / "projects", "t25ddry", confirm=False)
            self.assertTrue(res["dry_run"])
            self.assertIn("odm_texturing_25d", {Path(p).name for p in res["deletable"]})
            self.assertTrue((proj / "odm_texturing_25d" / "odm_textured_model_geo.obj").exists())

    def test_successful_cleanup_deletes_odm_texturing_25d_keeps_protected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "t25ddel")
            (proj / "odm_texturing_25d").mkdir()
            (proj / "odm_texturing_25d" / "odm_textured_model_geo.obj").write_text("mesh25d")
            (proj / "odm_texturing").mkdir(parents=True, exist_ok=True)
            (proj / "odm_texturing" / "odm_textured_model_geo.obj").write_text("mesh3d")
            res = run_cleanup(td / "runs", td / "projects", "t25ddel", confirm=True)
            self.assertFalse(res["dry_run"])
            deleted_names = {Path(p).name for p in res["deleted"]}
            # the 2.5D intermediate goes through the same successful-run gate...
            self.assertIn("odm_texturing_25d", deleted_names)
            self.assertFalse((proj / "odm_texturing_25d").exists())
            # ...while odm_texturing/, odm_georeferencing/ and the final LAZ survive
            self.assertTrue((proj / "odm_texturing" / "odm_textured_model_geo.obj").exists())
            self.assertEqual((proj / "odm_texturing" / "odm_textured_model_geo.obj").read_text(), "mesh3d")
            self.assertTrue((proj / "odm_georeferencing" / "odm_georeferenced_model.laz").exists())
            self.assertTrue((proj / "odm_georeferencing" / "coords.txt").exists())
            self.assertTrue((proj / "sih3d_final.laz").exists())
            self.assertTrue((work / "report.json").exists())

    def test_unsuccessful_run_keeps_odm_texturing_25d(self):
        # no extra gate and no bypass: the existing successful-run gate protects it
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work = td / "runs" / "bad25d"
            proj = td / "projects" / "bad25d"
            work.mkdir(parents=True)
            proj.mkdir(parents=True)
            # unreadable report is the deterministic way to make the run
            # unsuccessful here (every real LAZ is a success marker)
            (work / "report.json").write_text("{not valid json")
            (proj / "odm_texturing_25d").mkdir()
            (proj / "odm_texturing_25d" / "odm_textured_model_geo.obj").write_text("mesh25d")
            ok, reason = is_run_successful(work, proj)
            self.assertFalse(ok)
            with self.assertRaises(SystemExit) as cm:
                run_cleanup(td / "runs", td / "projects", "bad25d", confirm=True)
            self.assertIn("not marked successful", str(cm.exception))
            self.assertTrue((proj / "odm_texturing_25d" / "odm_textured_model_geo.obj").exists())

    def test_missing_odm_texturing_25d_graceful(self):
        # already-cleaned runs must not fail (same tolerance as the other targets)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_leveled_run(td / "runs", td / "projects", "t25dmiss")
            res = run_cleanup(td / "runs", td / "projects", "t25dmiss", confirm=True)
            self.assertNotIn("odm_texturing_25d", {Path(p).name for p in res["deleted"]})
            res2 = run_cleanup(td / "runs", td / "projects", "t25dmiss", confirm=True)
            self.assertEqual(res2["deleted"], [])


class TestCleanupUniqueSizing(unittest.TestCase):
    """The dry-run reclaimable total must count each (st_dev, st_ino) once:
    a no-op fill_gaps leaves sih3d_filled.laz as a hardlink of
    sih3d_classified.laz, and the old per-path sum double-counted that."""

    def test_hardlinked_laz_pair_counted_once_in_total(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "uniq1")
            for sub in DELETABLE_SUBDIRS:
                (proj / sub).mkdir(parents=True, exist_ok=True)
                (proj / sub / "a.txt").write_text("hello")  # 5 B each
            payload = b"c" * 4000
            (proj / "sih3d_classified.laz").write_bytes(payload)
            os.link(proj / "sih3d_classified.laz", proj / "sih3d_filled.laz")
            (proj / "sih3d_final.laz").write_bytes(b"final")
            listed, total, sized = collect_cleanup_info(proj)
            by_name = {p.name: s for p, s in sized}
            # both rows stay listed at their full logical size...
            self.assertEqual(by_name["sih3d_classified.laz"], 4000)
            self.assertEqual(by_name["sih3d_filled.laz"], 4000)
            # ...but the reclaimable total counts the shared inode once
            self.assertEqual(total, len(DELETABLE_SUBDIRS) * 5 + 4000)
            res = run_cleanup(td / "runs", td / "projects", "uniq1", confirm=False)
            self.assertEqual(res["total_bytes"], len(DELETABLE_SUBDIRS) * 5 + 4000)
            # deletion behavior unchanged: both names removed
            res = run_cleanup(td / "runs", td / "projects", "uniq1", confirm=True)
            self.assertFalse((proj / "sih3d_classified.laz").exists())
            self.assertFalse((proj / "sih3d_filled.laz").exists())
            self.assertTrue((proj / "sih3d_final.laz").exists())

    def test_hardlinked_file_across_two_deletable_dirs_counted_once(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "uniq2")
            (proj / "opensfm").mkdir()
            (proj / "openmvs").mkdir()
            src = proj / "opensfm" / "shared.bin"
            src.write_bytes(b"z" * 1234)
            os.link(src, proj / "openmvs" / "shared.bin")
            _, total, sized = collect_cleanup_info(proj)
            by_name = {p.name: s for p, s in sized}
            # each row shows what it contains...
            self.assertEqual(by_name["opensfm"], 1234)
            self.assertEqual(by_name["openmvs"], 1234)
            # ...the total counts the inode once
            self.assertEqual(total, 1234)

    def test_copies_with_different_inodes_counted_in_full(self):
        # shutil.copy2 fallback topology: identical content, different inodes
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            work, proj = _make_successful_run(td / "runs", td / "projects", "uniq3")
            payload = b"d" * 2500
            (proj / "sih3d_classified.laz").write_bytes(payload)
            (proj / "sih3d_filled.laz").write_bytes(payload)  # distinct inode
            (proj / "sih3d_final.laz").write_bytes(b"final")
            self.assertNotEqual(os.stat(proj / "sih3d_classified.laz").st_ino,
                                os.stat(proj / "sih3d_filled.laz").st_ino)
            _, total, _ = collect_cleanup_info(proj)
            self.assertEqual(total, 2500 + 2500)


if __name__ == "__main__":
    unittest.main()
