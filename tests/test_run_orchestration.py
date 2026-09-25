"""Tests for run orchestration: keyframe default, report persistence, missing-Docker failure,
and CLI diagnostics.

Deterministic and hermetic: no Docker, no ODM, no GPU, no network.  cmd_run is exercised
end-to-end with the real telemetry parser, real GPS fusion and real report writing; only the
Docker subprocess boundary and the (video-reading) keyframe extractor are stubbed.  A synthetic
three-shot video exercises the real extract_keyframes selection path.
"""

import inspect
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

# make pipeline importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sih3d import frames, odm
from sih3d.__main__ import build_parser, cmd_run
from sih3d.odm import DockerUnavailable

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aukerman_srt_excerpt.srt"


def _fake_keyframes(frames_dir, n=8, t0=4.0, dt=1.5):
    """Stub extract_keyframes: write n tiny JPEG keyframes inside the fixture's time range."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(frames_dir / name), np.full((8, 8, 3), 100 + i, np.uint8))
        records.append({"name": name, "frame": i * 30, "t": round(t0 + dt * i, 4), "sharpness": 100.0})
    report = {"video": "fake.mp4", "resolution": [1920, 1440], "fps": 30.0, "frames": 240,
              "crop_x0_y0_x1_y1": [0, 0, 1920, 1440], "cuts_s": [], "shots": 1, "shots_used": [0],
              "keyframes": n, "dropped_blurry": 0, "undistorted": False, "focal35": None,
              "exposure": None, "deblock": True}
    return records, report


def _run_args(tmp, name, extra=()):
    return build_parser().parse_args([
        "run", "--video", "fake.mp4", "--telemetry", str(FIXTURE), "--name", name,
        "--work", str(tmp / "runs"), "--projects", str(tmp / "projects"), "--no-ai", *extra])


class TestSegmentDefault(unittest.TestCase):
    """The default must not silently discard the majority of survey views (Aukerman kept
    10/77 views under the old `--segment longest` default)."""

    def test_cli_default_segment_is_all(self):
        a = build_parser().parse_args(["run", "--video", "v.mp4", "--telemetry", "t.srt", "--name", "x"])
        self.assertEqual(a.segment, "all")

    def test_explicit_modes_preserved(self):
        p = build_parser()
        base = ["run", "--video", "v", "--telemetry", "t", "--name", "x", "--segment"]
        self.assertEqual(p.parse_args(base + ["all"]).segment, "all")
        self.assertEqual(p.parse_args(base + ["longest"]).segment, "longest")
        self.assertEqual(p.parse_args(base + ["2"]).segment, "2")

    def test_help_documents_default(self):
        buf = StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stdout(buf):
            build_parser().parse_args(["run", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("all (default", buf.getvalue())

    def test_library_default_is_all(self):
        self.assertEqual(inspect.signature(frames.extract_keyframes).parameters["segment"].default, "all")


class TestKeyframeDefaultBehavior(unittest.TestCase):
    """Regression: with equal-length shots the old `longest` default reconstructed exactly
    one shot; the `all` default must keep coverage from every shot."""

    @staticmethod
    def _write_three_shot_video(path, w=320, h=240, fps=30, frames_per_shot=45):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        assert vw.isOpened()
        for shot in range(3):
            for i in range(frames_per_shot):
                frame = np.full((h, w, 3), 30 + 60 * shot, np.uint8)
                x = 20 + (i * 4) % (w - 100)
                frame[100:140, x:x + 60] = (0, max(200 - 60 * shot, 30), 255)  # moving bright rect
                vw.write(frame)
        vw.release()
        return frames_per_shot

    def test_default_keeps_frames_from_every_shot(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            video = td / "threeshot.avi"
            self._write_three_shot_video(video)
            records, report = frames.extract_keyframes(video, td / "kf")
            self.assertEqual(report["shots"], 3)
            kept = [r["frame"] for r in records]
            self.assertTrue(kept, "no keyframes selected")
            self.assertLess(min(kept), 45, f"default kept no frame from the first shot: {kept}")
            self.assertGreater(max(kept), 90, f"default kept no frame from the last shot: {kept}")

    def test_explicit_longest_still_restricts_to_one_shot(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            video = td / "threeshot.avi"
            self._write_three_shot_video(video)
            records, report = frames.extract_keyframes(video, td / "kf", segment="longest")
            self.assertEqual(report["shots"], 3)
            kept = [r["frame"] for r in records]
            self.assertTrue(kept)
            # equal-length shots: the first is the longest -> everything inside the first shot
            self.assertLess(max(kept), 45, kept)

    def test_deterministic_selection_across_runs(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            video = td / "threeshot.avi"
            self._write_three_shot_video(video)
            r1, rep1 = frames.extract_keyframes(video, td / "a")
            r2, rep2 = frames.extract_keyframes(video, td / "b")
            self.assertEqual([r["frame"] for r in r1], [r["frame"] for r in r2])
            self.assertEqual(rep1["keyframes"], rep2["keyframes"])


class TestDockerUnavailable(unittest.TestCase):
    def test_missing_binary_raises_docker_unavailable(self):
        with mock.patch("sih3d.odm.subprocess.run",
                        side_effect=FileNotFoundError(2, "No such file or directory: 'docker'")):
            with self.assertRaises(DockerUnavailable) as cm:
                odm.docker_threads()
        self.assertIn("not found on PATH", str(cm.exception))

    def test_daemon_down_raises_docker_unavailable(self):
        fake = SimpleNamespace(returncode=1, stdout="",
                               stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock")
        with mock.patch("sih3d.odm.subprocess.run", return_value=fake):
            with self.assertRaises(DockerUnavailable) as cm:
                odm.docker_threads()
        self.assertIn("Docker is not running", str(cm.exception))


class TestMissingDockerCleanFailure(unittest.TestCase):
    """A missing docker executable must end in a recorded, actionable failure - not an
    unhandled traceback and not a lost run record."""

    def test_file_not_found_becomes_recorded_failure_with_report(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch("sih3d.frames.extract_keyframes",
                            side_effect=lambda v, d, **k: _fake_keyframes(Path(d))), \
                 mock.patch("sih3d.odm.subprocess.run",
                            side_effect=FileNotFoundError(2, "No such file or directory: 'docker'")):
                with self.assertRaises(SystemExit) as cm:
                    cmd_run(_run_args(tmp, "nodocker"))
            self.assertIn("docker", str(cm.exception).lower())
            work = tmp / "runs" / "nodocker"
            # the run record survived, including the keyframes written before ODM was invoked
            self.assertTrue((work / "keyframes.json").exists())
            self.assertTrue(any((work / "frames").glob("*.jpg")))
            report = json.loads((work / "report.json").read_text())
            self.assertTrue(report["telemetry"]["ok"])
            self.assertTrue(report["keyframes"]["ok"])
            self.assertTrue(report["gps_fusion"]["ok"])
            rec = report["reconstruction"]
            self.assertTrue(rec["attempted"])
            self.assertFalse(rec["ok"])
            self.assertEqual(rec["docker"], "unavailable")
            self.assertIn("PATH", rec["reason"])
            self.assertNotIn("log", rec)  # same failure structure as a failed ODM run, minus the log

    def test_daemon_down_becomes_recorded_failure_with_report(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            fake = SimpleNamespace(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon")
            with mock.patch("sih3d.frames.extract_keyframes",
                            side_effect=lambda v, d, **k: _fake_keyframes(Path(d))), \
                 mock.patch("sih3d.odm.subprocess.run", return_value=fake):
                with self.assertRaises(SystemExit) as cm:
                    cmd_run(_run_args(tmp, "nodaemon"))
            self.assertIn("docker", str(cm.exception).lower())
            report = json.loads((tmp / "runs" / "nodaemon" / "report.json").read_text())
            self.assertFalse(report["reconstruction"]["ok"])
            self.assertEqual(report["reconstruction"]["docker"], "unavailable")


class TestReportPersistence(unittest.TestCase):
    def test_successful_run_records_every_stage(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            def fake_odm_run(project, **k):
                (project / "odm_georeferencing").mkdir(parents=True, exist_ok=True)
                (project / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 32N\n600000 5000000\n")
                (project / "odm_georeferencing" / "odm_georeferenced_model.laz").write_bytes(b"fake-laz")
                return {"ok": True, "threads": 2, "attempts": 1}

            with mock.patch("sih3d.frames.extract_keyframes",
                            side_effect=lambda v, d, **k: _fake_keyframes(Path(d))), \
                 mock.patch("sih3d.odm.run", side_effect=fake_odm_run), \
                 mock.patch("sih3d.export.export_points", return_value={"ok": True}):
                cmd_run(_run_args(tmp, "ok1"))
            report = json.loads((tmp / "runs" / "ok1" / "report.json").read_text())
            self.assertTrue(report["telemetry"]["ok"])
            self.assertTrue(report["keyframes"]["ok"])
            self.assertTrue(report["gps_fusion"]["ok"])
            self.assertTrue(report["reconstruction"]["attempted"])
            self.assertTrue(report["reconstruction"]["ok"])
            self.assertEqual(report["reconstruction"]["attempts"], 1)
            self.assertTrue((tmp / "runs" / "ok1" / "keyframes.json").exists())

    def test_fusion_diagnostics_recorded_in_report(self):
        # the fixture's real survey telemetry drives the filter off-model, so the report
        # must carry the fallback flag + reason + warnings (never a silent track)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            with mock.patch("sih3d.frames.extract_keyframes",
                            side_effect=lambda v, d, **k: _fake_keyframes(Path(d))), \
                 mock.patch("sih3d.odm.subprocess.run",
                            side_effect=FileNotFoundError(2, "No such file or directory: 'docker'")):
                with self.assertRaises(SystemExit):
                    cmd_run(_run_args(tmp, "diag"))
            gf = json.loads((tmp / "runs" / "diag" / "report.json").read_text())["gps_fusion"]
            for key in ("accepted", "rejected", "rejection_ratio", "fallback_used",
                        "fallback_reason", "quality", "warnings", "max_innovation_m"):
                self.assertIn(key, gf)
            self.assertTrue(gf["fallback_used"])
            self.assertTrue(gf["warnings"])
            self.assertEqual(gf["quality"], "fallback")

    def test_failing_odm_still_persists_keyframes_and_report(self):
        # same failure-reporting structure as before: reconstruction.ok=false + odm_run.log
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)

            def fake_subprocess_run(cmd, **k):
                if cmd[:2] == ["docker", "info"]:  # daemon is up
                    return SimpleNamespace(returncode=0, stdout="4294967296 2", stderr="")
                return SimpleNamespace(returncode=1, stdout="", stderr="")  # the ODM container fails

            with mock.patch("sih3d.frames.extract_keyframes",
                            side_effect=lambda v, d, **k: _fake_keyframes(Path(d))), \
                 mock.patch("sih3d.odm.subprocess.run", side_effect=fake_subprocess_run):
                with self.assertRaises(SystemExit) as cm:
                    cmd_run(_run_args(tmp, "odmfail"))
            self.assertIn("Reconstruction failed", str(cm.exception))
            work = tmp / "runs" / "odmfail"
            report = json.loads((work / "report.json").read_text())
            self.assertTrue(report["reconstruction"]["attempted"])
            self.assertFalse(report["reconstruction"]["ok"])
            self.assertEqual(report["reconstruction"]["attempts"], 1)
            self.assertTrue((work / "keyframes.json").exists())
            self.assertTrue((tmp / "projects" / "odmfail" / "odm_run.log").exists())


if __name__ == "__main__":
    unittest.main()
