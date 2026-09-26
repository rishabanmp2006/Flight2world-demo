"""Tests for single-pass flight validation (sih3d.flight) and the keyframe/flight distinction.

The pipeline assumes ONE continuous forward drone pass.  These tests pin down what that means in code:

  * what must stay valid — straight passes, curves (up to a half-circle), noisy GPS, pauses/hovering,
    slow flight, dense high-rate fixes, GPS dropouts, isolated spikes;
  * what must be caught — out-and-back flights, repeated survey lanes, orbits and multi-lap circles;
  * what must never be claimed — a trajectory that is not a continuous flight path at all (a photo-sequence
    slideshow such as the Aukerman fixture) or too short to judge: reported as `unknown` with the limitation
    spelled out, never as "single pass";
  * that a *video shot* (a run of frames between cuts) is not a *logical flight*: one continuous flight
    stays one flight however many cuts its footage has, and a video holding several flights is reported so;
  * that the policy (enforce / warn / off + --allow-multi-pass) only ever gates a run on evidence, and that
    keyframes stay in chronological order in every mode.

Deterministic, hermetic, CPU-only: no Docker, no ODM, no GPU, no network, no video decoding except for the
three tiny synthetic AVI clips written by the helpers below.
"""

import json
import sys
import tempfile
import unittest
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

from sih3d import evaluate, flight, fusion  # noqa: E402
from sih3d.__main__ import build_parser, cmd_run  # noqa: E402
from sih3d.frames import extract_keyframes  # noqa: E402
from sih3d.telemetry import Telemetry, from_srt  # noqa: E402

ORG = (41.0, -81.0, 300.0)  # near the real Aukerman site
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aukerman_srt_excerpt.srt"


# --------------------------------------------------------------------------- helpers

def telemetry_from_enu(e, n, u, dt=1.0, h_acc=None):
    """A Telemetry object whose positions follow the given local ENU trajectory."""
    e, n, u = np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)
    lat, lon, alt = fusion.enu_to_geodetic(np.column_stack([e, n, u]), ORG)
    ha = np.full(len(e), np.nan) if h_acc is None else np.full(len(e), float(h_acc))
    return Telemetry(np.arange(len(e)) * dt, lat, lon, alt, ha, np.full(len(e), np.nan), source="synthetic")


def analyse(tel):
    return flight.analyse_track(tel.lat, tel.lon, tel.alt, tel.t, source="telemetry_fused")


def straight(n=120, step=5.0, sigma=0.0, seed=0, dt=1.0):
    """A straight pass; `sigma` adds white GPS jitter to the fixes."""
    rng = np.random.default_rng(seed)
    e = np.arange(n) * step
    n_axis = np.zeros(n)
    if sigma:
        e = e + rng.normal(0, sigma, n)
        n_axis = n_axis + rng.normal(0, sigma, n)
    return telemetry_from_enu(e, n_axis, np.full(n, 50.0), dt=dt)


def arc(radius=80.0, degrees=120.0, n=90, sigma=2.0, seed=1):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, np.radians(degrees), n)
    e = radius * np.sin(theta)
    n_axis = radius * (1 - np.cos(theta))
    if sigma:
        e = e + rng.normal(0, sigma, n)
        n_axis = n_axis + rng.normal(0, sigma, n)
    return telemetry_from_enu(e, n_axis, np.full(n, 50.0))


def write_shot_video(path, per_shot=45, shots=3, w=320, h=240, fps=30):
    """Three visually distinct shots (same generator as tests/test_run_orchestration.py)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    assert vw.isOpened()
    for shot in range(shots):
        for i in range(per_shot):
            frame = np.full((h, w, 3), 30 + 60 * shot, np.uint8)
            x = 20 + (i * 4) % (w - 100)
            frame[100:140, x:x + 60] = (0, max(200 - 60 * shot, 30), 255)
            vw.write(frame)
    vw.release()
    return per_shot, fps


def write_srt(path, times, e, n, u, name="DJI_0001.MP4"):
    """A DJI-style .SRT for an ENU trajectory (what a real flight's telemetry looks like)."""
    lat, lon, alt = fusion.enu_to_geodetic(np.column_stack([e, n, u]), ORG)
    blocks = []
    for k, t in enumerate(times):
        start, end = t, t + 0.5
        fmt = lambda s: "%02d:%02d:%02d,%03d" % (int(s // 3600), int(s % 3600 // 60), int(s % 60),  # noqa: E731
                                                 int(round((s % 1) * 1000)))
        blocks.append(f"{k + 1}\n{fmt(start)} --> {fmt(end)}\nFrameCnt: {k + 1}, source: {name}\n"
                      f"[latitude: {lat[k]:.7f}] [longitude: {lon[k]:.7f}] "
                      f"[rel_alt: {alt[k] - alt[0]:.3f} abs_alt: {alt[k]:.3f}]\n")
    Path(path).write_text("\n".join(blocks))


def fake_keyframes(frames_dir, n=8, t0=4.0, dt=1.5):
    """Stub extract_keyframes: n tiny JPEGs spread over the fixture's telemetry time range."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(n):
        name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(frames_dir / name), np.full((8, 8, 3), 100 + i, np.uint8))
        records.append({"name": name, "frame": i * 30, "t": round(t0 + dt * i, 4), "sharpness": 100.0})
    report = {"video": "fake.mp4", "resolution": [1920, 1440], "fps": 30.0, "frames": 240,
              "crop_x0_y0_x1_y1": [0, 0, 1920, 1440], "cuts_s": [], "shots": 1, "shots_used": [0],
              "keyframes": n, "dropped_blurry": 0, "undistorted": False, "focal35": None,
              "exposure": None, "deblock": True, "logical_flights": 1, "flights_source": "assumed_single",
              "flights": [], "keyframes_chronological": True}
    return records, report


def run_args(tmp, name, extra=(), telemetry=FIXTURE):
    return build_parser().parse_args([
        "run", "--video", "fake.mp4", "--telemetry", str(telemetry), "--name", name,
        "--work", str(tmp / "runs"), "--projects", str(tmp / "projects"), "--no-ai", *extra])


def back_and_forth_srt(path, n=30, step=3.0):
    """A real out-and-back flight: 30 fixes out, 30 fixes back along the same line."""
    e = np.concatenate([np.arange(n) * step, n * step - np.arange(1, n + 1) * step])
    write_srt(path, np.arange(2 * n) * 0.5, e, np.zeros(2 * n), np.full(2 * n, 50.0))
    return path


def fake_odm_run(project, **k):
    """Stand in for OpenDroneMap: write the files the later stages read, change nothing else."""
    (project / "odm_georeferencing").mkdir(parents=True, exist_ok=True)
    (project / "odm_georeferencing" / "coords.txt").write_text("WGS84 UTM 32N\n600000 5000000\n")
    (project / "odm_georeferencing" / "odm_georeferenced_model.laz").write_bytes(b"fake-laz")
    return {"ok": True, "threads": 2, "attempts": 1}


def mocked_run(tmp, name, telemetry=FIXTURE, extra=()):
    """Run cmd_run with the keyframe extractor and Docker stubbed; returns the report dict."""
    fake = SimpleNamespace(returncode=0, stdout="4294967296 4\n", stderr="")
    with mock.patch("sih3d.frames.extract_keyframes", side_effect=lambda v, d, **k: fake_keyframes(Path(d))), \
         mock.patch("sih3d.odm.subprocess.run", return_value=fake), \
         mock.patch("sih3d.odm.run", side_effect=fake_odm_run), \
         mock.patch("sih3d.export.export_points", return_value={"ok": True}):
        cmd_run(run_args(tmp, name, extra, telemetry=telemetry))
    return json.loads((tmp / "runs" / name / "report.json").read_text())


# --------------------------------------------------------------------------- A/B/C/D: valid single passes

class TestValidSinglePasses(unittest.TestCase):
    """One continuous forward traversal, however it curves, slows or is measured."""

    def test_a_straight_pass_is_valid(self):
        m = analyse(straight())
        self.assertEqual(m["status"], flight.VALID)
        self.assertTrue(m["single_pass"])
        self.assertAlmostEqual(m["path_efficiency"], 1.0, places=2)
        self.assertEqual(m["direction_reversals"], 0)
        self.assertEqual(m["revisit_score"], 0.0)
        self.assertEqual(m["forward_fraction"], 1.0)
        self.assertEqual(m["reasons"], [])
        self.assertEqual(m["warnings"], [])

    def test_a_straight_pass_metrics(self):
        m = analyse(straight(n=60, step=5.0))
        self.assertAlmostEqual(m["path_length_m"], 295.0, places=1)
        self.assertAlmostEqual(m["displacement_m"], 295.0, places=1)
        self.assertEqual(m["points"], 60)
        self.assertEqual(m["dominant_heading_deg"], 90.0)  # heading east
        self.assertAlmostEqual(m["straightness"], 1.0, places=2)
        self.assertAlmostEqual(m["lateral_deviation_m"], 0.0, places=3)

    def test_b_curved_passes_are_valid(self):
        for degrees in (45.0, 90.0, 120.0, 180.0):
            with self.subTest(degrees=degrees):
                m = analyse(arc(radius=80.0, degrees=degrees))
                self.assertEqual(m["status"], flight.VALID)
                self.assertTrue(m["single_pass"])
                self.assertEqual(m["direction_reversals"], 0)
                self.assertGreater(m["path_efficiency"], flight.EFFICIENCY_SUSPICIOUS)

    def test_b_serpentine_pass_is_valid(self):
        x = np.linspace(0, 400, 120)
        rng = np.random.default_rng(3)
        m = analyse(telemetry_from_enu(x, 20 * np.sin(2 * np.pi * x / 200) + rng.normal(0, 2, 120),
                                       np.full(120, 50.0)))
        self.assertEqual(m["status"], flight.VALID)
        self.assertEqual(m["direction_reversals"], 0)

    def test_c_noisy_gps_stays_valid(self):
        for sigma in (2.0, 3.0, 5.0):
            with self.subTest(sigma=sigma):
                m = analyse(straight(n=120, step=5.0, sigma=sigma, seed=int(sigma)))
                self.assertEqual(m["status"], flight.VALID)
                self.assertGreater(m["path_efficiency"], 0.9)
                self.assertGreater(m["position_sigma_m"], 1.0)  # the jitter was really there
                self.assertEqual(m["gps_outliers_removed"], 0)

    def test_c_dense_high_rate_noisy_fixes_stay_valid(self):
        # 10 Hz fixes, tiny per-step motion, 3 m jitter: the classic case where naive path length explodes
        t = np.arange(600) * 0.1
        rng = np.random.default_rng(7)
        e = 6.0 * t + rng.normal(0, 3.0, 600)
        n_axis = rng.normal(0, 3.0, 600)
        lat, lon, alt = fusion.enu_to_geodetic(np.column_stack([e, n_axis, np.full(600, 50.0)]), ORG)
        tel = Telemetry(t, lat, lon, alt, np.full(600, np.nan), np.full(600, np.nan), source="dense")
        m = analyse(tel)
        self.assertEqual(m["status"], flight.VALID)
        self.assertGreater(m["path_efficiency"], 0.9)
        self.assertLess(m["max_speed_mps"], 25.0)  # judged over >= 1 s windows, not between adjacent fixes
        self.assertGreater(m["max_fix_speed_mps"], 50.0)  # ... while the per-fix speed is meaningless

    def test_d_pauses_and_slow_flight_stay_valid(self):
        rng = np.random.default_rng(4)
        e = np.concatenate([np.arange(40) * 5.0, np.full(15, 195.0), 195.0 + np.arange(1, 40) * 2.0])
        m = analyse(telemetry_from_enu(e + rng.normal(0, 2, len(e)), rng.normal(0, 2, len(e)),
                                       np.full(len(e), 50.0)))
        self.assertEqual(m["status"], flight.VALID)
        self.assertEqual(m["direction_reversals"], 0)

    def test_d_hover_does_not_count_as_a_revisit(self):
        # the drone stops in mid-flight for 20 s (GPS jitter only) and then continues: not a revisit
        rng = np.random.default_rng(6)
        e = np.concatenate([np.arange(60) * 5.0, np.full(20, 295.0), 295.0 + np.arange(1, 41) * 5.0])
        m = analyse(telemetry_from_enu(e + rng.normal(0, 2, len(e)), rng.normal(0, 2, len(e)),
                                       np.full(len(e), 50.0)))
        self.assertEqual(m["status"], flight.VALID)
        self.assertEqual(m["revisit_score"], 0.0)

    def test_gps_dropout_stays_valid(self):
        t = np.concatenate([np.arange(0, 30, 1.0), np.arange(52, 90, 1.0)])  # 22 s without fixes
        rng = np.random.default_rng(8)
        e = t * 8.0 + rng.normal(0, 3.0, len(t))
        m = analyse(telemetry_from_enu(e, rng.normal(0, 3.0, len(t)), np.full(len(t), 50.0),
                                       dt=1.0).__class__(t, *(lambda ll: ll)(
            fusion.enu_to_geodetic(np.column_stack([e, np.zeros(len(t)), np.full(len(t), 50.0)]), ORG)),
            np.full(len(t), np.nan), np.full(len(t), np.nan)))
        self.assertEqual(m["status"], flight.VALID)
        self.assertLess(m["discontinuous_step_fraction"], flight.DISCONTINUOUS_STEP_FRACTION)


# --------------------------------------------------------------------------- E/F: must be caught

class TestMultiPassAndOrbit(unittest.TestCase):
    """Behaviour that is outside the intended single-pass input model."""

    def test_e_out_and_back_along_one_line_is_not_valid(self):
        e = np.concatenate([np.arange(60) * 5.0, 300.0 - np.arange(1, 61) * 5.0])
        rng = np.random.default_rng(9)
        m = analyse(telemetry_from_enu(e + rng.normal(0, 2, len(e)), rng.normal(0, 2, len(e)),
                                       np.full(len(e), 50.0)))
        self.assertIn(m["status"], (flight.SUSPICIOUS, flight.INVALID))
        self.assertFalse(m["single_pass"])
        self.assertLess(m["path_efficiency"], 0.2)
        self.assertEqual(m["direction_reversals"], 1)
        self.assertGreater(m["revisit_score"], 0.0)
        self.assertTrue(m["reasons"])

    def test_e_repeated_survey_lanes_are_not_valid(self):
        e, n, x, y = [0.0], [0.0], 0.0, 0.0
        for lane in range(4):  # a lawnmower: straight lanes joined by short shifts
            for _ in range(40):
                y += 8.0 * (1 if lane % 2 == 0 else -1)
                e.append(x)
                n.append(y)
            if lane < 3:
                for _ in range(3):
                    x += 8.0 / 3
                    e.append(x)
                    n.append(y)
        e, n = np.array(e), np.array(n)
        rng = np.random.default_rng(10)
        m = analyse(telemetry_from_enu(e + rng.normal(0, 2, len(e)), n + rng.normal(0, 2, len(n)),
                                       np.full(len(e), 50.0)))
        self.assertIn(m["status"], (flight.SUSPICIOUS, flight.INVALID))
        self.assertLess(m["path_efficiency"], 0.3)
        self.assertGreaterEqual(m["direction_reversals"], 3)
        self.assertGreater(m["revisit_score"], flight.REVISIT_SUSPICIOUS)

    def test_f_orbit_is_not_valid(self):
        rng = np.random.default_rng(11)
        theta = np.linspace(0, 2 * np.pi, 200)
        e = 60 * np.sin(theta) + rng.normal(0, 2, 200)
        n = 60 * (1 - np.cos(theta)) + rng.normal(0, 2, 200)
        m = analyse(telemetry_from_enu(e, n, np.full(200, 50.0)))
        self.assertIn(m["status"], (flight.SUSPICIOUS, flight.INVALID))
        self.assertFalse(m["single_pass"])
        self.assertLess(m["path_efficiency"], flight.EFFICIENCY_INVALID)

    def test_f_two_lap_circle_is_not_valid(self):
        rng = np.random.default_rng(12)
        theta = np.linspace(0, 4 * np.pi, 400)
        e = 60 * np.sin(theta) + rng.normal(0, 2, 400)
        n = 60 * (1 - np.cos(theta)) + rng.normal(0, 2, 400)
        m = analyse(telemetry_from_enu(e, n, np.full(400, 50.0)))
        self.assertIn(m["status"], (flight.SUSPICIOUS, flight.INVALID))
        self.assertGreater(m["revisit_score"], flight.REVISIT_SUSPICIOUS)  # the second lap re-covers the first

    def test_gps_outliers_do_not_reject_a_valid_pass(self):
        e = np.arange(60) * 5.0
        rng = np.random.default_rng(13)
        n = rng.normal(0, 2, 60)
        e_bad, n_bad = e + rng.normal(0, 3, 60), n + rng.normal(0, 3, 60)
        for i in (13, 31, 47):  # three wild fixes
            e_bad[i] += 80
            n_bad[i] -= 60
        m = analyse(telemetry_from_enu(e_bad, n_bad, np.full(60, 50.0)))
        self.assertEqual(m["status"], flight.VALID)
        self.assertGreaterEqual(m["gps_outliers_removed"], 2)
        self.assertLess(m["discontinuous_step_fraction"], flight.DISCONTINUOUS_STEP_FRACTION)
        self.assertGreater(m["path_efficiency"], 0.9)

    def test_one_wild_fix_does_not_flip_a_verdict(self):
        e = np.arange(80) * 4.0
        rng = np.random.default_rng(14)
        n = rng.normal(0, 1.5, 80)
        e_bad = e + rng.normal(0, 1.5, 80)
        e_bad[40] += 300.0  # a single jump-out
        m = analyse(telemetry_from_enu(e_bad, n, np.full(80, 50.0)))
        self.assertEqual(m["status"], flight.VALID)
        self.assertEqual(m["gps_outliers_removed"], 1)


# --------------------------------------------------------------------------- H/I: limits of the evidence

class TestInsufficientEvidence(unittest.TestCase):
    def test_h_too_few_positions_is_unknown_with_a_warning(self):
        m = analyse(telemetry_from_enu(np.arange(4) * 4.0, np.zeros(4), np.full(4, 50.0)))
        self.assertEqual(m["status"], flight.UNKNOWN)
        self.assertIsNone(m["single_pass"])
        self.assertIn("4 positions", m["reasons"][0])
        self.assertTrue(m["warnings"])

    def test_h_very_short_track_is_unknown(self):
        m = analyse(straight(n=4, step=4.0))
        self.assertEqual(m["status"], flight.UNKNOWN)
        self.assertIsNone(m["single_pass"])
        self.assertTrue(m["warnings"])

    def test_h_short_but_judgeable_track_passes_with_a_warning(self):
        m = analyse(straight(n=9, step=5.0))
        self.assertEqual(m["status"], flight.VALID_WITH_WARNING)
        self.assertTrue(m["single_pass"])
        self.assertIn("short", " ".join(m["warnings"]))

    def test_h_hover_only_flight_is_unknown(self):
        rng = np.random.default_rng(15)
        m = analyse(telemetry_from_enu(rng.normal(0, 1, 30), rng.normal(0, 1, 30), np.full(30, 50.0)))
        self.assertEqual(m["status"], flight.UNKNOWN)
        self.assertIsNone(m["single_pass"])

    def test_i_missing_telemetry_is_reported_as_a_limitation(self):
        records = [{"name": f"f{i}.jpg", "t": float(i), "lat": np.nan, "lon": np.nan, "alt": np.nan}
                   for i in range(6)]
        block = flight.validate_run(records, None, times=np.arange(6, dtype=float))
        self.assertEqual(block["status"], flight.UNKNOWN)
        self.assertIsNone(block["single_pass"])
        self.assertEqual(block["action"], "continue")
        self.assertTrue(any("no telemetry" in w for w in block["warnings"]))
        self.assertTrue(any("no GPS position" in w for w in block["warnings"]))
        self.assertEqual(block["keyframes_with_position"], 0)

    def test_i_keyframes_outside_the_telemetry_window_are_flagged(self):
        tel = straight(n=20, step=5.0)
        records = [{"name": f"f{i}.jpg", "t": float(t), "lat": float(tel.lat[min(i, 19)]),
                    "lon": float(tel.lon[min(i, 19)]), "alt": float(tel.alt[min(i, 19)])}
                   for i, t in enumerate(np.arange(0, 100, 10.0))]
        block = flight.validate_run(records, tel, times=np.arange(0, 100, 10.0))
        self.assertLess(block["telemetry_coverage"], 1.0)
        self.assertGreater(block["keyframes_outside_telemetry"], 0)
        self.assertTrue(any("outside the telemetry time range" in w for w in block["warnings"]))

    def test_extrapolated_keyframes_never_earn_a_single_pass_verdict(self):
        # The keyframe positions describe a textbook straight pass, but most of them sit past the end of
        # the telemetry: they are extrapolated, so the block must say "unknown", not "valid".
        tel = straight(n=20, step=5.0)
        times = np.arange(0.0, 100.0, 10.0)
        enu = np.column_stack([times * 5.0, np.zeros(len(times)), np.full(len(times), 50.0)])
        lat, lon, alt = fusion.enu_to_geodetic(enu, ORG)
        records = [{"name": f"f{i}.jpg", "t": float(t), "lat": float(la), "lon": float(lo), "alt": float(al)}
                   for i, (t, la, lo, al) in enumerate(zip(times, lat, lon, alt))]
        block = flight.validate_run(records, tel, times=times)
        outside = block["keyframes_outside_telemetry"] / block["keyframes"]
        self.assertGreater(outside, flight.KEYFRAME_COVERAGE_MISSING)
        self.assertLess(block["telemetry_coverage"], 0.5)
        self.assertEqual(block["tracks"]["keyframes_fused"]["path_efficiency"], 1.0)
        self.assertEqual(block["status"], flight.UNKNOWN)
        self.assertIsNone(block["single_pass"])
        self.assertEqual(block["action"], "continue")

    def test_noise_limited_evidence_downgrades_a_negative_verdict(self):
        # a short, very noisy track that would otherwise look like a return: jitter must not reject it
        rng = np.random.default_rng(16)
        e = np.concatenate([np.arange(20) * 3.0, 60.0 - np.arange(1, 21) * 3.0])
        e = e + rng.normal(0, 8.0, len(e))
        m = analyse(telemetry_from_enu(e, rng.normal(0, 8.0, len(e)), np.full(len(e), 50.0)))
        self.assertNotEqual(m["status"], flight.SUSPICIOUS)
        self.assertNotEqual(m["status"], flight.INVALID)


# --------------------------------------------------------------------------- J: real fixture telemetry

class TestAukermanTelemetry(unittest.TestCase):
    """The Aukerman fixture is a photo-sequence slideshow: its telemetry is not a continuous flight path.

    The verdict must therefore be `unknown` (the limitation stated), never a fabricated "single pass" — and
    it must be identical on every run.
    """

    def setUp(self):
        self.tel = from_srt(FIXTURE)

    def test_j_photo_sequence_telemetry_is_reported_as_unjudgeable(self):
        m = analyse(self.tel)
        self.assertEqual(m["status"], flight.UNKNOWN)
        self.assertIsNone(m["single_pass"])
        self.assertGreater(m["discontinuous_step_fraction"], flight.DISCONTINUOUS_STEP_FRACTION)
        self.assertTrue(any("not one continuous flight path" in r for r in m["reasons"]))
        self.assertTrue(any("no single-pass" in w for w in m["warnings"]))

    def test_j_fused_track_gives_the_same_verdict(self):
        fused = fusion.fuse(self.tel, self.tel.t)
        m = flight.analyse_track(fused["lat"], fused["lon"], fused["alt"], self.tel.t,
                                 source="telemetry_fused")
        self.assertEqual(m["status"], flight.UNKNOWN)
        self.assertIsNone(m["single_pass"])

    def test_j_classification_is_deterministic(self):
        a = json.dumps(analyse(self.tel), sort_keys=True, default=str)
        b = json.dumps(analyse(from_srt(FIXTURE)), sort_keys=True, default=str)
        self.assertEqual(a, b)

    def test_j_validate_run_reports_the_limitation_and_continues(self):
        times = np.arange(8) * 1.5 + 4.0
        fused = fusion.fuse(self.tel, times)
        records = [{"name": f"frame_{i:06d}.jpg", "t": float(times[i]), "lat": float(fused["lat"][i]),
                    "lon": float(fused["lon"][i]), "alt": float(fused["alt"][i])} for i in range(len(times))]
        block = flight.validate_run(records, self.tel, times=times, fusion=fused["stats"], policy="enforce")
        self.assertEqual(block["status"], flight.UNKNOWN)
        self.assertIsNone(block["single_pass"])
        self.assertEqual(block["action"], "continue")
        self.assertEqual(block["telemetry_fixes"], len(self.tel.t))
        self.assertEqual(block["keyframes"], 8)
        self.assertTrue(block["warnings"])
        self.assertIn("keyframes_fused", block["tracks"])
        self.assertIn("telemetry_fused", block["tracks"])


# --------------------------------------------------------------------------- metrics + determinism

class TestMetrics(unittest.TestCase):
    def test_metrics_are_deterministic(self):
        tel = straight(n=80, step=4.0, sigma=3.0, seed=21)
        first = json.dumps(analyse(tel), sort_keys=True)
        for _ in range(3):
            self.assertEqual(json.dumps(analyse(tel), sort_keys=True), first)

    def test_metrics_do_not_modify_their_input(self):
        tel = straight(n=40, step=5.0, sigma=2.0, seed=22)
        before = (tel.lat.copy(), tel.lon.copy(), tel.alt.copy())
        analyse(tel)
        for original, after in zip(before, (tel.lat, tel.lon, tel.alt)):
            np.testing.assert_array_equal(original, after)

    def test_unsorted_times_are_judged_in_time_order(self):
        tel = straight(n=60, step=5.0, sigma=2.0, seed=23)
        order = np.argsort(tel.t)[::-1]
        m = flight.analyse_track(tel.lat[order], tel.lon[order], tel.alt[order], tel.t[order],
                                 source="telemetry_fused")
        self.assertEqual(m["status"], flight.VALID)
        self.assertAlmostEqual(m["path_efficiency"], 1.0, places=1)

    def test_thresholds_are_recorded_in_the_block(self):
        block = flight.validate_run([], None)
        self.assertEqual(block["thresholds"]["max_plausible_speed_mps"], flight.MAX_PLAUSIBLE_SPEED_MPS)
        self.assertEqual(block["thresholds"]["efficiency_suspicious"], flight.EFFICIENCY_SUSPICIOUS)
        self.assertEqual(block["mode"], "single_pass")

    def test_report_lines_describe_the_verdict(self):
        block = flight.validate_run([{"name": "f.jpg", "t": 0.0, "lat": 41.0, "lon": -81.0, "alt": 300.0}], None)
        text = "\n".join(flight.report_lines(block))
        self.assertIn("Flight validation:", text)
        self.assertIn("WARNING", text)


# --------------------------------------------------------------------------- policy

class TestPolicy(unittest.TestCase):
    def _block(self, status, **kw):
        return {"status": status, "single_pass": status in (flight.VALID, flight.VALID_WITH_WARNING),
                "reasons": ["because"], **kw}

    def test_valid_continues(self):
        for status in (flight.VALID, flight.VALID_WITH_WARNING, flight.UNKNOWN):
            action, overridden, _ = flight.decision(self._block(status), "enforce", False)
            self.assertEqual(action, "continue")
            self.assertFalse(overridden)

    def test_suspicious_and_invalid_stop_by_default(self):
        for status in (flight.SUSPICIOUS, flight.INVALID):
            action, overridden, reason = flight.decision(self._block(status), "enforce", False)
            self.assertEqual(action, "stop")
            self.assertIn("--allow-multi-pass", reason)
            action, overridden, reason = flight.decision(self._block(status), "enforce", True)
            self.assertEqual(action, "continue")
            self.assertTrue(overridden)
            self.assertIn("override", reason)

    def test_warn_policy_only_reports(self):
        for status in (flight.SUSPICIOUS, flight.INVALID):
            action, overridden, reason = flight.decision(self._block(status), "warn", False)
            self.assertEqual(action, "continue")
            self.assertFalse(overridden)
            self.assertIn(status, reason)

    def test_off_policy_skips_the_check(self):
        block = flight.validate_run([{"name": "f.jpg", "lat": 41.0, "lon": -81.0, "alt": 300.0}], None,
                                    policy="off")
        self.assertEqual(block["status"], "skipped")
        self.assertIsNone(block["single_pass"])
        self.assertEqual(block["action"], "continue")
        self.assertTrue(block["warnings"])

    def test_policy_names_are_the_cli_choices(self):
        a = build_parser().parse_args(["run", "--video", "v", "--telemetry", "t", "--name", "x"])
        self.assertEqual(a.flight_validation, "enforce")
        self.assertFalse(a.allow_multi_pass)
        for value in flight.POLICIES:
            args = build_parser().parse_args(["run", "--video", "v", "--telemetry", "t", "--name", "x",
                                              "--flight-validation", value])
            self.assertEqual(args.flight_validation, value)
        strict = build_parser().parse_args(["run", "--video", "v", "--telemetry", "t", "--name", "x",
                                            "--allow-multi-pass"])
        self.assertTrue(strict.allow_multi_pass)

    def test_help_documents_the_gate(self):
        import contextlib
        import io
        buf = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buf):
            build_parser().parse_args(["run", "--help"])
        text = buf.getvalue()
        self.assertIn("--allow-multi-pass", text)
        self.assertIn("single-pass", text)


# --------------------------------------------------------------------------- video shots vs logical flights

class TestLogicalFlights(unittest.TestCase):
    def _shots(self, specs):
        return [{"start_s": s, "end_s": e, "frames": f, "start_enu": a, "end_enu": b}
                for s, e, f, a, b in specs]

    def test_one_continuous_flight_with_cuts_is_one_flight(self):
        shots = self._shots([(0, 10, 30, [0, 0], [100, 0]), (10.2, 20, 30, [100, 0], [200, 0]),
                             (20.4, 30, 30, [200, 0], [300, 0])])
        flights = flight.logical_flights(shots)
        self.assertEqual(len(flights), 1)
        self.assertEqual(flights[0]["shots"], [0, 1, 2])
        self.assertEqual(flights[0]["frames"], 90)

    def test_shots_years_apart_are_separate_flights(self):
        shots = self._shots([(0, 10, 30, [0, 0], [100, 0]), (900, 920, 30, [5000, 0], [5100, 0])])
        self.assertEqual(len(flight.logical_flights(shots)), 2)

    def test_an_impossible_jump_splits_the_flight(self):
        shots = self._shots([(0, 10, 30, [0, 0], [100, 0]), (10.2, 20, 30, [900, 0], [1000, 0])])
        self.assertEqual(len(flight.logical_flights(shots)), 2)

    def test_without_positions_only_time_is_used(self):
        shots = [{"start_s": 0, "end_s": 10, "frames": 30}, {"start_s": 10.2, "end_s": 20, "frames": 30}]
        flights = flight.logical_flights(shots)
        self.assertEqual(len(flights), 1, "cuts alone must never create a second flight")

    def test_largest_flight_is_the_one_with_most_frames(self):
        shots = self._shots([(0, 10, 30, [0, 0], [100, 0]), (900, 950, 90, [5000, 0], [5100, 0])])
        flights = flight.logical_flights(shots)
        self.assertEqual(flight.largest_flight(flights), 1)
        self.assertIsNone(flight.largest_flight([]))

    def test_flights_are_reported_with_their_span(self):
        shots = self._shots([(0, 10, 30, [0, 0], [100, 0]), (10.2, 20, 30, [100, 0], [200, 0])])
        f = flight.logical_flights(shots)[0]
        self.assertEqual(f["start_s"], 0)
        self.assertEqual(f["end_s"], 20)
        self.assertAlmostEqual(f["duration_s"], 20.0)


class TestKeyframeSelectionIsSinglePassAware(unittest.TestCase):
    """The keyframe stage must keep chronological order, and must not read cuts as extra flights."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.td = Path(self.tmp.name)
        self.video = self.td / "threeshot.avi"
        per_shot, fps = write_shot_video(self.video)
        self.fps = fps
        self.n_frames = 3 * per_shot

    def tearDown(self):
        self.tmp.cleanup()

    def _continuous_telemetry(self):
        """Positions that really describe one continuous flight across every shot."""
        frames = np.arange(0, self.n_frames, 5)
        return telemetry_from_enu(frames * 0.5, np.zeros(len(frames)), np.full(len(frames), 50.0),
                                  dt=5.0 / self.fps)

    def _three_flight_telemetry(self):
        """Three separate flights: the second and third start 2 km away from the previous one."""
        frames = np.arange(0, self.n_frames, 5)
        e = frames * 0.5
        e[frames >= 45] += 2000.0
        e[frames >= 90] += 2000.0
        n = np.where(frames >= 90, 60.0, 0.0)
        return telemetry_from_enu(e + n * 0, n, np.full(len(frames), 50.0), dt=5.0 / self.fps)

    def test_cuts_do_not_create_extra_flights_for_a_continuous_flight(self):
        records, report = extract_keyframes(self.video, self.td / "kf_cont",
                                            telemetry=self._continuous_telemetry())
        self.assertGreaterEqual(report["shots"], 2, "the fixture video must really contain cuts")
        self.assertEqual(report["logical_flights"], 1)
        self.assertEqual(report["flights_source"], "telemetry")
        self.assertTrue(report["keyframes_chronological"])

    def test_without_telemetry_all_shots_are_still_one_flight(self):
        records, report = extract_keyframes(self.video, self.td / "kf_none")
        self.assertGreaterEqual(report["shots"], 2)
        self.assertEqual(report["logical_flights"], 1)
        self.assertEqual(report["flights_source"], "assumed_single")

    def test_separate_flights_are_reported_as_separate(self):
        records, report = extract_keyframes(self.video, self.td / "kf_split",
                                            telemetry=self._three_flight_telemetry())
        self.assertGreaterEqual(report["shots"], 2)
        self.assertEqual(report["logical_flights"], report["shots"])

    def test_segment_single_pass_keeps_one_logical_flight(self):
        tel = self._three_flight_telemetry()
        records, report = extract_keyframes(self.video, self.td / "kf_sp", segment="single-pass",
                                            telemetry=tel)
        self.assertEqual(len(report["shots_used"]), 1)
        self.assertIn("single-pass", report["segment_note"])
        bounds = [records[0]["frame"], records[-1]["frame"]]
        self.assertLess(bounds[1], 90, "kept keyframes from more than one flight")

    def test_segment_all_still_keeps_every_shot(self):
        records, report = extract_keyframes(self.video, self.td / "kf_all", telemetry=self._three_flight_telemetry())
        self.assertEqual(report["shots_used"], list(range(report["shots"])))
        self.assertEqual([r["frame"] for r in records], sorted(r["frame"] for r in records))
        self.assertTrue(report["keyframes_chronological"])

    def test_keyframes_are_chronological_in_every_mode(self):
        tel = self._continuous_telemetry()
        for segment in ("all", "longest", "single-pass", "0"):
            with self.subTest(segment=segment):
                records, report = extract_keyframes(self.video, self.td / f"kf_{segment}", segment=segment,
                                                    telemetry=tel)
                frames = [r["frame"] for r in records]
                times = [r["t"] for r in records]
                self.assertEqual(frames, sorted(frames))
                self.assertEqual(times, sorted(times))
                self.assertTrue(report["keyframes_chronological"])
                self.assertGreater(len(records), 1)

    def test_neighbouring_keyframes_keep_overlapping_coverage(self):
        # image-shift spacing: consecutive keyframes stay at least ~min_shift of the image apart, and no two
        # keyframes are duplicates of the same frame
        records, report = extract_keyframes(self.video, self.td / "kf_overlap", telemetry=self._continuous_telemetry())
        frames = [r["frame"] for r in records]
        self.assertEqual(len(frames), len(set(frames)))
        self.assertTrue(all(b > a for a, b in zip(frames, frames[1:])))


# --------------------------------------------------------------------------- report integration

class TestReportIntegration(unittest.TestCase):
    def test_report_contains_the_flight_validation_block(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = mocked_run(tmp, "fv1")
            fv = report["flight_validation"]
            for key in ("mode", "policy", "status", "single_pass", "path_length_m", "displacement_m",
                        "path_efficiency", "dominant_axis", "direction_reversals", "revisit_score",
                        "warnings", "thresholds", "action"):
                self.assertIn(key, fv, f"flight_validation is missing {key}")
            self.assertEqual(fv["mode"], "single_pass")
            self.assertEqual(fv["policy"], "enforce")
            self.assertEqual(fv["action"], "continue")
            self.assertIsNone(fv["single_pass"])  # the fixture telemetry cannot be judged
            self.assertTrue(fv["warnings"])
            # the pre-existing stages are untouched
            for key in ("telemetry", "keyframes", "gps_fusion", "reconstruction"):
                self.assertIn(key, report)
            self.assertTrue(report["keyframes"]["ok"])

    def test_report_records_shots_and_logical_flights(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = mocked_run(tmp, "fv2")
            self.assertIn("logical_flights", report["keyframes"])
            self.assertIn("video_shots", report["flight_validation"])

    def test_invalid_flight_stops_the_run_before_docker(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            srt = back_and_forth_srt(tmp / "back_and_forth.SRT")
            with mock.patch("sih3d.frames.extract_keyframes",
                            side_effect=lambda v, d, **k: fake_keyframes(Path(d))), \
                 mock.patch("sih3d.odm.subprocess.run", side_effect=AssertionError("docker must not be called")):
                with self.assertRaises(SystemExit) as cm:
                    cmd_run(run_args(tmp, "gate", (), telemetry=srt))
            self.assertIn("single-pass", str(cm.exception))
            report = json.loads((tmp / "runs" / "gate" / "report.json").read_text())
            self.assertIn(report["flight_validation"]["status"], ("suspicious", "invalid"))
            self.assertEqual(report["flight_validation"]["action"], "stop")
            self.assertFalse(report["reconstruction"]["attempted"])

    def test_allow_multi_pass_lets_the_run_continue(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            srt = back_and_forth_srt(tmp / "back_and_forth.SRT")
            report = mocked_run(tmp, "override", telemetry=srt, extra=("--allow-multi-pass",))
            fv = report["flight_validation"]
            self.assertEqual(fv["action"], "continue")
            self.assertTrue(fv["overridden"])
            self.assertIn(fv["status"], ("suspicious", "invalid"))
            self.assertTrue(report["reconstruction"]["attempted"])

    def test_warn_policy_keeps_the_run_going(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            srt = back_and_forth_srt(tmp / "back_and_forth.SRT")
            report = mocked_run(tmp, "warned", telemetry=srt, extra=("--flight-validation", "warn"))
            self.assertEqual(report["flight_validation"]["action"], "continue")
            self.assertEqual(report["flight_validation"]["policy"], "warn")
            self.assertIn(report["flight_validation"]["status"], ("suspicious", "invalid"))

    def test_off_policy_records_the_skipped_check(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = mocked_run(tmp, "off", extra=("--flight-validation", "off"))
            self.assertEqual(report["flight_validation"]["status"], "skipped")

    def test_quality_summary_is_written_with_the_report(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            report = mocked_run(tmp, "quality")
            quality = report["quality"]
            for key in ("reconstruction", "single_pass", "georeferencing", "point_cloud",
                        "ai_classification", "gap_filling"):
                self.assertIn(key, quality)
            self.assertEqual(quality["reconstruction"]["status"], "success")
            self.assertEqual(quality["reconstruction"]["attempts"], 1)
            self.assertEqual(quality["single_pass"]["basis"], "report.flight_validation")
            self.assertEqual(quality["ai_classification"]["status"], "not_run")  # --no-ai
            self.assertIsNone(quality["gap_filling"]["inferred_points"])
            # never an invented accuracy number without a reference dataset
            self.assertIsNone(quality["georeferencing"]["accuracy_vs_reference"])
            self.assertEqual(quality["georeferencing"]["status"], "gps_anchored_no_independent_reference")


class TestQualitySummary(unittest.TestCase):
    def _report(self, **over):
        report = {"reconstruction": {"attempted": True, "ok": True, "attempts": 1},
                  "outputs": {"point_cloud": "a.laz", "textured_mesh": "b.obj"},
                  "gps_fusion": {"quality": "ok", "median_h_std_m": 1.2, "fallback_used": False},
                  "flight_validation": {"status": "valid", "single_pass": True, "path_efficiency": 0.97,
                                        "revisit_score": 0.0, "direction_reversals": 0,
                                        "trajectory_source": "keyframes_fused", "policy": "enforce",
                                        "overridden": False, "warnings": []},
                  "classified_cloud": {"points": 1000, "labelled_by_ai": 0.999, "median_views": 8,
                                       "seen_by_2plus_cameras": 0.99, "class_share_of_labelled": {"building": 0.8}},
                  "gap_filling": {"inferred_points_added": 400, "inferred_share": 0.004, "measured_points": 1000,
                                  "median_depth_fit_error": 0.02}}
        report.update(over)
        return report

    def test_separates_the_six_claims(self):
        q = evaluate.quality_summary(self._report())
        self.assertEqual(q["reconstruction"]["status"], "success")
        self.assertTrue(q["single_pass"]["single_pass"])
        self.assertEqual(q["point_cloud"]["points"], 1000)
        self.assertEqual(q["ai_classification"]["labelled_share"], 0.999)
        self.assertEqual(q["gap_filling"]["inferred_points"], 400)
        self.assertEqual(q["georeferencing"]["status"], "gps_anchored_no_independent_reference")

    def test_reference_accuracy_is_reported_when_it_exists(self):
        q = evaluate.quality_summary(self._report(accuracy_vs_lidar={"as_georeferenced": {"median_m": 0.4}}))
        self.assertEqual(q["georeferencing"]["status"], "measured_against_reference")
        self.assertEqual(q["georeferencing"]["accuracy_vs_reference"]["as_georeferenced"]["median_m"], 0.4)

    def test_missing_stages_are_named_not_invented(self):
        q = evaluate.quality_summary({"reconstruction": {"attempted": False}})
        self.assertEqual(q["reconstruction"]["status"], "not_attempted")
        self.assertEqual(q["point_cloud"]["status"], "not_run")
        self.assertEqual(q["ai_classification"]["status"], "not_run")
        self.assertIsNone(q["gap_filling"]["inferred_points"])
        self.assertIsNone(q["single_pass"]["path_efficiency"])
        self.assertTrue(q["ai_classification"]["note"])

    def test_failed_reconstruction_is_not_reported_as_success(self):
        q = evaluate.quality_summary({"reconstruction": {"attempted": True, "ok": False}})
        self.assertEqual(q["reconstruction"]["status"], "failed")
        self.assertEqual(q["georeferencing"]["status"], "not_run")

    def test_quality_summary_is_json_serialisable(self):
        json.dumps(evaluate.quality_summary(self._report()))


if __name__ == "__main__":
    unittest.main()
