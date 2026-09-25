"""Tests for robust GPS/telemetry fusion (sih3d fusion).

Deterministic synthetic-trajectory tests plus a real-data regression fixture
(fixtures/aukerman_srt_excerpt.srt, an excerpt of the Aukerman survey's DJI-style
.SRT).  Together they protect against the observed Aukerman failure mode: the
Kalman gate mass-rejecting legitimate fixes (65/77 east) and the fused track
drifting 32-95 m from the real survey, silently.

Behaviour under test:
  * legitimate motion (straight flight, survey U-turns) is not systematically rejected;
  * an isolated genuine outlier is still rejected, without triggering fallback;
  * poor telemetry where the filter cannot track activates the deterministic fallback;
  * fallback output stays on the raw GPS track (no invented drift or precision);
  * diagnostics expose counts, ratios, innovation magnitudes, fallback flag, quality;
  * results are deterministic across runs.

No Docker, no network, no GPU; each test runs on the CPU in well under a second.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# make pipeline importable from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sih3d import fusion
from sih3d.telemetry import Telemetry, from_srt
from scripts.prepare_dataset import geodetic_to_enu

ORG = (41.0, -81.0, 300.0)  # near the real Aukerman site
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "aukerman_srt_excerpt.srt"


def _tel(e, n, u, dt, h_acc=None, source="synthetic"):
    """Telemetry built from an ENU trajectory around ORG (no accuracy unless given)."""
    e, n, u = np.asarray(e, float), np.asarray(n, float), np.asarray(u, float)
    lat, lon, alt = fusion.enu_to_geodetic(np.column_stack([e, n, u]), ORG)
    ha = np.full(len(e), np.nan) if h_acc is None else np.full(len(e), float(h_acc))
    return Telemetry(np.arange(len(e)) * dt, lat, lon, alt, ha, np.full(len(e), np.nan), source=source)


def _fused_enu(f):
    enu = geodetic_to_enu(f["lat"], f["lon"], f["alt"], ORG)
    return enu[:, 0], enu[:, 1], enu[:, 2]


def _straight_flight(n_fixes=180, speed=5.0, noise=1.5, seed=7):
    rng = np.random.default_rng(seed)
    e = speed * np.arange(n_fixes) + rng.normal(0, noise, n_fixes)
    n = 2.0 + rng.normal(0, noise, n_fixes)
    u = np.full(n_fixes, 50.0)
    return e, n, u


def _lawnmower(n_legs=4, leg_fixes=24, dt=1.0, noise=1.5, seed=11):
    """Realistic survey pattern: straight legs with curved U-turns between them."""
    rng = np.random.default_rng(seed)
    e, n = [0.0], [0.0]
    x, y = 0.0, 0.0
    for leg in range(n_legs):
        for _ in range(leg_fixes):
            y += 8.0 * dt
            e.append(x)
            n.append(y)
        if leg < n_legs - 1:  # curved 5-fix U-turn shifting 60 m east to the next line
            for i in range(5):
                x += 12.0
                y += 8.0 * dt * np.cos(np.pi * (i + 1) / 6)
                e.append(x)
                n.append(y)
    e = np.array(e) + rng.normal(0, noise, len(e))
    n = np.array(n) + rng.normal(0, noise, len(n))
    u = np.full(len(e), 50.0)
    return e, n, u, dt


class TestSmoothFlight(unittest.TestCase):
    def test_legitimate_motion_not_rejected_and_track_accurate(self):
        e, n, u = _straight_flight()
        tel = _tel(e, n, u, 1.0)
        f = fusion.fuse(tel, tel.t)
        s = f["stats"]
        self.assertLessEqual(s["rejection_ratio"], 0.05)
        self.assertGreaterEqual(s["accepted"], int(0.95 * len(e)))
        self.assertFalse(s["fallback_used"])
        self.assertEqual(s["quality"], "ok")
        self.assertEqual(s["warnings"], [])
        fe, fn, _ = _fused_enu(f)
        err = np.hypot(fe - e, fn - n)
        # the smoother must not sit systematically off the raw fixes (old bug: 32-95 m)
        self.assertLess(float(np.median(err)), 2.5)

    def test_deterministic_across_runs(self):
        e, n, u, dt = _lawnmower()
        tel = _tel(e, n, u, dt)
        a, b = fusion.fuse(tel, tel.t), fusion.fuse(tel, tel.t)
        for k in ("lat", "lon", "alt", "h_std", "v_std"):
            self.assertTrue(np.array_equal(a[k], b[k]), k)
        self.assertEqual(a["stats"], b["stats"])


class TestIsolatedOutlier(unittest.TestCase):
    def test_single_outlier_rejected_without_fallback(self):
        e, n, u = _straight_flight()
        truth_e = e.copy()
        e[90] += 80.0  # one impossible jump among clean fixes
        tel = _tel(e, n, u, 1.0)
        f = fusion.fuse(tel, tel.t)
        s = f["stats"]
        self.assertEqual(s["rejected_east"] + s["rejected_north"], 1)
        self.assertFalse(s["fallback_used"])
        self.assertEqual(s["quality"], "ok")
        # the innovation magnitude is visible in the diagnostics
        self.assertGreaterEqual(s["max_innovation_m"], 40.0)
        fe, fn, _ = _fused_enu(f)
        self.assertLess(float(np.hypot(fe[90] - truth_e[90], fn[90] - n[90])), 5.0)


class TestLegitimatePositionChanges(unittest.TestCase):
    def test_survey_turns_not_systematically_rejected(self):
        e, n, u, dt = _lawnmower()
        tel = _tel(e, n, u, dt)
        f = fusion.fuse(tel, tel.t)
        s = f["stats"]
        self.assertLessEqual(s["rejection_ratio"], 0.20)
        self.assertFalse(s["fallback_used"])
        self.assertIn(s["quality"], ("ok", "degraded"))
        fe, fn, _ = _fused_enu(f)
        err = np.hypot(fe - e, fn - n)
        self.assertLess(float(np.median(err)), 6.0)


class TestFallbackActivation(unittest.TestCase):
    def test_poor_telemetry_activates_fallback_and_stays_on_raw_track(self):
        rng = np.random.default_rng(5)
        n_fixes = 60
        step = np.where(np.arange(n_fixes) % 2 == 0, 95.0, -105.0) + rng.normal(0, 5, n_fixes)
        e = np.cumsum(step) * 0.5 + rng.normal(0, 2, n_fixes)
        n = rng.normal(0, 2, n_fixes)
        u = np.full(n_fixes, 50.0)
        tel = _tel(e, n, u, 0.5)
        f = fusion.fuse(tel, tel.t)
        s = f["stats"]
        self.assertTrue(s["fallback_used"])
        self.assertTrue(s["fallback_reason"])
        self.assertEqual(s["quality"], "fallback")
        self.assertTrue(s["warnings"])
        # the fallback output IS the raw GPS track: exact at the fix times, no drift
        fe, fn, _ = _fused_enu(f)
        self.assertLess(float(np.max(np.hypot(fe - e, fn - n))), 0.01)

    def test_excessive_rejection_always_means_fallback(self):
        # whatever the trigger, the invariant "rejection ratio > 30% => fallback" must hold
        e, n, u, dt = _lawnmower(n_legs=2, leg_fixes=10, dt=0.5, seed=23)
        tel = _tel(e, n, u, dt, h_acc=0.05)  # absurdly confident receiver + jumpy short track
        s = fusion.fuse(tel, tel.t)["stats"]
        if s["rejection_ratio"] > fusion.FALLBACK_REJECTION_RATIO:
            self.assertTrue(s["fallback_used"])
            self.assertTrue(s["warnings"])


class TestReportedAccuracyAdaptivity(unittest.TestCase):
    def test_thresholds_scale_with_reported_accuracy(self):
        rng = np.random.default_rng(3)
        n_fixes = 100
        e = 2.0 * np.arange(n_fixes) + rng.normal(0, 4.0, n_fixes)  # 4 m noise...
        n = rng.normal(0, 4.0, n_fixes)
        u = np.full(n_fixes, 50.0)
        tel = _tel(e, n, u, 1.0, h_acc=10.0)                        # ...receiver honestly says 10 m
        s = fusion.fuse(tel, tel.t)["stats"]
        self.assertEqual(s["rejected"], 0)
        self.assertFalse(s["fallback_used"])
        self.assertEqual(s["quality"], "ok")

    def test_overconfident_accuracy_cannot_starve_the_filter_silently(self):
        # reported 0.1 m accuracy on a noisier track: gates must still adapt via the
        # track's own dynamics floor instead of rejecting everything (and if they did,
        # the fallback + warning would surface it - never silent)
        rng = np.random.default_rng(19)
        n_fixes = 80
        e = 3.0 * np.arange(n_fixes) + rng.normal(0, 2.0, n_fixes)
        n = rng.normal(0, 2.0, n_fixes)
        u = np.full(n_fixes, 50.0)
        tel = _tel(e, n, u, 1.0, h_acc=0.1)
        s = fusion.fuse(tel, tel.t)["stats"]
        self.assertLessEqual(s["rejection_ratio"], 0.30)
        if s["rejection_ratio"] > fusion.WARN_REJECTION_RATIO:
            self.assertTrue(s["warnings"])


class TestStatsBackwardCompatible(unittest.TestCase):
    def test_historical_keys_unchanged_and_diagnostics_added(self):
        e, n, u = _straight_flight(n_fixes=60)
        tel = _tel(e, n, u, 1.0)
        s = fusion.fuse(tel, tel.t)["stats"]
        for legacy in ("gps_fixes", "rejected_east", "rejected_north", "rejected_up",
                       "barometer_used", "median_h_std_m", "median_v_std_m"):
            self.assertIn(legacy, s)
        for added in ("accepted", "accepted_east", "accepted_north", "accepted_up",
                      "rejected", "rejection_ratio", "max_innovation_m", "median_innovation_m",
                      "median_residual_m", "fallback_used", "fallback_reason", "quality", "warnings"):
            self.assertIn(added, s)
        self.assertEqual(s["gps_fixes"], 60)
        self.assertIsInstance(s["warnings"], list)

    def test_barometer_path_still_used(self):
        rng = np.random.default_rng(2)
        n_fixes = 120
        e = 3.0 * np.arange(n_fixes) + rng.normal(0, 1.5, n_fixes)
        n = rng.normal(0, 1.5, n_fixes)
        t = np.arange(n_fixes) * 1.0
        alt_gps = 100.0 + rng.normal(0, 2.0, n_fixes)
        bt = np.arange(0, n_fixes, 0.2)
        drift = np.cumsum(rng.normal(0, 0.05, len(bt)))
        lat, lon, _ = fusion.enu_to_geodetic(np.column_stack([e, n, np.full(n_fixes, 100.0)]), ORG)
        tel = Telemetry(t, lat, lon, alt_gps, np.full(n_fixes, np.nan), np.full(n_fixes, np.nan),
                        baro_t=bt, baro_alt=100.0 + drift, source="baro-test")
        s = fusion.fuse(tel, t)["stats"]
        self.assertTrue(s["barometer_used"])
        self.assertFalse(s["fallback_used"])
        self.assertLessEqual(s["median_v_std_m"], 1.0)  # smooth baro is trusted over noisy GPS alt


class TestAukermanRealDataRegression(unittest.TestCase):
    """Regression for the observed Aukerman failure mode: dozens of legitimate GPS fixes
    rejected, fused trajectory 32-95 m from the real survey, no warning anywhere.  The
    fixture is a small excerpt of that real run's DJI-style .SRT (no photos committed)."""

    def test_fixture_exists_and_parses(self):
        self.assertTrue(FIXTURE.exists())
        tel = from_srt(FIXTURE)
        self.assertGreaterEqual(len(tel.t), 20)

    def test_no_silent_drift_on_real_survey_telemetry(self):
        tel = from_srt(FIXTURE)
        f = fusion.fuse(tel, tel.t)
        s = f["stats"]
        fe, fn, _ = _fused_enu(f)
        ren = geodetic_to_enu(tel.lat, tel.lon, tel.alt, ORG)
        res = np.hypot(fe - ren[:, 0], fn - ren[:, 1])
        # the fused track never silently drifts away from the raw survey observations
        self.assertLess(float(np.median(res)), 5.0)
        # excessive rejection must always activate the declared fallback, never stay silent
        if s["rejection_ratio"] > fusion.FALLBACK_REJECTION_RATIO:
            self.assertTrue(s["fallback_used"])
            self.assertEqual(s["quality"], "fallback")
        # and the real telemetry IS the pathological case: the filter's own track is
        # materially off, so the run must say so loudly instead of trusting the filter
        self.assertTrue(s["fallback_used"])
        self.assertEqual(s["quality"], "fallback")
        self.assertTrue(s["warnings"])
        self.assertTrue(s["fallback_reason"])
        # vertical stayed healthy in the original failure and must remain metre-class
        self.assertLess(s["median_v_std_m"], 10.0)


if __name__ == "__main__":
    unittest.main()
