"""
Deterministic unit tests for frame-selection logic.

Inspected implementation:
  - sih3d_pipeline/sih3d/frames.py  (thin wrapper, delegates to prepare_dataset)
  - sih3d_pipeline/scripts/prepare_dataset.py  (real frame-selection logic)

Pure / helper functions identified that can be tested without the full drone pipeline:
  * scripts.prepare_dataset.split_segments(cands, cuts)
  * scripts.prepare_dataset.select_keyframes(cands, seg, gps, min_shift, min_move, max_gap, blur_ratio)
  * scripts.prepare_dataset.geodetic_to_ecef / geodetic_to_enu
  * scripts.prepare_dataset.interpolate_gps(rows, times, offset)
  * sih3d.frames.load_calibration(path)
  * sih3d.frames._undistort_maps(calib, frame_w, frame_h, crop)

All tests are fast, deterministic, CPU-only and avoid:
  GPU / MPS, Docker, COLMAP, OpenMVS, actual video files, network.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2

# Make sih3d_pipeline importable both from repo root and from inside sih3d_pipeline.
# Repo root is two parents up from this file: tests/test_frame_selection.py -> Flight2world-demo
# sih3d_pipeline is a subdirectory of repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SIH3D_PIPELINE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(SIH3D_PIPELINE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.prepare_dataset import (
    geodetic_to_ecef,
    geodetic_to_enu,
    interpolate_gps,
    select_keyframes,
    split_segments,
)

from sih3d.frames import _undistort_maps, load_calibration

import unittest


def _make_cands(n, sharpness=100.0, shift=0.05, t_step=0.2, frame_step=6):
    """Helper: n candidates with linear time and fixed sharpness/shift."""
    return [
        {"frame": i * frame_step, "t": i * t_step, "sharpness": float(sharpness), "shift": shift}
        for i in range(n)
    ]


class TestSplitSegments(unittest.TestCase):
    """Tests for scripts.prepare_dataset.split_segments."""

    def test_no_cuts_single_segment(self):
        cands = _make_cands(3)
        self.assertEqual(split_segments(cands, []), [(0, 3)])

    def test_one_cut_two_segments(self):
        # cands frames: 0,6,12,18 ; cut at frame 6 should split after index 1
        cands = _make_cands(4, frame_step=6)  # frames 0,6,12,18
        # cut at 6 -> cands[1].frame ==6 >=6 triggers split at i=1
        self.assertEqual(split_segments(cands, [6]), [(0, 1), (1, 4)])

    def test_multiple_cuts_three_segments(self):
        cands = _make_cands(4, frame_step=10)  # 0,10,20,30
        # cuts at 10 and 20
        self.assertEqual(split_segments(cands, [10, 20]), [(0, 1), (1, 2), (2, 4)])

    def test_unsorted_cuts_are_sorted(self):
        cands = _make_cands(3, frame_step=5)  # 0,5,10
        # unsorted cuts [10,5] should behave same as [5,10]
        self.assertEqual(split_segments(cands, [10, 5]), split_segments(cands, [5, 10]))
        self.assertEqual(split_segments(cands, [10, 5]), [(0, 1), (1, 2), (2, 3)])

    def test_duplicate_cuts_collapsed(self):
        cands = _make_cands(3, frame_step=5)  # 0,5,10
        # duplicate cut 5 appears twice but should only split once
        self.assertEqual(split_segments(cands, [5, 5]), [(0, 1), (1, 3)])

    def test_cut_at_candidate_boundary(self):
        cands = _make_cands(3, frame_step=5)  # 0,5,10
        # cut exactly at second candidate's frame
        segs = split_segments(cands, [5])
        self.assertEqual(segs, [(0, 1), (1, 3)])

    def test_cut_before_first_candidate_ignored(self):
        cands = [{"frame": 10}, {"frame": 20}, {"frame": 30}]
        # cut at 0 is before any candidate and should not create a split
        self.assertEqual(split_segments(cands, [0]), [(0, 3)])

    def test_cut_beyond_last_candidate_ignored(self):
        cands = _make_cands(3, frame_step=5)
        self.assertEqual(split_segments(cands, [100]), [(0, 3)])

    def test_between_candidates_splits_between(self):
        # cands 0,10,20 ; cuts 5 and 15 each lie between candidates
        cands = _make_cands(3, frame_step=10)
        self.assertEqual(split_segments(cands, [5, 15]), [(0, 1), (1, 2), (2, 3)])

    def test_empty_candidates_returns_empty_segment(self):
        # Current implementation returns [(0,0)] for empty cands.
        # This is arguably a bug (expected []), but we assert current behavior
        # and flag it in the final report without fixing.
        self.assertEqual(split_segments([], []), [(0, 0)])
        self.assertEqual(split_segments([], [5]), [(0, 0)])

    def test_single_candidate_no_cut(self):
        self.assertEqual(split_segments([{"frame": 0}], []), [(0, 1)])

    def test_single_candidate_cut_at_same_frame(self):
        # single cand frame 0, cut at 0 -> implementation does not split because i>start is False
        self.assertEqual(split_segments([{"frame": 0}], [0]), [(0, 1)])

    def test_deterministic(self):
        cands = _make_cands(5)
        cuts = [12, 6]
        a = split_segments(cands, cuts)
        b = split_segments(cands, cuts)
        self.assertEqual(a, b)

    def test_many_candidates_many_cuts(self):
        cands = _make_cands(10, frame_step=3)  # frames 0,3,6,9,12,15,18,21,24,27
        cuts = [6, 15, 22]
        segs = split_segments(cands, cuts)
        # expect splits at indices where frame >= cut
        # cands index:0:0,1:3,2:6->cut6,3:9,4:12,5:15->cut15,6:18,7:21,8:24->cut22,9:27
        # -> segs: (0,2),(2,5),(5,8),(8,10)
        self.assertEqual(segs, [(0, 2), (2, 5), (5, 8), (8, 10)])


class TestSelectKeyframes(unittest.TestCase):
    """Tests for scripts.prepare_dataset.select_keyframes.

    The function bins a shot by accumulated motion and keeps the sharpest frame per bin.
    """

    def test_single_frame_kept(self):
        cands = [{"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.05}]
        gps = np.full((1, 3), np.nan)
        kept, blurry, gaps = select_keyframes(cands, (0, 1), gps, 0.1, None, 2.0, 0.35)
        self.assertEqual(kept, [0])
        self.assertEqual(blurry, [])
        self.assertEqual(gaps, 0)

    def test_two_frames_each_own_bin(self):
        # shift 0.05 / min_shift 0.1 =0.5 ; dt 0.2/max_gap2=0.1 => step 0.5 each
        # motion [0,0.5] => cumsum [0,0.5] => bins [0,0] actually both in same bin? Let's set shift larger.
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.0},
            {"frame": 6, "t": 0.2, "sharpness": 90.0, "shift": 0.2},
        ]
        gps = np.full((2, 3), np.nan)
        kept, blurry, gaps = select_keyframes(cands, (0, 2), gps, 0.1, None, 2.0, 0.35)
        # shift 0.2/0.1=2.0 => bins [0,2] => two bins each with one element
        self.assertEqual(kept, [0, 1])
        self.assertEqual(blurry, [])
        self.assertEqual(gaps, 0)

    def test_sharpest_per_bin_keeps_sharpest(self):
        # 3 cands in same bin: bins [0,0,0] if small shift, then only sharpest kept
        # To force two bins: we need motion that puts first two in bin0, third in bin1
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.0},
            {"frame": 1, "t": 0.2, "sharpness": 10.0, "shift": 0.05},
            {"frame": 2, "t": 0.4, "sharpness": 100.0, "shift": 0.05},
        ]
        gps = np.full((3, 3), np.nan)
        kept, blurry, gaps = select_keyframes(cands, (0, 3), gps, 0.1, None, 2.0, 0.35)
        # motion: [0,0.5,0.5] => cumsum [0,0.5,1.0] => bins [0,0,1]
        # bin0 members 0,1 => best 0 (100 vs10) => kept 0
        # bin1 member 2 => kept 2
        self.assertEqual(kept, [0, 2])
        self.assertEqual(blurry, [])

    def test_blurry_detection(self):
        # A bin where the single best is far below median*blur_ratio should be marked blurry.
        # Design: 3 candidates each in own bin (step 1.0 each), median 200, threshold 70,
        # last candidate sharpness 10 => blurry.
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 200.0, "shift": 0.0},
            {"frame": 1, "t": 0.2, "sharpness": 200.0, "shift": 0.1},  # step 1.0 -> bin 1
            {"frame": 2, "t": 0.4, "sharpness": 10.0, "shift": 0.1},  # step 1.0 -> bin 2, low sharpness
        ]
        gps = np.full((3, 3), np.nan)
        kept, blurry, _ = select_keyframes(cands, (0, 3), gps, 0.1, None, 10.0, 0.35)
        # median 200, threshold 70, bin2 best 10 <70 => blurry
        self.assertIn(2, blurry)
        self.assertNotIn(2, kept)
        self.assertIn(0, kept)
        self.assertIn(1, kept)

    def test_shift_none_fallback_to_one(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.0},
            {"frame": 6, "t": 0.2, "sharpness": 90.0, "shift": None},
        ]
        gps = np.full((2, 3), np.nan)
        kept, blurry, _ = select_keyframes(cands, (0, 2), gps, 0.1, None, 2.0, 0.35)
        # shift None => step 1.0 => bins [0,1] => two bins
        self.assertEqual(kept, [0, 1])

    def test_max_gap_triggers_new_bin_despite_small_shift(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.01},
            {"frame": 1, "t": 5.0, "sharpness": 100.0, "shift": 0.01},  # dt 5 /2 =2.5
        ]
        gps = np.full((2, 3), np.nan)
        kept, _, gaps = select_keyframes(cands, (0, 2), gps, 0.1, None, 2.0, 0.35)
        # motion second = max(0.1,2.5)=2.5 => bins [0,2] => still both kept, gaps counts jumps >2? diff 2 not >2 =>0
        self.assertEqual(kept, [0, 1])
        self.assertEqual(gaps, 0)

    def test_gps_min_move_mode(self):
        # 3 cands with GPS moving ~11m per step, min_move 10m => step ~1.1
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.01},
            {"frame": 1, "t": 0.2, "sharpness": 100.0, "shift": 0.01},
            {"frame": 2, "t": 0.4, "sharpness": 100.0, "shift": 0.01},
        ]
        gps = np.array([[48.0, 11.0, 500], [48.0001, 11.0, 500], [48.0002, 11.0, 500]], float)
        kept, blurry, _ = select_keyframes(cands, (0, 3), gps, 0.1, 10.0, 2.0, 0.35)
        # _with min_move_, shift is ignored and GPS distance drives binning
        self.assertEqual(len(kept) + len(blurry), 3)  # all in separate bins due to ~1.1 step
        self.assertEqual(kept, [0, 1, 2])

    def test_gps_nan_fallback_to_shift(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.05},
            {"frame": 1, "t": 0.2, "sharpness": 100.0, "shift": 0.05},
            {"frame": 2, "t": 0.4, "sharpness": 100.0, "shift": 0.05},
        ]
        gps = np.array([[48.0, 11.0, 500], [np.nan, np.nan, np.nan], [48.0002, 11.0, 500]], float)
        kept, _, _ = select_keyframes(cands, (0, 3), gps, 0.1, 10.0, 2.0, 0.35)
        # second step has NaN => fallback to shift 0.05/0.1=0.5
        # third step: prev has NaN => fallback as well? The code checks both prev and cur
        # So both steps use shift => motion ~0.5 each => bins [0,0,1] => kept [0,2] expected
        self.assertEqual(kept, [0, 2])

    def test_empty_segment_raises_index_error(self):
        # Current implementation crashes on empty segment; we assert this behavior without fixing.
        cands = [{"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.05}]
        gps = np.full((1, 3), np.nan)
        with self.assertRaises(IndexError):
            select_keyframes(cands, (0, 0), gps, 0.1, None, 2.0, 0.35)

    def test_min_shift_zero_raises(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.05},
            {"frame": 1, "t": 0.2, "sharpness": 100.0, "shift": 0.05},
        ]
        gps = np.full((2, 3), np.nan)
        with self.assertRaises(ZeroDivisionError):
            select_keyframes(cands, (0, 2), gps, 0.0, None, 2.0, 0.35)

    def test_max_gap_zero_raises(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.05},
            {"frame": 1, "t": 0.2, "sharpness": 100.0, "shift": 0.05},
        ]
        gps = np.full((2, 3), np.nan)
        with self.assertRaises(ZeroDivisionError):
            select_keyframes(cands, (0, 2), gps, 0.1, None, 0.0, 0.35)

    def test_blur_ratio_zero_keeps_all(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.0},
            {"frame": 1, "t": 0.2, "sharpness": 1.0, "shift": 0.2},
            {"frame": 2, "t": 0.4, "sharpness": 1.0, "shift": 0.2},
        ]
        gps = np.full((3, 3), np.nan)
        kept, blurry, _ = select_keyframes(cands, (0, 3), gps, 0.1, None, 2.0, 0.0)
        # threshold 0 => nothing is blurry
        self.assertEqual(len(blurry), 0)
        self.assertGreater(len(kept), 0)

    def test_blur_ratio_one_uses_median_threshold(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 10.0, "shift": 0.0},
            {"frame": 1, "t": 0.2, "sharpness": 10.0, "shift": 0.5},
            {"frame": 2, "t": 0.4, "sharpness": 100.0, "shift": 0.5},
        ]
        gps = np.full((3, 3), np.nan)
        kept, blurry, _ = select_keyframes(cands, (0, 3), gps, 0.1, None, 10.0, 1.0)
        # median 10 => threshold 10, best 10 in first bins not <10 => kept, best 10 in second bin etc.
        # All best >=10 => none blurry
        self.assertEqual(blurry, [])

    def test_all_sharpness_equal(self):
        cands = _make_cands(4, sharpness=50.0, shift=0.05)
        gps = np.full((4, 3), np.nan)
        kept, blurry, _ = select_keyframes(cands, (0, 4), gps, 0.1, None, 2.0, 0.35)
        self.assertEqual(len(blurry), 0)
        self.assertGreater(len(kept), 0)

    def test_deterministic(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.07},
            {"frame": 5, "t": 0.5, "sharpness": 80.0, "shift": 0.12},
            {"frame": 10, "t": 1.0, "sharpness": 90.0, "shift": 0.09},
            {"frame": 15, "t": 1.5, "sharpness": 110.0, "shift": 0.11},
        ]
        gps = np.full((4, 3), np.nan)
        a = select_keyframes(cands, (0, 4), gps, 0.1, None, 2.0, 0.35)
        b = select_keyframes(cands, (0, 4), gps, 0.1, None, 2.0, 0.35)
        self.assertEqual(a, b)

    def test_gap_count(self):
        # Force a large jump: shift huge => bins jump 5 each => kept bins will be sparse
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 100.0, "shift": 0.5},  # step 5
            {"frame": 1, "t": 0.5, "sharpness": 100.0, "shift": 0.5},
            {"frame": 2, "t": 1.0, "sharpness": 100.0, "shift": 0.5},
            {"frame": 3, "t": 1.5, "sharpness": 100.0, "shift": 0.5},
        ]
        gps = np.full((4, 3), np.nan)
        kept, _, gaps = select_keyframes(cands, (0, 4), gps, 0.1, None, 10.0, 0.35)
        # Each step 5 => bins [0,5,10,15] kept all => diff 5 each => gaps count 3 (since >2)
        self.assertEqual(gaps, 3)
        # If we fake blur to drop middle frames, gaps might be different; but we test kept case
        self.assertEqual(kept, [0, 1, 2, 3])

    def test_very_small_two_cands(self):
        cands = [
            {"frame": 0, "t": 0.0, "sharpness": 50.0, "shift": 0.0},
            {"frame": 1, "t": 0.1, "sharpness": 60.0, "shift": 0.09},
        ]
        gps = np.full((2, 3), np.nan)
        kept, blurry, _ = select_keyframes(cands, (0, 2), gps, 0.1, None, 2.0, 0.35)
        # median 55 threshold 19.25 => both above => kept both if in separate bins? shift 0.09/0.1=0.9 => bins [0,0] => one bin best is 60 => kept [1]
        self.assertEqual(kept, [1])
        self.assertEqual(blurry, [])


class TestGeodetic(unittest.TestCase):
    """Tests for geodetic helpers."""

    def test_same_point_zero_enu(self):
        enu = geodetic_to_enu(48.0, 11.0, 500, (48.0, 11.0, 500))
        np.testing.assert_allclose(enu, np.zeros(3), atol=1e-9)

    def test_ecef_equator_prime_meridian(self):
        # At lat 0, lon 0, alt 0 => ECEF x = a, y=0, z=0
        ecef = geodetic_to_ecef(0, 0, 0)
        self.assertAlmostEqual(float(ecef[0]), 6378137.0, places=1)
        self.assertAlmostEqual(float(ecef[1]), 0.0, places=1)
        self.assertAlmostEqual(float(ecef[2]), 0.0, places=1)

    def test_enu_north_displacement(self):
        # Small north displacement ~ 0.0001 deg ~ 11m
        enu = geodetic_to_enu(np.array([48.0001]), np.array([11.0]), np.array([500]), (48.0, 11.0, 500))
        # north should be positive, east ~0
        self.assertGreater(enu[0, 1], 5.0)  # north
        self.assertAlmostEqual(enu[0, 0], 0.0, delta=1.0)  # east ~0

    def test_enu_mixed_array(self):
        lats = np.array([48.0, 48.0001])
        lons = np.array([11.0, 11.0001])
        alts = np.array([500.0, 505.0])
        enu = geodetic_to_enu(lats, lons, alts, (48.0, 11.0, 500))
        self.assertEqual(enu.shape, (2, 3))
        np.testing.assert_allclose(enu[0], np.zeros(3), atol=1e-6)

    def test_deterministic(self):
        a = geodetic_to_enu(48.1, 11.2, 510, (48.0, 11.0, 500))
        b = geodetic_to_enu(48.1, 11.2, 510, (48.0, 11.0, 500))
        np.testing.assert_array_equal(a, b)


class TestInterpolateGps(unittest.TestCase):
    """Tests for interpolate_gps."""

    def test_interpolation_inside(self):
        rows = [(0.0, 48.0, 11.0, 500), (1.0, 48.0001, 11.0, 500), (2.0, 48.0002, 11.0, 500)]
        times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        out = interpolate_gps(rows, times, 0.0)
        # middle value interpolated
        self.assertAlmostEqual(out[1, 0], 48.00005, places=7)
        self.assertAlmostEqual(out[3, 0], 48.00015, places=7)

    def test_outside_range_is_nan(self):
        rows = [(0.0, 48.0, 11.0, 500), (1.0, 48.0001, 11.0, 500)]
        times = np.array([-1.0, 0.0, 1.0, 2.0])
        out = interpolate_gps(rows, times, 0.0)
        self.assertTrue(np.isnan(out[0, 0]))
        self.assertTrue(np.isnan(out[3, 0]))
        self.assertFalse(np.isnan(out[1, 0]))

    def test_offset_shifts_interpolation(self):
        rows = [(0.0, 48.0, 11.0, 500), (1.0, 48.0001, 11.0, 500)]
        times = np.array([0.5])
        out_no_offset = interpolate_gps(rows, times, 0.0)
        out_offset = interpolate_gps(rows, times, 0.5)  # rows shifted +0.5 => aligns with 0.0 row
        # With offset 0.5, telemetry times become 0.5 and 1.5, so time 0.5 should exactly match first row
        self.assertAlmostEqual(out_offset[0, 0], 48.0, places=7)
        # without offset, time 0.5 is midpoint
        self.assertAlmostEqual(out_no_offset[0, 0], 48.00005, places=7)

    def test_empty_times(self):
        rows = [(0.0, 48.0, 11.0, 500)]
        times = np.array([])
        out = interpolate_gps(rows, times, 0.0)
        self.assertEqual(out.shape, (0, 3))

    def test_empty_rows_raises_index_error(self):
        # Current implementation crashes on empty rows; assert this behavior
        with self.assertRaises(IndexError):
            interpolate_gps([], np.array([0.0]), 0.0)

    def test_deterministic(self):
        rows = [(0.0, 48.0, 11.0, 500), (2.0, 48.0002, 11.0, 500)]
        times = np.array([1.0])
        a = interpolate_gps(rows, times, 0.0)
        b = interpolate_gps(rows, times, 0.0)
        np.testing.assert_array_equal(a, b)

    def test_unsorted_rows_handled(self):
        rows = [(2.0, 48.0002, 11.0, 500), (0.0, 48.0, 11.0, 500), (1.0, 48.0001, 11.0, 500)]
        times = np.array([0.5, 1.5])
        out = interpolate_gps(rows, times, 0.0)
        self.assertAlmostEqual(out[0, 0], 48.00005, places=7)
        self.assertAlmostEqual(out[1, 0], 48.00015, places=7)


class TestLoadCalibration(unittest.TestCase):
    """Tests for sih3d.frames.load_calibration."""

    def test_load_npz(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "calib.npz"
            K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], float)
            dist = np.array([0.1, -0.05, 0.0, 0.0], float)
            np.savez(p, intrinsic_matrix=K, distCoeff=dist)
            calib = load_calibration(p)
            np.testing.assert_allclose(calib["K"], K)
            np.testing.assert_allclose(calib["dist"], dist)
            self.assertIsNone(calib["size"])

    def test_load_json_without_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "calib.json"
            data = {"K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], "dist": [0.1, -0.05, 0.0, 0.0]}
            p.write_text(json.dumps(data))
            calib = load_calibration(p)
            np.testing.assert_allclose(calib["K"], np.array(data["K"], float))
            np.testing.assert_allclose(calib["dist"], np.array(data["dist"], float))
            self.assertIsNone(calib.get("size"))

    def test_load_json_with_size(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "calib.json"
            data = {"K": [[800, 0, 640], [0, 800, 360], [0, 0, 1]], "dist": [0.05, 0.0, 0.0, 0.0], "size": [1280, 720]}
            p.write_text(json.dumps(data))
            calib = load_calibration(p)
            self.assertEqual(calib["size"], [1280, 720])
            np.testing.assert_allclose(calib["K"], np.array(data["K"], float))

    def test_load_invalid_path_raises(self):
        with self.assertRaises((FileNotFoundError, OSError)):
            load_calibration(Path("/nonexistent/calib.npz"))

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            data = {"K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], "dist": [0, 0, 0, 0]}
            p.write_text(json.dumps(data))
            a = load_calibration(p)
            b = load_calibration(p)
            np.testing.assert_array_equal(a["K"], b["K"])


class TestUndistortMaps(unittest.TestCase):
    """Tests for sih3d.frames._undistort_maps."""

    def setUp(self):
        self.calib_no_dist = {"K": np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], float),
                              "dist": np.array([0, 0, 0, 0], float), "size": None}
        self.calib_with_dist = {"K": np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], float),
                                "dist": np.array([0.1, -0.05, 0, 0], float), "size": None}

    def test_no_distortion_maps_shape(self):
        maps, new_k, roi = _undistort_maps(self.calib_no_dist, 1920, 1080, (0, 0, 1920, 1080))
        self.assertEqual(len(maps), 2)
        self.assertEqual(new_k.shape, (3, 3))
        self.assertEqual(len(roi), 4)
        # maps should be usable by cv2.remap; check dtypes
        self.assertEqual(maps[0].dtype, np.int16)  # CV_16SC2

    def test_with_distortion_preserves_size(self):
        maps, new_k, roi = _undistort_maps(self.calib_with_dist, 1920, 1080, (0, 0, 1920, 1080))
        self.assertEqual(maps[0].shape[:2], (1080, 1920))

    def test_scaled_with_size_field(self):
        # calib recorded at 1920x1080 but frame is 1280x720
        calib = {"K": np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], float),
                 "dist": np.array([0.05, 0, 0, 0], float), "size": [1920, 1080]}
        maps, new_k, _ = _undistort_maps(calib, 1280, 720, (0, 0, 1280, 720))
        # focal should be scaled by 1280/1920 == 0.666...
        self.assertAlmostEqual(new_k[0, 0], 1000 * 1280 / 1920, delta=50)  # optimal new K may differ slightly

    def test_crop_adjusts_principal_point(self):
        maps1, new_k1, _ = _undistort_maps(self.calib_no_dist, 1920, 1080, (0, 0, 1920, 1080))
        maps2, new_k2, _ = _undistort_maps(self.calib_no_dist, 1920, 1080, (10, 20, 1910, 1060))
        # cropping should shift principal point and change map size
        self.assertEqual(maps2[0].shape[:2], (1040, 1900))
        # new_k principal point should be shifted by crop x0,y0
        self.assertLess(new_k2[0, 2], new_k1[0, 2])
        self.assertLess(new_k2[1, 2], new_k1[1, 2])

    def test_deterministic(self):
        a = _undistort_maps(self.calib_with_dist, 640, 480, (0, 0, 640, 480))
        b = _undistort_maps(self.calib_with_dist, 640, 480, (0, 0, 640, 480))
        np.testing.assert_array_equal(a[1], b[1])

    def test_zero_distortion_new_k_close_to_original(self):
        # With no distortion and alpha 0, new_k should be close to original
        maps, new_k, _ = _undistort_maps(self.calib_no_dist, 640, 480, (0, 0, 640, 480))
        np.testing.assert_allclose(new_k[0, 0], 1000, atol=10)
        np.testing.assert_allclose(new_k[1, 1], 1000, atol=10)


class TestNormaliseExposure(unittest.TestCase):
    """Smoke test for _normalise_exposure with synthetic images – ensures no crash and returns dict."""

    def test_with_three_synthetic_images(self):
        from sih3d.frames import _normalise_exposure

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # create 3 images with increasing brightness
            for i in range(3):
                img = np.full((64, 64, 3), 30 + i * 80, dtype=np.uint8)
                cv2.imwrite(str(td / f"img{i}.jpg"), img)
            paths = [td / f"img{i}.jpg" for i in range(3)]
            res = _normalise_exposure(paths, strength=0.7)
            self.assertIn("target_lightness", res)
            self.assertIn("lightness_range_before", res)
            self.assertIsInstance(res["target_lightness"], float)
            self.assertEqual(len(res["lightness_range_before"]), 2)

    def test_deterministic(self):
        from sih3d.frames import _normalise_exposure
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # use copies of same image; second run should give same target
            for i in range(2):
                img = np.full((32, 32, 3), 100, dtype=np.uint8)
                cv2.imwrite(str(td / f"a{i}.jpg"), img)
            # we need fresh copies for deterministic check because normalise mutates files
            # create two identical sets
            import shutil
            td2 = Path(tempfile.mkdtemp())
            td3 = Path(tempfile.mkdtemp())
            for src in [td / "a0.jpg", td / "a1.jpg"]:
                shutil.copy(src, td2 / src.name)
                shutil.copy(src, td3 / src.name)
            res1 = _normalise_exposure([td2 / "a0.jpg", td2 / "a1.jpg"])
            res2 = _normalise_exposure([td3 / "a0.jpg", td3 / "a1.jpg"])
            self.assertEqual(res1["target_lightness"], res2["target_lightness"])


if __name__ == "__main__":
    unittest.main()
