"""
Deterministic unit tests for the fill_gaps() no-op hardlink optimization.

Inspected implementation:
  - sih3d_pipeline/sih3d/holes.py  (fill_gaps() no-op path uses odm._hardlink_or_copy)
  - sih3d_pipeline/sih3d/odm.py     (_hardlink_or_copy helper, shutil.copy2 fallback)

Covered behaviour:
  * no-op fill (zero new points) creates the output file
  * on the same filesystem, output is a hard link of the input (same device + inode)
  * input stays readable and byte-identical; output content equals input content
  * OSError from Path.hardlink_to() falls back to shutil.copy2 (different inode, identical content)
  * the points-added path still re-encodes via laspy (new inode) with unchanged stats/report shape

All tests are fast, deterministic, CPU-only and avoid:
  GPU, Docker, ODM, COLMAP/OpenMVS, real reconstruction runs, network, model downloads.
  Only the heavy, unused-here modules (torch, scipy, transformers) are stubbed;
  laspy/numpy/cv2 are real. Only temporary directories are used.
"""

import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import laspy
import numpy as np

# make pipeline importable from repo root (same bootstrap as test_odm_setup.py,
# works both from tests/ and from sih3d_pipeline/tests/)
REPO_ROOT = Path(__file__).resolve().parents[1]
PIPE = REPO_ROOT / "sih3d_pipeline"
for p in (str(REPO_ROOT), str(PIPE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sih3d.holes import fill_gaps


# ---------------------------------------------------------------------------
# heavy-module stubs (torch / scipy / transformers are never genuinely needed:
# the no-op path runs no inference, and the add-path test drives one
# deterministic fake frame through the real projection/geometry code)
# ---------------------------------------------------------------------------

class _FakeDepthTensor:
    """Mimics the tiny slice of the torch API fill_gaps touches."""

    def __init__(self, value=1.0, shape=None):
        self._value = value
        self._shape = shape

    def unsqueeze(self, _dim):
        return self

    def __getitem__(self, _key):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.full(self._shape, self._value, dtype=np.float32)


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _heavy_stubs():
    """Fresh {sys.modules name: stub} mapping for one fill_gaps() call."""
    torch_stub = types.ModuleType("torch")
    torch_stub.no_grad = lambda: _FakeNoGrad()
    torch_stub.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch_stub.nn = SimpleNamespace(
        functional=SimpleNamespace(
            interpolate=lambda t, size=None, mode=None: _FakeDepthTensor(shape=size)
        )
    )

    transformers_stub = types.ModuleType("transformers")

    class _Proc:
        @classmethod
        def from_pretrained(cls, *a, **k):
            def proc(images=None, return_tensors=None):
                return SimpleNamespace(to=lambda device: {})

            return proc

    class _Model:
        @classmethod
        def from_pretrained(cls, *a, **k):
            return _Model()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(predicted_depth=_FakeDepthTensor())

    transformers_stub.AutoImageProcessor = _Proc
    transformers_stub.AutoModelForDepthEstimation = _Model

    scipy_pkg = types.ModuleType("scipy")
    scipy_pkg.__path__ = []
    scipy_spatial = types.ModuleType("scipy.spatial")

    class _FakeKDTree:
        def __init__(self, pts):
            self.n = len(pts)

        def query(self, pts, distance_upper_bound=None):
            m = len(np.asarray(pts))
            return np.full(m, 0.5), np.zeros(m, dtype=np.int64)

    scipy_spatial.cKDTree = _FakeKDTree

    return {
        "torch": torch_stub,
        "transformers": transformers_stub,
        "scipy": scipy_pkg,
        "scipy.spatial": scipy_spatial,
    }


# ---------------------------------------------------------------------------
# fixture helpers (temporary directories only)
# ---------------------------------------------------------------------------

OFF_X, OFF_Y = 1000.0, 2000.0


def _write_project(root, shots, cameras):
    root = Path(root)
    (root / "opensfm").mkdir(parents=True, exist_ok=True)
    (root / "opensfm" / "reconstruction.json").write_text(
        json.dumps([{"shots": shots, "cameras": cameras}])
    )
    (root / "odm_georeferencing").mkdir(parents=True, exist_ok=True)
    (root / "odm_georeferencing" / "coords.txt").write_text(
        f"WGS84 UTM 32N\n{OFF_X} {OFF_Y} 0\n"
    )
    (root / "frames").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    return root


def _write_cloud(path, x, y, z, classification=2):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.asarray(x, dtype=np.float64)
    las.y = np.asarray(y, dtype=np.float64)
    las.z = np.asarray(z, dtype=np.float64)
    n = len(las.points)
    las.red = np.full(n, 1000, dtype=np.uint16)
    las.green = np.full(n, 2000, dtype=np.uint16)
    las.blue = np.full(n, 3000, dtype=np.uint16)
    las.classification = np.full(n, classification, dtype=np.uint8)
    las.write(path)
    return path


def _links_to(src, dst):
    """True if dst is a real hard link to src (same device + inode, not a symlink)."""
    src, dst = Path(src), Path(dst)
    if not (src.is_file() and dst.is_file()) or dst.is_symlink():
        return False
    s, d = os.stat(src), os.stat(dst)
    return (s.st_dev, s.st_ino) == (d.st_dev, d.st_ino)


def _points_of(path):
    las = laspy.read(path)
    return (
        np.asarray(las.x),
        np.asarray(las.y),
        np.asarray(las.z),
        np.asarray(las.classification),
    )


class FillGapsNoopBase(unittest.TestCase):
    """Shared fixture: project with no shots, so fill_gaps() adds zero points."""

    def _noop_project(self, td, n_points=100, seed=7):
        project = _write_project(Path(td) / "project", shots={}, cameras={})
        rng = np.random.default_rng(seed)
        cloud = project / "sih3d_classified.laz"
        _write_cloud(
            cloud,
            OFF_X + rng.uniform(-5, 5, n_points),
            OFF_Y + rng.uniform(-5, 5, n_points),
            rng.uniform(9, 11, n_points),
        )
        return project, cloud

    def _run_noop(self, project, out_name="sih3d_filled.laz"):
        out = project / out_name
        with mock.patch.dict(sys.modules, _heavy_stubs()):
            res = fill_gaps(
                project, project / "frames", project / "labels", out, device="cpu"
            )
        return res, out


class TestFillGapsNoop(FillGapsNoopBase):
    def test_noop_creates_output_with_unchanged_stats_shape(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            res, out = self._run_noop(project)
            self.assertTrue(out.is_file())
            # report.json-facing stats dict is unchanged by the optimization
            self.assertEqual(res["inferred_points_added"], 0)
            self.assertEqual(res["output"], str(out))
            self.assertEqual(
                set(res),
                {"model", "frames_used", "median_depth_fit_error",
                 "inferred_points_added", "output"},
            )
            self.assertEqual(res["frames_used"], 0)
            self.assertIsNone(res["median_depth_fit_error"])

    def test_noop_output_is_hardlink_to_input(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            _, out = self._run_noop(project)
            self.assertTrue(_links_to(cloud, out))
            self.assertGreaterEqual(os.stat(cloud).st_nlink, 2)

    def test_noop_input_readable_and_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            before_bytes = cloud.read_bytes()
            before_points = _points_of(cloud)
            self._run_noop(project)
            self.assertEqual(cloud.read_bytes(), before_bytes)
            after_points = _points_of(cloud)  # still readable by laspy
            for before, after in zip(before_points, after_points):
                np.testing.assert_array_equal(after, before)

    def test_noop_output_content_matches_input(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            _, out = self._run_noop(project)
            self.assertEqual(out.read_bytes(), cloud.read_bytes())
            for out_arr, in_arr in zip(_points_of(out), _points_of(cloud)):
                np.testing.assert_array_equal(out_arr, in_arr)

    def test_noop_accepts_str_out_path(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            out = project / "sih3d_filled.laz"
            with mock.patch.dict(sys.modules, _heavy_stubs()):
                res = fill_gaps(
                    project, project / "frames", project / "labels",
                    str(out), device="cpu",
                )
            self.assertEqual(res["output"], str(out))
            self.assertTrue(_links_to(cloud, out))


class TestFillGapsNoopFallback(FillGapsNoopBase):
    def test_oserror_falls_back_to_copy2(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            out = project / "sih3d_filled.laz"
            with mock.patch.dict(sys.modules, _heavy_stubs()):
                with mock.patch.object(
                    Path, "hardlink_to", side_effect=OSError("cross-device")
                ):
                    with mock.patch(
                        "shutil.copy2", wraps=shutil.copy2
                    ) as copy_spy:
                        res = fill_gaps(
                            project, project / "frames", project / "labels",
                            out, device="cpu",
                        )
            self.assertEqual(copy_spy.call_count, 1)
            self.assertEqual(res["inferred_points_added"], 0)
            self.assertEqual(res["output"], str(out))
            self.assertTrue(out.is_file())

    def test_fallback_output_has_different_inode_but_identical_content(self):
        with tempfile.TemporaryDirectory() as td:
            project, cloud = self._noop_project(td)
            before_bytes = cloud.read_bytes()
            out = project / "sih3d_filled.laz"
            with mock.patch.dict(sys.modules, _heavy_stubs()):
                with mock.patch.object(
                    Path, "hardlink_to", side_effect=OSError("cross-device")
                ):
                    fill_gaps(
                        project, project / "frames", project / "labels",
                        out, device="cpu",
                    )
            s, d = os.stat(cloud), os.stat(out)
            self.assertFalse((s.st_dev, s.st_ino) == (d.st_dev, d.st_ino))
            self.assertEqual(out.read_bytes(), before_bytes)
            self.assertEqual(cloud.read_bytes(), before_bytes)
            for out_arr, in_arr in zip(_points_of(out), _points_of(cloud)):
                np.testing.assert_array_equal(out_arr, in_arr)


class TestFillGapsPointsAdded(unittest.TestCase):
    """The points-added path must be byte-for-byte behaviourally unchanged."""

    def test_added_points_still_reencode_via_laspy(self):
        from sih3d.semantics import NAMES

        w = h = 400
        n_points = 600
        with tempfile.TemporaryDirectory() as td:
            project = _write_project(
                Path(td) / "project",
                shots={
                    "frame_000000.jpg": {
                        "rotation": [0.0, 0.0, 0.0],
                        "translation": [0.0, 0.0, 0.0],
                        "camera": "cam0",
                    }
                },
                cameras={
                    "cam0": {
                        "width": w,
                        "height": h,
                        "focal": 0.5,
                        "k1": 0.0,
                        "k2": 0.0,
                        "projection_type": "perspective",
                    }
                },
            )
            rng = np.random.default_rng(3)
            cloud = project / "sih3d_classified.laz"
            _write_cloud(
                cloud,
                OFF_X + rng.uniform(-9, 9, n_points),
                OFF_Y + rng.uniform(-9, 9, n_points),
                rng.uniform(9.5, 10.5, n_points),
            )
            before_bytes = cloud.read_bytes()
            cv2.imwrite(
                str(project / "frames" / "frame_000000.jpg"),
                rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
            )
            cv2.imwrite(
                str(project / "labels" / "frame_000000_labels.png"),
                np.full((h, w), NAMES.index("ground"), dtype=np.uint8),
            )
            out = project / "sih3d_filled.laz"
            with mock.patch.dict(sys.modules, _heavy_stubs()):
                with warnings.catch_warnings():
                    # constant fake AI depth makes the polyfit ill-conditioned;
                    # the production code handles it, the warning is just noise here
                    warnings.simplefilter("ignore")
                    res = fill_gaps(
                        project, project / "frames", project / "labels", out,
                        device="cpu", max_share=10.0,
                    )
            m = res["inferred_points_added"]
            self.assertGreater(m, 0)
            self.assertEqual(res["measured_points"], n_points)
            self.assertEqual(res["output"], str(out))
            self.assertIn("inferred_share", res)
            self.assertIn("capped_at_share", res)
            # re-encoded output: a new file, not a link of the input
            s, d = os.stat(cloud), os.stat(out)
            self.assertFalse((s.st_dev, s.st_ino) == (d.st_dev, d.st_ino))
            # measured points first and unchanged, inferred points appended + flagged
            las = laspy.read(out)
            self.assertEqual(len(las.points), n_points + m)
            self.assertIn("inferred", set(las.point_format.dimension_names))
            inferred = np.asarray(las.inferred)
            self.assertEqual(int(inferred.sum()), m)
            np.testing.assert_array_equal(inferred[:n_points], np.zeros(n_points))
            for band, in_band in zip(
                (las.x, las.y, las.z), _points_of(cloud)[:3]
            ):
                np.testing.assert_array_equal(np.asarray(band)[:n_points], in_band)
            # input file itself untouched
            self.assertEqual(cloud.read_bytes(), before_bytes)


if __name__ == "__main__":
    unittest.main()
