"""tests/test_lct.py"""

import sys
import pathlib
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.lct import (
    tukey_2d,
    build_tukey_kernel,
    get_ccf,
    get_flow_velocity,
    _fit_ellipsoid_peak,
)


class TestTukey2D(unittest.TestCase):

    def test_output_shape(self):
        w = tukey_2d(34, alpha=0.8)
        self.assertEqual(w.shape, (34, 34))

    def test_values_between_zero_and_one(self):
        w = tukey_2d(34, alpha=0.8)
        self.assertTrue(np.all(w >= 0) and np.all(w <= 1))

    def test_centre_is_one(self):
        w = tukey_2d(35, alpha=0.5)
        cy, cx = 35 // 2, 35 // 2
        self.assertAlmostEqual(w[cy, cx], 1.0)

    def test_corners_near_zero(self):
        w = tukey_2d(34, alpha=0.8)
        self.assertAlmostEqual(w[0, 0], 0.0)
        self.assertAlmostEqual(w[-1, -1], 0.0)

    def test_circular_symmetry(self):
        w = tukey_2d(34, alpha=0.8)
        np.testing.assert_array_almost_equal(w, w[::-1, :])
        np.testing.assert_array_almost_equal(w, w[:, ::-1])


class TestGetCCF(unittest.TestCase):

    def setUp(self):
        self.patch_size = 34
        self.kernel     = build_tukey_kernel(self.patch_size, alpha=0.8)
        rng             = np.random.default_rng(42)
        self.patch1     = rng.standard_normal((self.patch_size, self.patch_size))

    def test_output_shape(self):
        ccf, _, _ = get_ccf(self.patch1, self.patch1, self.kernel)
        self.assertEqual(ccf.shape, (self.patch_size, self.patch_size))

    def test_identical_patches_peak_at_centre(self):
        ccf, _, _ = get_ccf(self.patch1, self.patch1, self.kernel)
        cy, cx    = self.patch_size // 2, self.patch_size // 2
        peak_pos  = np.unravel_index(ccf.argmax(), ccf.shape)
        self.assertEqual(peak_pos, (cy, cx))

    def test_ccf_finite(self):
        ccf, _, _ = get_ccf(self.patch1, self.patch1, self.kernel)
        self.assertTrue(np.all(np.isfinite(ccf)))

    def test_zero_patches_give_zero_ccf(self):
        zeros     = np.zeros((self.patch_size, self.patch_size))
        ccf, _, _ = get_ccf(zeros, zeros, self.kernel)
        np.testing.assert_array_almost_equal(ccf, 0)

    def test_shifted_patch_moves_peak(self):
        """A laterally shifted copy should shift the CCF peak."""
        shift_px = 3
        patch2   = np.roll(self.patch1, shift_px, axis=1)
        ccf, _, _ = get_ccf(self.patch1, patch2, self.kernel)
        peak_pos  = np.unravel_index(ccf.argmax(), ccf.shape)
        cy, cx    = self.patch_size // 2, self.patch_size // 2
        self.assertNotEqual(peak_pos[1], cx)


class TestFitEllipsoidPeak(unittest.TestCase):

    def test_integer_peak_at_centre(self):
        """A CCF with peak at exact centre should return near-zero sub-pixel offset."""
        size = 35   # odd so centre is unambiguous
        cy, cx = size // 2, size // 2
        y, x = np.mgrid[:size, :size]
        ccf  = np.exp(-((x - cx)**2 + (y - cy)**2) / 4.0)
        xpar, ypar = _fit_ellipsoid_peak(ccf, grid_len=5)
        self.assertAlmostEqual(xpar, 0.0, places=3)
        self.assertAlmostEqual(ypar, 0.0, places=3)


class TestGetFlowVelocity(unittest.TestCase):

    def _make_ccf_with_peak_at(self, patch_size, dx, dy):
        """Build a synthetic Gaussian CCF with peak offset from centre."""
        cy, cx = patch_size // 2 + dy, patch_size // 2 + dx
        y, x   = np.mgrid[:patch_size, :patch_size]
        return np.exp(-((x - cx)**2 + (y - cy)**2) / 2.0)

    def test_zero_displacement_gives_near_zero_velocity(self):
        ps  = 35   # odd so centre is unambiguous
        ccf = self._make_ccf_with_peak_at(ps, dx=0, dy=0)
        dx, dy, ux, uy = get_flow_velocity(
            ccf, patch_size=ps, pixel_size_deg=0.03,
            cadence_interp=45, R_sun_Mm=695.7, grid_len=5, ntry=2)
        self.assertAlmostEqual(ux, 0.0, places=0)
        self.assertAlmostEqual(uy, 0.0, places=0)

    def test_positive_displacement_gives_positive_ux(self):
        ps  = 34
        ccf = self._make_ccf_with_peak_at(ps, dx=3, dy=0)
        dx, dy, ux, uy = get_flow_velocity(
            ccf, patch_size=ps, pixel_size_deg=0.03,
            cadence_interp=45, R_sun_Mm=695.7, grid_len=5, ntry=2)
        self.assertGreater(ux, 0)

    def test_output_types(self):
        ps  = 34
        ccf = self._make_ccf_with_peak_at(ps, dx=0, dy=0)
        result = get_flow_velocity(
            ccf, patch_size=ps, pixel_size_deg=0.03,
            cadence_interp=45, R_sun_Mm=695.7, grid_len=5, ntry=1)
        self.assertEqual(len(result), 4)
        for val in result:
            self.assertIsInstance(val, float)


if __name__ == '__main__':
    unittest.main()
