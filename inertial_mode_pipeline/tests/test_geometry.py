"""
tests/test_geometry.py
----------------------
Unit tests for inertial_mode_pipeline.geometry
"""

import unittest
import numpy as np
import sys
import pathlib

# Allow import without installing the package
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.geometry import (
    make_lon_lat_grids,
    clip_flow_data,
    apodize_flow_data,
    apply_symmetry,
    fill_carrington_gaps,
    get_correction_factor,
)


class TestMakeLonLatGrids(unittest.TestCase):

    def test_default_shape(self):
        lon, lat = make_lon_lat_grids()
        self.assertEqual(len(lon), 73)
        self.assertEqual(len(lat), 73)

    def test_range(self):
        lon, lat = make_lon_lat_grids()
        self.assertAlmostEqual(lon[0],  -90.0)
        self.assertAlmostEqual(lon[-1],  90.0)
        self.assertAlmostEqual(lat[0],  -90.0)
        self.assertAlmostEqual(lat[-1],  90.0)

    def test_custom_shape(self):
        lon, lat = make_lon_lat_grids((-45, 45, 10), (-45, 45, 5))
        self.assertEqual(len(lon), 10)
        self.assertEqual(len(lat), 5)


class TestClipFlowData(unittest.TestCase):

    def setUp(self):
        self.nt, self.nlat, self.nlng = 5, 10, 10
        rng = np.random.default_rng(42)
        self.arr      = rng.standard_normal((self.nt, self.nlat, self.nlng))
        self.rsun_obs = np.full(self.nt, 960.0)
        # radius_arr: inner half of pixels at 0.5 * rsun, outer at 1.5 * rsun
        self.radius_arr = np.zeros_like(self.arr)
        self.radius_arr[:, :, :5]  = 0.5 * 960.0   # inside clip
        self.radius_arr[:, :, 5:]  = 1.5 * 960.0   # outside clip

    def test_outside_pixels_are_nan(self):
        out = clip_flow_data(self.arr, self.radius_arr, self.rsun_obs,
                             radius_ratio=0.99, pad=False)
        self.assertTrue(np.all(np.isnan(out[:, :, 5:])))

    def test_inside_pixels_preserved(self):
        out = clip_flow_data(self.arr, self.radius_arr, self.rsun_obs,
                             radius_ratio=0.99, pad=False)
        np.testing.assert_array_equal(out[:, :, :5], self.arr[:, :, :5])

    def test_pad_increases_longitude_axis(self):
        out = clip_flow_data(self.arr, self.radius_arr, self.rsun_obs,
                             radius_ratio=0.99, pad=True)
        self.assertEqual(out.shape[2], self.nlng + 36 + 35)

    def test_output_is_copy(self):
        out = clip_flow_data(self.arr, self.radius_arr, self.rsun_obs,
                             radius_ratio=0.99, pad=False)
        out[:] = 0
        self.assertFalse(np.all(self.arr == 0))


class TestApodizeFlowData(unittest.TestCase):

    def setUp(self):
        self.nt, self.nlat, self.nlng = 3, 8, 8
        self.arr      = np.ones((self.nt, self.nlat, self.nlng))
        self.rsun_obs = np.full(self.nt, 960.0)

    def test_inner_pixels_unchanged(self):
        radius_arr = np.full_like(self.arr, 0.90 * 960.0)   # inside r_min
        out = apodize_flow_data(self.arr, radius_arr, self.rsun_obs,
                                r_min=0.96, r_max=0.99)
        # Only the non-padded part; inside region should still be 1
        np.testing.assert_array_almost_equal(out[:, :, 36:-35], 1.0)

    def test_outer_pixels_zero(self):
        radius_arr = np.full_like(self.arr, 1.10 * 960.0)   # outside r_max
        out = apodize_flow_data(self.arr, radius_arr, self.rsun_obs,
                                r_min=0.96, r_max=0.99)
        np.testing.assert_array_almost_equal(out[:, :, 36:-35], 0.0)

    def test_transition_between_zero_and_one(self):
        radius_arr = np.full_like(self.arr, 0.975 * 960.0)  # in transition
        out = apodize_flow_data(self.arr, radius_arr, self.rsun_obs,
                                r_min=0.96, r_max=0.99)
        vals = out[:, :, 36:-35]
        self.assertTrue(np.all(vals > 0) and np.all(vals < 1))


class TestApplySymmetry(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(7)
        self.arr = rng.standard_normal((4, 6, 8))

    def test_sym_uphi_is_symmetric(self):
        uphi, _ = apply_symmetry(self.arr, self.arr.copy(), 'sym')
        np.testing.assert_array_almost_equal(uphi, uphi[:, ::-1, :])

    def test_anti_uphi_is_antisymmetric(self):
        uphi, _ = apply_symmetry(self.arr, self.arr.copy(), 'anti')
        np.testing.assert_array_almost_equal(uphi, -uphi[:, ::-1, :])

    def test_all_returns_unchanged(self):
        uphi, uthe = apply_symmetry(self.arr, self.arr * 2, 'all')
        np.testing.assert_array_equal(uphi, self.arr)
        np.testing.assert_array_equal(uthe, self.arr * 2)

    def test_invalid_symmetry_raises(self):
        with self.assertRaises(ValueError):
            apply_symmetry(self.arr, self.arr, 'bad')


class TestFillCarringtonGaps(unittest.TestCase):

    def test_no_nans_unchanged(self):
        crln = np.array([10.0, 9.5, 9.0, 8.5])
        out  = fill_carrington_gaps(crln)
        np.testing.assert_array_almost_equal(out, crln)

    def test_single_nan_filled(self):
        crln = np.array([10.0, np.nan, 9.0, 8.5])
        out  = fill_carrington_gaps(crln)
        self.assertFalse(np.any(np.isnan(out)))

    def test_consecutive_nans_filled(self):
        # First element valid so dphi can be computed from later differences
        crln = np.array([10.0, np.nan, np.nan, 8.5, 8.0])
        out  = fill_carrington_gaps(crln)
        self.assertFalse(np.any(np.isnan(out)))

    def test_original_unchanged(self):
        crln = np.array([10.0, np.nan, 9.0])
        _    = fill_carrington_gaps(crln)
        self.assertTrue(np.isnan(crln[1]))   # original not mutated


class TestGetCorrectionFactor(unittest.TestCase):

    def test_all_finite_returns_ones(self):
        arr = np.ones((10, 5, 8))
        cft, cfl = get_correction_factor(arr, nlng_carr=16)
        self.assertTrue(np.all(np.isfinite(cft)))
        self.assertTrue(np.all(np.isfinite(cfl)))

    def test_shape(self):
        arr = np.ones((10, 5, 8))
        cft, cfl = get_correction_factor(arr, nlng_carr=16)
        # cft is (1, nlat, 1) — scalar per latitude after time reduction
        self.assertEqual(cft.shape[1], 5)
        self.assertEqual(cft.shape[2], 1)
        self.assertEqual(cfl.shape[1], 5)
        self.assertEqual(cfl.shape[2], 1)

    def test_nan_pixels_increase_correction(self):
        arr_full = np.ones((10, 5, 8))
        arr_half = arr_full.copy()
        arr_half[:, :, 4:] = np.nan
        _, cfl_full = get_correction_factor(arr_full, nlng_carr=16)
        _, cfl_half = get_correction_factor(arr_half, nlng_carr=16)
        self.assertTrue(np.all(cfl_half >= cfl_full))


if __name__ == '__main__':
    unittest.main()
