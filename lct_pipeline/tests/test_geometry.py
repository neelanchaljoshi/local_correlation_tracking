"""tests/test_geometry.py"""

import sys
import pathlib
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.geometry import (
    compute_airy_radius_pixels,
    airy_disk_psf,
    simulate_pmi_from_hmi,
    differential_rotation_rate,
    carrington_longitude_shift,
)


class TestAiryRadius(unittest.TestCase):

    def test_larger_aperture_gives_smaller_radius(self):
        r_large = compute_airy_radius_pixels(617e-9, 0.14, 0.5)
        r_small = compute_airy_radius_pixels(617e-9, 0.075, 0.5)
        self.assertLess(r_large, r_small)

    def test_larger_wavelength_gives_larger_radius(self):
        r1 = compute_airy_radius_pixels(617e-9, 0.14, 0.5)
        r2 = compute_airy_radius_pixels(700e-9, 0.14, 0.5)
        self.assertLess(r1, r2)

    def test_positive_result(self):
        r = compute_airy_radius_pixels(617e-9, 0.14, 0.5)
        self.assertGreater(r, 0)


class TestAiryDiskPsf(unittest.TestCase):

    def test_normalised_to_one(self):
        psf = airy_disk_psf((64, 64), airy_radius_pixels=2.5)
        self.assertAlmostEqual(psf.sum(), 1.0, places=10)

    def test_output_shape(self):
        psf = airy_disk_psf((32, 32), airy_radius_pixels=2.0)
        self.assertEqual(psf.shape, (32, 32))

    def test_peak_at_centre(self):
        psf = airy_disk_psf((65, 65), airy_radius_pixels=3.0)
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        self.assertEqual(psf.argmax(), cy * psf.shape[1] + cx)

    def test_non_negative(self):
        psf = airy_disk_psf((32, 32), airy_radius_pixels=2.0)
        self.assertTrue(np.all(psf >= 0))

    def test_symmetric(self):
        # Use odd size so centre pixel is exact
        psf = airy_disk_psf((65, 65), airy_radius_pixels=3.0)
        np.testing.assert_array_almost_equal(psf, psf[::-1, :], decimal=10)
        np.testing.assert_array_almost_equal(psf, psf[:, ::-1], decimal=10)


class TestSimulatePmi(unittest.TestCase):

    def test_output_half_size(self):
        img    = np.random.default_rng(0).standard_normal((64, 64))
        psf    = airy_disk_psf((32, 32), airy_radius_pixels=2.0)
        result = simulate_pmi_from_hmi(img, psf)
        self.assertEqual(result.shape, (32, 32))

    def test_output_is_finite(self):
        img    = np.ones((32, 32))
        psf    = airy_disk_psf((16, 16), airy_radius_pixels=1.5)
        result = simulate_pmi_from_hmi(img, psf)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_odd_dimensions_handled(self):
        """Odd-sized images should be handled by cropping."""
        img    = np.ones((33, 33))
        psf    = airy_disk_psf((16, 16), airy_radius_pixels=1.5)
        result = simulate_pmi_from_hmi(img, psf)
        self.assertEqual(result.shape, (16, 16))


class TestDifferentialRotation(unittest.TestCase):

    def test_equator_rate(self):
        """At equator sin(lat)=0 so rate = A."""
        rate = differential_rotation_rate(0.0, A=14.034, B=-1.702, C=-2.494)
        self.assertAlmostEqual(rate, 14.034)

    def test_slower_at_poles(self):
        eq   = differential_rotation_rate(0.0,  A=14.034, B=-1.702, C=-2.494)
        pole = differential_rotation_rate(60.0, A=14.034, B=-1.702, C=-2.494)
        self.assertLess(pole, eq)

    def test_symmetric_about_equator(self):
        r_n = differential_rotation_rate(30.0,  A=14.034, B=-1.702, C=-2.494)
        r_s = differential_rotation_rate(-30.0, A=14.034, B=-1.702, C=-2.494)
        self.assertAlmostEqual(r_n, r_s)


if __name__ == '__main__':
    unittest.main()
