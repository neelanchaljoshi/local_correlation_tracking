"""tests/test_geometry.py"""

import sys
import pathlib
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.geometry import (
    compute_airy_radius_pixels,
    airy_disk_psf,
    build_psfs,
    simulate_pmi_from_hmi,
    differential_rotation_rate,
    carrington_longitude_shift,
    compute_b0_correction,
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


class TestBuildPsfs(unittest.TestCase):

    def _cfg(self):
        return SimpleNamespace(
            psf_size=32,
            wavelength_m=617.3e-9,
            aperture_hmi_m=0.14,
            aperture_pmi_m=0.075,
            pixel_scale_arcsec=0.5,
        )

    def test_output_shapes(self):
        psf_hmi, psf_pmi, psf_rel = build_psfs(self._cfg())
        self.assertEqual(psf_hmi.shape, (32, 32))
        self.assertEqual(psf_pmi.shape, (32, 32))
        self.assertEqual(psf_rel.shape, (32, 32))

    def test_individual_psfs_normalised(self):
        psf_hmi, psf_pmi, _ = build_psfs(self._cfg())
        self.assertAlmostEqual(psf_hmi.sum(), 1.0, places=8)
        self.assertAlmostEqual(psf_pmi.sum(), 1.0, places=8)

    def test_relative_psf_is_finite(self):
        _, _, psf_rel = build_psfs(self._cfg())
        self.assertTrue(np.all(np.isfinite(psf_rel)))

    def test_smaller_aperture_gives_broader_pmi_psf(self):
        """
        PMI's smaller aperture (0.075m vs HMI's 0.14m) diffracts more at
        the same wavelength/pixel scale, so its PSF should be broader
        (lower peak, since both are normalised to unit sum).
        """
        psf_hmi, psf_pmi, _ = build_psfs(self._cfg())
        self.assertLess(psf_pmi.max(), psf_hmi.max())


class TestCarringtonLongitudeShift(unittest.TestCase):

    def _cfg(self, A=14.034, B=-1.702, C=-2.494, CRrate=14.184):
        return SimpleNamespace(A=A, B=B, C=C, CRrate=CRrate)

    def test_matches_manual_formula(self):
        cfg = self._cfg()
        lat, dt_seconds = 30.0, 86400.0
        rate = differential_rotation_rate(lat, cfg.A, cfg.B, cfg.C)
        expected = (rate - cfg.CRrate) * (dt_seconds / 86400.0)
        self.assertAlmostEqual(
            carrington_longitude_shift(lat, dt_seconds, cfg), expected)

    def test_zero_when_local_rate_matches_crrate(self):
        # At the equator (sin=0) rate=A; set A=CRrate so the shift is 0.
        cfg = self._cfg(A=14.184, CRrate=14.184)
        self.assertAlmostEqual(
            carrington_longitude_shift(0.0, 86400.0, cfg), 0.0)

    def test_scales_linearly_with_time(self):
        cfg = self._cfg()
        shift_1day = carrington_longitude_shift(30.0, 86400.0, cfg)
        shift_2day = carrington_longitude_shift(30.0, 2 * 86400.0, cfg)
        self.assertAlmostEqual(shift_2day, 2 * shift_1day)


class TestComputeB0Correction(unittest.TestCase):

    def _cfg(self, dI=-0.08, t_ref=None):
        return SimpleNamespace(
            dI=dI, t_ref_b0=t_ref or datetime(2010, 6, 7, 14, 17, 20))

    def test_zero_dB_at_reference_epoch(self):
        cfg = self._cfg()
        dB, dP = compute_b0_correction(cfg.t_ref_b0, cfg)
        self.assertAlmostEqual(dB, 0.0)
        self.assertAlmostEqual(dP, -cfg.dI)

    def test_quarter_year_gives_peak_dB_and_zero_dP(self):
        cfg = self._cfg()
        t = cfg.t_ref_b0 + timedelta(days=365.25 / 4)
        dB, dP = compute_b0_correction(t, cfg)
        self.assertAlmostEqual(dB, cfg.dI, places=6)
        self.assertAlmostEqual(dP, 0.0, places=6)

    def test_half_year_gives_zero_dB_and_flipped_dP(self):
        cfg = self._cfg()
        t = cfg.t_ref_b0 + timedelta(days=365.25 / 2)
        dB, dP = compute_b0_correction(t, cfg)
        self.assertAlmostEqual(dB, 0.0, places=6)
        self.assertAlmostEqual(dP, cfg.dI, places=6)

    def test_full_year_returns_to_start(self):
        cfg = self._cfg()
        t = cfg.t_ref_b0 + timedelta(days=365.25)
        dB, dP = compute_b0_correction(t, cfg)
        self.assertAlmostEqual(dB, 0.0, places=6)
        self.assertAlmostEqual(dP, -cfg.dI, places=6)


if __name__ == '__main__':
    unittest.main()
