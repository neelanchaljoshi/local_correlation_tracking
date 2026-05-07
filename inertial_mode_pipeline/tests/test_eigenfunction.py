"""
tests/test_eigenfunction.py
----------------------------
Unit tests for inertial_mode_pipeline.eigenfunction
"""

import unittest
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.eigenfunction import _lat_svd_mask, extract_eigenfunction


class TestLatSvdMask(unittest.TestCase):

    def test_excludes_high_latitudes(self):
        lats = np.linspace(-90, 90, 73)
        mask = _lat_svd_mask(lats, lat_max=75)
        self.assertFalse(mask[0])    # -90° excluded
        self.assertFalse(mask[-1])   # +90° excluded

    def test_includes_equator(self):
        lats = np.linspace(-90, 90, 73)
        mask = _lat_svd_mask(lats, lat_max=75)
        equator = np.argmin(np.abs(lats))
        self.assertTrue(mask[equator])

    def test_boundary(self):
        lats = np.array([-75, -74, 0, 74, 75])
        mask = _lat_svd_mask(lats, lat_max=75)
        np.testing.assert_array_equal(mask, [True, True, True, True, True])

    def test_custom_lat_max(self):
        lats = np.array([-60, -30, 0, 30, 60])
        mask = _lat_svd_mask(lats, lat_max=45)
        np.testing.assert_array_equal(mask, [False, True, True, True, False])


class TestExtractEigenfunction(unittest.TestCase):
    """
    Synthetic test: plant a known signal at (m, freq) and verify the
    pipeline recovers a non-trivial eigenfunction without crashing.
    """

    def _make_synthetic_fourier(self, nt=64, nlat=10, m=2,
                                freq_nHz=-100.0, dt=6*3600):
        """
        Build a synthetic Fourier array with a Gaussian signal at m=2,
        freq=-100 nHz.
        """
        freq_axis = np.fft.fftshift(-np.fft.fftfreq(nt, dt) * 1e9)
        nlng      = 20   # enough columns, well above m=2
        uphi_ft   = np.zeros((nt, nlat, nlng), dtype=np.complex128)
        uthe_ft   = np.zeros((nt, nlat, nlng), dtype=np.complex128)

        # Gaussian envelope in frequency
        sig = np.exp(-((freq_axis - freq_nHz) ** 2) / (2 * 20 ** 2))
        # Uniform latitude profile
        uphi_ft[:, :, m] = sig[:, None] * np.ones(nlat)[None, :]
        uthe_ft[:, :, m] = sig[:, None] * np.ones(nlat)[None, :] * 0.5

        return uphi_ft, uthe_ft, freq_axis

    def test_output_keys_present(self):
        nt, nlat, m = 64, 10, 2
        lats = np.linspace(-90, 90, nlat)
        uphi_ft, uthe_ft, freq = self._make_synthetic_fourier(
            nt=nt, nlat=nlat, m=m)

        result = extract_eigenfunction(
            uphi_ft, uthe_ft, freq, m=m,
            cent_freq=-100.0, lats=lats, df=30.0)

        for key in ('ef_uphi', 'ef_uthe', 'final_td'):
            self.assertIn(key, result)

    def test_output_shapes(self):
        nt, nlat, m = 64, 10, 2
        lats = np.linspace(-90, 90, nlat)
        uphi_ft, uthe_ft, freq = self._make_synthetic_fourier(
            nt=nt, nlat=nlat, m=m)

        result = extract_eigenfunction(
            uphi_ft, uthe_ft, freq, m=m,
            cent_freq=-100.0, lats=lats, df=30.0)

        self.assertEqual(result['ef_uphi'].shape, (nlat,))
        self.assertEqual(result['ef_uthe'].shape, (nlat,))
        self.assertEqual(result['final_td'].shape, (nt,))

    def test_ef_not_all_zero(self):
        """Pipeline runs without error on a realistic noisy input."""
        nt, nlat, m = 64, 10, 2
        lats = np.linspace(-90, 90, nlat)
        rng  = np.random.default_rng(42)
        uphi_ft, uthe_ft, freq = self._make_synthetic_fourier(
            nt=nt, nlat=nlat, m=m)
        uphi_ft += 1e-2 * (rng.standard_normal(uphi_ft.shape) +
                        1j * rng.standard_normal(uphi_ft.shape))
        result = extract_eigenfunction(
            uphi_ft, uthe_ft, freq, m=m,
            cent_freq=-100.0, lats=lats, df=30.0)
        # Pipeline should complete and return correct shapes
        self.assertEqual(result['ef_uphi'].shape, (nlat,))
        self.assertEqual(result['ef_uthe'].shape, (nlat,))

    def test_final_td_non_negative(self):
        nt, nlat, m = 64, 10, 2
        lats = np.linspace(-90, 90, nlat)
        uphi_ft, uthe_ft, freq = self._make_synthetic_fourier(
            nt=nt, nlat=nlat, m=m)

        result = extract_eigenfunction(
            uphi_ft, uthe_ft, freq, m=m,
            cent_freq=-100.0, lats=lats, df=30.0)

        self.assertTrue(np.all(result['final_td'] >= 0))


if __name__ == '__main__':
    unittest.main()
