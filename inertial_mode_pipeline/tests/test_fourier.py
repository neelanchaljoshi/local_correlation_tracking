"""
tests/test_fourier.py
---------------------
Unit tests for inertial_mode_pipeline.fourier
"""

import unittest
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.fourier import (
    tukeywin,
    bandpass_filter,
    inverse_time_transform,
)


class TestTukeywin(unittest.TestCase):

    def test_alpha_zero_is_rectangular(self):
        w = tukeywin(100, alpha=0)
        np.testing.assert_array_equal(w, np.ones(100))

    def test_alpha_one_is_hanning(self):
        w = tukeywin(100, alpha=1)
        np.testing.assert_array_almost_equal(w, np.hanning(100))

    def test_length(self):
        for N in [10, 50, 128]:
            self.assertEqual(len(tukeywin(N, 0.5)), N)

    def test_values_between_zero_and_one(self):
        w = tukeywin(64, alpha=0.5)
        self.assertTrue(np.all(w >= 0) and np.all(w <= 1))

    def test_symmetric(self):
        w = tukeywin(64, alpha=0.3)
        np.testing.assert_array_almost_equal(w, w[::-1])

    def test_centre_is_one(self):
        w = tukeywin(101, alpha=0.5)
        # Middle section should be exactly 1
        self.assertAlmostEqual(w[50], 1.0)


class TestBandpassFilter(unittest.TestCase):

    def setUp(self):
        self.nfreq = 200
        self.nlat  = 10
        self.freq  = np.linspace(-500, 500, self.nfreq)
        rng        = np.random.default_rng(0)
        self.uphi  = rng.standard_normal((self.nfreq, self.nlat)) + \
                     1j * rng.standard_normal((self.nfreq, self.nlat))
        self.uthe  = rng.standard_normal((self.nfreq, self.nlat)) + \
                     1j * rng.standard_normal((self.nfreq, self.nlat))

    def test_outside_band_is_zero(self):
        uphi_f, uthe_f = bandpass_filter(
            self.uphi, self.uthe, self.freq, cent_freq=0.0, df=50.0)
        outside = (self.freq < -50) | (self.freq > 50)
        np.testing.assert_array_equal(uphi_f[outside], 0)
        np.testing.assert_array_equal(uthe_f[outside], 0)

    def test_output_shape_unchanged(self):
        uphi_f, uthe_f = bandpass_filter(
            self.uphi, self.uthe, self.freq, cent_freq=0.0, df=50.0)
        self.assertEqual(uphi_f.shape, self.uphi.shape)
        self.assertEqual(uthe_f.shape, self.uthe.shape)

    def test_no_amplification(self):
        """Filter should never increase power."""
        uphi_f, _ = bandpass_filter(
            self.uphi, self.uthe, self.freq, cent_freq=0.0, df=50.0)
        self.assertTrue(np.all(np.abs(uphi_f) <= np.abs(self.uphi) + 1e-12))

    def test_narrow_band_zeroes_most(self):
        uphi_f, _ = bandpass_filter(
            self.uphi, self.uthe, self.freq, cent_freq=0.0, df=5.0)
        frac_nonzero = np.count_nonzero(uphi_f) / uphi_f.size
        self.assertLess(frac_nonzero, 0.15)


class TestInverseTimeTransform(unittest.TestCase):

    def test_output_shape(self):
        nt, nlat, nlng, m = 50, 10, 30, 3
        filt_m = np.ones((nt, nlat), dtype=np.complex128)
        out    = inverse_time_transform(filt_m, nt, nlat, nlng, m)
        self.assertEqual(out.shape, (nt, nlat, nlng))

    def test_only_m_column_nonzero(self):
        nt, nlat, nlng, m = 20, 5, 15, 2
        filt_m = np.ones((nt, nlat), dtype=np.complex128)
        out    = inverse_time_transform(filt_m, nt, nlat, nlng, m)
        # All columns other than m should be zero
        for col in range(nlng):
            if col != m:
                np.testing.assert_array_almost_equal(out[:, :, col], 0)

    def test_invertibility_roundtrip(self):
        """IFFT of FFT of a signal should recover the original."""
        nt, nlat, nlng, m = 32, 4, 16, 1
        rng    = np.random.default_rng(99)
        signal = rng.standard_normal((nt, nlat)) + \
                 1j * rng.standard_normal((nt, nlat))
        # Forward FFT + fftshift
        ft     = np.fft.fftshift(np.fft.fft(signal, axis=0), axes=0)
        # Inverse via our function
        out    = inverse_time_transform(ft, nt, nlat, nlng, m)
        np.testing.assert_array_almost_equal(out[:, :, m], signal, decimal=10)


if __name__ == '__main__':
    unittest.main()
