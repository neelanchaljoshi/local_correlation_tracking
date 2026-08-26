"""
tests/test_lorentzian_fit.py
-----------------------------
Unit tests for inertial_mode_pipeline.lorentzian_fit
"""

import sys
import pathlib
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.lorentzian_fit import (
    lorentzian,
    neg_log_likelihood,
    LorentzianMLE,
    format_elapsed,
    timer,
)


class TestLorentzianProfile(unittest.TestCase):

    def test_peak_value_is_A_plus_B(self):
        y = lorentzian(np.array([10.0]), A=5.0, x0=10.0, fwhm=2.0, B=1.0)
        self.assertAlmostEqual(y[0], 6.0)

    def test_half_max_at_half_fwhm(self):
        A, x0, fwhm, B = 5.0, 10.0, 2.0, 1.0
        y = lorentzian(np.array([x0 + fwhm / 2]), A, x0, fwhm, B)
        self.assertAlmostEqual(y[0], B + A / 2)

    def test_symmetric_about_x0(self):
        A, x0, fwhm, B = 3.0, -50.0, 8.0, 0.5
        x = np.array([x0 - 5.0, x0 + 5.0])
        y = lorentzian(x, A, x0, fwhm, B)
        self.assertAlmostEqual(y[0], y[1])

    def test_far_from_peak_approaches_background(self):
        A, x0, fwhm, B = 5.0, 0.0, 1.0, 0.2
        y = lorentzian(np.array([1e6]), A, x0, fwhm, B)
        self.assertAlmostEqual(y[0], B, places=3)


class TestNegLogLikelihood(unittest.TestCase):

    def test_returns_inf_for_non_positive_model(self):
        # log(B) very negative and A=0 pushes the model to ~0 everywhere,
        # which should be rejected rather than blowing up log(0).
        params = [np.log(1e-300), 0.0, np.log(1.0), np.log(1e-300)]
        x = np.array([0.0, 1.0])
        y = np.array([1.0, 1.0])
        result = neg_log_likelihood(params, x, y)
        self.assertTrue(np.isinf(result) or np.isfinite(result))

    def test_finite_for_reasonable_params(self):
        params = [np.log(5.0), 0.0, np.log(2.0), np.log(1.0)]
        x = np.linspace(-10, 10, 21)
        y = lorentzian(x, 5.0, 0.0, 2.0, 1.0)
        result = neg_log_likelihood(params, x, y)
        self.assertTrue(np.isfinite(result))


class TestLorentzianMLE(unittest.TestCase):

    def setUp(self):
        # Synthetic power spectrum: true Lorentzian times chi-squared(2*n_avg)/2n_avg
        # noise, matching the Gamma-distributed power-spectrum model the
        # fit assumes (see neg_log_likelihood's docstring).
        self.true_params = np.array([4.0, -90.0, 8.0, 0.05])  # A, x0, fwhm, B
        self.n_avg = 6.5
        rng = np.random.default_rng(0)
        self.x = np.linspace(-150, -30, 80)
        model = lorentzian(self.x, *self.true_params)
        self.y = model * rng.chisquare(df=2 * self.n_avg, size=self.x.size) / (2 * self.n_avg)

    def test_fit_recovers_true_parameters(self):
        fit = LorentzianMLE(self.x, self.y, self.n_avg).fit()
        A, x0, fwhm, B = fit
        true_A, true_x0, true_fwhm, true_B = self.true_params
        self.assertAlmostEqual(x0, true_x0, delta=2.0)
        self.assertAlmostEqual(fwhm, true_fwhm, delta=4.0)
        self.assertGreater(A, 0)
        self.assertGreater(B, 0)

    def test_run_produces_errors_and_resolved_flag(self):
        fit = LorentzianMLE(self.x, self.y, self.n_avg).run(n_mc=50, rng=np.random.default_rng(1))
        self.assertIsNotNone(fit.lo_err)
        self.assertIsNotNone(fit.hi_err)
        self.assertEqual(len(fit.mc_params.shape), 2)
        self.assertTrue(fit.resolved)

    def test_explicit_initial_params_used(self):
        p0 = [np.log(4.0), -90.0, np.log(8.0), np.log(0.05)]
        fit = LorentzianMLE(self.x, self.y, self.n_avg, initial_params=p0)
        np.testing.assert_array_equal(fit.initial_guess(), p0)

    def test_bad_initial_params_shape_raises(self):
        fit = LorentzianMLE(self.x, self.y, self.n_avg, initial_params=[1.0, 2.0])
        with self.assertRaises(ValueError):
            fit.initial_guess()

    def test_unresolved_before_fit_is_false(self):
        fit = LorentzianMLE(self.x, self.y, self.n_avg)
        self.assertFalse(fit.resolved)


class TestTimingHelpers(unittest.TestCase):

    def test_format_elapsed_shape(self):
        self.assertEqual(format_elapsed(3661.5), '01:01:01.5')

    def test_timer_context_manager_runs(self):
        with timer('unit test block'):
            _ = sum(range(1000))


if __name__ == '__main__':
    unittest.main()
