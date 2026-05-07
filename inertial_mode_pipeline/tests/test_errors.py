"""
tests/test_errors.py
--------------------
Unit tests for inertial_mode_pipeline.errors
"""

import unittest
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.errors import (
    monte_carlo_phase,
    monte_carlo_amp_phase,
    fl_sum_errors,
    compute_errors,
)

# Common test fixtures
THETA = np.deg2rad(90 - np.linspace(-90, 90, 73))
L_DISCARD  = [3, 5, 7]
FL_DISCARD = np.array([0.5 + 0.1j, 0.3 - 0.2j, 0.1 + 0.4j])


class TestMonteCarloPhaseFn(unittest.TestCase):

    def test_output_shape(self):
        std_r, std_i = monte_carlo_phase(THETA, L_DISCARD, FL_DISCARD, num_samples=50)
        self.assertEqual(std_r.shape, THETA.shape)
        self.assertEqual(std_i.shape, THETA.shape)

    def test_non_negative(self):
        std_r, std_i = monte_carlo_phase(THETA, L_DISCARD, FL_DISCARD, num_samples=50)
        self.assertTrue(np.all(std_r >= 0))
        self.assertTrue(np.all(std_i >= 0))

    def test_empty_discard_returns_zeros(self):
        std_r, std_i = monte_carlo_phase(THETA, [], np.array([]), num_samples=10)
        np.testing.assert_array_equal(std_r, 0)
        np.testing.assert_array_equal(std_i, 0)

    def test_larger_coefficients_give_larger_errors(self):
        fl_small = FL_DISCARD * 0.01
        fl_large = FL_DISCARD * 10.0
        std_r_small, _ = monte_carlo_phase(THETA, L_DISCARD, fl_small, num_samples=200)
        std_r_large, _ = monte_carlo_phase(THETA, L_DISCARD, fl_large, num_samples=200)
        self.assertGreater(np.mean(std_r_large), np.mean(std_r_small))


class TestMonteCarloAmpPhase(unittest.TestCase):

    def test_output_shape(self):
        std_r, std_i = monte_carlo_amp_phase(THETA, L_DISCARD, FL_DISCARD, num_samples=50)
        self.assertEqual(std_r.shape, THETA.shape)
        self.assertEqual(std_i.shape, THETA.shape)

    def test_non_negative(self):
        std_r, std_i = monte_carlo_amp_phase(THETA, L_DISCARD, FL_DISCARD, num_samples=50)
        self.assertTrue(np.all(std_r >= 0))
        self.assertTrue(np.all(std_i >= 0))

    def test_empty_discard_returns_zeros(self):
        std_r, std_i = monte_carlo_amp_phase(THETA, [], np.array([]), num_samples=10)
        np.testing.assert_array_equal(std_r, 0)
        np.testing.assert_array_equal(std_i, 0)


class TestFlSumErrors(unittest.TestCase):

    def test_output_shape(self):
        err_r, err_i = fl_sum_errors(THETA, L_DISCARD, FL_DISCARD)
        self.assertEqual(err_r.shape, THETA.shape)
        self.assertEqual(err_i.shape, THETA.shape)

    def test_real_and_imag_equal(self):
        err_r, err_i = fl_sum_errors(THETA, L_DISCARD, FL_DISCARD)
        np.testing.assert_array_equal(err_r, err_i)

    def test_non_negative(self):
        err_r, err_i = fl_sum_errors(THETA, L_DISCARD, FL_DISCARD)
        self.assertTrue(np.all(err_r >= 0))

    def test_empty_discard_returns_zeros(self):
        err_r, err_i = fl_sum_errors(THETA, [], np.array([]))
        np.testing.assert_array_equal(err_r, 0)
        np.testing.assert_array_equal(err_i, 0)

    def test_zero_coefficient_gives_zero_error(self):
        fl_zero = np.zeros(len(L_DISCARD), dtype=np.complex128)
        err_r, err_i = fl_sum_errors(THETA, L_DISCARD, fl_zero)
        np.testing.assert_array_almost_equal(err_r, 0)


class TestComputeErrors(unittest.TestCase):

    def test_monte_carlo_dispatches(self):
        std_r, std_i = compute_errors(
            THETA, L_DISCARD, FL_DISCARD, method='monte_carlo', num_samples=20)
        self.assertEqual(std_r.shape, THETA.shape)

    def test_monte_carlo_amp_dispatches(self):
        std_r, std_i = compute_errors(
            THETA, L_DISCARD, FL_DISCARD, method='monte_carlo_amp', num_samples=20)
        self.assertEqual(std_r.shape, THETA.shape)

    def test_fl_sum_dispatches(self):
        err_r, err_i = compute_errors(
            THETA, L_DISCARD, FL_DISCARD, method='fl_sum')
        self.assertEqual(err_r.shape, THETA.shape)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            compute_errors(THETA, L_DISCARD, FL_DISCARD, method='bad_method')


if __name__ == '__main__':
    unittest.main()
