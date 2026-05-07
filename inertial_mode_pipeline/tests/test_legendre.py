"""
tests/test_legendre.py
----------------------
Unit tests for inertial_mode_pipeline.legendre
"""

import unittest
import numpy as np
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from inertial_mode_pipeline.legendre import (
    project_to_legendre_coefficients,
    enforce_symmetry,
    compute_keep_mask,
    reconstruct_from_coefficients,
    reconstruct_full,
    align_phase_at_equator,
    project_and_clean,
)

LATS  = np.linspace(-90, 90, 73)
THETA = np.deg2rad(90 - LATS)
L_ARRAY = np.arange(36)


class TestProjectToLegendreCoefficients(unittest.TestCase):

    def test_output_length(self):
        ef = np.ones(len(LATS), dtype=np.complex128)
        fl = project_to_legendre_coefficients(ef, THETA, L_ARRAY)
        self.assertEqual(len(fl), len(L_ARRAY))

    def test_complex_output(self):
        ef = np.ones(len(LATS), dtype=np.complex128) * (1 + 1j)
        fl = project_to_legendre_coefficients(ef, THETA, L_ARRAY)
        self.assertTrue(np.iscomplexobj(fl))

    def test_zero_input_gives_zero_coefficients(self):
        ef = np.zeros(len(LATS), dtype=np.complex128)
        fl = project_to_legendre_coefficients(ef, THETA, L_ARRAY)
        np.testing.assert_array_almost_equal(np.abs(fl), 0)


class TestEnforceSymmetry(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.fl_uphi = rng.standard_normal(36) + 1j * rng.standard_normal(36)
        self.fl_uthe = rng.standard_normal(36) + 1j * rng.standard_normal(36)

    def test_anti_zeros_even_uphi(self):
        fl_uphi, _, _, _, _, _ = enforce_symmetry(
            self.fl_uphi, self.fl_uthe, L_ARRAY, 'anti')
        np.testing.assert_array_equal(fl_uphi[0::2], 0)

    def test_anti_zeros_odd_uthe(self):
        _, fl_uthe, _, _, _, _ = enforce_symmetry(
            self.fl_uphi, self.fl_uthe, L_ARRAY, 'anti')
        np.testing.assert_array_equal(fl_uthe[1::2], 0)

    def test_sym_zeros_odd_uphi(self):
        fl_uphi, _, _, _, _, _ = enforce_symmetry(
            self.fl_uphi, self.fl_uthe, L_ARRAY, 'sym')
        np.testing.assert_array_equal(fl_uphi[1::2], 0)

    def test_all_does_not_zero_anything(self):
        fl_uphi_out, fl_uthe_out, _, _, _, _ = enforce_symmetry(
            self.fl_uphi, self.fl_uthe, L_ARRAY, 'all')
        np.testing.assert_array_equal(fl_uphi_out, self.fl_uphi)

    def test_invalid_symmetry_raises(self):
        with self.assertRaises(ValueError):
            enforce_symmetry(self.fl_uphi, self.fl_uthe, L_ARRAY, 'bad')

    def test_returns_copy_not_view(self):
        fl_uphi_out, _, _, _, _, _ = enforce_symmetry(
            self.fl_uphi, self.fl_uthe, L_ARRAY, 'anti')
        fl_uphi_out[:] = 999
        self.assertFalse(np.all(self.fl_uphi == 999))


class TestComputeKeepMask(unittest.TestCase):

    def test_low_ell_always_kept(self):
        power  = np.ones(18)   # 18 odd ell values: 1,3,5,...,35
        l_vals = np.arange(1, 36, 2)
        mask   = compute_keep_mask(power, l_vals, l_theory_cutoff=15)
        self.assertTrue(np.all(mask[l_vals <= 15]))

    def test_output_length_matches_input(self):
        power  = np.random.rand(18)
        l_vals = np.arange(1, 36, 2)
        mask   = compute_keep_mask(power, l_vals)
        self.assertEqual(len(mask), len(power))

    def test_boolean_output(self):
        power  = np.random.rand(18)
        l_vals = np.arange(1, 36, 2)
        mask   = compute_keep_mask(power, l_vals)
        self.assertEqual(mask.dtype, bool)

    def test_pure_noise_above_cutoff_mostly_rejected(self):
        rng    = np.random.default_rng(42)
        l_vals = np.arange(1, 36, 2)
        # Flat noise power — modes above cutoff should be mostly rejected
        power  = rng.exponential(scale=1.0, size=len(l_vals))
        mask   = compute_keep_mask(power, l_vals, l_theory_cutoff=15)
        high_l_keep_frac = mask[l_vals > 15].mean()
        self.assertLess(high_l_keep_frac, 0.8)


class TestReconstructFromCoefficients(unittest.TestCase):

    def test_output_shape(self):
        fl     = np.ones(36, dtype=np.complex128)
        l_keep = np.array([1, 3, 5])
        recon  = reconstruct_from_coefficients(fl, L_ARRAY, l_keep, THETA)
        self.assertEqual(recon.shape, THETA.shape)

    def test_empty_keep_gives_zero(self):
        fl    = np.ones(36, dtype=np.complex128)
        recon = reconstruct_from_coefficients(fl, L_ARRAY, np.array([]), THETA)
        np.testing.assert_array_almost_equal(recon, 0)

    def test_l_max_respected(self):
        """Modes at or above l_max should not contribute."""
        fl        = np.zeros(36, dtype=np.complex128)
        fl[22]    = 1.0   # exactly at l_max — should be excluded
        fl[23]    = 1.0   # above l_max — should be excluded
        l_keep    = np.array([22, 23])
        recon     = reconstruct_from_coefficients(fl, L_ARRAY, l_keep, THETA, l_max=22)
        np.testing.assert_array_almost_equal(recon, 0)


class TestReconstructFull(unittest.TestCase):

    def test_output_shape(self):
        fl    = np.ones(36, dtype=np.complex128)
        recon = reconstruct_full(fl, L_ARRAY, THETA)
        self.assertEqual(recon.shape, THETA.shape)

    def test_zero_coefficients_give_zero(self):
        fl    = np.zeros(36, dtype=np.complex128)
        recon = reconstruct_full(fl, L_ARRAY, THETA)
        np.testing.assert_array_almost_equal(recon, 0)


class TestAlignPhaseAtEquator(unittest.TestCase):

    def test_uthe_real_at_equator(self):
        uphi   = np.ones(len(THETA), dtype=np.complex128) * (1 + 1j)
        uthe   = np.ones(len(THETA), dtype=np.complex128) * (0.5 + 0.5j)
        _, uthe_aligned = align_phase_at_equator(uphi, uthe, THETA)
        equator_idx = np.argmin(np.abs(np.rad2deg(THETA) - 90))
        self.assertAlmostEqual(uthe_aligned[equator_idx].imag, 0.0, places=10)

    def test_shape_preserved(self):
        uphi = np.ones(len(THETA), dtype=np.complex128)
        uthe = np.ones(len(THETA), dtype=np.complex128)
        u, v = align_phase_at_equator(uphi, uthe, THETA)
        self.assertEqual(u.shape, uphi.shape)
        self.assertEqual(v.shape, uthe.shape)


class TestProjectAndClean(unittest.TestCase):
    """Integration test for the full project_and_clean pipeline."""

    def setUp(self):
        rng = np.random.default_rng(42)
        self.ef_uphi = (rng.standard_normal(len(LATS)) +
                        1j * rng.standard_normal(len(LATS))) * 0.1
        self.ef_uthe = (rng.standard_normal(len(LATS)) +
                        1j * rng.standard_normal(len(LATS))) * 0.05

    def test_output_keys(self):
        result = project_and_clean(
            m=2, ef_uphi=self.ef_uphi, ef_uthe=self.ef_uthe,
            lats=LATS, symmetryuphi='anti',
            num_mc_samples=20, error_method='monte_carlo')
        expected_keys = {
            'ef_uphi', 'ef_uthe', 'ef_uphi_sm', 'ef_uthe_sm',
            'uphi_err_real', 'uphi_err_imag',
            'uthe_err_real', 'uthe_err_imag',
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_output_shapes(self):
        result = project_and_clean(
            m=2, ef_uphi=self.ef_uphi, ef_uthe=self.ef_uthe,
            lats=LATS, symmetryuphi='anti',
            num_mc_samples=20, error_method='monte_carlo')
        for key, val in result.items():
            self.assertEqual(val.shape, (len(LATS),),
                             msg=f'{key} has wrong shape')

    def test_errors_non_negative(self):
        result = project_and_clean(
            m=2, ef_uphi=self.ef_uphi, ef_uthe=self.ef_uthe,
            lats=LATS, symmetryuphi='anti',
            num_mc_samples=20, error_method='monte_carlo')
        for key in ('uphi_err_real', 'uphi_err_imag',
                    'uthe_err_real', 'uthe_err_imag'):
            self.assertTrue(np.all(result[key] >= 0), msg=f'{key} has negative values')

    def test_uthe_real_at_equator(self):
        result = project_and_clean(
            m=2, ef_uphi=self.ef_uphi, ef_uthe=self.ef_uthe,
            lats=LATS, symmetryuphi='anti',
            num_mc_samples=20, error_method='monte_carlo')
        equator_idx = np.argmin(np.abs(LATS))
        self.assertAlmostEqual(
            result['ef_uthe'][equator_idx].imag, 0.0, places=8)

    def test_fl_sum_method_works(self):
        result = project_and_clean(
            m=2, ef_uphi=self.ef_uphi, ef_uthe=self.ef_uthe,
            lats=LATS, symmetryuphi='anti',
            error_method='fl_sum')
        self.assertIn('ef_uphi', result)


if __name__ == '__main__':
    unittest.main()
