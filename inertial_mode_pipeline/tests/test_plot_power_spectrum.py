"""
tests/test_plot_power_spectrum.py
-----------------------------------
Unit tests for plot_power_spectrum.py's pure helper logic.

compute_power_spectrum/fit_and_plot need real flow data and are
exercised manually (same as run_pipeline.py/check_span.py — see
INERTIAL_MODE_PIPELINE.md); resolve_lat_band has no such dependency
and is worth covering directly, especially its partial-override
behaviour.
"""

import sys
import pathlib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from plot_power_spectrum import resolve_lat_band


class TestResolveLatBand(unittest.TestCase):

    def test_known_mode_uses_default_band(self):
        self.assertEqual(resolve_lat_band('highlat', None, None), (45.0, 75.0))
        self.assertEqual(resolve_lat_band('critlat', None, None), (15.0, 45.0))
        self.assertEqual(resolve_lat_band('rossby', None, None), (0.0, 30.0))
        self.assertEqual(resolve_lat_band('hfr', None, None), (0.0, 30.0))

    def test_explicit_both_overrides_known_mode(self):
        self.assertEqual(resolve_lat_band('highlat', 10.0, 20.0), (10.0, 20.0))

    def test_explicit_lat_min_only_overrides_default_max_kept(self):
        lat_min, lat_max = resolve_lat_band('highlat', 50.0, None)
        self.assertEqual((lat_min, lat_max), (50.0, 75.0))

    def test_explicit_lat_max_only_overrides_default_min_kept(self):
        lat_min, lat_max = resolve_lat_band('highlat', None, 70.0)
        self.assertEqual((lat_min, lat_max), (45.0, 70.0))

    def test_unknown_mode_requires_explicit_band(self):
        with self.assertRaises(ValueError):
            resolve_lat_band('some_custom_mode', None, None)

    def test_unknown_mode_with_explicit_band_is_fine(self):
        self.assertEqual(resolve_lat_band('some_custom_mode', 1.0, 2.0), (1.0, 2.0))

    def test_unknown_mode_with_only_one_bound_still_raises(self):
        with self.assertRaises(ValueError):
            resolve_lat_band('some_custom_mode', 1.0, None)


if __name__ == '__main__':
    unittest.main()
