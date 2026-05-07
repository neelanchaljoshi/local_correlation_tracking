"""tests/test_interpolation.py"""

import sys
import pathlib
import unittest
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.interpolation import interpolate_image_stack


class TestInterpolateImageStack(unittest.TestCase):

    def test_exact_at_knot_points(self):
        """Cubic spline should return exact values at the knot times."""
        H, W = 8, 8
        imgs = np.random.default_rng(0).standard_normal((4, H, W))
        times = [0, 45, 90, 135]
        for t, img in zip(times, imgs):
            out = interpolate_image_stack(imgs, target_time=t, times=times)
            np.testing.assert_array_almost_equal(out, img, decimal=10)

    def test_output_shape(self):
        imgs = np.ones((4, 16, 16))
        out  = interpolate_image_stack(imgs, target_time=60)
        self.assertEqual(out.shape, (16, 16))

    def test_midpoint_between_constant_images(self):
        """Interpolation of constant images should return that constant."""
        imgs = np.ones((4, 8, 8)) * 5.0
        out  = interpolate_image_stack(imgs, target_time=67.5)
        np.testing.assert_array_almost_equal(out, 5.0)

    def test_invalid_ndim_raises(self):
        with self.assertRaises(ValueError):
            interpolate_image_stack(np.ones((4, 8)), target_time=60)

    def test_mismatched_times_raises(self):
        with self.assertRaises(ValueError):
            interpolate_image_stack(np.ones((4, 8, 8)),
                                    target_time=60, times=[0, 1, 2])

    def test_default_times(self):
        """Default times [0, 45, 90, 135] should be used if not specified."""
        imgs = np.random.default_rng(1).standard_normal((4, 4, 4))
        out1 = interpolate_image_stack(imgs, target_time=67.5)
        out2 = interpolate_image_stack(imgs, target_time=67.5,
                                       times=[0, 45, 90, 135])
        np.testing.assert_array_equal(out1, out2)

    def test_linear_signal_interpolated_exactly(self):
        """A signal linear in time should be recovered exactly."""
        H, W  = 6, 6
        times = [0, 45, 90, 135]
        base  = np.ones((H, W))
        imgs  = np.array([base * t for t in times])
        out   = interpolate_image_stack(imgs, target_time=67.5, times=times)
        np.testing.assert_array_almost_equal(out, base * 67.5, decimal=8)


if __name__ == '__main__':
    unittest.main()
