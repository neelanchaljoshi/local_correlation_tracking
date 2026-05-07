"""
interpolation.py
----------------
Cubic spline interpolation of a 4-image stack to an arbitrary target time.
Isolated here so it can be tested independently of MPI and I/O.
"""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


def interpolate_image_stack(
    img_stack: np.ndarray,
    target_time: float,
    times: list[float] | None = None,
) -> np.ndarray:
    """
    Interpolate a stack of images to a target time using cubic splines.

    Each pixel is interpolated independently along the time axis.
    This is efficient because scipy CubicSpline operates on the full
    flattened array in one call.

    Parameters
    ----------
    img_stack   : (N, H, W) array of N images taken at `times`
    target_time : time [same units as times] at which to interpolate
    times       : list of N time points. Defaults to [0, 45, 90, 135].

    Returns
    -------
    (H, W) interpolated image at target_time

    Raises
    ------
    ValueError if img_stack does not have exactly N images matching times
    """
    if times is None:
        times = [0, 45, 90, 135]

    N = len(times)
    if img_stack.ndim != 3:
        raise ValueError(f'img_stack must be 3-D (N, H, W), got shape {img_stack.shape}')
    if img_stack.shape[0] != N:
        raise ValueError(
            f'img_stack has {img_stack.shape[0]} images but {N} times were given')

    H, W = img_stack.shape[1], img_stack.shape[2]
    flat = img_stack.reshape(N, -1)      # (N, H*W)
    cs   = CubicSpline(times, flat, axis=0)
    return cs(target_time).reshape(H, W)
