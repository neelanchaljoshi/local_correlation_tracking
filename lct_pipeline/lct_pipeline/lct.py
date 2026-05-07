"""
lct.py
------
Core LCT computations: Tukey window, cross-correlation function (CCF),
and sub-pixel peak fitting via ellipsoid fitting.

All functions are pure computation — no I/O, no MPI, no config dependency
beyond what is passed as arguments. This makes them fully unit-testable.
"""
from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.linalg import lstsq
from scipy.ndimage import shift


# ── Tukey window ──────────────────────────────────────────────────────────

def tukey_2d(width: int, alpha: float = 0.8) -> np.ndarray:
    """
    2D Tukey (tapered cosine) window with circular support.

    The window is 1 inside radius width/2 and tapers to 0 at the edge.

    Parameters
    ----------
    width : window size in pixels (square)
    alpha : taper fraction [0=rectangular, 1=Hanning]

    Returns
    -------
    (width, width) float array
    """
    base   = np.zeros((width, width))
    tukey  = signal.windows.tukey(width, alpha)
    tukey  = tukey[len(tukey) // 2 - 1:]   # second half only

    x = np.linspace(-width / 2, width / 2, width)
    y = np.linspace(-width / 2, width / 2, width)

    for xi in range(width):
        for yi in range(width):
            r = np.hypot(x[xi], y[yi])
            if r <= width / 2:
                base[xi, yi] = tukey[int(r)]
    return base


def build_tukey_kernel(patch_size: int, alpha: float) -> np.ndarray:
    """
    Pre-compute the Tukey kernel for a given patch size and alpha.
    Call once at startup and reuse in get_ccf.
    """
    return tukey_2d(patch_size, alpha)


# ── Cross-correlation ─────────────────────────────────────────────────────

def get_ccf(
    patch1: np.ndarray,
    patch2: np.ndarray,
    kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the normalised cross-correlation function (CCF) between two
    image patches using FFT convolution.

    Parameters
    ----------
    patch1, patch2 : (H, W) image patches
    kernel         : (H, W) Tukey apodization window

    Returns
    -------
    ccf     : (H, W) cross-correlation function (fftshifted)
    patch1w : apodized patch1
    patch2w : apodized patch2
    """
    patch1w = kernel * (patch1 - np.nanmean(patch1))
    patch2w = kernel * (patch2 - np.nanmean(patch2))
    f       = np.fft.rfft2(patch1w)
    g       = np.fft.rfft2(patch2w)
    c       = np.conj(f) * g
    ccf     = np.fft.irfft2(c, s=patch1.shape, axes=[0, 1])
    ccf     = np.fft.fftshift(ccf, axes=[0, 1])
    return ccf, patch1w, patch2w


# ── Sub-pixel peak fitting ────────────────────────────────────────────────

def _fit_ellipsoid_peak(ccf: np.ndarray, grid_len: int) -> tuple[float, float]:
    """
    Fit a 2D quadratic surface to the neighbourhood of the CCF peak to
    find the sub-pixel peak location.

    Parameters
    ----------
    ccf      : 2D CCF array
    grid_len : size of fitting neighbourhood (must be odd)

    Returns
    -------
    xpar, ypar : sub-pixel offsets from the integer peak location
    """
    ym, xm = np.where(ccf == ccf[1:-1, 1:-1].max())
    xmax, ymax = int(xm[0]), int(ym[0])

    half = grid_len // 2
    fxy  = ccf[ymax - half: ymax + half + 1,
               xmax - half: xmax + half + 1]
    vals = fxy.flatten()

    coords = np.arange(-half, half + 1)   # e.g. [-2,-1,0,1,2] for grid_len=5
    coeff_arr = [
        [i**2, j**2, i, j, i*j, 1]
        for i in coords for j in coords
    ]
    p, *_ = lstsq(np.array(coeff_arr), vals)
    a, b, c, d, e, _ = p

    denom = 4 * a * b - e**2
    if abs(denom) < 1e-12:
        return 0.0, 0.0

    ypar = (e * c - 2 * a * d) / denom
    xpar = (-e * ypar - c) / (2 * a)
    return xpar, ypar


def get_flow_velocity(
    ccf_av: np.ndarray,
    patch_size: int,
    pixel_size_deg: float,
    cadence_interp: int,
    R_sun_Mm: float,
    grid_len: int = 5,
    ntry: int = 4,
) -> tuple[float, float, float, float]:
    """
    Extract flow velocities (ux, uy) from an averaged CCF via iterative
    ellipsoid peak fitting.

    Parameters
    ----------
    ccf_av         : (patch_size, patch_size) averaged CCF
    patch_size     : size of the CCF array [pixels]
    pixel_size_deg : pixel scale of the remapped image [deg/pixel]
    cadence_interp : effective cadence for velocity calculation [seconds]
    R_sun_Mm       : solar radius [Mm]
    grid_len       : fitting neighbourhood size (odd integer)
    ntry           : number of iterative refinement steps

    Returns
    -------
    dx_tot, dy_tot : total pixel displacement
    ux, uy         : flow velocities [m/s]
    """
    assert grid_len % 2 == 1, 'grid_len must be odd'

    dx_tot = 0.0
    dy_tot = 0.0
    ccf    = ccf_av.copy()

    for _ in range(ntry):
        xpar, ypar = _fit_ellipsoid_peak(ccf, grid_len)

        ym, xm = np.where(ccf == ccf[1:-1, 1:-1].max())
        xmax, ymax = int(xm[0]), int(ym[0])

        del_x = xmax - patch_size // 2 + xpar
        del_y = ymax - patch_size // 2 + ypar

        ccf    = shift(ccf, [-del_y, -del_x], mode='reflect')
        dx_tot += del_x
        dy_tot += del_y

    ux = R_sun_Mm * dx_tot * np.deg2rad(pixel_size_deg) / cadence_interp * 1e6
    uy = R_sun_Mm * dy_tot * np.deg2rad(pixel_size_deg) / cadence_interp * 1e6
    return dx_tot, dy_tot, ux, uy


# ── FFT-based sub-pixel shift (alternative) ───────────────────────────────

def fft_shift(img: np.ndarray, shift_xy: tuple[float, float]) -> np.ndarray:
    """
    Sub-pixel image shift via phase ramp in Fourier space.

    Parameters
    ----------
    img      : 2D image array
    shift_xy : (shift_x, shift_y) in pixels

    Returns
    -------
    Shifted image (real part)
    """
    try:
        import pyfftw.interfaces.numpy_fft as fft # type: ignore
    except ImportError:
        import numpy.fft as fft

    sz     = img.shape
    yf     = fft.fftfreq(sz[1], d=1 / sz[0])
    xf     = fft.fftfreq(sz[0], d=1 / sz[1])
    img_f  = fft.fft2(img)
    phase  = np.exp(-2j * np.pi * (
        yf[:, None] * shift_xy[1] / sz[1] +
        xf[None, :] * shift_xy[0] / sz[0]))
    return fft.ifft2(img_f * phase).real
