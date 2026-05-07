"""
eigenfunction.py
----------------
SVD-based extraction of the dominant inertial mode eigenfunction
from the filtered time-domain flow maps.
"""

import numpy as np
from numpy import linalg

from .config import LAT_SVD_MAX
from .fourier import extract_m_slice, inverse_time_transform


def _lat_svd_mask(lats: np.ndarray, lat_max: float = LAT_SVD_MAX) -> np.ndarray:
    """Return boolean mask selecting latitudes within ±lat_max degrees."""
    return np.abs(lats) <= lat_max


def extract_eigenfunction(
    uphi_ft: np.ndarray,
    uthe_ft: np.ndarray,
    freq_nHz: np.ndarray,
    m: int,
    cent_freq: float,
    lats: np.ndarray,
    df: float = 10.0,
    lat_for_scaling: float = 0.0,
) -> dict:
    """
    Extract the dominant eigenfunction at azimuthal order m via SVD.

    The leading singular vector in time is used as the mode's time dependence.
    Both u_phi and u_theta are projected onto it to yield the spatial
    eigenfunction.

    Parameters
    ----------
    uphi_ft, uthe_ft : (nfreq, nlat, nm) full Fourier arrays
    freq_nHz         : (nfreq,) frequency axis [nHz]
    m                : azimuthal order
    cent_freq        : central frequency [nHz]
    lats             : (nlat,) latitude array [degrees]
    df               : bandpass half-width [nHz]
    lat_for_scaling  : latitude used to normalise the time dependence amplitude

    Returns
    -------
    dict with keys:
        ef_uphi  : (nlat,) complex eigenfunction for u_phi
        ef_uthe  : (nlat,) complex eigenfunction for u_theta
        final_td : (nt,) amplitude of the time dependence
    """
    nfreq, nlat, nlng = uphi_ft.shape[0], uphi_ft.shape[1], uphi_ft.shape[2]
    nt = nfreq  # same number of time steps after fftshift

    # ── Bandpass filter and return to time domain ─────────────────────
    uphi_filt, uthe_filt = extract_m_slice(
        uphi_ft, uthe_ft, m, cent_freq, freq_nHz, df=df)

    uphi_t = inverse_time_transform(uphi_filt, nt, nlat, nlng, m)
    uthe_t = inverse_time_transform(uthe_filt, nt, nlat, nlng, m)

    # m-th column of each array: shape (nt, nlat)
    uphi_tm = uphi_t[:, :, m]
    uthe_tm = uthe_t[:, :, m]

    # ── SVD on the joint (uphi, uthe) matrix restricted to |lat| ≤ 75° ──
    lat_mask = _lat_svd_mask(lats)
    arr_svd  = np.concatenate(
        (uphi_tm[:, lat_mask], uthe_tm[:, lat_mask]), axis=1)
    U, s, Vh = linalg.svd(arr_svd, full_matrices=False)

    # ── Project onto leading singular vector ──────────────────────────
    time_dep = U[:, 0]
    m_factor = 2.0 / nlng / np.sqrt(np.mean(np.abs(time_dep) ** 2))

    ef_uphi  = np.mean(uphi_tm * np.conj(time_dep[:, None]) * m_factor, axis=0)
    ef_uthe  = np.mean(uthe_tm * np.conj(time_dep[:, None]) * m_factor, axis=0)

    # ── Time dependence amplitude at the scaling latitude ─────────────
    scale_idx = np.argmin(np.abs(lats - lat_for_scaling))
    final_td  = np.abs(s[0] * Vh[0, scale_idx] * np.abs(time_dep) * 2.0 / nlng)

    return {
        'ef_uphi':  ef_uphi,
        'ef_uthe':  ef_uthe,
        'final_td': final_td,
    }
