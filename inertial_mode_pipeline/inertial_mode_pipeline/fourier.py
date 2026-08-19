"""
fourier.py
----------
Fourier transforms, Carrington-frame conversion, and frequency filtering.
Pure computation — no I/O, no disk geometry.
"""

import numpy as np # type: ignore

from .config import DT_SEC


# ── Window functions ──────────────────────────────────────────────────────

def tukeywin(N: int, alpha: float = 0.0) -> np.ndarray:
    """
    Tukey (tapered cosine) window of length N with taper fraction alpha.

    alpha=0 → rectangular, alpha=1 → Hanning.
    """
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    x = np.linspace(0, 1, N)
    w = np.ones(N)
    left  = x < alpha / 2
    right = x >= 1 - alpha / 2
    w[left]  = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[left]  - alpha / 2)))
    w[right] = 0.5 * (1 + np.cos(2 * np.pi / alpha * (x[right] - 1 + alpha / 2)))
    return w


# ── Main transform ────────────────────────────────────────────────────────

def transform_to_fourier(
    arr: np.ndarray,
    crln: np.ndarray,
    cft: np.ndarray,
    cfl: np.ndarray,
    span: np.ndarray,
    dt: float = DT_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a flow array to the (frequency, latitude, m) Fourier space
    in the Carrington rotating frame.

    Steps
    -----
    1. Real FFT in longitude → azimuthal order m
    2. Rotate to Carrington frame via per-timestep phase shift
    3. Complex FFT in time
    4. fftshift to centre zero frequency

    Parameters
    ----------
    arr  : (nt, nlat, nlng) flow array, NaNs filled with zeros
    crln : (nt,) Carrington longitude [degrees]
    cft  : time correction factor from geometry.get_correction_factor
    cfl  : longitude correction factor from geometry.get_correction_factor
    span : boolean mask of length nt selecting the time range
    dt   : cadence in seconds (default: 6 h)

    Returns
    -------
    ft       : (nt_span, nlat, nm) complex Fourier array (fftshifted in time)
    freq_nHz : (nt_span,) frequency axis in nHz (fftshifted)
    """
    data        = np.nan_to_num(arr[span])
    fft_m       = np.fft.rfft(data, axis=2) * np.nan_to_num(cfl[span])
    M_arr       = np.arange(fft_m.shape[2])
    carr_phase  = np.exp(-1j * np.deg2rad(crln[span])[:, None] * M_arr[None, :])
    fft_m_carr  = fft_m * carr_phase[:, None, :]
    ft          = np.fft.fft(fft_m_carr, axis=0) * np.sqrt(np.nan_to_num(cft))
    freq_nHz    = np.fft.fftshift(-np.fft.fftfreq(ft.shape[0], dt) * 1e9)
    ft          = np.fft.fftshift(ft, axes=0)
    return ft, freq_nHz


# ── Frequency filtering ───────────────────────────────────────────────────

def bandpass_filter(
    uphi_ft_m: np.ndarray,
    uthe_ft_m: np.ndarray,
    freq_nHz: np.ndarray,
    cent_freq: float,
    df: float = 20.0,
    tukey_alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a Tukey bandpass filter around cent_freq ± df nHz.

    Parameters
    ----------
    uphi_ft_m, uthe_ft_m : (nfreq, nlat) complex arrays at fixed m
    freq_nHz             : (nfreq,) frequency axis in nHz
    cent_freq            : centre frequency [nHz]
    df                   : half-bandwidth [nHz]
    tukey_alpha          : taper fraction for Tukey window

    Returns
    -------
    uphi_filt, uthe_filt : bandpass-filtered arrays, same shape as input
    """
    band   = (freq_nHz > cent_freq - df) & (freq_nHz < cent_freq + df)
    window = np.zeros_like(freq_nHz)
    window[band] = tukeywin(band.sum(), tukey_alpha)
    uphi_filt = uphi_ft_m * window[:, np.newaxis]
    uthe_filt = uthe_ft_m * window[:, np.newaxis]
    return uphi_filt, uthe_filt


def extract_m_slice(
    uphi_ft: np.ndarray,
    uthe_ft: np.ndarray,
    m: int,
    cent_freq: float,
    freq_nHz: np.ndarray,
    df: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract and bandpass-filter the m-th azimuthal slice from the full
    Fourier array.

    Parameters
    ----------
    uphi_ft, uthe_ft : (nfreq, nlat, nm) full Fourier arrays
    m                : azimuthal order
    cent_freq        : central frequency [nHz]
    freq_nHz         : frequency axis [nHz]
    df               : half-bandwidth [nHz]

    Returns
    -------
    uphi_filt, uthe_filt : (nfreq, nlat) filtered slices
    """
    return bandpass_filter(
        uphi_ft[:, :, m],
        uthe_ft[:, :, m],
        freq_nHz, cent_freq, df=df)


def inverse_time_transform(
    filt_m: np.ndarray,
    nt: int,
    nlat: int,
    nlng: int,
    m: int,
) -> np.ndarray:
    """
    Inverse FFT in time for a single m-slice, returning (nt, nlat, nlng).

    Parameters
    ----------
    filt_m : (nfreq, nlat) filtered Fourier slice
    nt, nlat, nlng : output dimensions
    m              : azimuthal order (column index to place the slice)

    Returns
    -------
    arr_t : (nt, nlat, nlng) complex time-domain array
    """
    arr_f      = np.zeros((nt, nlat, nlng), dtype=np.complex128)
    arr_f[:, :, m] = filt_m
    return np.fft.ifft(np.fft.ifftshift(arr_f, axes=0), axis=0)
