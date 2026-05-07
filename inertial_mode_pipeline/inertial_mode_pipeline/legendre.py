"""
legendre.py
-----------
Legendre polynomial projection of eigenfunctions, noise-mode filtering,
phase alignment, and orchestration of error estimation.
"""

import numpy as np
from scipy import integrate
from scipy.special import legendre
from scipy.stats import chi2

from .config import L_ARRAY_MAX, L_MAX_RECON, L_THEORY_CUTOFF, NOISE_CONFIDENCE
from .errors import compute_errors


# ── Projection ────────────────────────────────────────────────────────────

def project_to_legendre_coefficients(
    ef: np.ndarray,
    theta: np.ndarray,
    l_array: np.ndarray,
) -> np.ndarray:
    """
    Compute Legendre coefficients f_ell for a 1-D complex eigenfunction.

    f_ell = ∫ ef(θ) * P_ell(cosθ) * sinθ * norm dθ

    Parameters
    ----------
    ef      : (nlat,) complex eigenfunction
    theta   : (nlat,) colatitude in radians
    l_array : 1-D array of ell values to compute

    Returns
    -------
    fl : complex array, same length as l_array
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    fl = np.zeros(len(l_array), dtype=np.complex128)
    for i, l in enumerate(l_array):
        norm  = np.sqrt((2 * l + 1) / 2)
        P_l   = legendre(l)(cos_theta)
        fl[i] = integrate.simpson(ef * P_l * sin_theta * norm, x=theta)
    return fl


def enforce_symmetry(
    fl_uphi: np.ndarray,
    fl_uthe: np.ndarray,
    l_array: np.ndarray,
    symmetryuphi: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """
    Zero out wrong-parity Legendre coefficients and return the parity-selected
    sub-arrays used for power spectrum analysis.

    Parameters
    ----------
    fl_uphi, fl_uthe : full complex coefficient arrays (len = L_ARRAY_MAX)
    l_array          : full ell array
    symmetryuphi     : 'sym' | 'anti' | 'all'

    Returns
    -------
    fl_uphi, fl_uthe     : zeroed-out coefficient arrays (modified in-place copy)
    l_uphi, l_uthe       : ell values for each component
    fl_uphi_sym, fl_uthe_sym : parity-selected sub-arrays for power analysis
    """
    fl_uphi = fl_uphi.copy()
    fl_uthe = fl_uthe.copy()

    if symmetryuphi == 'anti':
        fl_uphi[0::2] = 0      # u_phi: odd ell only
        fl_uthe[1::2] = 0      # u_theta: even ell only
        l_uphi, l_uthe           = l_array[1::2], l_array[0::2]
        fl_uphi_sym, fl_uthe_sym = fl_uphi[1::2], fl_uthe[0::2]
    elif symmetryuphi == 'sym':
        fl_uphi[1::2] = 0
        fl_uthe[0::2] = 0
        l_uphi, l_uthe           = l_array[0::2], l_array[1::2]
        fl_uphi_sym, fl_uthe_sym = fl_uphi[0::2], fl_uthe[1::2]
    elif symmetryuphi == 'all':
        l_uphi = l_uthe           = l_array
        fl_uphi_sym = fl_uthe_sym = fl_uphi   # both use the same array
    else:
        raise ValueError(f"symmetryuphi must be 'sym', 'anti', or 'all', got {symmetryuphi!r}")

    return fl_uphi, fl_uthe, l_uphi, l_uthe, fl_uphi_sym, fl_uthe_sym


# ── Noise filtering ───────────────────────────────────────────────────────

def _confidence_threshold(
    power: np.ndarray,
    l_theory_cutoff: int,
    confidence: float,
) -> float:
    """
    Compute the power threshold at a given confidence level using a chi-squared
    noise model estimated from modes above l_theory_cutoff.

    Parameters
    ----------
    power           : 1-D array of |f_ell|^2 values
    l_theory_cutoff : modes at or below this are trusted signal; above is noise
    confidence      : confidence level (e.g. 0.90)

    Returns
    -------
    threshold power (float)
    """
    noise_region = power[int(l_theory_cutoff // 2):]
    noise_floor  = np.median(noise_region)
    n_bins       = len(noise_region)
    cdf          = confidence ** (1.0 / n_bins)
    z_conf       = chi2.ppf(cdf, df=2)
    return z_conf * noise_floor / 2


def compute_keep_mask(
    power: np.ndarray,
    l_vals: np.ndarray,
    l_theory_cutoff: int = L_THEORY_CUTOFF,
    confidence: float = NOISE_CONFIDENCE,
) -> np.ndarray:
    """
    Return a boolean mask of which ell modes to keep.

    Modes at or below l_theory_cutoff are always kept.
    Modes above it are kept only if their power exceeds the noise threshold
    AND they fall within the 99th percentile cumulative power.

    Parameters
    ----------
    power           : 1-D array of |f_ell|^2 for the parity-selected modes
    l_vals          : corresponding ell values
    l_theory_cutoff : always-keep boundary
    confidence      : confidence level for chi-squared threshold

    Returns
    -------
    keep_mask : boolean array, same length as power
    """
    threshold   = _confidence_threshold(power, l_theory_cutoff, confidence)
    cumulative  = np.cumsum(power) / np.sum(power)
    idx_99      = np.argmax(cumulative >= 0.99)
    mask_low    = l_vals <= l_theory_cutoff
    mask_high   = ((l_vals > l_theory_cutoff) &
                   (np.arange(len(power)) <= idx_99) &
                   (power > threshold))
    return mask_low | mask_high


# ── Reconstruction ────────────────────────────────────────────────────────

def reconstruct_from_coefficients(
    fl: np.ndarray,
    l_array: np.ndarray,
    l_keep: np.ndarray,
    theta: np.ndarray,
    l_max: int = L_MAX_RECON,
) -> np.ndarray:
    """
    Reconstruct a 1-D eigenfunction from a subset of Legendre coefficients.

    Parameters
    ----------
    fl      : full complex coefficient array (indexed by ell)
    l_array : full ell array
    l_keep  : ell values to include in the reconstruction
    theta   : (nlat,) colatitude in radians
    l_max   : hard upper limit on ell (modes at or above this are skipped)

    Returns
    -------
    recon : (nlat,) complex reconstructed eigenfunction
    """
    cos_theta = np.cos(theta)
    recon     = np.zeros(len(theta), dtype=np.complex128)
    for l in l_array:
        if l >= l_max:
            continue
        if l in l_keep:
            norm   = np.sqrt((2 * l + 1) / 2)
            recon += fl[l] * norm * legendre(l)(cos_theta)
    return recon


def reconstruct_full(
    fl: np.ndarray,
    l_array: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Reconstruct using all Legendre coefficients (no noise filtering)."""
    cos_theta = np.cos(theta)
    return sum(
        fl[l] * np.sqrt((2 * l + 1) / 2) * legendre(l)(cos_theta)
        for l in l_array)


# ── Phase alignment ───────────────────────────────────────────────────────

def align_phase_at_equator(
    uphi: np.ndarray,
    uthe: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate both eigenfunctions so that u_theta is real at the equator.

    Parameters
    ----------
    uphi, uthe : complex 1-D arrays
    theta      : (nlat,) colatitude in radians

    Returns
    -------
    uphi_aligned, uthe_aligned
    """
    equator_idx = np.argmin(np.abs(np.rad2deg(theta) - 90))
    phase       = -1j * np.angle(uthe[equator_idx])
    return uphi * np.exp(phase), uthe * np.exp(phase)


# ── Top-level orchestrator ────────────────────────────────────────────────

def project_and_clean(
    m: int,
    ef_uphi: np.ndarray,
    ef_uthe: np.ndarray,
    lats: np.ndarray,
    symmetryuphi: str = 'anti',
    l_max: int = L_MAX_RECON,
    l_theory_cutoff: int = L_THEORY_CUTOFF,
    confidence: float = NOISE_CONFIDENCE,
    num_mc_samples: int = 500,
    error_method: str = 'monte_carlo',
) -> dict:
    """
    Full Legendre projection, noise filtering, reconstruction, and error
    estimation pipeline for one eigenfunction.

    Parameters
    ----------
    m              : azimuthal order (for logging)
    ef_uphi, ef_uthe : (nlat,) raw complex eigenfunctions from SVD
    lats           : (nlat,) latitude array [degrees]
    symmetryuphi   : equatorial symmetry of u_phi ('sym' | 'anti' | 'all')
    l_max          : maximum ell in reconstruction
    l_theory_cutoff: always-keep ell boundary
    confidence     : noise confidence level
    num_mc_samples : MC trials for error estimation
    error_method   : 'monte_carlo' | 'monte_carlo_amp' | 'fl_sum'

    Returns
    -------
    dict with keys:
        ef_uphi, ef_uthe         – noise-filtered reconstructions
        ef_uphi_sm, ef_uthe_sm   – full Legendre-smoothed eigenfunctions
        uphi_err_real, uphi_err_imag,
        uthe_err_real, uthe_err_imag – 1-sigma errors
    """
    theta     = np.deg2rad(90 - lats)
    l_array   = np.arange(L_ARRAY_MAX)

    # ── Project ─────────────────────────────────────────────────────────
    fl_uphi_raw = project_to_legendre_coefficients(ef_uphi, theta, l_array)
    fl_uthe_raw = project_to_legendre_coefficients(ef_uthe, theta, l_array)

    # ── Symmetry enforcement ─────────────────────────────────────────────
    (fl_uphi, fl_uthe,
     l_uphi, l_uthe,
     fl_uphi_sym, fl_uthe_sym) = enforce_symmetry(
        fl_uphi_raw, fl_uthe_raw, l_array, symmetryuphi)

    # ── Full smoothed reconstruction (all modes, no noise cut) ──────────
    uphi_sm = reconstruct_full(fl_uphi, l_array, theta)
    uthe_sm = reconstruct_full(fl_uthe, l_array, theta)

    # ── Noise filtering ──────────────────────────────────────────────────
    keep_uphi = compute_keep_mask(
        np.abs(fl_uphi_sym) ** 2, l_uphi, l_theory_cutoff, confidence)
    keep_uthe = compute_keep_mask(
        np.abs(fl_uthe_sym) ** 2, l_uthe, l_theory_cutoff, confidence)

    l_uphi_keep = l_uphi[keep_uphi]
    l_uthe_keep = l_uthe[keep_uthe]
    print(f'  m={m} | u_phi  keeping ℓ = {l_uphi_keep}')
    print(f'  m={m} | u_theta keeping ℓ = {l_uthe_keep}')

    # ── Filtered reconstruction ──────────────────────────────────────────
    uphi_recon = reconstruct_from_coefficients(fl_uphi, l_array, l_uphi_keep, theta, l_max)
    uthe_recon = reconstruct_from_coefficients(fl_uthe, l_array, l_uthe_keep, theta, l_max)

    # ── Discarded coefficients ───────────────────────────────────────────
    l_disc_uphi  = [l for l in l_uphi if l not in l_uphi_keep]
    l_disc_uthe  = [l for l in l_uthe if l not in l_uthe_keep]
    fl_disc_uphi = np.array([fl_uphi[l] for l in l_disc_uphi])
    fl_disc_uthe = np.array([fl_uthe[l] for l in l_disc_uthe])

    # ── Error estimation ─────────────────────────────────────────────────
    uphi_err_r, uphi_err_i = compute_errors(
        theta, l_disc_uphi, fl_disc_uphi, error_method, num_mc_samples)
    uthe_err_r, uthe_err_i = compute_errors(
        theta, l_disc_uthe, fl_disc_uthe, error_method, num_mc_samples)

    # ── Phase alignment ──────────────────────────────────────────────────
    uphi_recon, uthe_recon = align_phase_at_equator(uphi_recon, uthe_recon, theta)
    uphi_sm,    uthe_sm    = align_phase_at_equator(uphi_sm,    uthe_sm,    theta)

    return {
        'ef_uphi':       uphi_recon,
        'ef_uthe':       uthe_recon,
        'ef_uphi_sm':    uphi_sm,
        'ef_uthe_sm':    uthe_sm,
        'uphi_err_real': uphi_err_r,
        'uphi_err_imag': uphi_err_i,
        'uthe_err_real': uthe_err_r,
        'uthe_err_imag': uthe_err_i,
    }
