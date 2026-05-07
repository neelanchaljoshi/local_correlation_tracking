"""
errors.py
---------
Error estimation for Legendre-projected eigenfunctions.
Three methods available:
  - monte_carlo           : phase-randomised MC over discarded coefficients
  - monte_carlo_amp       : amplitude + phase randomised MC
  - fl_sum                : deterministic sum of discarded modes
"""

import numpy as np
from scipy.special import legendre


def _legendre_basis(theta: np.ndarray, l_vals: list) -> np.ndarray:
    """
    Compute normalised Legendre polynomials at colatitudes theta.

    Returns
    -------
    P : (len(l_vals), len(theta)) array
    """
    return np.array([
        np.sqrt((2 * l + 1) / 2) * legendre(l)(np.cos(theta))
        for l in l_vals
    ])


def monte_carlo_phase(
    theta: np.ndarray,
    l_discard: list,
    fl_discard: np.ndarray,
    num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate errors by randomising the phase of discarded coefficients.

    Parameters
    ----------
    theta      : (nlat,) colatitude in radians
    l_discard  : list of discarded ell values
    fl_discard : complex array of discarded Legendre coefficients
    num_samples: number of Monte Carlo trials

    Returns
    -------
    std_real, std_imag : (nlat,) 1-sigma errors on real and imaginary parts
    """
    if len(l_discard) == 0:
        z = np.zeros(len(theta))
        return z, z

    P = _legendre_basis(theta, l_discard)   # (n_l, nlat)
    samples = np.array([
        np.sum(
            (np.abs(fl_discard) * np.exp(1j * 2 * np.pi * np.random.rand(len(fl_discard))))[:, None] * P,
            axis=0)
        for _ in range(num_samples)
    ])
    return np.std(samples.real, axis=0), np.std(samples.imag, axis=0)


def monte_carlo_amp_phase(
    theta: np.ndarray,
    l_discard: list,
    fl_discard: np.ndarray,
    num_samples: int = 500,
    amp_std_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate errors by randomising both amplitude and phase of discarded
    coefficients. Amplitude is drawn from N(|fl|, amp_std_frac * |fl|).

    Parameters
    ----------
    theta         : (nlat,) colatitude in radians
    l_discard     : list of discarded ell values
    fl_discard    : complex array of discarded coefficients
    num_samples   : Monte Carlo trials
    amp_std_frac  : fractional std of amplitude perturbation (default 0.2)

    Returns
    -------
    std_real, std_imag : (nlat,) 1-sigma errors
    """
    if len(l_discard) == 0:
        z = np.zeros(len(theta))
        return z, z

    P      = _legendre_basis(theta, l_discard)
    trials = np.zeros((num_samples, len(theta)), dtype=np.complex128)

    for i in range(num_samples):
        amps   = np.random.normal(np.abs(fl_discard),
                                  amp_std_frac * np.abs(fl_discard))
        phases = np.random.uniform(0, 2 * np.pi, len(fl_discard))
        trials[i] = np.sum((amps * np.exp(1j * phases))[:, None] * P, axis=0)

    return np.std(trials.real, axis=0), np.std(trials.imag, axis=0)


def fl_sum_errors(
    theta: np.ndarray,
    l_discard: list,
    fl_discard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic error: absolute value of the sum of discarded modes.

    Parameters
    ----------
    theta      : (nlat,) colatitude in radians
    l_discard  : list of discarded ell values
    fl_discard : complex array of discarded coefficients

    Returns
    -------
    err_real, err_imag : (nlat,) errors (identical for this method)
    """
    if len(l_discard) == 0:
        z = np.zeros(len(theta))
        return z, z

    P       = _legendre_basis(theta, l_discard)
    u_total = np.sum(fl_discard[:, None] * P, axis=0)
    err     = np.abs(u_total)
    return err, err


# ── Dispatcher ────────────────────────────────────────────────────────────

def compute_errors(
    theta: np.ndarray,
    l_discard: list,
    fl_discard: np.ndarray,
    method: str = 'monte_carlo',
    num_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch to the requested error estimation method.

    Parameters
    ----------
    theta      : (nlat,) colatitude in radians
    l_discard  : list of discarded ell values
    fl_discard : complex array of discarded Legendre coefficients
    method     : 'monte_carlo' | 'monte_carlo_amp' | 'fl_sum'
    num_samples: MC trials (ignored for fl_sum)

    Returns
    -------
    std_real, std_imag : (nlat,) 1-sigma errors
    """
    fl_discard = np.asarray(fl_discard)

    if method == 'monte_carlo':
        return monte_carlo_phase(theta, l_discard, fl_discard, num_samples)
    elif method == 'monte_carlo_amp':
        return monte_carlo_amp_phase(theta, l_discard, fl_discard, num_samples)
    elif method == 'fl_sum':
        return fl_sum_errors(theta, l_discard, fl_discard)
    else:
        raise ValueError(
            f"Unknown error method {method!r}. "
            "Choose 'monte_carlo', 'monte_carlo_amp', or 'fl_sum'.")
