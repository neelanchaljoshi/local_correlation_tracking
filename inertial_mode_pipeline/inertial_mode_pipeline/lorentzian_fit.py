"""
lorentzian_fit.py
------------------
Maximum-likelihood Lorentzian fitting of a power spectrum, with
parametric Monte Carlo error estimates. Pure fitting code — no I/O,
no Fourier transforms, no disk geometry.

Ported from Zhi-Chao's rewrite of plotting_scripts/table_lor_fit.py
(see /data/seismo/zhichao/codes/Joshi/git_lorentzian_fit_joshi1).
"""

from contextlib import contextmanager
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, minimize


# ── Timing helper ─────────────────────────────────────────────────────────

def format_elapsed(seconds: float) -> str:
    """Return elapsed seconds as HH:MM:SS.s."""
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f'{int(hours):02d}:{int(minutes):02d}:{seconds:04.1f}'


@contextmanager
def timer(label: str, logger=None):
    """Report wall-clock elapsed time for a code block."""
    tic = perf_counter()
    try:
        yield
    finally:
        message = f'{label} elapsed {format_elapsed(perf_counter() - tic)}'
        if logger is None:
            print(message)
        else:
            logger.info(message)


# ── Lorentzian model ──────────────────────────────────────────────────────

def lorentzian(x: np.ndarray, A: float, x0: float, fwhm: float, B: float) -> np.ndarray:
    """
    Lorentzian profile with flat background.

    Parameters
    ----------
    A    : peak amplitude above background (>0)
    x0   : centre frequency [nHz]
    fwhm : full-width at half-maximum (>0) [nHz]
    B    : flat background level (>0)

    Peak value = A + B. HWHM = fwhm / 2.
    """
    half_width = 0.5 * fwhm
    return A * half_width**2 / ((x - x0)**2 + half_width**2) + B


def neg_log_likelihood(params, x, y):
    """
    Negative log-likelihood for Gamma-distributed power spectra using
    [log(A), x0, log(fwhm), log(B)].

    NLL = sum_i [ ln(S_i) + P_i / S_i ]
    """
    log_A, x0, log_fwhm, log_B = params
    A = np.exp(log_A)
    fwhm = np.exp(log_fwhm)
    B = np.exp(log_B)
    model = lorentzian(x, A, x0, fwhm, B)
    if np.any(model <= 0):
        return np.inf
    return np.sum(np.log(model) + y / model)


# ── Fit object ────────────────────────────────────────────────────────────

class LorentzianMLE:
    """
    Lorentzian maximum-likelihood fit with Monte Carlo error estimates.

    Parameters
    ----------
    x, y : frequency [nHz] and power spectrum arrays to fit
    n_avg : effective number of independent samples averaged into each
        power-spectrum bin (drives the chi-squared degrees of freedom
        used by the Monte Carlo error estimate)
    label, mode, method : free-text metadata for printing/plotting
    fwhm_max_factor : upper bound on fitted FWHM, as a fraction of the
        fitted x-range
    use_differential_evolution : if True, run a global optimizer first
        to find the initial guess instead of the automatic heuristic
    initial_params : optional [log(A), x0, log(fwhm), log(B)] override
        for the initial guess
    """

    def __init__(self, x, y, n_avg, label='', mode='', method='',
                 fwhm_max_factor=1.0, use_differential_evolution=False,
                 initial_params=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.n_avg = n_avg
        self.label = label
        self.mode = mode
        self.method = method
        self.freq_res = np.abs(np.diff(self.x)).mean()
        self.x_range = self.x.max() - self.x.min()
        self.fwhm_max = fwhm_max_factor * self.x_range
        self.use_differential_evolution = use_differential_evolution
        self.initial_params = initial_params
        y_floor = max(1e-300, 1e-8 * np.nanmax(self.y))
        y_min_positive = max(y_floor, np.nanmin(self.y))

        self.de_bounds = [
            (np.log(1e-8 * self.y.max()), np.log(100.0 * self.y.max())),
            (self.x.min(), self.x.max()),
            (np.log(2.0 * self.freq_res), np.log(self.fwhm_max)),
            (np.log(y_min_positive), np.log(self.y.max())),
        ]
        self.popt_log = None
        self.popt = None
        self.nll_min = None
        self.lo_err = None
        self.hi_err = None
        self.mc_params = None
        self.fit_seconds = np.nan
        self.mc_seconds = np.nan

    def initial_guess(self):
        if self.initial_params is not None:
            p0 = np.asarray(self.initial_params, dtype=float)
            if p0.shape != (4,):
                raise ValueError('initial_params must be [log(A), x0, log(fwhm), log(B)]')
        else:
            c0 = np.nanmedian(self.y)
            a0 = max(np.nanmax(self.y) - c0, np.exp(self.de_bounds[0][0]))
            x0 = self.x[np.nanargmax(self.y)]
            fwhm0 = min(max(0.2 * self.x_range, np.exp(self.de_bounds[2][0])), np.exp(self.de_bounds[2][1]))
            c0 = max(c0, np.exp(self.de_bounds[3][0]))
            p0 = np.array([np.log(a0), x0, np.log(fwhm0), np.log(c0)], dtype=float)

        return p0

    def fit(self):
        if self.label:
            print(f'\n--- {self.label} ---')

        tic = perf_counter()
        try:
            if self.use_differential_evolution:
                result_global = differential_evolution(
                    neg_log_likelihood, self.de_bounds, args=(self.x, self.y),
                    seed=42, maxiter=3000, tol=1e-12, polish=True,
                    popsize=20, mutation=(0.5, 1.5), recombination=0.7,
                    workers=1,
                )
                p0 = result_global.x
            else:
                p0 = self.initial_guess()

            result = minimize(
                neg_log_likelihood, p0, args=(self.x, self.y),
                method='L-BFGS-B',
                options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 100000},
            )
        finally:
            self.fit_seconds = perf_counter() - tic

        self.popt_log = result.x
        log_A, x0, log_fwhm, log_B = result.x
        self.popt = np.array([np.exp(log_A), x0, np.exp(log_fwhm), np.exp(log_B)])
        self.nll_min = result.fun
        return self.popt

    def monte_carlo_errors(self, n_mc=1000, rng=None):
        """
        Parametric Monte Carlo error estimate using rng.chisquare.
        """
        if self.popt is None:
            self.fit()
        if rng is None:
            rng = np.random.default_rng(42)

        model_best = lorentzian(self.x, *self.popt)
        mc_params_log = np.full((n_mc, 4), np.nan)

        tic = perf_counter()
        try:
            for i in range(n_mc):
                y_sim = model_best * rng.chisquare(
                    df=2 * self.n_avg, size=self.x.size) / (2 * self.n_avg)
                res_mc = minimize(
                    neg_log_likelihood, self.popt_log, args=(self.x, y_sim),
                    method='L-BFGS-B',
                    options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 10000},
                )
                if np.isfinite(res_mc.fun):
                    mc_params_log[i] = res_mc.x
        finally:
            self.mc_seconds = perf_counter() - tic

        good_mc = np.isfinite(mc_params_log).all(axis=1)
        if good_mc.any():
            mc_params = mc_params_log[good_mc].copy()
            mc_params[:, [0, 2, 3]] = np.exp(mc_params[:, [0, 2, 3]])
            q_lo, q_hi = np.nanpercentile(mc_params, [15.865, 84.135], axis=0)
            lo_err = np.maximum(self.popt - q_lo, 0.0)
            hi_err = np.maximum(q_hi - self.popt, 0.0)
        else:
            mc_params = np.empty((0, 4))
            lo_err = np.full(4, np.nan)
            hi_err = np.full(4, np.nan)

        self.lo_err = lo_err
        self.hi_err = hi_err
        self.mc_params = mc_params
        return lo_err, hi_err, mc_params

    def run(self, n_mc=1000, rng=None):
        """
        Fit the model and estimate errors with parametric Monte Carlo.
        """
        self.fit()
        self.monte_carlo_errors(n_mc=n_mc, rng=rng)
        self.print_summary(n_mc=n_mc)
        return self

    def print_summary(self, n_mc=None):
        if self.popt is None or self.lo_err is None or self.hi_err is None:
            return

        A_fit, x0_fit, fwhm_fit, B_fit = self.popt
        snr = A_fit / B_fit
        fit_time = format_elapsed(self.fit_seconds)
        mc_time = format_elapsed(self.mc_seconds) if np.isfinite(self.mc_seconds) else 'n/a'
        print(f'  A     = {A_fit:.5f}  -{self.lo_err[0]:.5f} +{self.hi_err[0]:.5f}  [m2/s2/nHz]')
        print(f'  x0    = {x0_fit:.3f}   -{self.lo_err[1]:.3f} +{self.hi_err[1]:.3f}  [nHz]')
        print(f'  fwhm  = {fwhm_fit:.3f}   -{self.lo_err[2]:.3f} +{self.hi_err[2]:.3f}  [nHz]')
        print(f'  B     = {B_fit:.5f}  -{self.lo_err[3]:.5f} +{self.hi_err[3]:.5f}  [m2/s2/nHz]')
        print(f'  SNR   = {snr:.2f}  (A/B, n_avg={self.n_avg})')
        if n_mc is not None:
            print(f'  MC    = {len(self.mc_params)}/{n_mc} successful realisations')
        print(f'  time  = fit {fit_time}, MC {mc_time}')
        if not self.resolved:
            print(f'  *** WARNING: unresolved - fwhm/2={0.5 * fwhm_fit:.4f} < freq_res={self.freq_res:.4f} nHz ***')

    @property
    def resolved(self):
        if self.popt is None:
            return False
        return 0.5 * self.popt[2] > self.freq_res
