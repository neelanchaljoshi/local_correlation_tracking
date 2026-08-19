# %% imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import minimize, differential_evolution

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 18

# %% Load data
uphi_ft_anti_mag  = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_anti_hmi_m_720s_dt_1h.npy')
uthe_ft_sym_mag   = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_m_720s_dt_1h.npy')
uthe_ft_anti_mag  = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_anti_hmi_m_720s_dt_1h.npy')

uphi_ft_anti_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uphi_ft_2010_2024_anti_hmi_ic_45s_granule.npy')
uthe_ft_sym_gran  = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_sym_hmi_ic_45s_granule.npy')
uthe_ft_anti_gran = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/uthe_ft_2010_2024_anti_hmi_ic_45s_granule.npy')

rda_list       = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/for_rda_mask_comparison/rda_fft.npz')
rda_list_anti  = np.load('/data/seismo/joshin/pipeline-test/local_correlation_tracking/data/processed_data/for_rda_mask_comparison/rda_fft_ns_anti.npz')
uphi_ft_anti_rda = rda_list['fup']
uthe_ft_sym_rda  = rda_list['fut']
uphi_ft_sym_rda  = rda_list_anti['fup']
uthe_ft_anti_rda = rda_list_anti['fut']
freqs_rda        = rda_list['freq']

# %% Frequency and latitude setup
lat_og = np.linspace(-90, 90, 73)
freqs  = np.fft.fftfreq(len(uphi_ft_anti_mag), d=6*3600)
freqs  = -np.fft.fftshift(freqs) * 1e9
freqs_rda = np.fft.fftshift(freqs_rda)

lat_eq_rossby = (np.abs(lat_og) <= 30)
lat_eq_hl     = (np.abs(lat_og) >= 45) & (np.abs(lat_og) <= 75)
lat_eq_cl     = (np.abs(lat_og) >= 15) & (np.abs(lat_og) <= 45)

nt = len(uphi_ft_anti_mag)
dt = 6. * 3600
print('Number of time steps: {}'.format(nt))
conv_factor = 2 / nt * 1e-9 * dt / 144 / 144

nt_rda = len(uphi_ft_anti_rda[:, -1, 0, 0])
print('Number of time steps RDA: {}'.format(nt_rda))
dt_rda = 27.3 / 3 * 3600
conv_factor_rda = 2 / nt_rda * 1e-9 * dt_rda / 144 / 144

# %% Compute latitude-averaged power spectra
power_uphi_m1_mag   = np.nanmean(np.abs(uphi_ft_anti_mag[:, lat_eq_hl,     :])**2, axis=1) * conv_factor
power_uphi_m2_mag   = np.nanmean(np.abs(uphi_ft_anti_mag[:, lat_eq_cl,     :])**2, axis=1) * conv_factor
power_uthe_m8_mag   = np.nanmean(np.abs(uthe_ft_sym_mag[:,  lat_eq_rossby, :])**2, axis=1) * conv_factor
power_uthe_m13_mag  = np.nanmean(np.abs(uthe_ft_anti_mag[:, lat_eq_rossby, :])**2, axis=1) * conv_factor

power_uphi_m1_gran  = np.nanmean(np.abs(uphi_ft_anti_gran[:, lat_eq_hl,     :])**2, axis=1) * conv_factor
power_uphi_m2_gran  = np.nanmean(np.abs(uphi_ft_anti_gran[:, lat_eq_cl,     :])**2, axis=1) * conv_factor
power_uthe_m8_gran  = np.nanmean(np.abs(uthe_ft_sym_gran[:,  lat_eq_rossby, :])**2, axis=1) * conv_factor
power_uthe_m13_gran = np.nanmean(np.abs(uthe_ft_anti_gran[:, lat_eq_rossby, :])**2, axis=1) * conv_factor

power_uphi_m1_rda   = np.nanmean(np.abs(uphi_ft_anti_rda[:, -1, lat_eq_hl,     :])**2, axis=1) * conv_factor_rda
power_uphi_m2_rda   = np.nanmean(np.abs(uphi_ft_anti_rda[:, -1, lat_eq_cl,     :])**2, axis=1) * conv_factor_rda
power_uthe_m8_rda   = np.nanmean(np.abs(uthe_ft_sym_rda[:,  -1, lat_eq_rossby, :])**2, axis=1) * conv_factor_rda
power_uthe_m13_rda  = np.nanmean(np.abs(uthe_ft_anti_rda[:, -1, lat_eq_rossby, :])**2, axis=1) * conv_factor_rda

for arr in [power_uphi_m1_rda, power_uphi_m2_rda, power_uthe_m8_rda, power_uthe_m13_rda]:
    arr[:] = np.fft.fftshift(arr, axes=0)

# %% Frequency windows
freq_window_uphi_m1  = (freqs >= -200) & (freqs <= 0)
freq_window_rda_m1   = (freqs_rda >= -200) & (freqs_rda <= 0)
freq_window_uphi_m2  = (freqs >= -110) & (freqs <= -50)
freq_window_rda_m2   = (freqs_rda >= -110) & (freqs_rda <= -50)
freq_window_uthe_m8  = (freqs >= -180) & (freqs <= -50)
freq_window_rda_m8   = (freqs_rda >= -180) & (freqs_rda <= -50)
freq_window_uthe_m13 = (freqs >= -250) & (freqs <= -150)
freq_window_rda_m13  = (freqs_rda >= -250) & (freqs_rda <= -150)

power_m1_plot_lct_mag   = power_uphi_m1_mag[freq_window_uphi_m1,   1]
power_m1_plot_lct_gran  = power_uphi_m1_gran[freq_window_uphi_m1,  1]
power_m1_plot_rda       = power_uphi_m1_rda[freq_window_rda_m1,    1]
power_m2_plot_lct_mag   = power_uphi_m2_mag[freq_window_uphi_m2,   2]
power_m2_plot_lct_gran  = power_uphi_m2_gran[freq_window_uphi_m2,  2]
power_m2_plot_rda       = power_uphi_m2_rda[freq_window_rda_m2,    2]
power_m8_plot_lct_mag   = power_uthe_m8_mag[freq_window_uthe_m8,   8]
power_m8_plot_lct_gran  = power_uthe_m8_gran[freq_window_uthe_m8,  8]
power_m8_plot_rda       = power_uthe_m8_rda[freq_window_rda_m8,    8]
power_m13_plot_lct_mag  = power_uthe_m13_mag[freq_window_uthe_m13,  13]
power_m13_plot_lct_gran = power_uthe_m13_gran[freq_window_uthe_m13, 13]
power_m13_plot_rda      = power_uthe_m13_rda[freq_window_rda_m13,   13]

# =============================================================================
# Fitting: Maximum Likelihood Estimation for chi-squared distributed power
# =============================================================================
#
# Physical setup
# --------------
# Each power estimate P_i is the mean over n_avg independent latitude bins
# of |F_i|^2. Each |F_i|^2 is exponentially distributed (chi^2_2 / 2) with
# mean S_i (the true PSD at frequency i). The mean of n_avg such values
# follows a Gamma distribution:
#
#   P_i ~ Gamma(n_avg,  S_i / n_avg)
#
#   p(P_i | S_i) = (n_avg / S_i)^n_avg * P_i^(n_avg-1) * exp(-n_avg P_i / S_i)
#                  / Gamma(n_avg)
#
# The log-likelihood (dropping terms constant in the model parameters) is:
#
#   ln L = -n_avg * sum_i [ ln S_i + P_i / S_i ]
#
# So the negative log-likelihood to minimise is:
#
#   NLL = n_avg * sum_i [ ln S_i + P_i / S_i ]
#
# This is the standard "M-estimator" used in helioseismology since
# Anderson, Duvall & Jefferies (1990) and formalised in
# Appourchaux et al. (1998, A&A 338, 1049).
#
# Profile likelihood confidence intervals (Wilks 1938)
# -----------------------------------------------------
# For each scalar parameter theta_k, the profile NLL is:
#
#   NLL_prof(theta_k) = min_{all others} NLL(theta_k, nuisance params)
#
# The 1-sigma interval is where:
#
#   NLL_prof(theta_k) - NLL_min  <=  0.5
#
# because Wilks' theorem states that 2 * [NLL_prof - NLL_min] ~ chi^2_1,
# and chi^2_1(68.27%) = 1.0, so the threshold on NLL differences is 0.5.
#
# This gives correct asymmetric intervals that respect physical bounds
# (A > 0, gamma > 0, C > 0) without any Gaussian approximation.
#
# Reference: Appourchaux et al. (1998) use exactly this MLE + profile
# likelihood approach for p-mode fitting in GOLF/VIRGO data.
# The same method is used in Gizon & Solanki (2003, ApJ 589, 1009) for
# local helioseismology, and in Loptien et al. (2018, NatAstron) and
# Loeptien et al. / Gizon et al. inertial mode papers for spectral fits
# to latitude-averaged power spectra of solar surface flows.

def lorentzian(x, A, x0, gamma, C):
    """
    Lorentzian profile with flat background.
      A     : peak amplitude above background  (>0)
      x0    : centre frequency                 (nHz)
      gamma : half-width at half-maximum HWHM  (>0, nHz)
      C     : flat background level            (>0)
    Peak value = A + C.  FWHM = 2*gamma.
    """
    return A * gamma**2 / ((x - x0)**2 + gamma**2) + C


def neg_log_likelihood(params, x, y, n_avg):
    """
    Negative log-likelihood for Gamma-distributed power spectra.

    NLL = n_avg * sum_i [ ln(S_i) + P_i / S_i ]

    where S_i = lorentzian(x_i; params) is the model PSD.
    Returns +inf for unphysical parameters (A, gamma, C <= 0).
    """
    A, x0, gamma, C = params
    if A <= 0 or gamma <= 0 or C <= 0:
        return np.inf
    model = lorentzian(x, A, x0, gamma, C)
    if np.any(model <= 0):
        return np.inf
    return n_avg * np.sum(np.log(model) + y / model)


def fit_lorentzian_mle(x, y, n_avg, label='', gamma_max_factor=0.5, n_scan=300):
    """
    Fit a Lorentzian to a power spectrum using Gamma MLE (Anderson et al. 1990),
    then compute asymmetric 1-sigma confidence intervals via profile likelihood
    (Wilks 1938; Appourchaux et al. 1998).

    Parameters
    ----------
    x                : 1-D array of frequencies (nHz), monotonically increasing
    y                : 1-D array of power values (m^2/s^2/nHz), all > 0
    n_avg            : number of independent latitude bins averaged
    label            : string for printed output
    gamma_max_factor : upper bound on gamma as fraction of x-range
    n_scan           : number of steps in each direction of profile scan

    Returns
    -------
    popt    : ndarray (4,)  best-fit [A, x0, gamma, C]
    lo_err  : ndarray (4,)  lower 1-sigma errors (positive quantities)
    hi_err  : ndarray (4,)  upper 1-sigma errors (positive quantities)
    resolved: bool  True if gamma > freq_res (mode is spectrally resolved)
    """
    if label:
        print(f'\n--- {label} ---')

    freq_res  = np.abs(np.diff(x)).mean()   # frequency bin width (nHz)
    x_range   = x.max() - x.min()
    gamma_max = gamma_max_factor * x_range

    # Parameter bounds: [A, x0, gamma, C]
    # gamma_min = 1 frequency bin (modes narrower than this are unresolved)
    bounds = [
        (1e-8 * y.max(), 100.0 * y.max()),   # A
        (x.min(),        x.max()),            # x0
        (freq_res,       gamma_max),          # gamma >= 1 bin
        (1e-8 * y.min(), y.max()),            # C
    ]

    # ---- Step 1: global optimisation to avoid the spike-degeneracy local minimum
    result_global = differential_evolution(
        neg_log_likelihood, bounds, args=(x, y, n_avg),
        seed=42, maxiter=3000, tol=1e-12, polish=True,
        popsize=20, mutation=(0.5, 1.5), recombination=0.7,
        workers=1,
    )

    # ---- Step 2: local refinement from global best (L-BFGS-B respects bounds)
    result = minimize(
        neg_log_likelihood, result_global.x, args=(x, y, n_avg),
        method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 100000},
    )

    popt    = result.x
    nll_min = result.fun
    A_fit, x0_fit, gamma_fit, C_fit = popt

    # ---- Step 3: profile likelihood intervals
    # Wilks threshold: Delta(NLL) = 0.5  <=>  Delta(-2 ln L) = 1.0 = chi^2_1(68.27%)
    DELTA_NLL_1SIGMA = 0.5

    def profile_interval(param_idx, param_best, param_bound_lo, param_bound_hi):
        """
        Scan parameter param_idx from its best-fit value toward each bound,
        optimising over all nuisance parameters at each step using L-BFGS-B
        with warm-starting from the previous step's solution.

        Returns (lo_err, hi_err) as positive quantities, or np.nan if the
        profile never crosses the threshold within the search range.
        """
        other_idx  = [i for i in range(4) if i != param_idx]
        bds_other  = [bounds[i] for i in other_idx]

        def profile_cost(val):
            """NLL minimised over nuisance params with param_idx fixed at val."""
            nonlocal p0_other
            fixed = popt.copy()
            fixed[param_idx] = val

            def cost(p_other):
                p = fixed.copy()
                for k, idx in enumerate(other_idx):
                    p[idx] = p_other[k]
                return neg_log_likelihood(p, x, y, n_avg)

            res = minimize(cost, p0_other, method='L-BFGS-B', bounds=bds_other,
                           options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 10000})
            if res.success or res.fun < neg_log_likelihood(
                    [popt[i] if i != param_idx else val for i in range(4)], x, y, n_avg):
                p0_other = res.x.tolist()   # warm-start for next step
            return res.fun

        # --- scan downward (lower bound)
        lo_err = np.nan
        p0_other = [popt[i] for i in other_idx]     # reset warm-start
        lo_vals  = np.linspace(param_best, param_bound_lo, n_scan)
        prev_dnll = 0.0
        for j, val in enumerate(lo_vals[1:], start=1):
            dnll = profile_cost(val) - nll_min
            if dnll >= DELTA_NLL_1SIGMA:
                # Linear interpolation between steps j-1 and j for precision
                val_prev = lo_vals[j - 1]
                dnll_prev = prev_dnll
                frac = (DELTA_NLL_1SIGMA - dnll_prev) / (dnll - dnll_prev)
                crossing = val_prev + frac * (val - val_prev)
                lo_err = param_best - crossing
                break
            prev_dnll = dnll

        # --- scan upward (upper bound)
        hi_err = np.nan
        p0_other = [popt[i] for i in other_idx]     # reset warm-start
        hi_vals  = np.linspace(param_best, param_bound_hi, n_scan)
        prev_dnll = 0.0
        for j, val in enumerate(hi_vals[1:], start=1):
            dnll = profile_cost(val) - nll_min
            if dnll >= DELTA_NLL_1SIGMA:
                val_prev = hi_vals[j - 1]
                dnll_prev = prev_dnll
                frac = (DELTA_NLL_1SIGMA - dnll_prev) / (dnll - dnll_prev)
                crossing = val_prev + frac * (val - val_prev)
                hi_err = crossing - param_best
                break
            prev_dnll = dnll

        return lo_err, hi_err

    lo_err = np.full(4, np.nan)
    hi_err = np.full(4, np.nan)
    for i, bnd in enumerate(bounds):
        lo_err[i], hi_err[i] = profile_interval(i, popt[i], bnd[0], bnd[1])

    # Spectral resolution flag: gamma must exceed one frequency bin
    resolved = gamma_fit > freq_res
    snr      = A_fit / C_fit

    print(f"  A     = {A_fit:.5f}  -{lo_err[0]:.5f} +{hi_err[0]:.5f}  [m²/s²/nHz]")
    print(f"  x0    = {x0_fit:.3f}   -{lo_err[1]:.3f} +{hi_err[1]:.3f}  [nHz]")
    print(f"  gamma = {gamma_fit:.3f}   -{lo_err[2]:.3f} +{hi_err[2]:.3f}  [nHz]  (HWHM)")
    print(f"  C     = {C_fit:.5f}  -{lo_err[3]:.5f} +{hi_err[3]:.5f}  [m²/s²/nHz]")
    print(f"  SNR   = {snr:.2f}  (A/C, n_avg={n_avg})")
    if not resolved:
        print(f"  *** WARNING: unresolved — gamma={gamma_fit:.4f} < freq_res={freq_res:.4f} nHz ***")

    return popt, lo_err, hi_err, resolved


# %% Run all fits
n_avg_hl     = int(lat_eq_hl.sum())
n_avg_cl     = int(lat_eq_cl.sum())
n_avg_rossby = int(lat_eq_rossby.sum())

fits = {}

fits['lct_mag_m1'],  fits['lct_mag_m1_lo'],  fits['lct_mag_m1_hi'],  _ = fit_lorentzian_mle(freqs[freq_window_uphi_m1],     power_m1_plot_lct_mag,  n_avg_hl,     'LCTMag  m=1 uphi')
fits['lct_gran_m1'], fits['lct_gran_m1_lo'], fits['lct_gran_m1_hi'], _ = fit_lorentzian_mle(freqs[freq_window_uphi_m1],     power_m1_plot_lct_gran, n_avg_hl,     'LCTGran m=1 uphi')
fits['rda_m1'],      fits['rda_m1_lo'],      fits['rda_m1_hi'],      _ = fit_lorentzian_mle(freqs_rda[freq_window_rda_m1],  power_m1_plot_rda,      n_avg_hl,     'RDA     m=1 uphi')

fits['lct_mag_m2'],  fits['lct_mag_m2_lo'],  fits['lct_mag_m2_hi'],  _ = fit_lorentzian_mle(freqs[freq_window_uphi_m2],     power_m2_plot_lct_mag,  n_avg_cl,     'LCTMag  m=2 uphi')
fits['lct_gran_m2'], fits['lct_gran_m2_lo'], fits['lct_gran_m2_hi'], _ = fit_lorentzian_mle(freqs[freq_window_uphi_m2],     power_m2_plot_lct_gran, n_avg_cl,     'LCTGran m=2 uphi')
fits['rda_m2'],      fits['rda_m2_lo'],      fits['rda_m2_hi'],      _ = fit_lorentzian_mle(freqs_rda[freq_window_rda_m2],  power_m2_plot_rda,      n_avg_cl,     'RDA     m=2 uphi')

fits['lct_mag_m8'],  fits['lct_mag_m8_lo'],  fits['lct_mag_m8_hi'],  _ = fit_lorentzian_mle(freqs[freq_window_uthe_m8],     power_m8_plot_lct_mag,  n_avg_rossby, 'LCTMag  m=8 uthe')
fits['lct_gran_m8'], fits['lct_gran_m8_lo'], fits['lct_gran_m8_hi'], _ = fit_lorentzian_mle(freqs[freq_window_uthe_m8],     power_m8_plot_lct_gran, n_avg_rossby, 'LCTGran m=8 uthe')
fits['rda_m8'],      fits['rda_m8_lo'],      fits['rda_m8_hi'],      _ = fit_lorentzian_mle(freqs_rda[freq_window_rda_m8],  power_m8_plot_rda,      n_avg_rossby, 'RDA     m=8 uthe')

fits['lct_mag_m13'],  fits['lct_mag_m13_lo'],  fits['lct_mag_m13_hi'],  fits['res_lct_mag_m13']  = fit_lorentzian_mle(freqs[freq_window_uthe_m13],    power_m13_plot_lct_mag,  n_avg_rossby, 'LCTMag  m=13 uthe')
fits['lct_gran_m13'], fits['lct_gran_m13_lo'], fits['lct_gran_m13_hi'], fits['res_lct_gran_m13'] = fit_lorentzian_mle(freqs[freq_window_uthe_m13],    power_m13_plot_lct_gran, n_avg_rossby, 'LCTGran m=13 uthe')
fits['rda_m13'],      fits['rda_m13_lo'],      fits['rda_m13_hi'],      fits['res_rda_m13']      = fit_lorentzian_mle(freqs_rda[freq_window_rda_m13], power_m13_plot_rda,      n_avg_rossby, 'RDA     m=13 uthe')

# %% Summary table
print("\n====  Peak frequencies x0 (nHz)  ====")
print(f"{'Mode':<12}  {'Method':<10}  {'x0':>10}  {'-err':>8}  {'+err':>8}")
entries = [
    ('m=1 HL',   'LCTMag',  fits['lct_mag_m1'],   fits['lct_mag_m1_lo'],   fits['lct_mag_m1_hi']),
    ('m=1 HL',   'LCTGran', fits['lct_gran_m1'],  fits['lct_gran_m1_lo'],  fits['lct_gran_m1_hi']),
    ('m=1 HL',   'RDA',     fits['rda_m1'],       fits['rda_m1_lo'],       fits['rda_m1_hi']),
    ('m=2 CL',   'LCTMag',  fits['lct_mag_m2'],   fits['lct_mag_m2_lo'],   fits['lct_mag_m2_hi']),
    ('m=2 CL',   'LCTGran', fits['lct_gran_m2'],  fits['lct_gran_m2_lo'],  fits['lct_gran_m2_hi']),
    ('m=2 CL',   'RDA',     fits['rda_m2'],       fits['rda_m2_lo'],       fits['rda_m2_hi']),
    ('m=8 Ros',  'LCTMag',  fits['lct_mag_m8'],   fits['lct_mag_m8_lo'],   fits['lct_mag_m8_hi']),
    ('m=8 Ros',  'LCTGran', fits['lct_gran_m8'],  fits['lct_gran_m8_lo'],  fits['lct_gran_m8_hi']),
    ('m=8 Ros',  'RDA',     fits['rda_m8'],       fits['rda_m8_lo'],       fits['rda_m8_hi']),
    ('m=13 HFR', 'LCTMag',  fits['lct_mag_m13'],  fits['lct_mag_m13_lo'],  fits['lct_mag_m13_hi']),
    ('m=13 HFR', 'LCTGran', fits['lct_gran_m13'], fits['lct_gran_m13_lo'], fits['lct_gran_m13_hi']),
    ('m=13 HFR', 'RDA',     fits['rda_m13'],      fits['rda_m13_lo'],      fits['rda_m13_hi']),
]
for mode, method, popt, lo, hi in entries:
    print(f"{mode:<12}  {method:<10}  {popt[1]:>10.3f}  {lo[1]:>8.3f}  {hi[1]:>8.3f}")

# %% Plotting
freqs_lorentzian_m1  = np.linspace(-200,   0, 500)
freqs_lorentzian_m2  = np.linspace(-110, -50, 500)
freqs_lorentzian_m8  = np.linspace(-250,  -50, 500)
freqs_lorentzian_m13 = np.linspace(-260, -170, 500)

fig = plt.figure(figsize=(10, 8))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

def plot_panel(ax, xdata_list, ydata_list, popt_list, lo_list, hi_list,
               freqs_lor, colors, labels, linestyles, title,
               xlabel=None, ylabel=None, xlim=None, ylim=None):
    for xd, yd, col, lab, ls in zip(xdata_list, ydata_list, colors, labels, linestyles):
        ax.plot(xd, yd, color=col, lw=2, label=lab, ls=ls)
    for popt, lo, hi, col in zip(popt_list, lo_list, hi_list, colors):
        ax.plot(freqs_lor, lorentzian(freqs_lor, *popt), color=col, lw=1.5, ls=':')
        ax.axvline(popt[1], color=col, lw=0.8, ls='--', alpha=0.5)
        ax.axvspan(popt[1] - lo[1], popt[1] + hi[1], alpha=0.08, color=col)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(alpha=0.4)
    ax.legend(loc='upper right', fontsize=9)

colors  = ['cyan', 'blue', 'green']
ls_data = ['-', '-', '--']

ax1 = fig.add_subplot(gs[0, 0])
plot_panel(ax1,
    [freqs, freqs, freqs_rda],
    [power_uphi_m1_gran[:, 1], power_uphi_m1_mag[:, 1], power_uphi_m1_rda[:, 1]],
    [fits['lct_gran_m1'], fits['lct_mag_m1'], fits['rda_m1']],
    [fits['lct_gran_m1_lo'], fits['lct_mag_m1_lo'], fits['rda_m1_lo']],
    [fits['lct_gran_m1_hi'], fits['lct_mag_m1_hi'], fits['rda_m1_hi']],
    freqs_lorentzian_m1, colors,
    [r'LCTGran $u_\phi^-$', r'LCTMag $u_\phi^-$', r'RDA $u_\phi^-$'], ls_data,
    title='m=1  High Latitude', ylabel=r'Power [$\rm m^2\,s^{-2}\,nHz^{-1}$]',
    xlim=(-200, 0),ylim=(0, 7))

ax2 = fig.add_subplot(gs[0, 1])
plot_panel(ax2,
    [freqs, freqs, freqs_rda],
    [power_uphi_m2_gran[:, 2], power_uphi_m2_mag[:, 2], power_uphi_m2_rda[:, 2]],
    [fits['lct_gran_m2'], fits['lct_mag_m2'], fits['rda_m2']],
    [fits['lct_gran_m2_lo'], fits['lct_mag_m2_lo'], fits['rda_m2_lo']],
    [fits['lct_gran_m2_hi'], fits['lct_mag_m2_hi'], fits['rda_m2_hi']],
    freqs_lorentzian_m2, colors,
    [r'LCTGran $u_\phi^-$', r'LCTMag $u_\phi^-$', r'RDA $u_\phi^-$'], ls_data,
    title='m=2  Critical Latitude', ylabel=r'Power [$\rm m^2\,s^{-2}\,nHz^{-1}$]',
    xlim=(-200, 50), ylim=(0, 0.10))

ax3 = fig.add_subplot(gs[1, 0])
plot_panel(ax3,
    [freqs, freqs, freqs_rda],
    [power_uthe_m8_gran[:, 8], power_uthe_m8_mag[:, 8], power_uthe_m8_rda[:, 8]],
    [fits['lct_gran_m8'], fits['lct_mag_m8'], fits['rda_m8']],
    [fits['lct_gran_m8_lo'], fits['lct_mag_m8_lo'], fits['rda_m8_lo']],
    [fits['lct_gran_m8_hi'], fits['lct_mag_m8_hi'], fits['rda_m8_hi']],
    freqs_lorentzian_m8, colors,
    [r'LCTGran $u_\theta^+$', r'LCTMag $u_\theta^+$', r'RDA $u_\theta^+$'], ls_data,
    title='m=8  Equatorial Rossby',
    xlabel='Frequency (nHz)', ylabel=r'Power [$\rm m^2\,s^{-2}\,nHz^{-1}$]',
    xlim=(-200, 0))

ax4 = fig.add_subplot(gs[1, 1])
plot_panel(ax4,
    [freqs, freqs, freqs_rda],
    [power_uthe_m13_gran[:, 13], power_uthe_m13_mag[:, 13], power_uthe_m13_rda[:, 13]],
    [fits['lct_gran_m13'], fits['lct_mag_m13'], fits['rda_m13']],
    [fits['lct_gran_m13_lo'], fits['lct_mag_m13_lo'], fits['rda_m13_lo']],
    [fits['lct_gran_m13_hi'], fits['lct_mag_m13_hi'], fits['rda_m13_hi']],
    freqs_lorentzian_m13, colors,
    [r'LCTGran $u_\theta^-$', r'LCTMag $u_\theta^-$', r'RDA $u_\theta^-$'], ls_data,
    title='m=13  HFR',
    xlabel='Frequency (nHz)', ylabel=r'Power [$\rm m^2\,s^{-2}\,nHz^{-1}$]',
    xlim=(-260, -170))

plt.tight_layout()
# fig.savefig('/data/seismo/joshin/pipeline-test/local_correlation_tracking/pdfs/figure_lct_rda_compare_ps.pdf',
#             dpi=300, bbox_inches='tight')
plt.show()
# %%
