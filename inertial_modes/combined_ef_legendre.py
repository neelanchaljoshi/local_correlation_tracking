"""
extract_eigenfunction.py
========================
End-to-end pipeline for a single solar inertial mode:
  1. Load LCT flow data
  2. Clip / apodize the disk
  3. Transform to (m, frequency) space
  4. Extract the eigenfunction via SVD
  5. Project onto Legendre polynomials and remove noise modes
  6. Estimate errors via Monte Carlo over discarded coefficients
  7. Save the cleaned eigenfunction + errors to an .npz file

Usage
-----
    python combined_ef_legendre.py <m> <cent_freq> <mode> <data> <symmetry> \
        [--l_max L_MAX] [--l_cutoff L_CUTOFF] [--noise_factor NF] \
        [--mc_samples N] [--error_method METHOD] \
        [--span_lower YEAR] [--span_upper YEAR] \
        [--reject_type {clip,noclip}]

Arguments
---------
  m             Azimuthal order (integer)
  cent_freq     Central frequency in nHz (float)
  mode          Mode label string, e.g. 'highlat', 'critlat', 'rossby', 'hfr'
  data          Data product name, e.g. 'hmi.m_720s_dt_1h'
  symmetry      'sym', 'anti', or 'all'

Optional
--------
  --l_max           Maximum ell kept in reconstruction  (default: 22)
  --l_cutoff        Theory cutoff ell                   (default: 15)
  --noise_factor    Noise floor multiplier              (default: 3)
  --mc_samples      Monte Carlo samples for errors      (default: 500)
  --error_method    'monte_carlo' | 'fl_sum' | 'monte_carlo_amp'
                                                        (default: monte_carlo)
  --span_lower      Start year (inclusive)              (default: 2010)
  --span_upper      End year   (exclusive)              (default: 2025)
  --reject_type     'clip' or 'noclip'                  (default: clip)

Output
------
  <ef_path>/eigenfunction_clean_m<m>_<cent_freq>_<mode>_<symmetry>_<data>.npz

  Keys in the output file:
    ef_uphi       – complex 1-D eigenfunction u_phi  (shape: nlat)
    ef_uthe       – complex 1-D eigenfunction u_theta (shape: nlat)
    ef_uphi_sm    – full Legendre-smoothed u_phi
    ef_uthe_sm    – full Legendre-smoothed u_theta
    uphi_err_real – real-part 1-sigma error on u_phi
    uphi_err_imag – imag-part 1-sigma error on u_phi
    uthe_err_real – real-part 1-sigma error on u_theta
    uthe_err_imag – imag-part 1-sigma error on u_theta
    lats          – latitude array (degrees)
    final_td      – time-dependence amplitude array
"""

# ── Standard library ───────────────────────────────────────────────────────
import argparse
import pathlib
import sys
from datetime import datetime

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from numpy import linalg
from scipy import integrate
from scipy.special import legendre
from scipy.stats import chi2
from tqdm import tqdm

# ── MPS-internal ───────────────────────────────────────────────────────────
sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
from zclpy3.remap import get_tan_from_lnglat  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change paths here if the data layout changes
# ══════════════════════════════════════════════════════════════════════════
DATA_ROOT   = '/data/seismo/joshin/pipeline-test/local_correlation_tracking/data'
EF_OUT_PATH = pathlib.Path(DATA_ROOT) / 'eigenfunctions'
EF_OUT_PATH.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional
    p.add_argument('m',          type=int,   help='Azimuthal order')
    p.add_argument('cent_freq',  type=float, help='Central frequency [nHz]')
    p.add_argument('mode',       type=str,   help='Mode label (highlat / rossby / …)')
    p.add_argument('data',       type=str,   help='Data product name')
    p.add_argument('symmetry',   type=str,   choices=['sym', 'anti', 'all'],
                   help='Equatorial symmetry of u_phi')

    # Optional
    p.add_argument('--l_max',        type=int,   default=22)
    p.add_argument('--l_cutoff',     type=int,   default=15,
                   help='Theory cutoff ell — modes up to this are always kept')
    p.add_argument('--noise_factor', type=float, default=3.0)
    p.add_argument('--mc_samples',   type=int,   default=500)
    p.add_argument('--error_method', type=str,   default='monte_carlo',
                   choices=['monte_carlo', 'fl_sum', 'monte_carlo_amp'])
    p.add_argument('--span_lower',   type=int,   default=2010)
    p.add_argument('--span_upper',   type=int,   default=2025)
    p.add_argument('--reject_type',  type=str,   default='clip',
                   choices=['clip', 'noclip'])

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — disk geometry
# ══════════════════════════════════════════════════════════════════════════
def year_fraction(date: datetime) -> float:
    start       = datetime(date.year, 1, 1).toordinal()
    year_length = datetime(date.year + 1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


def build_radius_array(crlt_obs, rsun_obs, lon_og, lat_og):
    """Compute the projected disk radius at every (time, lat, lon) pixel."""
    nt   = len(crlt_obs)
    nlat = len(lat_og)
    nlng = len(lon_og)
    r    = np.zeros((nt, nlat, nlng))
    for i, b_angle in tqdm(enumerate(np.nan_to_num(crlt_obs)),
                           total=nt, desc='Building radius array'):
        lng_, lat_ = np.meshgrid(lon_og, lat_og)
        xdisk, ydisk = get_tan_from_lnglat(
            lng_.flatten(), lat_.flatten(), rsun_obs[i], b_angle, dP=0)
        r[i] = np.hypot(xdisk.reshape(nlat, nlng),
                        ydisk.reshape(nlat, nlng))
    return r


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — disk masking
# ══════════════════════════════════════════════════════════════════════════
def clip_flow_data(arr, radius_arr, radius_ratio, rsun_obs, pad=True):
    """Set pixels beyond radius_ratio * R_sun to NaN and optionally pad."""
    arr = arr.copy()
    clipradius = radius_ratio * rsun_obs
    arr[~(radius_arr < clipradius[:, None, None])] = np.nan
    if pad:
        arr = np.pad(arr, [(0, 0), (0, 0), (36, 35)],
                     mode='constant', constant_values=np.nan)
    return arr


def apodize_flow_data(arr, radius_arr, r_min, r_max, r_sun):
    """Cosine apodization between r_min and r_max (in units of R_sun)."""
    arr    = arr.copy()
    r_frac = np.clip(radius_arr / r_sun[:, None, None], 0, 1.0)
    apod   = np.zeros_like(r_frac)
    apod[r_frac < r_min] = 1.0
    span   = (r_frac >= r_min) & (r_frac < r_max)
    apod[span] = 0.5 * (1 + np.cos(
        np.pi * (r_frac[span] - r_min) / (r_max - r_min)))
    arr   *= apod
    arr    = np.pad(arr, [(0, 0), (0, 0), (36, 35)],
                    mode='constant', constant_values=0)
    return arr


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — amplitude correction factors
# ══════════════════════════════════════════════════════════════════════════
def get_correction_factor(arr, nlng_carr):
    win    = np.isfinite(arr).astype(int)
    nlon_p = np.sum(np.nan_to_num(win), axis=2)[:, :, None]
    nt_p   = np.sum(nlon_p > 0, axis=0)[None, :]
    cft    = win.shape[0] / nt_p
    cfl    = np.nan_to_num(nlng_carr / nlon_p)
    cft[cft > 1e200] = np.inf
    cfl[cfl > 1e200] = np.inf
    return cft, cfl


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — Fourier / Carrington transform
# ══════════════════════════════════════════════════════════════════════════
def tukeywin(N, alpha=0.0):
    if alpha <= 0:
        return np.ones(N)
    if alpha >= 1:
        return np.hanning(N)
    x = np.linspace(0, 1, N)
    w = np.ones(N)
    w[x < alpha / 2] = 0.5 * (
        1 + np.cos(2 * np.pi / alpha * (x[x < alpha / 2] - alpha / 2)))
    w[x >= 1 - alpha / 2] = 0.5 * (
        1 + np.cos(2 * np.pi / alpha * (x[x >= 1 - alpha / 2] - 1 + alpha / 2)))
    return w


def transform_fourier(arr, crln, cft, cfl, span, dt=6.0 * 3600):
    """FFT in longitude → Carrington frame → FFT in time. Returns (ft, freqs_nHz)."""
    print('Transforming to Fourier space...')
    data        = np.nan_to_num(arr[span])
    uphi_fft_m  = np.fft.rfft(data, axis=2) * np.nan_to_num(cfl[span])
    M_arr       = np.arange(uphi_fft_m.shape[2])
    carr_conv   = np.exp(-1j * np.deg2rad(crln[span])[:, None] * M_arr[None, :])[:, None, :]
    uphi_m_carr = uphi_fft_m * carr_conv
    uphi_f      = np.fft.fft(uphi_m_carr, axis=0) * np.sqrt(np.nan_to_num(cft))
    freq        = -np.fft.fftfreq(uphi_f.shape[0], dt) * 1e9   # nHz
    freq_ffts   = np.fft.fftshift(freq)
    uphi_ft     = np.fft.fftshift(uphi_f, axes=0)
    return uphi_ft, freq_ffts


def filter_in_freq(uphi_ft_m, uthe_ft_m, freq_ffts, cent_freq, df=20):
    """Bandpass filter around cent_freq ± df nHz with a Tukey window."""
    s      = (freq_ffts > cent_freq - df) & (freq_ffts < cent_freq + df)
    wl     = s.sum()
    window = np.zeros_like(freq_ffts)
    window[s] = tukeywin(wl, 0.1)
    uphi_filt = uphi_ft_m * window[:, np.newaxis]
    uthe_filt = uthe_ft_m * window[:, np.newaxis]
    return uphi_filt, uthe_filt


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — eigenfunction extraction (SVD)
# ══════════════════════════════════════════════════════════════════════════
def extract_eigenfunction_lats(uphi_ft, uthe_ft, m, cent_freq, freq_ffts,
                               lat_for_scaling=0, nlng=144, df=10):
    """
    Extract the dominant eigenfunction at azimuthal order m via SVD.

    Returns
    -------
    ef_uphi, ef_uthe : complex 1-D arrays (nlat,)
    final_td         : float 1-D array — time-dependence amplitude
    """
    uphi_f_m = uphi_ft[:, :, m]
    uthe_f_m = uthe_ft[:, :, m]
    uphi_f_m_filt, uthe_f_m_filt = filter_in_freq(
        uphi_f_m, uthe_f_m, freq_ffts, cent_freq, df=df)

    nt, nlat = uphi_ft.shape[0], uphi_ft.shape[1]
    uphi_filt = np.zeros((nt, nlat, nlng), dtype=np.complex128)
    uthe_filt = np.zeros((nt, nlat, nlng), dtype=np.complex128)
    uphi_filt[:, :, m] = uphi_f_m_filt
    uthe_filt[:, :, m] = uthe_f_m_filt

    lats      = np.linspace(-90, 90, nlat)
    lat_mask  = np.where(np.abs(lats) <= 75)[0]

    uphi_t_m  = np.fft.ifft(np.fft.ifftshift(uphi_filt, axes=0), axis=0)
    uthe_t_m  = np.fft.ifft(np.fft.ifftshift(uthe_filt, axes=0), axis=0)

    arr_svd   = np.concatenate(
        (uphi_t_m[:, lat_mask, m], uthe_t_m[:, lat_mask, m]), axis=1)
    U, s, Vh  = linalg.svd(arr_svd, full_matrices=False)

    time_dep  = U[:, 0]
    m_factor  = 2 / nlng / np.sqrt(np.mean(np.abs(time_dep) ** 2))

    ef_uphi   = np.mean(uphi_t_m[:, :, m] * np.conj(time_dep[:, None]) * m_factor, axis=0)
    ef_uthe   = np.mean(uthe_t_m[:, :, m] * np.conj(time_dep[:, None]) * m_factor, axis=0)

    index    = np.where(lats == lat_for_scaling)[0][0]
    final_td = np.abs(s[0] * Vh[0, index] * np.abs(time_dep) * 2 / nlng)
    return ef_uphi, ef_uthe, final_td


# ══════════════════════════════════════════════════════════════════════════
# HELPERS — Legendre projection and noise filtering
# ══════════════════════════════════════════════════════════════════════════
def confidence_level_to_power(conf, B, M, dof):
    cdf    = conf ** (1 / M)
    z_conf = chi2.ppf(cdf, df=dof)
    return z_conf * B / dof


def monte_carlo_errorbars(theta, l_discard, fl_discard, num_samples=500):
    """Phase-randomised MC error on discarded Legendre coefficients."""
    if len(l_discard) == 0:
        return np.zeros(len(theta)), np.zeros(len(theta))
    P = np.array([np.sqrt((2 * l + 1) / 2) * legendre(l)(np.cos(theta))
                  for l in l_discard])
    samples_real, samples_imag = [], []
    for _ in range(num_samples):
        phases   = np.exp(1j * 2 * np.pi * np.random.rand(len(fl_discard)))
        f_random = np.abs(fl_discard) * phases
        u_sample = np.sum(f_random[:, None] * P, axis=0)
        samples_real.append(u_sample.real)
        samples_imag.append(u_sample.imag)
    return np.std(samples_real, axis=0), np.std(samples_imag, axis=0)


def monte_carlo_legendre_error_random_amp_phase(theta_array, discarded_fl,
                                                l_array, num_trials=500):
    """Amplitude- and phase-randomised MC error."""
    if len(discarded_fl) == 0:
        return np.zeros(len(theta_array)), np.zeros(len(theta_array))
    n_theta = len(theta_array)
    trials  = np.zeros((num_trials, n_theta), dtype=np.complex128)
    for i in range(num_trials):
        u_recon = np.zeros(n_theta, dtype=np.complex128)
        for fl, l in zip(discarded_fl, l_array):
            amp_sample = np.random.normal(np.abs(fl), 0.2 * np.abs(fl))
            phase      = np.random.uniform(0, 2 * np.pi)
            norm       = np.sqrt((2 * l + 1) / 2)
            u_recon   += amp_sample * np.exp(1j * phase) * norm * legendre(l)(np.cos(theta_array))
        trials[i] = u_recon
    return np.std(trials.real, axis=0), np.std(trials.imag, axis=0)


def fl_errorbars(theta, l_discard, fl_discard):
    """Deterministic error from summing discarded modes."""
    if len(l_discard) == 0:
        return np.zeros(len(theta)), np.zeros(len(theta))
    P       = np.array([np.sqrt((2 * l + 1) / 2) * legendre(l)(np.cos(theta))
                        for l in l_discard])
    u_total = np.sum([f * p for f, p in zip(fl_discard, P)], axis=0)
    return np.abs(u_total), np.abs(u_total)


def project_onto_legendre(m, ef_uphi, ef_uthe, lats,
                           symmetryuphi='anti', l_max=22,
                           l_theory_cutoff=15, noise_factor=3,
                           num_mc_samples=500, error_method='monte_carlo'):
    """
    Project the raw eigenfunction onto Legendre polynomials, discard noise
    modes above l_theory_cutoff, reconstruct the cleaned eigenfunction, and
    estimate errors from the discarded coefficients.

    Returns
    -------
    uphi_sm, uthe_sm         : full Legendre-smoothed eigenfunctions
    uphi_recon, uthe_recon   : noise-filtered reconstructions
    uphi_err_real, uphi_err_imag,
    uthe_err_real, uthe_err_imag : 1-sigma errors (real and imaginary parts)
    """
    theta     = np.deg2rad(90 - lats)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    l_array   = np.arange(36)

    # ── Project onto Legendre basis ─────────────────────────────────────
    fl_uphi = np.zeros(len(l_array), dtype=np.complex128)
    fl_uthe = np.zeros(len(l_array), dtype=np.complex128)
    for l in l_array:
        norm      = np.sqrt((2 * l + 1) / 2)
        P_l       = legendre(l)(cos_theta)
        fl_uphi[l] = integrate.simpson(ef_uphi * P_l * sin_theta * norm, x=theta)
        fl_uthe[l] = integrate.simpson(ef_uthe * P_l * sin_theta * norm, x=theta)

    # ── Enforce symmetry — zero out wrong-parity modes ──────────────────
    if symmetryuphi == 'anti':
        fl_uphi[0::2] = 0   # u_phi: odd ell only
        fl_uthe[1::2] = 0   # u_theta: even ell only
        l_uphi        = l_array[1::2]
        l_uthe        = l_array[0::2]
        fl_uphi_sym   = fl_uphi[1::2]
        fl_uthe_sym   = fl_uthe[0::2]
    elif symmetryuphi == 'sym':
        fl_uphi[1::2] = 0
        fl_uthe[0::2] = 0
        l_uphi        = l_array[0::2]
        l_uthe        = l_array[1::2]
        fl_uphi_sym   = fl_uphi[0::2]
        fl_uthe_sym   = fl_uthe[1::2]
    else:   # 'all' — no symmetry enforcement
        l_uphi      = l_array
        l_uthe      = l_array
        fl_uphi_sym = fl_uphi
        fl_uthe_sym = fl_uthe

    # ── Full Legendre-smoothed eigenfunction ────────────────────────────
    uphi_sm = sum(fl_uphi[l] * np.sqrt((2*l+1)/2) * legendre(l)(cos_theta)
                  for l in l_array)
    uthe_sm = sum(fl_uthe[l] * np.sqrt((2*l+1)/2) * legendre(l)(cos_theta)
                  for l in l_array)

    # ── Noise-threshold filter above l_theory_cutoff ───────────────────
    def filter_l(power, l_vals):
        noise_floor      = np.median(power[int(l_theory_cutoff // 2):])
        len_bins         = len(power[int(l_theory_cutoff // 2):])
        confidence_power = confidence_level_to_power(0.9, noise_floor, len_bins, 2)
        cumulative       = np.cumsum(power) / np.sum(power)
        idx_99           = np.argmax(cumulative >= 0.99)
        mask_low         = l_vals <= l_theory_cutoff
        mask_high        = (l_vals > l_theory_cutoff) & \
                           (np.arange(len(power)) <= idx_99) & \
                           (power > confidence_power)
        return mask_low | mask_high, confidence_power

    keep_uphi, _ = filter_l(np.abs(fl_uphi_sym) ** 2, l_uphi)
    keep_uthe, _ = filter_l(np.abs(fl_uthe_sym) ** 2, l_uthe)

    l_uphi_keep = l_uphi[keep_uphi]
    l_uthe_keep = l_uthe[keep_uthe]
    print(f'  u_phi:  keeping ℓ = {l_uphi_keep}')
    print(f'  u_theta: keeping ℓ = {l_uthe_keep}')

    # ── Reconstruct from kept modes only ───────────────────────────────
    uphi_recon = np.zeros_like(theta, dtype=np.complex128)
    uthe_recon = np.zeros_like(theta, dtype=np.complex128)
    for l in l_array:
        if l >= l_max:
            continue
        norm = np.sqrt((2 * l + 1) / 2)
        P_l  = legendre(l)(cos_theta)
        if l in l_uphi_keep:
            uphi_recon += fl_uphi[l] * norm * P_l
        if l in l_uthe_keep:
            uthe_recon += fl_uthe[l] * norm * P_l

    # ── Discarded coefficients → error estimate ─────────────────────────
    l_discard_uphi  = [l for l in l_uphi if l not in l_uphi_keep]
    l_discard_uthe  = [l for l in l_uthe if l not in l_uthe_keep]
    fl_discard_uphi = np.array([fl_uphi[l] for l in l_discard_uphi])
    fl_discard_uthe = np.array([fl_uthe[l] for l in l_discard_uthe])

    if error_method == 'monte_carlo':
        uphi_err_real, uphi_err_imag = monte_carlo_errorbars(
            theta, l_discard_uphi, fl_discard_uphi, num_samples=num_mc_samples)
        uthe_err_real, uthe_err_imag = monte_carlo_errorbars(
            theta, l_discard_uthe, fl_discard_uthe, num_samples=num_mc_samples)
    elif error_method == 'fl_sum':
        uphi_err_real, uphi_err_imag = fl_errorbars(theta, l_discard_uphi, fl_discard_uphi)
        uthe_err_real, uthe_err_imag = fl_errorbars(theta, l_discard_uthe, fl_discard_uthe)
    elif error_method == 'monte_carlo_amp':
        uphi_err_real, uphi_err_imag = monte_carlo_legendre_error_random_amp_phase(
            theta, fl_discard_uphi, l_discard_uphi, num_trials=num_mc_samples)
        uthe_err_real, uthe_err_imag = monte_carlo_legendre_error_random_amp_phase(
            theta, fl_discard_uthe, l_discard_uthe, num_trials=num_mc_samples)
    else:
        raise ValueError(f"Unknown error_method: {error_method!r}")

    # ── Phase alignment: make u_theta real at the equator ───────────────
    equator_idx = np.argmin(np.abs(np.rad2deg(theta) - 90))
    phase       = -1j * np.angle(uthe_recon[equator_idx])
    uphi_recon *= np.exp(phase)
    uthe_recon *= np.exp(phase)
    uphi_sm    *= np.exp(phase)
    uthe_sm    *= np.exp(phase)

    return (uphi_sm, uthe_sm,
            uphi_recon, uthe_recon,
            uphi_err_real, uphi_err_imag,
            uthe_err_real, uthe_err_imag)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    m           = args.m
    cent_freq   = args.cent_freq
    mode        = args.mode
    data        = args.data
    symmetry    = args.symmetry
    data_name   = data.replace('.', '_')

    # Derived symmetry labels
    if symmetry == 'sym':
        sym_uphi, sym_uthe = 'sym', 'anti'
    elif symmetry == 'anti':
        sym_uphi, sym_uthe = 'anti', 'sym'
    else:
        sym_uphi = sym_uthe = 'all'

    print(f'\n{"="*60}')
    print(f'  Extracting eigenfunction: m={m}, freq={cent_freq} nHz')
    print(f'  Mode: {mode}  |  Data: {data}  |  Symmetry: {symmetry}')
    print(f'{"="*60}\n')

    # ── Load flow data ──────────────────────────────────────────────────
    print('Loading flow data...')
    proc = pathlib.Path(DATA_ROOT) / 'processed_data'
    uphi_all  = np.load(proc / f'uphi_{data_name}_processed.npy')
    uthe_all  = np.load(proc / f'utheta_{data_name}_processed.npy')
    t_raw     = np.load(pathlib.Path(DATA_ROOT) / 't_rec.npy')
    crln_obs  = np.load(pathlib.Path(DATA_ROOT) / 'crln_obs.npy')
    crlt_obs  = np.load(pathlib.Path(DATA_ROOT) / 'crlt_obs.npy')
    rsun_obs  = np.load(pathlib.Path(DATA_ROOT) / 'rsun_obs.npy')

    # ── Time array ──────────────────────────────────────────────────────
    t_array_dt = [datetime.strptime(str(t, 'utf-8'), '%Y.%m.%d_%H:%M:%S_TAI')
                  for t in t_raw]
    t_array    = np.array([year_fraction(d) for d in t_array_dt])

    # ── Interpolate missing metadata ────────────────────────────────────
    df_meta  = pd.DataFrame({'t': t_array, 'crln': crln_obs,
                              'crlt': crlt_obs, 'rsun': rsun_obs})
    df_meta.interpolate(method='linear', inplace=True)
    rsun_obs = df_meta['rsun'].values

    # ── Fill Carrington longitude gaps ──────────────────────────────────
    crln   = crln_obs.copy()
    dcrln  = crln[1:] - crln[:-1]
    dphi   = np.nanmean(dcrln[dcrln < 0.])
    nan_pos = np.where(np.isnan(crln))[0]
    while len(nan_pos) > 0:
        for j in nan_pos:
            crln[j] = crln[j - 1] + dphi
            crln[j] += 360 if crln[j] < 0. else 0
        nan_pos = np.where(np.isnan(crln))[0].tolist()

    # ── Grid ────────────────────────────────────────────────────────────
    lon_og      = np.linspace(-90, 90, 73)
    lat_og      = np.linspace(-90, 90, 73)
    nt, nlat, nlng_stony = uthe_all.shape
    nlng_carr   = 2 * (nlng_stony - 1)
    dt_sec      = 6 * 3600

    # ── Apply equatorial symmetry ────────────────────────────────────────
    if sym_uphi == 'sym':
        uphi = (uphi_all + uphi_all[:, ::-1, :]) / 2
    elif sym_uphi == 'anti':
        uphi = (uphi_all - uphi_all[:, ::-1, :]) / 2
    else:
        uphi = uphi_all

    if sym_uthe == 'sym':
        uthe = (uthe_all + uthe_all[:, ::-1, :]) / 2
    elif sym_uthe == 'anti':
        uthe = (uthe_all - uthe_all[:, ::-1, :]) / 2
    else:
        uthe = uthe_all

    # ── Build radius array ──────────────────────────────────────────────
    r = build_radius_array(crlt_obs, rsun_obs, lon_og, lat_og)

    # ── Disk masking ────────────────────────────────────────────────────
    if args.reject_type == 'clip':
        uphi = clip_flow_data(uphi, r, 0.99, rsun_obs, pad=True)
        uthe = clip_flow_data(uthe, r, 0.99, rsun_obs, pad=True)
    else:
        uphi = apodize_flow_data(uphi, r, 0.96, 0.99, rsun_obs)
        uthe = apodize_flow_data(uthe, r, 0.96, 0.99, rsun_obs)

    # ── Amplitude correction ─────────────────────────────────────────────
    cft_uphi, cfl_uphi = get_correction_factor(uphi, nlng_carr)
    cft_uthe, cfl_uthe = get_correction_factor(uthe, nlng_carr)

    # ── Time span mask ───────────────────────────────────────────────────
    span = (t_array >= args.span_lower) & (t_array < args.span_upper)

    # ── Fourier transform ────────────────────────────────────────────────
    uphi_ft, freq_ffts = transform_fourier(uphi, crln, cft_uphi, cfl_uphi,
                                           span, dt=dt_sec)
    uthe_ft, _         = transform_fourier(uthe, crln, cft_uthe, cfl_uthe,
                                           span, dt=dt_sec)

    # ── Extract eigenfunction via SVD ────────────────────────────────────
    print(f'\nExtracting eigenfunction (SVD)...')
    ef_uphi, ef_uthe, final_td = extract_eigenfunction_lats(
        uphi_ft, uthe_ft, m, cent_freq, freq_ffts,
        lat_for_scaling=0, nlng=nlng_carr, df=10)

    # ── Legendre projection + noise filtering + error estimation ─────────
    print('\nProjecting onto Legendre polynomials...')
    lats = np.linspace(-90, 90, nlat)
    (uphi_sm, uthe_sm,
     uphi_recon, uthe_recon,
     uphi_err_real, uphi_err_imag,
     uthe_err_real, uthe_err_imag) = project_onto_legendre(
        m, ef_uphi, ef_uthe, lats,
        symmetryuphi=sym_uphi,
        l_max=args.l_max,
        l_theory_cutoff=args.l_cutoff,
        noise_factor=args.noise_factor,
        num_mc_samples=args.mc_samples,
        error_method=args.error_method,
    )

    # ── Save output ──────────────────────────────────────────────────────
    out_file = EF_OUT_PATH / (
        f'eigenfunction_clean_m{m}_{cent_freq}_{mode}_{symmetry}_{data_name}.npz')
    np.savez(out_file,
             ef_uphi       = uphi_recon,
             ef_uthe       = uthe_recon,
             ef_uphi_sm    = uphi_sm,
             ef_uthe_sm    = uthe_sm,
             uphi_err_real = uphi_err_real,
             uphi_err_imag = uphi_err_imag,
             uthe_err_real = uthe_err_real,
             uthe_err_imag = uthe_err_imag,
             lats          = lats,
             final_td      = final_td)

    print(f'\n✓ Saved: {out_file}\n')


if __name__ == '__main__':
    main()
