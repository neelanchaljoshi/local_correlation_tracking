"""
extract_paper_eigenfunctions.py
---------------------------------
Regenerates data/for_edmond/paper_eigenfunctions/ from the three raw
sources compared in the paper -- LCT on magnetic-feature tracking
(hmi.m_720s), LCT on granulation tracking (hmi.ic_45s), and
ring-diagram analysis (RDA) -- for the four modes the paper highlights
(m=1 high-latitude, m=2 critical-latitude, m=8 equatorial Rossby,
m=13 high-frequency retrograde).

The Legendre-projection + Monte Carlo error-bar logic below
(confidence_level_to_power, monte_carlo_errorbars,
monte_carlo_legendre_error_random_amp_phase, fl_errorbars,
project_onto_legendre) is ported from
plotting_scripts/rda_comparison_lct.ipynb, which computed the same
thing interactively for one-off plots. This script just runs it for
all (mode, source) combinations and saves the result, instead of
leaving it in a notebook with commented-out np.savez calls.

Output layout, one subfolder per source, filenames following the same
eigenfunction_clean_m{m}_{freq}_{mode}_{symmetry}_{data}.npz
convention as rossby_eigenfunctions/ (m/freq/mode/symmetry/data parsed
straight out of the filename by anything downstream that reads it,
same as inertial_mode_pipeline.config.EF_FILENAME):

    data/for_edmond/paper_eigenfunctions/
        lctmag/eigenfunction_clean_m1_-88.0_highlat_anti_hmi_m_720s_dt_1h.npz
        lctgran/eigenfunction_clean_m1_-88.0_highlat_anti_hmi_ic_45s_granule.npz
        rda/eigenfunction_clean_m1_-88.0_highlat_anti_rda.npz
        ... (m=2, 8, 13, same pattern)

Each .npz holds: ef_uphi, ef_uthe (cleaned, noise-filtered
reconstruction), ef_uphi_sm, ef_uthe_sm (fully-smoothed, no noise cut),
uphi_err_real, uphi_err_imag, uthe_err_real, uthe_err_imag (1-sigma
Monte Carlo errors over the discarded coefficients), and lats. No
final_td -- that's the SVD time-dependence from the real extraction
pipeline (inertial_mode_pipeline.eigenfunction), which this
Legendre-projection-only code path never computes.

Usage
-----
    python extract_paper_eigenfunctions.py
"""

import pathlib
import shutil

import numpy as np
from scipy import integrate
from scipy.special import legendre
from scipy.stats import chi2

HERE = pathlib.Path(__file__).parent
EIGENFUNCTIONS_DIR = HERE.parent / 'data' / 'eigenfunctions'
RDA_DIR = pathlib.Path('/data/seismo/joshin/pipeline-test/paper_lct/rda_efs/rda_efs_75')
OUT_DIR = HERE.parent / 'data' / 'for_edmond' / 'paper_eigenfunctions'

LATS_LCT = np.linspace(-90, 90, 73)

# (m, freq, mode, symmetry, lct_mag_file, lct_gran_file, rda_file) --
# the four modes plotting_scripts/rda_comparison_lct.ipynb compares,
# with the exact filenames it loads them from.
MODES = [
    (1, -88.0, 'highlat', 'anti',
     'eigenfunction_m1_-88.0_highlat_anti_hmi_m_720s_dt_1h.npz',
     'eigenfunction_m1_-88.0_highlat_anti_hmi_ic_45s_granule.npz',
     'hmi_rda_05_new_300_1_+_[-108, -68]_2010_2025.npz'),
    (2, -73.0, 'critlat', 'anti',
     'eigenfunction_m2_-73.0_critlat_anti_hmi_m_720s_dt_1h.npz',
     'eigenfunction_m2_-73.0_critlat_anti_hmi_ic_45s_granule.npz',
     'hmi_rda_05_new_300_2_+_[-83, -63]_2010_2025.npz'),
    (8, -115.0, 'rossby', 'anti',
     'eigenfunction_m8_-115.0_rossby_anti_hmi_m_720s_dt_1h.npz',
     'eigenfunction_m8_-115.0_rossby_anti_hmi_ic_45s_granule.npz',
     'hmi_rda_05_new_300_8_+_[-130, -90]_2010_2025.npz'),
    (13, -214.0, 'hfr', 'sym',
     'eigenfunction_m13_-214.0_hfr_sym_hmi_m_720s_dt_1h.npz',
     'eigenfunction_m13_-214.0_hfr_sym_hmi_ic_45s_granule.npz',
     'hmi_rda_05_new_300_13_-_[-224, -204]_2010_2025.npz'),
]

L_THEORY_CUTOFF = 15
L_MAX = 22
NUM_MC_SAMPLES = 500
ERROR_METHOD = 'monte_carlo'


# ── Legendre projection + error bars (ported from the notebook) ────────────

def confidence_level_to_power(conf, B, M, dof):
    """conf = 1 - FAP (false alarm probability), B = background power, M = n frequency bins."""
    cdf = conf ** (1 / M)
    z_conf = chi2.ppf(cdf, df=dof)
    return z_conf * B / dof


def monte_carlo_errorbars(theta, l_discard, fl_discard, num_samples=1000):
    """Randomize only the phase of each discarded coefficient; std of the reconstruction over trials."""
    P = np.array([
        np.sqrt((2 * l + 1) / 2) * legendre(l)(np.cos(theta)) for l in l_discard
    ])
    samples_real, samples_imag = [], []
    for _ in range(num_samples):
        phases = np.exp(1j * 2 * np.pi * np.random.rand(len(fl_discard)))
        f_random = np.abs(fl_discard) * phases
        u_sample = np.sum(f_random[:, None] * P, axis=0)
        samples_real.append(np.real(u_sample))
        samples_imag.append(np.imag(u_sample))
    return np.std(samples_real, axis=0), np.std(samples_imag, axis=0)


def monte_carlo_legendre_error_random_amp_phase(theta_array, discarded_fl, l_array, num_trials=500):
    """Randomize both amplitude (20% std) and phase of each discarded coefficient."""
    theta_array = np.array(theta_array)
    n_theta = len(theta_array)
    trials = np.zeros((num_trials, n_theta), dtype=np.complex128)
    for i in range(num_trials):
        u_recon = np.zeros(n_theta, dtype=np.complex128)
        for fl, l in zip(discarded_fl, l_array):
            amp_mean = np.abs(fl)
            amp_sample = np.random.normal(loc=amp_mean, scale=0.2 * amp_mean)
            phase = np.random.uniform(0, 2 * np.pi)
            fl_sample = amp_sample * np.exp(1j * phase)
            norm = np.sqrt((2 * l + 1) / 2)
            u_recon += fl_sample * norm * legendre(l)(np.cos(theta_array))
        trials[i] = u_recon
    return np.std(trials.real, axis=0), np.std(trials.imag, axis=0)


def fl_errorbars(theta, l_discard, fl_discard):
    """Deterministic: absolute value of the direct sum of all discarded modes."""
    P = np.array([
        np.sqrt((2 * l + 1) / 2) * legendre(l)(np.cos(theta)) for l in l_discard
    ])
    u_total = np.sum(np.array([f * p for f, p in zip(fl_discard, P)]), axis=0)
    return np.abs(u_total), np.abs(u_total)


def project_onto_legendre(
    ef_uphi, ef_uthe, lats,
    symmetryuphi='anti', l_max=L_MAX,
    l_theory_cutoff=L_THEORY_CUTOFF, noise_factor=3,
    num_mc_samples=NUM_MC_SAMPLES, error_method=ERROR_METHOD,
):
    """
    Project onto Legendre polynomials, keep only theoretically-expected
    (l <= l_theory_cutoff) plus statistically-significant higher-l
    coefficients, reconstruct both the fully-smoothed (all l) and
    noise-filtered (kept l only, l < l_max) eigenfunctions, and
    estimate 1-sigma errors from the discarded coefficients.

    Returns
    -------
    uphi_sm, uthe_sm, uphi_clean, uthe_clean,
    uphi_err_real, uphi_err_imag, uthe_err_real, uthe_err_imag
    """
    theta = np.deg2rad(90 - lats)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    l_array = np.arange(36)
    fl_uphi = np.zeros_like(l_array, dtype=np.complex128)
    fl_uthe = np.zeros_like(l_array, dtype=np.complex128)
    for l in l_array:
        norm = np.sqrt((2 * l + 1) / 2)
        P_l = legendre(l)(cos_theta)
        fl_uphi[l] = integrate.simpson(ef_uphi * P_l * sin_theta * norm, theta)
        fl_uthe[l] = integrate.simpson(ef_uthe * P_l * sin_theta * norm, theta)

    if symmetryuphi == 'anti':
        l_uphi, l_uthe = l_array[1::2], l_array[0::2]
        fl_uphi[0::2] = 0
        fl_uthe[1::2] = 0
        fl_uphi_sym, fl_uthe_sym = fl_uphi[1::2], fl_uthe[0::2]
    else:
        l_uphi, l_uthe = l_array[0::2], l_array[1::2]
        fl_uphi[1::2] = 0
        fl_uthe[0::2] = 0
        fl_uphi_sym, fl_uthe_sym = fl_uphi[0::2], fl_uthe[1::2]

    uphi_sm = np.zeros_like(theta, dtype=np.complex128)
    uthe_sm = np.zeros_like(theta, dtype=np.complex128)
    for l in l_array:
        norm = np.sqrt((2 * l + 1) / 2)
        P_l = legendre(l)(cos_theta)
        uphi_sm += fl_uphi[l] * norm * P_l
        uthe_sm += fl_uthe[l] * norm * P_l

    def filter_l(power, l_vals):
        cumulative_power = np.cumsum(power) / np.sum(power)
        idx_95 = np.argmax(cumulative_power >= 0.99)
        noise_floor = np.median(power[int(l_theory_cutoff // 2):])
        confidence_power = confidence_level_to_power(0.9, noise_floor, len(power), 2)
        mask_95 = np.arange(len(power)) <= idx_95
        mask_noise = power > confidence_power
        mask_low = l_vals <= l_theory_cutoff
        mask_high = (l_vals > l_theory_cutoff) & (mask_95 & mask_noise)
        return mask_low | mask_high

    keep_mask_uphi = filter_l(np.abs(fl_uphi_sym) ** 2, l_uphi)
    keep_mask_uthe = filter_l(np.abs(fl_uthe_sym) ** 2, l_uthe)
    l_uphi_keep, l_uthe_keep = l_uphi[keep_mask_uphi], l_uthe[keep_mask_uthe]

    uphi_clean = np.zeros_like(theta, dtype=np.complex128)
    uthe_clean = np.zeros_like(theta, dtype=np.complex128)
    for l in l_array:
        if l >= l_max:
            continue
        norm = np.sqrt((2 * l + 1) / 2)
        P_l = legendre(l)(cos_theta)
        if l in l_uphi_keep:
            uphi_clean += fl_uphi[l] * norm * P_l
        if l in l_uthe_keep:
            uthe_clean += fl_uthe[l] * norm * P_l

    l_discard_uphi = [l for l in l_uphi if l not in l_uphi_keep]
    l_discard_uthe = [l for l in l_uthe if l not in l_uthe_keep]
    fl_discard_uphi = [fl_uphi[l] for l in l_discard_uphi]
    fl_discard_uthe = [fl_uthe[l] for l in l_discard_uthe]

    if error_method == 'monte_carlo':
        uphi_err_real, uphi_err_imag = monte_carlo_errorbars(
            theta, l_discard_uphi, fl_discard_uphi, num_samples=num_mc_samples)
        uthe_err_real, uthe_err_imag = monte_carlo_errorbars(
            theta, l_discard_uthe, fl_discard_uthe, num_samples=num_mc_samples)
    elif error_method == 'monte_carlo_amp':
        uphi_err_real, uphi_err_imag = monte_carlo_legendre_error_random_amp_phase(
            theta, fl_discard_uphi, l_discard_uphi, num_trials=num_mc_samples)
        uthe_err_real, uthe_err_imag = monte_carlo_legendre_error_random_amp_phase(
            theta, fl_discard_uthe, l_discard_uthe, num_trials=num_mc_samples)
    elif error_method == 'fl_sum':
        uphi_err_real, uphi_err_imag = fl_errorbars(theta, l_discard_uphi, fl_discard_uphi)
        uthe_err_real, uthe_err_imag = fl_errorbars(theta, l_discard_uthe, fl_discard_uthe)
    else:
        raise ValueError(f'Unknown error_method: {error_method!r}')

    # Phase-align so u_theta is purely real at the equator.
    theta_deg = np.rad2deg(theta)
    equator_idx = np.argmin(np.abs(theta_deg - 90))
    phase = -1j * np.angle(uthe_clean[equator_idx])
    uphi_clean = uphi_clean * np.exp(phase)
    uthe_clean = uthe_clean * np.exp(phase)
    uphi_sm = uphi_sm * np.exp(phase)
    uthe_sm = uthe_sm * np.exp(phase)

    return (uphi_sm, uthe_sm, uphi_clean, uthe_clean,
            uphi_err_real, uphi_err_imag, uthe_err_real, uthe_err_imag)


# ── Driver ───────────────────────────────────────────────────────────────

def load_lct(filename: str) -> tuple:
    d = np.load(EIGENFUNCTIONS_DIR / filename)
    return d['ef_uphi'], d['ef_uthe'], LATS_LCT


def load_rda(filename: str) -> tuple:
    d = np.load(RDA_DIR / filename)
    return d['up_ef'][0], d['ut_ef'][0], d['lat']


def save_result(subdir: str, m: int, freq: float, mode: str, symmetry: str, data_tag: str,
                 result: tuple, lats: np.ndarray) -> pathlib.Path:
    uphi_sm, uthe_sm, uphi_clean, uthe_clean, uphi_er, uphi_ei, uthe_er, uthe_ei = result
    out_dir = OUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'eigenfunction_clean_m{m}_{freq}_{mode}_{symmetry}_{data_tag}.npz'
    np.savez(
        out_path,
        ef_uphi=uphi_clean, ef_uthe=uthe_clean,
        ef_uphi_sm=uphi_sm, ef_uthe_sm=uthe_sm,
        uphi_err_real=uphi_er, uphi_err_imag=uphi_ei,
        uthe_err_real=uthe_er, uthe_err_imag=uthe_ei,
        lats=lats,
    )
    return out_path


def main() -> None:
    if OUT_DIR.exists():
        print(f'Removing existing {OUT_DIR} (previous flat, mixed-schema copy)...')
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for m, freq, mode, symmetry, lct_mag_file, lct_gran_file, rda_file in MODES:
        print(f'\n=== m={m} ({mode}, {symmetry}) ===')

        uphi, uthe, lats = load_lct(lct_mag_file)
        result = project_onto_legendre(uphi, uthe, lats, symmetryuphi=symmetry)
        out = save_result('lctmag', m, freq, mode, symmetry, 'hmi_m_720s_dt_1h', result, lats)
        print(f'  lctmag  -> {out}')

        uphi, uthe, lats = load_lct(lct_gran_file)
        result = project_onto_legendre(uphi, uthe, lats, symmetryuphi=symmetry)
        out = save_result('lctgran', m, freq, mode, symmetry, 'hmi_ic_45s_granule', result, lats)
        print(f'  lctgran -> {out}')

        uphi, uthe, lat_rda = load_rda(rda_file)
        result = project_onto_legendre(uphi, uthe, lat_rda, symmetryuphi=symmetry)
        out = save_result('rda', m, freq, mode, symmetry, 'rda', result, lat_rda)
        print(f'  rda     -> {out}')

    print(f'\nDone. Wrote lctmag/, lctgran/, rda/ under {OUT_DIR}')


if __name__ == '__main__':
    main()
