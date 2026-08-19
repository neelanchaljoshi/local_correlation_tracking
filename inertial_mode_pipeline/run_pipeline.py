"""
run_pipeline.py
---------------
Command-line entrypoint for the inertial modes eigenfunction pipeline.

Usage
-----
    python run_pipeline.py <m> <cent_freq> <mode> <data> <symmetry> [options]

Example
-------
    python run_pipeline.py 2 -171.0 highlat hmi.m_720s_dt_1h sym \
        --l_max 22 --l_cutoff 15 --mc_samples 500

Run with --help for full argument list.
"""

import argparse
import pathlib
import sys

import numpy as np

from inertial_mode_pipeline.config import (
    LON_OG, LAT_OG, DT_SEC,
    L_MAX_RECON, L_THEORY_CUTOFF, NOISE_CONFIDENCE,
    SPAN_LOWER, SPAN_UPPER,
)
from inertial_mode_pipeline.io import load_flow_data, save_eigenfunction
from inertial_mode_pipeline.geometry import (
    make_lon_lat_grids,
    build_radius_array,
    clip_flow_data,
    apodize_flow_data,
    apply_symmetry,
    fill_carrington_gaps,
    get_correction_factor,
)
from inertial_mode_pipeline.fourier import transform_to_fourier
from inertial_mode_pipeline.eigenfunction import extract_eigenfunction
from inertial_mode_pipeline.legendre import project_and_clean


def resolve_df(n_span: int, dt_seconds: float, cent_freq: float,
                df_arg, min_bins: int = 2,
                margin: float = 1.2) -> float:
    """
    Pick a safe --df for the given span, or validate a user-supplied one.

    The frequency grid bandpass_filter builds has bins spaced
    Δf = 1e9 / (n_span * dt_seconds) apart. bandpass_filter's Tukey
    window defaults to tukey_alpha=0.0 (rectangular, no edge zeroing) on
    this code path, so a single surviving passband bin is technically
    enough for a non-zero result — min_bins (default 2) is a small
    statistical margin against a lone, fragile surviving bin, not a
    taper requirement. See tests/test_span_resolution.py and
    tests/min_df_reference.md for the full derivation.

    If df_arg is None, auto-selects df = min_bins * Δf / 2 * margin and
    prints what it chose. If df_arg is given but doesn't clear the
    min_bins threshold, prints a warning with a suggested value and
    returns df_arg unchanged (the user's explicit choice is respected).
    """
    freq_res = 1e9 / (n_span * dt_seconds)
    df_needed = min_bins * freq_res / 2 * margin

    if df_arg is None:
        print(f'  --df not given: auto-selected df={df_needed:.2f} nHz for '
              f'this span (resolution {freq_res:.2f} nHz/bin, targeting '
              f'>= {min_bins} passband bins).')
        return df_needed

    freq_nHz = np.fft.fftshift(-np.fft.fftfreq(n_span, dt_seconds) * 1e9)
    band = (freq_nHz > cent_freq - df_arg) & (freq_nHz < cent_freq + df_arg)
    n_bins = int(band.sum())
    if n_bins < min_bins:
        print(f'  WARNING: --df={df_arg} gives only {n_bins} passband bin(s) '
              f'for this span (resolution {freq_res:.2f} nHz/bin) — fewer '
              f'than {min_bins} risks a zero or unstable eigenfunction. '
              f'Consider --df {df_needed:.1f} or higher.')
    return df_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Positional
    p.add_argument('m',         type=int,   help='Azimuthal order')
    p.add_argument('cent_freq', type=float, help='Central frequency [nHz]')
    p.add_argument('mode',      type=str,   help='Mode label (highlat/rossby/…)')
    p.add_argument('data',      type=str,   help='Data product name')
    p.add_argument('symmetry',  type=str,   choices=['sym', 'anti', 'all'],
                   help='Equatorial symmetry of u_phi')

    # Optional
    p.add_argument('--l_max',        type=int,   default=L_MAX_RECON)
    p.add_argument('--l_cutoff',     type=int,   default=L_THEORY_CUTOFF)
    p.add_argument('--mc_samples',   type=int,   default=500)
    p.add_argument('--error_method', type=str,   default='monte_carlo',
                   choices=['monte_carlo', 'monte_carlo_amp', 'fl_sum'])
    p.add_argument('--span_lower',   type=float, default=SPAN_LOWER)
    p.add_argument('--span_upper',   type=float, default=SPAN_UPPER)
    p.add_argument('--reject_type',  type=str,   default='clip',
                   choices=['clip', 'noclip'])
    p.add_argument('--df',           type=float, default=None,
                   help='Bandpass half-width [nHz]. If omitted, a safe '
                        'value is auto-selected from --span_lower/--span_upper '
                        '(see resolve_df()).')
    p.add_argument('--min_bins',     type=int,   default=2,
                   help='Minimum passband bins required for a non-zero '
                        'result (default 2; bare mathematical minimum is 1 '
                        'under the pipeline\'s default rectangular window).')

    return p.parse_args()


def main() -> None:
    args      = parse_args()
    data_name = args.data.replace('.', '_')

    print(f'\n{"="*60}')
    print(f'  m={args.m}  freq={args.cent_freq} nHz  '
          f'mode={args.mode}  data={args.data}  sym={args.symmetry}')
    print(f'{"="*60}\n')

    # ── Load data ────────────────────────────────────────────────────────
    print('Loading flow data...')
    raw = load_flow_data(data_name)

    # ── Geometry setup ───────────────────────────────────────────────────
    lon_og, lat_og = make_lon_lat_grids(LON_OG, LAT_OG)
    crln           = fill_carrington_gaps(raw['crln_obs'])
    t_array        = raw['t_array']
    rsun_obs       = raw['rsun_obs']

    nt, nlat, nlng_stony = raw['uthe_all'].shape
    nlng_carr            = 2 * (nlng_stony - 1)
    lats                 = np.linspace(-90, 90, nlat)

    # ── Symmetry ─────────────────────────────────────────────────────────
    uphi, uthe = apply_symmetry(raw['uphi_all'], raw['uthe_all'], args.symmetry)

    # ── Radius array and disk masking ────────────────────────────────────
    print('Computing disk geometry...')
    r = build_radius_array(raw['crlt_obs'], rsun_obs, lon_og, lat_og)

    if args.reject_type == 'clip':
        uphi = clip_flow_data(uphi, r, rsun_obs)
        uthe = clip_flow_data(uthe, r, rsun_obs)
    else:
        uphi = apodize_flow_data(uphi, r, rsun_obs)
        uthe = apodize_flow_data(uthe, r, rsun_obs)

    # ── Amplitude correction ─────────────────────────────────────────────
    cft_u, cfl_u = get_correction_factor(uphi, nlng_carr)
    cft_v, cfl_v = get_correction_factor(uthe, nlng_carr)

    # ── Time span ────────────────────────────────────────────────────────
    span = (t_array >= args.span_lower) & (t_array < args.span_upper)
    n_span = int(span.sum())
    if n_span == 0:
        raise ValueError(
            f'No timesteps fall inside [--span_lower={args.span_lower}, '
            f'--span_upper={args.span_upper}) — widen the span.')

    df = resolve_df(n_span, DT_SEC, args.cent_freq, args.df,
                     min_bins=args.min_bins)

    # ── Fourier transform ────────────────────────────────────────────────
    print('Transforming to Fourier space...')
    uphi_ft, freq_nHz = transform_to_fourier(uphi, crln, cft_u, cfl_u, span, DT_SEC)
    uthe_ft, _        = transform_to_fourier(uthe, crln, cft_v, cfl_v, span, DT_SEC)

    # ── Eigenfunction extraction ─────────────────────────────────────────
    print('Extracting eigenfunction (SVD)...')
    ef_result = extract_eigenfunction(
        uphi_ft, uthe_ft, freq_nHz,
        m=args.m, cent_freq=args.cent_freq,
        lats=lats, df=df)

    # ── Legendre projection and cleaning ─────────────────────────────────
    print('Projecting onto Legendre polynomials...')

    # Determine u_phi symmetry label for Legendre projection
    sym_uphi = {'sym': 'sym', 'anti': 'anti', 'all': 'all'}[args.symmetry]

    clean = project_and_clean(
        m=args.m,
        ef_uphi=ef_result['ef_uphi'],
        ef_uthe=ef_result['ef_uthe'],
        lats=lats,
        symmetryuphi=sym_uphi,
        l_max=args.l_max,
        l_theory_cutoff=args.l_cutoff,
        num_mc_samples=args.mc_samples,
        error_method=args.error_method,
    )

    # ── Save ─────────────────────────────────────────────────────────────
    result = {
        **clean,
        'lats':     lats,
        'final_td': ef_result['final_td'],
    }
    out_path = save_eigenfunction(
        result, args.m, args.cent_freq,
        args.mode, args.symmetry, data_name)

    print(f'\n✓ Saved: {out_path}\n')


if __name__ == '__main__':
    main()
