"""
plot_power_spectrum.py
-----------------------
Visualize the latitude-averaged power spectrum for one (m, component)
combination and fit it with a Lorentzian, using the same geometry and
Fourier-transform steps as run_pipeline.py — but without the bandpass
filter, so the full spectrum (not just the passband run_pipeline.py
would extract the mode from) is available to look at and fit.

Usage
-----
    python plot_power_spectrum.py <m> <component> <mode> <data> <symmetry> \
        --fit_range LOW HIGH [options]

Example
-------
    python plot_power_spectrum.py 1 uphi highlat hmi.m_720s_dt_1h anti \
        --fit_range -150 50

Run with --help for full argument list.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from inertial_mode_pipeline.config import (
    LON_OG, LAT_OG, DT_SEC, SPAN_LOWER, SPAN_UPPER,
    TILE_SIZE_DEG, PS_MODE_LAT_BANDS, PS_OUT, PS_FILENAME,
)
from inertial_mode_pipeline.io import load_flow_data
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
from inertial_mode_pipeline.lorentzian_fit import LorentzianMLE, lorentzian, timer


def resolve_lat_band(mode: str, lat_min, lat_max):
    """
    Fall back to the mode's default latitude band (PS_MODE_LAT_BANDS)
    if --lat_min/--lat_max were not both given explicitly.
    """
    if lat_min is not None and lat_max is not None:
        return lat_min, lat_max
    if mode in PS_MODE_LAT_BANDS:
        default_min, default_max = PS_MODE_LAT_BANDS[mode]
        return (lat_min if lat_min is not None else default_min,
                lat_max if lat_max is not None else default_max)
    raise ValueError(
        f"mode '{mode}' has no default latitude band "
        f'(known: {sorted(PS_MODE_LAT_BANDS)}) — pass both '
        f'--lat_min and --lat_max explicitly.')


def compute_power_spectrum(
    m: int, component: str, data: str, symmetry: str,
    lat_min: float, lat_max: float,
    span_lower: float, span_upper: float, reject_type: str,
) -> dict:
    """
    Run the pipeline's own geometry + Fourier-transform steps (no
    bandpass, no SVD) and return the latitude-averaged power spectrum
    at azimuthal order m for one flow component.

    Returns
    -------
    dict with keys: freq_nHz, power, lat_mask, lats, n_avg, nt
    """
    data_name = data.replace('.', '_')
    raw = load_flow_data(data_name)

    lon_og, lat_og = make_lon_lat_grids(LON_OG, LAT_OG)
    crln     = fill_carrington_gaps(raw['crln_obs'])
    t_array  = raw['t_array']
    rsun_obs = raw['rsun_obs']

    nt, nlat, nlng_stony = raw['uthe_all'].shape
    nlng_carr = 2 * (nlng_stony - 1)
    lats      = np.linspace(-90, 90, nlat)

    uphi, uthe = apply_symmetry(raw['uphi_all'], raw['uthe_all'], symmetry)
    arr = uphi if component == 'uphi' else uthe

    r = build_radius_array(raw['crlt_obs'], rsun_obs, lon_og, lat_og)
    if reject_type == 'clip':
        arr = clip_flow_data(arr, r, rsun_obs)
    else:
        arr = apodize_flow_data(arr, r, rsun_obs)

    cft, cfl = get_correction_factor(arr, nlng_carr)

    span = (t_array >= span_lower) & (t_array < span_upper)
    if span.sum() == 0:
        raise ValueError(
            f'No timesteps fall inside [--span_lower={span_lower}, '
            f'--span_upper={span_upper}) — widen the span.')

    ft, freq_nHz = transform_to_fourier(arr, crln, cft, cfl, span, DT_SEC)

    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    if not lat_mask.any():
        raise ValueError(
            f'No latitude grid points fall inside [{lat_min}, {lat_max}] deg.')

    lat_grid_spacing_deg = np.abs(np.diff(lats)).mean()
    lat_overlap_factor = TILE_SIZE_DEG / lat_grid_spacing_deg
    n_avg = lat_mask.sum() / lat_overlap_factor

    nt_span = int(span.sum())
    conv_factor = 2 / nt_span * 1e-9 * DT_SEC / nlng_carr / nlng_carr
    power = np.nanmean(np.abs(ft[:, lat_mask, m])**2, axis=1) * conv_factor

    return {
        'freq_nHz': freq_nHz,
        'power': power,
        'lat_mask': lat_mask,
        'lats': lats,
        'n_avg': n_avg,
        'nt': nt_span,
    }


def fit_and_plot(
    freq_nHz, power, n_avg, m, component, mode, symmetry, data,
    fit_range, lat_min, lat_max,
    use_differential_evolution=False, initial_params=None,
    n_mc=2000, rng=None, xlim=None,
):
    """
    Fit a Lorentzian over fit_range and plot the spectrum + fit.

    Returns
    -------
    fig, fit (LorentzianMLE)
    """
    freq_window = (freq_nHz >= fit_range[0]) & (freq_nHz <= fit_range[1])
    if freq_window.sum() < 4:
        raise ValueError(
            f'Only {int(freq_window.sum())} frequency bins fall inside '
            f'fit_range={fit_range} — widen it or check the span/cadence.')

    label = f'm={m} {component} {mode} {data} ({symmetry})'
    with timer(f'fit {label}'):
        fit = LorentzianMLE(
            freq_nHz[freq_window], power[freq_window], n_avg,
            label=label, mode=mode, method=data,
            use_differential_evolution=use_differential_evolution,
            initial_params=initial_params,
        ).run(n_mc=n_mc, rng=rng)

    fig, ax = plt.subplots(figsize=(6, 4))
    freq_lor = np.linspace(fit_range[0], fit_range[1], 300)
    ax.plot(freq_nHz, power, color='tab:blue', lw=1.5, label='power spectrum')
    ax.plot(freq_lor, lorentzian(freq_lor, *fit.popt), color='tab:orange', lw=1.5,
             label='Lorentzian fit')
    ax.axvline(fit.popt[1], color='tab:orange', lw=1, alpha=0.6)
    ax.axvspan(fit.popt[1] - fit.lo_err[1], fit.popt[1] + fit.hi_err[1],
               alpha=0.1, color='tab:orange')
    ax.set_xlim(xlim or fit_range)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Frequency [nHz]')
    ax.set_ylabel(f'{component} power  [$m^2\\,s^{{-2}}\\,nHz^{{-1}}$]')
    ax.set_title(
        f'$m={m}$, {mode}, {data} ({symmetry})\n'
        f'lat {lat_min:g}\N{DEGREE SIGN}-{lat_max:g}\N{DEGREE SIGN}, '
        f'$\\nu_0$={fit.popt[1]:.1f} nHz, SNR={fit.popt[0] / fit.popt[3]:.1f}')
    ax.legend(fontsize='small')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, fit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument('m',         type=int,   help='Azimuthal order')
    p.add_argument('component', type=str,   choices=['uphi', 'uthe'],
                   help='Flow component to plot/fit')
    p.add_argument('mode',      type=str,   help='Mode label (highlat/rossby/'
                                                  'critlat/hfr/…) — used as '
                                                  'the default latitude band '
                                                  'and as plot/filename metadata')
    p.add_argument('data',      type=str,   help='Data product name')
    p.add_argument('symmetry',  type=str,   choices=['sym', 'anti', 'all'],
                   help='Equatorial symmetry of u_phi (same convention as '
                        "run_pipeline.py's symmetry argument — u_theta gets "
                        'the opposite parity via geometry.apply_symmetry)')

    p.add_argument('--fit_range', type=float, nargs=2, required=True,
                   metavar=('LOW', 'HIGH'),
                   help='Frequency range [nHz] to fit the Lorentzian over '
                        'and to set the default plot x-limits')
    p.add_argument('--lat_min', type=float, default=None,
                   help='Latitude band lower bound [deg]. Defaults from '
                        '--mode (highlat/critlat/rossby/hfr) if omitted.')
    p.add_argument('--lat_max', type=float, default=None,
                   help='Latitude band upper bound [deg]. Defaults from '
                        '--mode (highlat/critlat/rossby/hfr) if omitted.')
    p.add_argument('--span_lower', type=float, default=SPAN_LOWER)
    p.add_argument('--span_upper', type=float, default=SPAN_UPPER)
    p.add_argument('--reject_type', type=str, default='clip',
                   choices=['clip', 'noclip'])
    p.add_argument('--n_mc', type=int, default=2000,
                   help='Monte Carlo realisations for the fit error bars '
                        '(default 2000; table_lor_fit.py-scale runs use '
                        '10000 but that is slow for an interactive tool)')
    p.add_argument('--use_differential_evolution', action='store_true',
                   help='Run a global optimizer for the initial guess '
                        'instead of the automatic heuristic')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for the Monte Carlo error estimate')
    p.add_argument('--xlim', type=float, nargs=2, default=None,
                   metavar=('LOW', 'HIGH'),
                   help='Plot x-limits [nHz] (default: --fit_range)')
    p.add_argument('--outfile', type=str, default=None,
                   help='Output figure path. Defaults to PS_OUT/'
                        + PS_FILENAME)
    p.add_argument('--no-show', action='store_true',
                   help='Do not call plt.show() even on an interactive backend')

    return p.parse_args()


def main() -> None:
    args = parse_args()
    lat_min, lat_max = resolve_lat_band(args.mode, args.lat_min, args.lat_max)

    print(f'\n{"=" * 60}')
    print(f'  m={args.m}  component={args.component}  mode={args.mode}  '
          f'data={args.data}  sym={args.symmetry}')
    print(f'  latitude band: {lat_min:g} to {lat_max:g} deg')
    print(f'{"=" * 60}\n')

    print('Loading flow data and transforming to Fourier space...')
    spectrum = compute_power_spectrum(
        args.m, args.component, args.data, args.symmetry,
        lat_min, lat_max, args.span_lower, args.span_upper, args.reject_type)

    fig, fit = fit_and_plot(
        spectrum['freq_nHz'], spectrum['power'], spectrum['n_avg'],
        args.m, args.component, args.mode, args.symmetry, args.data,
        args.fit_range, lat_min, lat_max,
        use_differential_evolution=args.use_differential_evolution,
        n_mc=args.n_mc, rng=np.random.default_rng(args.seed),
        xlim=args.xlim,
    )

    if args.outfile:
        out_path = pathlib.Path(args.outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        PS_OUT.mkdir(parents=True, exist_ok=True)
        data_name = args.data.replace('.', '_')
        out_path = PS_OUT / PS_FILENAME.format(
            m=args.m, component=args.component, mode=args.mode,
            symmetry=args.symmetry, data=data_name)
    fig.savefig(out_path, bbox_inches='tight')
    print(f'\n✓ Saved figure: {out_path}')

    summary_path = out_path.with_suffix('.json')
    A, x0, fwhm, B = fit.popt
    summary = {
        'm': args.m, 'component': args.component, 'mode': args.mode,
        'data': args.data, 'symmetry': args.symmetry,
        'lat_min': lat_min, 'lat_max': lat_max, 'n_avg': spectrum['n_avg'],
        'fit_range_nHz': list(args.fit_range),
        'amp': A, 'amp_lo_err': fit.lo_err[0], 'amp_hi_err': fit.hi_err[0],
        'freq_nHz': x0, 'freq_lo_err': fit.lo_err[1], 'freq_hi_err': fit.hi_err[1],
        'fwhm_nHz': fwhm, 'fwhm_lo_err': fit.lo_err[2], 'fwhm_hi_err': fit.hi_err[2],
        'background': B, 'background_lo_err': fit.lo_err[3], 'background_hi_err': fit.hi_err[3],
        'snr': A / B, 'resolved': bool(fit.resolved),
    }
    with open(summary_path, 'w', encoding='utf-8') as fobj:
        json.dump(summary, fobj, indent=2)
    print(f'✓ Saved fit summary: {summary_path}')

    if not args.no_show and plt.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':
    main()
