"""
plot_observed_eigenfunction.py
--------------------------------
Example: load one of the published observed inertial-mode
eigenfunctions from data/sup_eigenfunctions/observed_eigenfunctions/
and plot it — u_phi and u_theta, real and imaginary parts, vs
latitude, with 1-sigma error shading.

The .npz schema matches inertial_mode_pipeline.io.save_eigenfunction()
exactly (ef_uphi, ef_uthe, ef_uphi_sm, ef_uthe_sm, uphi_err_real/imag,
uthe_err_real/imag, lats, final_td); filenames follow
inertial_mode_pipeline.config.EF_FILENAME
('eigenfunction_clean_m{m}_{freq}_{mode}_{symmetry}_{data}.npz').

Usage
-----
    python plot_observed_eigenfunction.py                # default example: m=8 (Rossby)
    python plot_observed_eigenfunction.py --m 13          # pick by azimuthal order
    python plot_observed_eigenfunction.py --npz /path/to/some_eigenfunction.npz
    python plot_observed_eigenfunction.py --list           # list what's available
"""

import argparse
import pathlib
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

EF_DIR = (pathlib.Path(__file__).parent.parent / 'data' / 'sup_eigenfunctions'
          / 'observed_eigenfunctions')

FILENAME_RE = re.compile(
    r'eigenfunction_clean_m(-?\d+)_(-?[\d.]+)_(.+)_(sym|anti|all)_(.+)\.npz')


def parse_filename(path: pathlib.Path) -> dict:
    """Pull (m, freq, mode, symmetry, data) metadata out of the filename."""
    match = FILENAME_RE.match(path.name)
    if not match:
        return {'m': None, 'freq': None, 'mode': None, 'symmetry': None, 'data': None}
    m, freq, mode, symmetry, data = match.groups()
    return {'m': int(m), 'freq': float(freq), 'mode': mode,
            'symmetry': symmetry, 'data': data}


def find_by_m(m: int) -> pathlib.Path:
    """Find the top-level (full 2010-2024 span) eigenfunction file for azimuthal order m."""
    candidates = [p for p in EF_DIR.glob(f'eigenfunction_clean_m{m}_*.npz')
                  if parse_filename(p)['m'] == m]
    if not candidates:
        raise FileNotFoundError(
            f'No top-level eigenfunction found for m={m} in {EF_DIR} — '
            f'use --list to see what is available.')
    return candidates[0]


def list_available() -> None:
    files = sorted(EF_DIR.glob('eigenfunction_clean_*.npz'), key=lambda p: p.name)
    print(f'Top-level eigenfunctions in {EF_DIR}:\n')
    for f in files:
        meta = parse_filename(f)
        print(f"  m={meta['m']:<4} freq={meta['freq']:<9.1f} mode={meta['mode']:<10} "
              f"sym={meta['symmetry']:<5} data={meta['data']}")
    year_dirs = sorted(d for d in EF_DIR.iterdir() if d.is_dir())
    if year_dirs:
        print(f'\n...plus {len(year_dirs)} sliding-window subdirectories '
              f'(year_YYYY.Y/, each with its own per-window eigenfunctions).')


def plot_eigenfunction(npz_path: pathlib.Path, outfile: pathlib.Path = None,
                        show_smoothed: bool = True) -> plt.Figure:
    data = np.load(npz_path)
    meta = parse_filename(npz_path)
    lats = data['lats']

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

    for ax, comp, label in [(axes[0], 'uphi', r'$u_\phi$'),
                             (axes[1], 'uthe', r'$u_\theta$')]:
        ef = data[f'ef_{comp}']
        err_r = data[f'{comp}_err_real']
        err_i = data[f'{comp}_err_imag']

        ax.plot(lats, ef.real, color='tab:blue', lw=2, label='Re')
        ax.fill_between(lats, ef.real - err_r, ef.real + err_r,
                         color='tab:blue', alpha=0.2)
        ax.plot(lats, ef.imag, color='tab:orange', lw=2, ls='--', label='Im')
        ax.fill_between(lats, ef.imag - err_i, ef.imag + err_i,
                         color='tab:orange', alpha=0.2)

        if show_smoothed and f'ef_{comp}_sm' in data:
            ef_sm = data[f'ef_{comp}_sm']
            ax.plot(lats, ef_sm.real, color='tab:blue', lw=1, alpha=0.5, ls=':')
            ax.plot(lats, ef_sm.imag, color='tab:orange', lw=1, alpha=0.5, ls=':')

        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_ylabel(f'{label}  [m/s]')
        ax.grid(alpha=0.3)
        ax.legend(fontsize='small', loc='upper right')

    axes[-1].set_xlabel('Latitude [deg]')
    axes[0].set_title(
        f"m={meta['m']}  {meta['mode']}  ({meta['symmetry']})  "
        f"$\\nu$={meta['freq']:.1f} nHz\n{meta['data']}",
        fontsize='medium')
    fig.tight_layout()

    if outfile:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, bbox_inches='tight')
        print(f'Saved: {outfile}')

    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', type=str, default=None,
                    help='Path to a specific eigenfunction .npz. Overrides --m.')
    p.add_argument('--m', type=int, default=8,
                    help='Azimuthal order to look up among the top-level '
                         '(full 2010-2024 span) eigenfunctions (default: 8, '
                         'the equatorial Rossby mode).')
    p.add_argument('--list', action='store_true',
                    help='List available top-level eigenfunctions and exit.')
    p.add_argument('--outfile', type=str, default=None,
                    help='Output figure path (default: next to this script, '
                         'named after the input file).')
    p.add_argument('--no-smoothed', action='store_true',
                    help='Skip overlaying the fully-smoothed (no noise cut) curve.')
    p.add_argument('--no-show', action='store_true',
                    help='Do not call plt.show() even on an interactive backend.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_available()
        return

    try:
        npz_path = pathlib.Path(args.npz) if args.npz else find_by_m(args.m)
    except FileNotFoundError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    if args.npz and not npz_path.exists():
        print(f'ERROR: {npz_path} does not exist', file=sys.stderr)
        sys.exit(1)

    print(f'Loading: {npz_path}')

    outfile = (pathlib.Path(args.outfile) if args.outfile
               else pathlib.Path(__file__).parent / f'{npz_path.stem}.pdf')
    plot_eigenfunction(npz_path, outfile=outfile, show_smoothed=not args.no_smoothed)

    if not args.no_show and plt.get_backend().lower() != 'agg':
        plt.show()


if __name__ == '__main__':
    main()
