"""
check_reading.py
-----------------
Stand-alone diagnostic: exercises ONLY the FITS-reading step of the
LCT pipeline (load_keys_table + read_fits_pair/read_fits_quad) for a
sample of timesteps, without doing any correlation tracking, CCF
averaging, or HDF5 writing. Useful to quickly confirm a config/year
actually reads before submitting a full (much slower) SLURM run.

Also specifically checks for the newline-poisoned-SUMS-path bug fixed
in io.py/read_fits_image: reports, for every timestep it looks at,
whether the *raw* keys-table path (before io.py's .strip()) contains a
trailing newline or other whitespace — so you can confirm the anomaly
either isn't present, or is present but handled.

Covers both pipeline entrypoints with one script, since they share
the exact same reading code (load_keys_table/read_fits_pair/
read_fits_quad from io.py) and only differ in how they resolve a time
window:

  Month mode (mirrors main.py, and main_chunk.py without --chunk):
      python check_reading.py config/magnetic.ini 2018 1

  Chunk mode (mirrors main_chunk.py with --chunk, 1-indexed):
      python check_reading.py config/magnetic.ini 2018 1 --chunk 1

  Range mode (mirrors main_chunk.py with no year/month, using
  range_start/range_end from the config's [job] section):
      python check_reading.py config/one_day_hourly.ini --chunk 1

By default, samples --sample evenly-spaced timesteps within the
resolved window rather than reading every single one (a full month at
45s cadence is tens of thousands of reads); pass --full to check every
timestep instead.

Exits 0 if every attempted read succeeded, 1 otherwise.
"""

import argparse
import sys
from datetime import datetime

import numpy as np

from lct_pipeline.config import load_config
from lct_pipeline.geometry import build_psfs, simulate_pmi_from_hmi
from lct_pipeline.io import load_keys_table, read_fits_pair, read_fits_quad
from lct_pipeline.mpi_utils import get_loglevel, setup_logging
from lct_pipeline.pipeline import _month_bounds, _count_chunks, _iter_times
from lct_pipeline.pipeline_chunk import (
    resolve_chunk_bounds,
    resolve_range_chunk_bounds,
)


def resolve_window(cfg, year, month, chunk):
    """
    Return (dstart, dstop, label) for the requested mode, using the
    exact same bounds-resolution functions main.py/main_chunk.py use.
    """
    month_mode = year is not None
    if month_mode:
        if chunk is not None:
            bounds = resolve_chunk_bounds(cfg, year, month, chunk - 1)
            if bounds is None:
                dstart, dstop = _month_bounds(year, month, cfg)
                nt = _count_chunks(dstart, dstop, cfg.dspan)
                raise ValueError(
                    f'--chunk {chunk} out of range [1, {nt}] for {year}-{month:02d}')
            dstart, dstop = bounds
            label = f'{year}-{month:02d} chunk {chunk} (chunk mode)'
        else:
            dstart, dstop = _month_bounds(year, month, cfg)
            label = f'{year}-{month:02d} full month (month mode)'
    else:
        if not cfg.has_range:
            raise ValueError(
                'no year/month given, and range_start/range_end are not '
                'set in [job] — pass year and month, or set a range')
        if chunk is not None:
            bounds = resolve_range_chunk_bounds(cfg, chunk - 1)
            if bounds is None:
                nt = _count_chunks(cfg.range_start, cfg.range_end, cfg.dspan)
                raise ValueError(
                    f'--chunk {chunk} out of range [1, {nt}] for '
                    f'{cfg.range_start} - {cfg.range_end}')
            dstart, dstop = bounds
            label = f'{cfg.range_start} - {cfg.range_end} chunk {chunk} (range mode)'
        else:
            dstart, dstop = cfg.range_start, cfg.range_end
            label = f'{cfg.range_start} - {cfg.range_end} full range (range mode)'
    return dstart, dstop, label


def sample_timesteps(dstart, dstop, dstep, sample, full):
    """Evenly-spaced sample of timesteps in [dstart, dstop), or all of them if full."""
    all_times = list(_iter_times(dstart, dstop, dstep))
    if full or sample is None or sample >= len(all_times):
        return all_times
    if sample <= 0:
        return []
    idx = np.linspace(0, len(all_times) - 1, sample).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [all_times[i] for i in idx]


def check_reading(cfg, dstart, dstop, sample, full, log):
    keys = load_keys_table(dstart.year, cfg)
    dat0 = datetime.strptime(keys['t_rec'][0], '%Y.%m.%d_%H:%M:%S_TAI')

    _, _, psf_rel = build_psfs(cfg)

    max_offset = 3 * cfg.njump if cfg.interpolate else cfg.njump
    times = sample_timesteps(dstart, dstop, cfg.dstep, sample, full)

    n_checked = 0
    n_isbad = 0
    n_ok = 0
    n_failed = 0
    n_anomaly = 0
    n_out_of_range = 0

    for dat in times:
        ii_f = (dat - dat0).total_seconds() / cfg.cadence_keys
        if not ii_f.is_integer():
            log.warning('%s: non-integer frame index (%.3f) — skipping',
                        dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'), ii_f)
            continue
        ii = int(ii_f)

        if ii < 0 or ii + max_offset >= len(keys):
            log.warning('%s: index %d (+%d) out of range for a %d-row keys '
                        'table — skipping', dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'),
                        ii, max_offset, len(keys))
            n_out_of_range += 1
            continue

        n_checked += 1

        offsets = [0, cfg.njump, 2 * cfg.njump, 3 * cfg.njump] if cfg.interpolate \
            else [0, cfg.njump]
        for off in offsets:
            raw_path = keys['path'][ii + off]
            if raw_path != raw_path.strip():
                n_anomaly += 1
                log.warning('%s: raw path has leading/trailing whitespace '
                            '(e.g. the newline-poisoned-SUMS-path bug): %r',
                            dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'), raw_path)

        if keys['isbad'][ii] or keys['isbad'][ii + cfg.njump]:
            n_isbad += 1
            log.info('%s: marked isbad in keys table — skipping read',
                     dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'))
            continue

        log.info('--- %s ---', dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'))
        if cfg.interpolate:
            isbad, img1, img2, img3, img4 = read_fits_quad(
                keys, ii, cfg.njump, cfg.segname, cfg.Ntry,
                cfg.downsample, psf_rel, simulate_pmi_from_hmi)
        else:
            isbad, img1, img2 = read_fits_pair(
                keys, ii, cfg.njump, cfg.segname, cfg.Ntry,
                cfg.downsample, psf_rel, simulate_pmi_from_hmi)

        if isbad:
            n_failed += 1
        else:
            n_ok += 1

    return {
        'n_rows': len(keys),
        'n_checked': n_checked,
        'n_isbad': n_isbad,
        'n_ok': n_ok,
        'n_failed': n_failed,
        'n_anomaly': n_anomaly,
        'n_out_of_range': n_out_of_range,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config_file', type=str, help='Path to .ini config file')
    p.add_argument('year',  type=int, nargs='?', default=None,
                   help='Year (month mode). Omit, along with month, for range mode.')
    p.add_argument('month', type=int, nargs='?', default=None,
                   help='Month, 1-12 (month mode). Omit, along with year, for range mode.')
    p.add_argument('--chunk', '-c', type=int, default=None,
                   help='1-indexed chunk within the month/range (matches '
                        '$SLURM_ARRAY_TASK_ID) — checks only that chunk\'s '
                        'window, same bounds main_chunk.py would use. Omit '
                        'to check the whole month/range instead.')
    p.add_argument('--sample', type=int, default=20,
                   help='Number of evenly-spaced timesteps to check '
                        '(default 20). Ignored if --full is given.')
    p.add_argument('--full', action='store_true',
                   help='Check every timestep in the window instead of sampling.')
    p.add_argument('--loglevel', '-l', type=str, default='info',
                   help='Logging level (default: info)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        loglevel = get_loglevel(args.loglevel)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    try:
        cfg = load_config(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        print(f'ERROR loading config: {e}', file=sys.stderr)
        sys.exit(1)

    if (args.year is None) != (args.month is None):
        print('ERROR: year and month must both be given (month mode) or '
              'both omitted (range mode)', file=sys.stderr)
        sys.exit(1)

    log = setup_logging(rank=0, loglevel=loglevel)

    try:
        dstart, dstop, label = resolve_window(cfg, args.year, args.month, args.chunk)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'\n{"=" * 60}')
    print(f'  config: {args.config_file}')
    print(f'  window: {label}')
    print(f'  {dstart} to {dstop}')
    print(f'{"=" * 60}\n')

    try:
        result = check_reading(cfg, dstart, dstop, args.sample, args.full, log)
    except FileNotFoundError as e:
        print(f'\nERROR: {e}', file=sys.stderr)
        sys.exit(1)

    print(f'\n{"=" * 60}')
    print(f'  keys table rows        : {result["n_rows"]}')
    print(f'  timesteps checked      : {result["n_checked"]}')
    print(f'  out of keys-table range: {result["n_out_of_range"]}')
    print(f'  marked isbad (skipped) : {result["n_isbad"]}')
    print(f'  read OK                : {result["n_ok"]}')
    print(f'  read FAILED            : {result["n_failed"]}')
    print(f'  raw paths with newline/whitespace anomaly: {result["n_anomaly"]}')
    print(f'{"=" * 60}')

    if result['n_anomaly'] > 0:
        print(f"\nNOTE: {result['n_anomaly']} raw path(s) had leading/trailing "
              f"whitespace (the newline-poisoned-SUMS-path pattern) — this is "
              f"expected for keys files generated before the get_hmi_keys/"
              f"fetch_keys.py fix, and io.py's read_fits_image.strip() handles "
              f"it transparently (see read OK/FAILED counts above).")

    if result['n_failed'] > 0:
        print(f"\nFAIL: {result['n_failed']} read(s) failed.")
        sys.exit(1)
    if result['n_checked'] == 0:
        print('\nFAIL: nothing was actually checked (see out-of-range/isbad counts).')
        sys.exit(1)
    print(f"\nOK: all {result['n_ok']} attempted read(s) succeeded.")


if __name__ == '__main__':
    main()
