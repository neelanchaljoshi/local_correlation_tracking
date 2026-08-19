"""
main_chunk.py
-------------
Command-line entrypoint for the embarrassingly-parallel (non-MPI) LCT
pipeline. Each invocation processes exactly one time chunk (a
dspan-length window — e.g. one day at dspan_hours=24, one hour at
dspan_hours=1). Meant to be driven by a SLURM array where each array
task is one independent chunk.

Two modes, chosen by whether year/month are given on the command line:

Month mode — chunks span one calendar month:
    python main_chunk.py config/granulation.ini 2019 6 --print-nchunks
    python main_chunk.py config/granulation.ini 2019 6 --chunk 15 -l info
    sbatch --array=1-30 run_slurm_chunk.sh config/granulation.ini 2019 6

Range mode — chunks span an explicit range_start/range_end set in the
config's [job] section (no year/month on the command line at all).
This is the way to get, say, "one day of hourly files" as a clean
--array=1-24 with no day-offset arithmetic: set range_start/range_end
to just that one day and dspan_hours=1.
    python main_chunk.py config/one_day_hourly.ini --print-nchunks
    python main_chunk.py config/one_day_hourly.ini --chunk 7 -l info
    sbatch --array=1-24 run_slurm_chunk.sh config/one_day_hourly.ini

--chunk is 1-indexed, matching $SLURM_ARRAY_TASK_ID directly.
"""

import argparse
import sys

from lct_pipeline.config import load_config
from lct_pipeline.mpi_utils import get_loglevel
from lct_pipeline.pipeline import _month_bounds, _count_chunks
from lct_pipeline import pipeline_chunk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config_file', type=str, help='Path to .ini config file')
    p.add_argument('year',        type=int, nargs='?', default=None,
                   help='Year to process (month mode). Omit, along with '
                        'month, to use range_start/range_end from the '
                        'config instead.')
    p.add_argument('month',       type=int, nargs='?', default=None,
                   help='Month to process, 1-12 (month mode). Omit, '
                        'along with year, for range mode.')
    p.add_argument('--chunk', '-c', type=int, default=None,
                   help='1-indexed chunk within the month or range '
                        '(matches $SLURM_ARRAY_TASK_ID). Required unless '
                        '--print-nchunks is given.')
    p.add_argument('--print-nchunks', action='store_true',
                   help='Print the number of chunks (for sizing '
                        '--array=1-N) and exit.')
    p.add_argument('--loglevel', '-l', type=str, default='info',
                   help='Logging level (default: info)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        cfg = load_config(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        print(f'ERROR loading config: {e}', file=sys.stderr)
        sys.exit(1)

    if (args.year is None) != (args.month is None):
        print('ERROR: year and month must both be given (month mode) or '
              'both omitted (range mode)', file=sys.stderr)
        sys.exit(1)

    month_mode = args.year is not None
    if month_mode and not (1 <= args.month <= 12):
        print(f'ERROR: month must be 1-12, got {args.month}', file=sys.stderr)
        sys.exit(1)

    if not month_mode and not cfg.has_range:
        print('ERROR: no year/month given, and range_start/range_end are '
              'not set in [job] — either pass year and month, or set '
              'range_start/range_end in the config', file=sys.stderr)
        sys.exit(1)

    if args.print_nchunks:
        if month_mode:
            dstart, dstop = _month_bounds(args.year, args.month, cfg)
        else:
            dstart, dstop = cfg.range_start, cfg.range_end
        print(_count_chunks(dstart, dstop, cfg.dspan))
        return

    if args.chunk is None:
        print('ERROR: --chunk is required unless --print-nchunks is given',
              file=sys.stderr)
        sys.exit(1)

    try:
        loglevel = get_loglevel(args.loglevel)
    except ValueError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    if month_mode:
        pipeline_chunk.run_chunk(cfg, args.year, args.month, args.chunk - 1,
                                  loglevel)
    else:
        pipeline_chunk.run_chunk_range(cfg, args.chunk - 1, loglevel)


if __name__ == '__main__':
    main()
