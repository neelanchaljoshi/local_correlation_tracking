"""
main_chunk.py
-------------
Command-line entrypoint for the embarrassingly-parallel (non-MPI) LCT
pipeline. Each invocation processes exactly one time chunk (a
dspan-length window — e.g. one day at dspan_hours=24, one hour at
dspan_hours=1) for a given year/month. Meant to be driven by a SLURM
array where each array task is one independent chunk.

Usage
-----
    # How many chunks does this month have? (sizes the --array range)
    python main_chunk.py config/granulation.ini 2019 6 --print-nchunks

    # Process one chunk directly
    python main_chunk.py config/granulation.ini 2019 6 --chunk 15 -l info

    # Typical SLURM submission
    sbatch --array=1-30 run_slurm_chunk.sh config/granulation.ini 2019 6

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
    p.add_argument('year',        type=int, help='Year to process')
    p.add_argument('month',       type=int, help='Month to process (1-12)')
    p.add_argument('--chunk', '-c', type=int, default=None,
                   help='1-indexed chunk within the month (matches '
                        '$SLURM_ARRAY_TASK_ID). Required unless '
                        '--print-nchunks is given.')
    p.add_argument('--print-nchunks', action='store_true',
                   help='Print the number of chunks in this month '
                        '(for sizing --array=1-N) and exit.')
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

    if not (1 <= args.month <= 12):
        print(f'ERROR: month must be 1-12, got {args.month}', file=sys.stderr)
        sys.exit(1)

    if args.print_nchunks:
        dstart, dstop = _month_bounds(args.year, args.month, cfg)
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

    pipeline_chunk.run_chunk(cfg, args.year, args.month, args.chunk - 1, loglevel)


if __name__ == '__main__':
    main()
