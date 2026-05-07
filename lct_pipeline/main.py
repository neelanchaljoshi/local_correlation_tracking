"""
main.py
-------
Command-line entrypoint for the LCT pipeline.

Usage
-----
    python main.py config/granulation.ini 2019 --month 6 -l info
    mpirun -n 500 python main.py config/granulation.ini 2019 --month 6

Arguments
---------
  config_file   Path to the .ini configuration file
  year          Year to process (integer)

Optional
--------
  --month, -m   Month to process (1–12). If omitted, processes all 12 months
                sequentially (single-node use only).
  --loglevel, -l  Logging level: debug | info | warning | error | critical
                  (default: info)
"""

import argparse
import logging
import sys

from lct_pipeline.config import load_config
from lct_pipeline.mpi_utils import get_loglevel
from lct_pipeline import pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config_file',    type=str, help='Path to .ini config file')
    p.add_argument('year',           type=int, help='Year to process')
    p.add_argument('--month', '-m',  type=int, default=None,
                   help='Month to process (1-12). Omit to run all months.')
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

    months = [args.month] if args.month is not None else list(range(1, 13))

    for month in months:
        if not cfg.validate_month(args.year, month):
            continue
        pipeline.run(cfg, args.year, month, loglevel)


if __name__ == '__main__':
    main()
