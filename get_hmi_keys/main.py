"""
main.py
-------
Command-line entrypoint for get_hmi_keys.

Usage
-----
    python main.py <config_file> [--year Y]

Example
-------
    python main.py config/hmi_v_45s.ini
    python main.py config/hmi_v_45s.ini --year 2018

If --year is omitted, processes every year in [yr_start, yr_stop]
(inclusive) from the config's [job] section.
"""
import argparse
import sys
from datetime import datetime

from settings import load_config
from process import process_year


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config_file', type=str, help='Path to .ini config file')
    p.add_argument('--year', type=int, default=None,
                    help='Single year to process. Omit to process every '
                         'year in [yr_start, yr_stop] from the config.')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        cfg = load_config(args.config_file)
    except (FileNotFoundError, ValueError) as e:
        print(f'ERROR loading config: {e}', file=sys.stderr)
        sys.exit(1)

    years = [args.year] if args.year is not None else list(range(cfg.yr_start, cfg.yr_stop + 1))

    T = [datetime.now()]
    for yr in years:
        process_year(yr, cfg)
        T.append(datetime.now())
        print(f"Total time for {yr}: {T[-1] - T[-2]}")


if __name__ == "__main__":
    main()
