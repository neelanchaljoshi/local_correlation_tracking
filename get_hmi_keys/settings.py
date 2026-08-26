"""
settings.py
------------
Parses `.ini` files into a typed `Config` dataclass. All other modules
take a `Config` and never touch `configparser` directly — same
convention as lct_pipeline/inertial_mode_pipeline's config.py.

Usage
-----
    from settings import load_config
    cfg = load_config('config/hmi_v_45s.ini')
    cfg.seriesname   # -> 'hmi.v_45s'
"""
from __future__ import annotations
import configparser
import pathlib
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


# ── DRMS keyword schema ─────────────────────────────────────────────────
# The keywords fetched and their dtypes. This is a fixed contract with
# downstream code (lct_pipeline reads t_rec/crln_obs/crlt_obs/rsun_obs/
# quality/path/isbad by name), not a per-run tunable, so it lives here
# as a constant rather than in the .ini. `extra_keys` in [series] can
# append additional DRMS keywords on top of this baseline without
# touching code.
BASE_KEY_LIST = [
    ('t_rec', bytes),
    ('t_obs', bytes),
    ('obs_vr', float),
    ('quality', str),
    ('crpix1', float),
    ('crpix2', float),
    ('crval1', float),
    ('crval2', float),
    ('cdelt1', float),
    ('cdelt2', float),
    ('crota2', float),
    ('crln_obs', float),
    ('crlt_obs', float),
    ('rsun_obs', float),
]


@dataclass
class Config:
    """All get_hmi_keys parameters in one typed object."""

    # ── Job ──────────────────────────────────────────────────────────
    yr_start: int
    yr_stop: int

    # ── Series ───────────────────────────────────────────────────────
    seriesname: str
    cadence_seconds: int
    key_list: list = field(default_factory=lambda: list(BASE_KEY_LIST))

    # ── Data availability ────────────────────────────────────────────
    data_start: Optional[date] = None
    # If a requested year is data_start's year, the query window starts
    # at data_start instead of Jan 1 of that year. None means every
    # year runs Jan 1 - Jan 1 with no special case.

    # ── Quality ──────────────────────────────────────────────────────
    qbits_pass: int = 0

    # ── Paths ────────────────────────────────────────────────────────
    outdir: pathlib.Path = None
    outfile_fmt: str = 'keys-%Y.fits'
    # strftime-format filename, relative to outdir. Override this (e.g.
    # 'keys_new_swan/keys-%Y.fits') to write directly into whatever
    # subdirectory an lct_pipeline .ini's infile_fmt_4k/2k expects,
    # instead of the flat outdir/keys-%Y.fits this stage wrote before.

    def __post_init__(self):
        self.outdir.mkdir(parents=True, exist_ok=True)

    def output_path(self, year: int) -> pathlib.Path:
        """Return the output FITS path for a given year."""
        rel = date(year, 1, 1).strftime(self.outfile_fmt)
        path = self.outdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | pathlib.Path) -> Config:
    """
    Parse a .ini file into a Config.

    Raises
    ------
    FileNotFoundError  if path doesn't exist
    ValueError          on missing/invalid required fields
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    parser = configparser.RawConfigParser()
    parser.read(path)

    def get(section, key, fallback=None):
        if fallback is None and not parser.has_option(section, key):
            raise ValueError(f'{path}: missing required [{section}] {key}')
        return parser.get(section, key, fallback=fallback)

    def getint(section, key, fallback=None):
        return int(get(section, key, fallback=fallback))

    # [job]
    yr_start = getint('job', 'yr_start')
    yr_stop = getint('job', 'yr_stop')
    if yr_stop < yr_start:
        raise ValueError(f'{path}: [job] yr_stop ({yr_stop}) < yr_start ({yr_start})')

    # [series]
    seriesname = get('series', 'seriesname')
    cadence_seconds = getint('series', 'cadence_seconds')

    key_list = list(BASE_KEY_LIST)
    extra_keys_raw = get('series', 'extra_keys', fallback='').strip()
    if extra_keys_raw:
        existing = {name for name, _ in key_list}
        for name in (n.strip() for n in extra_keys_raw.split(',')):
            if name and name not in existing:
                key_list.append((name, float))
                existing.add(name)

    # [data_availability]
    data_start_raw = get('data_availability', 'data_start', fallback='').strip()
    data_start = date.fromisoformat(data_start_raw) if data_start_raw else None

    # [quality]
    qbits_pass_raw = get('quality', 'qbits_pass', fallback='0')
    qbits_pass = int(qbits_pass_raw, 0)  # base=0 autodetects 0x../0b../plain decimal

    # [paths]
    outdir = pathlib.Path(get('paths', 'outdir'))
    outfile_fmt = get('paths', 'outfile_fmt', fallback='keys-%Y.fits')

    return Config(
        yr_start=yr_start, yr_stop=yr_stop,
        seriesname=seriesname, cadence_seconds=cadence_seconds,
        key_list=key_list,
        data_start=data_start,
        qbits_pass=qbits_pass,
        outdir=outdir, outfile_fmt=outfile_fmt,
    )
