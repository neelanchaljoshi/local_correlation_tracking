"""
config.py
---------
Parse, validate, and expose all pipeline configuration from a .ini file.

Usage
-----
    from lct_pipeline.config import load_config
    cfg = load_config('config/granulation.ini')
    cfg.patch_size   # → 68 or 34 depending on downsample
"""
from __future__ import annotations
import configparser
import pathlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

# Data availability start — warn if requested range is before this
DATA_AVAILABLE_FROM = datetime(2010, 5, 1)


# ── Dataclass ──────────────────────────────────────────────────────────────

@dataclass
class Config:
    """All pipeline parameters in one typed object."""

    # ── Job ──────────────────────────────────────────────────────────────
    yr_start:               int
    yr_stop:                int
    dspan:                  timedelta
    dstep:                  timedelta

    # ── Instrument ───────────────────────────────────────────────────────
    segname:                str
    cadence:                int
    cadence_keys:           int
    NX:                     int
    NY:                     int
    Ntry:                   int
    downsample:             bool
    interpolate:            bool
    cadence_interp:         int
    njump:                  int

    # ── PSF ──────────────────────────────────────────────────────────────
    wavelength_m:           float
    aperture_hmi_m:         float
    aperture_pmi_m:         float
    pixel_scale_arcsec:     float
    psf_size:               int

    # ── Grid ─────────────────────────────────────────────────────────────
    clat_arr:               np.ndarray
    clng_arr:               np.ndarray

    # ── LCT ──────────────────────────────────────────────────────────────
    patch_size:             int
    pixel_size:             float
    alpha:                  float
    R_sun_Mm:               float
    grid_len:               int
    ntry_fit:               int

    # ── Tracking ─────────────────────────────────────────────────────────
    change_track:           bool
    A:                      float
    B:                      float
    C:                      float
    CRrate:                 float

    # ── B0 correction ────────────────────────────────────────────────────
    dI:                     float
    t_ref_b0:               datetime

    # ── Paths ────────────────────────────────────────────────────────────
    infile_fmt:             str
    rootdir_out:            pathlib.Path
    ccf_dir:                Optional[pathlib.Path]

    # ── Output ───────────────────────────────────────────────────────────
    save_ccf:               bool
    ccf_lat_threshold:      float
    ccf_lng_threshold:      float

    # ── Source ───────────────────────────────────────────────────────────
    config_path:            pathlib.Path = field(repr=False)

    def __post_init__(self):
        self.rootdir_out.mkdir(parents=True, exist_ok=True)
        if self.ccf_dir is not None:
            self.ccf_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_magnetic(self) -> bool:
        return 'magnetogram' in self.segname

    @property
    def resolution_label(self) -> str:
        return '2k' if self.downsample else '4k'

    def output_filename(self, year: int, month: int) -> pathlib.Path:
        """Return the HDF5 output path for a given year/month."""
        seg = 'mag' if self.is_magnetic else 'gran'
        fname = (f'{year}_{month:02d}_{seg}'
                 f'_dspan{int(self.dspan.total_seconds()//3600)}h'
                 f'_dstep{int(self.dstep.total_seconds()//60)}m'
                 f'_{self.resolution_label}.hdf5')
        return self.rootdir_out / fname

    def validate_month(self, year: int, month: int) -> bool:
        """
        Return True if data is available for this year/month.
        Warns and returns False if before May 2010.
        """
        requested = datetime(year, month, 1)
        if requested < DATA_AVAILABLE_FROM:
            warnings.warn(
                f'Requested {year}-{month:02d} is before data availability '
                f'({DATA_AVAILABLE_FROM.strftime("%Y-%m")}). Skipping.',
                UserWarning, stacklevel=2)
            return False
        return True


# ── Loader ────────────────────────────────────────────────────────────────

def load_config(path: str | pathlib.Path) -> Config:
    """
    Parse a .ini file and return a validated Config object.

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ValueError         if required fields are missing or invalid
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config file not found: {path}')

    cp = configparser.RawConfigParser()
    cp.read(path)

    def get(section, key, fallback=None):
        try:
            return cp.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return str(fallback)
            raise ValueError(f'Missing required config key: [{section}] {key}')

    def getint(s, k, fb=None):   return int(get(s, k, fb))
    def getfloat(s, k, fb=None): return float(get(s, k, fb))
    def getbool(s, k, fb=None):
        val = get(s, k, fb).strip().lower()
        if val in ('1', 'true', 'yes', 'on'):   return True
        if val in ('0', 'false', 'no', 'off'):  return False
        raise ValueError(f'Invalid boolean [{s}] {k}: {val!r}')

    # ── Job ──────────────────────────────────────────────────────────────
    yr_start = getint('job', 'yr_start')
    yr_stop  = getint('job', 'yr_stop')
    dspan    = timedelta(hours=getint('job', 'dspan_hours'))
    dstep    = timedelta(minutes=getint('job', 'dstep_minutes'))
    if yr_start > yr_stop:
        raise ValueError(f'yr_start ({yr_start}) > yr_stop ({yr_stop})')

    # ── Instrument ───────────────────────────────────────────────────────
    segname      = get('instrument', 'segname')
    cadence      = getint('instrument', 'cadence_seconds')
    cadence_keys = getint('instrument', 'dataset_cadence_seconds')
    NX           = getint('instrument', 'NX')
    NY           = getint('instrument', 'NY')
    Ntry         = getint('instrument', 'Ntry')
    downsample   = getbool('instrument', 'downsample')
    interpolate  = getbool('instrument', 'interpolate')
    njump          = cadence // cadence_keys
    cadence_interp = 60 if interpolate else cadence_keys

    # ── PSF ──────────────────────────────────────────────────────────────
    wavelength_m       = getfloat('psf', 'wavelength_nm') * 1e-9
    aperture_hmi_m     = getfloat('psf', 'aperture_hmi_m')
    aperture_pmi_m     = getfloat('psf', 'aperture_pmi_m')
    pixel_scale_arcsec = getfloat('psf', 'pixel_scale_arcsec')
    psf_size           = getint('psf', 'psf_size')

    # ── Grid ─────────────────────────────────────────────────────────────
    clat_arr = np.arange(
        getfloat('grid', 'clat_start'),
        getfloat('grid', 'clat_stop') + 1e-9,
        getfloat('grid', 'clat_step'))
    clng_arr = np.arange(
        getfloat('grid', 'clng_start'),
        getfloat('grid', 'clng_stop') + 1e-9,
        getfloat('grid', 'clng_step'))

    # ── LCT ──────────────────────────────────────────────────────────────
    patch_key  = 'patch_size_2k' if downsample else 'patch_size_4k'
    pixel_key  = 'pixel_size_2k' if downsample else 'pixel_size_4k'
    patch_size = getint('lct', patch_key)
    pixel_size = getfloat('lct', pixel_key)
    alpha      = getfloat('lct', 'alpha')
    R_sun_Mm   = getfloat('lct', 'R_sun_Mm')
    grid_len   = getint('lct', 'grid_len')
    ntry_fit   = getint('lct', 'ntry_fit')
    if grid_len % 2 == 0:
        raise ValueError(f'grid_len must be odd, got {grid_len}')

    # ── Tracking ─────────────────────────────────────────────────────────
    change_track = getbool('tracking', 'change_track')
    A      = getfloat('tracking', 'A')
    B      = getfloat('tracking', 'B')
    C      = getfloat('tracking', 'C')
    CRrate = getfloat('tracking', 'CRrate')

    # ── B0 correction ────────────────────────────────────────────────────
    dI       = getfloat('b0_correction', 'dI')
    t_ref_b0 = datetime.fromisoformat(get('b0_correction', 't_ref_b0'))

    # ── Paths ────────────────────────────────────────────────────────────
    fmt_key     = 'infile_fmt_2k' if downsample else 'infile_fmt_4k'
    infile_fmt  = get('paths', fmt_key)
    rootdir_out = pathlib.Path(get('paths', 'rootdir_out'))
    ccf_raw     = get('paths', 'ccf_dir', fallback='').strip()
    ccf_dir     = pathlib.Path(ccf_raw) if ccf_raw else None

    # ── Output ───────────────────────────────────────────────────────────
    save_ccf          = getbool('output', 'save_ccf')
    ccf_lat_threshold = getfloat('output', 'ccf_lat_threshold')
    ccf_lng_threshold = getfloat('output', 'ccf_lng_threshold')
    if save_ccf and ccf_dir is None:
        raise ValueError('save_ccf=true but ccf_dir is empty in [paths]')

    return Config(
        yr_start=yr_start, yr_stop=yr_stop, dspan=dspan, dstep=dstep,
        segname=segname, cadence=cadence, cadence_keys=cadence_keys,
        NX=NX, NY=NY, Ntry=Ntry,
        downsample=downsample, interpolate=interpolate,
        cadence_interp=cadence_interp, njump=njump,
        wavelength_m=wavelength_m,
        aperture_hmi_m=aperture_hmi_m, aperture_pmi_m=aperture_pmi_m,
        pixel_scale_arcsec=pixel_scale_arcsec, psf_size=psf_size,
        clat_arr=clat_arr, clng_arr=clng_arr,
        patch_size=patch_size, pixel_size=pixel_size,
        alpha=alpha, R_sun_Mm=R_sun_Mm,
        grid_len=grid_len, ntry_fit=ntry_fit,
        change_track=change_track, A=A, B=B, C=C, CRrate=CRrate,
        dI=dI, t_ref_b0=t_ref_b0,
        infile_fmt=infile_fmt, rootdir_out=rootdir_out, ccf_dir=ccf_dir,
        save_ccf=save_ccf,
        ccf_lat_threshold=ccf_lat_threshold,
        ccf_lng_threshold=ccf_lng_threshold,
        config_path=path,
    )
