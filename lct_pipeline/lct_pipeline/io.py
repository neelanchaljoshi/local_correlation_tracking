"""
io.py
-----
All file I/O for the LCT pipeline:
  - Loading FITS keys tables
  - Reading FITS images with retry logic
  - Creating and writing HDF5 output files
  - Saving CCF arrays to .npy files

No physics or MPI here — pure I/O only.
"""
from __future__ import annotations
import logging
import os
import pathlib
import time
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .config import Config

logger = logging.getLogger(__name__)


# ── Keys table ────────────────────────────────────────────────────────────

def load_keys_table(year: int, cfg: Config) -> Table:
    """
    Load the FITS keys table for a given year.

    Parameters
    ----------
    year : observation year
    cfg  : Config (provides infile_fmt)

    Returns
    -------
    astropy Table

    Raises
    ------
    FileNotFoundError if the keys file does not exist
    """
    infile = datetime(year, 1, 1).strftime(cfg.infile_fmt)
    if not os.path.exists(infile):
        raise FileNotFoundError(f'Keys file not found: {infile}')
    logger.critical('Reading keys: %s', infile)
    return Table.read(infile)


# ── FITS image reading ────────────────────────────────────────────────────

def read_fits_image(path: str, segname: str) -> np.ndarray:
    """Read a single FITS image from a directory path."""
    full_path = os.path.join(path.rstrip('/'), segname)
    return fits.getdata(full_path)


def read_fits_pair(
    keys: Table,
    ii: int,
    njump: int,
    segname: str,
    Ntry: int,
    downsample: bool,
    psf_rel: Optional[np.ndarray],
    simulate_pmi_fn,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read a pair of FITS images (at ii and ii+njump) with retry logic.

    Parameters
    ----------
    keys          : FITS keys table
    ii            : index of the first image
    njump         : frame step to the second image
    segname       : filename of the image segment
    Ntry          : number of read retries
    downsample    : whether to simulate PMI from HMI
    psf_rel       : relative PSF (only used if downsample=True)
    simulate_pmi_fn : callable(image, psf_rel) → simulated image

    Returns
    -------
    isbad  : True if the images could not be read
    img1   : first image array (None if isbad)
    img2   : second image array (None if isbad)
    """
    if keys['isbad'][ii] or keys['isbad'][ii + njump]:
        logger.error('Skipping bad frames: %s and %s',
                     keys['t_rec'][ii], keys['t_rec'][ii + njump])
        return True, None, None

    for attempt in range(Ntry):
        try:
            img1 = read_fits_image(keys['path'][ii], segname)
            img2 = read_fits_image(keys['path'][ii + njump], segname)
            if downsample:
                img1 = simulate_pmi_fn(img1, psf_rel)
                img2 = simulate_pmi_fn(img2, psf_rel)
            return False, img1, img2
        except IOError as e:
            logger.warning('Read attempt %d/%d failed: %s', attempt + 1, Ntry, e)
            if attempt < Ntry - 1:
                time.sleep(5)

    logger.error('All %d read attempts failed for %s and %s',
                 Ntry, keys['t_rec'][ii], keys['t_rec'][ii + njump])
    return True, None, None


def read_fits_quad(
    keys: Table,
    ii: int,
    njump: int,
    segname: str,
    Ntry: int,
    downsample: bool,
    psf_rel: Optional[np.ndarray],
    simulate_pmi_fn,
) -> tuple[bool, Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Read four consecutive FITS images for interpolation mode.

    Returns
    -------
    isbad, img1, img2, img3, img4
    """
    indices = [ii, ii + njump, ii + 2 * njump, ii + 3 * njump]
    if any(keys['isbad'][i] for i in indices):
        logger.error('Skipping bad frames at index %d', ii)
        return True, None, None, None, None

    for attempt in range(Ntry):
        try:
            imgs = [read_fits_image(keys['path'][i], segname) for i in indices]
            if downsample:
                imgs = [simulate_pmi_fn(im, psf_rel) for im in imgs]
            return (False, *imgs)
        except IOError as e:
            logger.warning('Read attempt %d/%d failed: %s', attempt + 1, Ntry, e)
            if attempt < Ntry - 1:
                time.sleep(5)

    logger.error('All %d quad-read attempts failed at index %d', Ntry, ii)
    return True, None, None, None, None


# ── HDF5 output ───────────────────────────────────────────────────────────

def create_output_hdf5(
    outfile: pathlib.Path,
    nt: int,
    cfg: Config,
) -> tuple[h5py.File, h5py.Dataset, h5py.Dataset, h5py.Dataset]:
    """
    Create the HDF5 output file with pre-allocated datasets.

    Parameters
    ----------
    outfile : output file path
    nt      : number of time steps
    cfg     : Config (provides grid arrays)

    Returns
    -------
    h5file, utheta_ds, uphi_ds, tstart_ds
    """
    nlat = len(cfg.clat_arr)
    nlng = len(cfg.clng_arr)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    h5file  = h5py.File(outfile, 'w')
    utheta  = h5file.create_dataset('utheta',   (nt, nlat, nlng), dtype='f4')
    uphi    = h5file.create_dataset('uphi',     (nt, nlat, nlng), dtype='f4')
    tstart  = h5file.create_dataset('tstart',   (nt,),            dtype='S19')
    h5file.create_dataset('longitude', data=cfg.clng_arr, dtype='f8')
    h5file.create_dataset('latitude',  data=cfg.clat_arr, dtype='f8')
    logger.critical('Created output file: %s', outfile)
    return h5file, utheta, uphi, tstart


def write_chunk_velocities(
    utheta_ds: h5py.Dataset,
    uphi_ds: h5py.Dataset,
    it: int,
    ux_all: np.ndarray,
    uy_all: np.ndarray,
    ijlist: list,
) -> None:
    """
    Write one chunk of gathered velocities to the HDF5 datasets.

    Parameters
    ----------
    utheta_ds, uphi_ds : HDF5 datasets (nt, nlat, nlng)
    it                 : time index of this chunk
    ux_all, uy_all     : gathered velocity arrays
    ijlist             : list of (i, j) index pairs for each patch
    """
    for (i, j), ux, uy in zip(ijlist, ux_all, uy_all):
        utheta_ds[it, j, i] = -uy
        uphi_ds[it, j, i]   =  ux


# ── CCF saving ───────────────────────────────────────────────────────────

def save_ccf(
    ccf: np.ndarray,
    dat: datetime,
    clat: float,
    clng: float,
    cfg: Config,
    averaged: bool = False,
) -> None:
    """
    Save a CCF array to a .npy file if it falls within the configured
    lat/lng threshold and saving is enabled.

    Parameters
    ----------
    ccf      : CCF array to save
    dat      : observation datetime
    clat     : patch centre latitude [degrees]
    clng     : patch centre longitude [degrees]
    cfg      : Config
    averaged : True if this is a time-averaged CCF (different filename suffix)
    """
    if not cfg.save_ccf:
        return
    if abs(clat) > cfg.ccf_lat_threshold or abs(clng) > cfg.ccf_lng_threshold:
        return
    if cfg.ccf_dir is None:
        return

    suffix = 'av' if averaged else 'no_av'
    dspan_h = int(cfg.dspan.total_seconds() // 3600)
    dstep_m = int(cfg.dstep.total_seconds() // 60)
    fname   = (f'{dat.strftime("%Y%m%d_%H%M%S")}_ccf'
               f'_dspan{dspan_h}_dstep{dstep_m}'
               f'_{round(clat, 1)}_{round(clng, 1)}'
               f'_{cfg.resolution_label}_{suffix}.npy')
    np.save(cfg.ccf_dir / fname, ccf)
