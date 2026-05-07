"""
pipeline.py
-----------
Main processing loop for the LCT pipeline.

Orchestrates MPI distribution, year/chunk/patch iteration,
CCF accumulation, velocity extraction, and HDF5 output.

No physics logic lives here — this is pure orchestration.
"""
from __future__ import annotations
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from .config import Config
from .geometry import (
    build_psfs,
    simulate_pmi_from_hmi,
    compute_b0_correction,
    carrington_longitude_shift,
    remap_patches,
)
from .interpolation import interpolate_image_stack
from .io import (
    load_keys_table,
    read_fits_pair,
    read_fits_quad,
    create_output_hdf5,
    write_chunk_velocities,
    save_ccf as save_ccf_fn,
)
from .lct import build_tukey_kernel, get_ccf, get_flow_velocity
from .mpi_utils import gather_bigsize, log_mpi_info, setup_logging

logger = logging.getLogger(__name__)

# Sentinel for when MPI is not available (local testing)
_MPI_AVAILABLE = True
try:
    from mpi4py import MPI
except ImportError:
    _MPI_AVAILABLE = False


# ── Time iteration helpers ────────────────────────────────────────────────

def _iter_times(start: datetime, stop: datetime, step: timedelta):
    """Yield datetimes from start to stop (exclusive) in step increments."""
    t = start
    while t < stop:
        yield t
        t += step


def _month_bounds(year: int, month: int, cfg: Config):
    """Return (dstart, dstop) for a calendar month."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    dstart   = datetime(year, month, 1)
    dstop    = datetime(year, month, last_day, 23, 59, 59)
    return dstart, dstop


def _count_chunks(dstart, dstop, dspan):
    total  = (dstop - dstart).total_seconds()
    n      = total / dspan.total_seconds()
    return int(np.ceil(n))


# ── Per-patch processing ──────────────────────────────────────────────────

def _process_patch(
    img1: np.ndarray,
    img2: np.ndarray,
    clng: float,
    clat: float,
    keys,
    ii: int,
    dat: datetime,
    kernel: np.ndarray,
    cfg: Config,
    img3: Optional[np.ndarray] = None,
    img4: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Remap and cross-correlate one patch. Returns the CCF or None on failure.
    """
    try:
        # ── Astrometry ────────────────────────────────────────────────
        njump = cfg.njump
        if cfg.interpolate:
            crpix1   = tuple(keys['crpix1'][ii + k*njump] for k in range(4))
            crpix2   = tuple(keys['crpix2'][ii + k*njump] for k in range(4))
            cdelt1   = tuple(keys['cdelt1'][ii + k*njump] for k in range(4))
            cdelt2   = tuple(keys['cdelt2'][ii + k*njump] for k in range(4))
            rsun_obs = tuple(keys['rsun_obs'][ii + k*njump] for k in range(4))

            t_recs   = [datetime.strptime(keys['t_rec'][ii + k*njump],
                                          '%Y.%m.%d_%H:%M:%S_TAI')
                        for k in range(4)]
            dB_vals  = tuple(keys['crlt_obs'][ii + k*njump] +
                             compute_b0_correction(t_recs[k], cfg)[0]
                             for k in range(4))
            dP_vals  = tuple(-keys['crota2'][ii + k*njump] +
                             compute_b0_correction(t_recs[k], cfg)[1]
                             for k in range(4))
            dL_vals  = tuple(
                0 if k == 0 else
                keys['crln_obs'][ii] - keys['crln_obs'][ii + k*njump] +
                (carrington_longitude_shift(clat, k * njump * cfg.cadence_keys, cfg)
                 if cfg.change_track else 0.0)
                for k in range(4))

            imgs_r = remap_patches(
                [img1, img2, img3, img4],
                crpix1, crpix2, cdelt1, cdelt2, rsun_obs,
                dB_vals, dP_vals, dL_vals, cfg, clng, clat)
            img1r, img2r, img3r, img4r = imgs_r
            img2_interp = interpolate_image_stack(
                np.array([img1r, img2r, img3r, img4r]),
                target_time=cfg.cadence_interp,
                times=[0, cfg.cadence_keys, 2*cfg.cadence_keys, 3*cfg.cadence_keys])
            ccf, _, _ = get_ccf(img1r, img2_interp, kernel)

        else:
            crpix1   = keys['crpix1'][ii], keys['crpix1'][ii + njump]
            crpix2   = keys['crpix2'][ii], keys['crpix2'][ii + njump]
            cdelt1   = keys['cdelt1'][ii], keys['cdelt1'][ii + njump]
            cdelt2   = keys['cdelt2'][ii], keys['cdelt2'][ii + njump]
            rsun_obs = keys['rsun_obs'][ii], keys['rsun_obs'][ii + njump]

            t1 = datetime.strptime(keys['t_rec'][ii], '%Y.%m.%d_%H:%M:%S_TAI')
            dB1, dP1 = compute_b0_correction(t1, cfg)
            dB  = (keys['crlt_obs'][ii] + dB1, keys['crlt_obs'][ii + njump] + dB1)
            dP  = (-keys['crota2'][ii] + dP1,  -keys['crota2'][ii + njump] + dP1)
            dL_shift = (carrington_longitude_shift(
                clat, njump * cfg.cadence_keys, cfg) if cfg.change_track else 0.0)
            dL  = (0, keys['crln_obs'][ii] - keys['crln_obs'][ii + njump] + dL_shift)

            img1r, img2r = remap_patches(
                [img1, img2],
                crpix1, crpix2, cdelt1, cdelt2, rsun_obs,
                dB, dP, dL, cfg, clng, clat)
            ccf, _, _ = get_ccf(img1r, img2r, kernel)

        if np.isnan(ccf).any():
            return None
        return ccf

    except Exception as e:
        logger.warning('Patch (%.1f, %.1f) failed: %s', clat, clng, e)
        return None


# ── Main run function ─────────────────────────────────────────────────────

def run(cfg: Config, year: int, month: int, loglevel: int) -> None:
    """
    Run the full LCT pipeline for one year-month.

    Parameters
    ----------
    cfg      : validated Config object
    year     : year to process
    month    : month to process (1–12)
    loglevel : logging level for rank 0
    """
    # ── MPI setup ─────────────────────────────────────────────────────
    if _MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm, rank, size = None, 0, 1

    log = setup_logging(rank, loglevel)

    if _MPI_AVAILABLE:
        log_mpi_info(comm, log)

    # ── Data availability check ────────────────────────────────────────
    if not cfg.validate_month(year, month):
        if rank == 0:
            log.warning('Exiting cleanly — no data for %d-%02d', year, month)
        return

    # ── Startup: PSF and Tukey kernel ─────────────────────────────────
    _, _, psf_rel = build_psfs(cfg)
    kernel        = build_tukey_kernel(cfg.patch_size, cfg.alpha)

    # ── Build patch list and MPI distribution ─────────────────────────
    sys.path.insert(0, '/data/seismo/zhichao/codes/pypkg')
    from zclpy3.mpi_util import get_delimiters

    xylist = [(x, y) for y in cfg.clat_arr for x in cfg.clng_arr]
    ijlist = [(i, j) for j in range(len(cfg.clat_arr))
              for i in range(len(cfg.clng_arr))]
    delim, chunks = get_delimiters(len(xylist), size, return_chunks=True)
    lo, hi = delim[rank]

    # ── Load keys ─────────────────────────────────────────────────────
    if rank == 0:
        keys = load_keys_table(year, cfg)
    else:
        keys = None
    if _MPI_AVAILABLE:
        keys = comm.bcast(keys, root=0)

    # ── Time bounds for this month ─────────────────────────────────────
    dstart, dstop = _month_bounds(year, month, cfg)
    nt            = _count_chunks(dstart, dstop, cfg.dspan)

    # ── Create HDF5 output (rank 0 only) ──────────────────────────────
    if rank == 0:
        outfile = cfg.output_filename(year, month)
        h5file, utheta_ds, uphi_ds, tstart_ds = create_output_hdf5(
            outfile, nt, cfg)

    dat0 = datetime.strptime(keys['t_rec'][0], '%Y.%m.%d_%H:%M:%S_TAI')

    # ── Main loop: chunks ─────────────────────────────────────────────
    for it, dstart_chunk in enumerate(_iter_times(dstart, dstop, cfg.dspan)):
        ccfs = np.zeros((hi - lo, cfg.patch_size, cfg.patch_size), dtype='f8')
        nums = np.zeros(hi - lo, dtype='i4')

        if rank == 0:
            tstart_ds[it] = bytes(dstart_chunk.strftime('%Y.%m.%d_%H:%M:%S'),
                                  encoding='utf-8')

        # ── Inner loop: time steps within this chunk ───────────────────
        for dat in _iter_times(dstart_chunk, dstart_chunk + cfg.dspan, cfg.dstep):
            try:
                isbad = False
                img1 = img2 = img3 = img4 = None

                if rank == 0:
                    log.info('--- %s ---', dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'))
                    ii_f = (dat - dat0).total_seconds() / cfg.cadence_keys
                    assert ii_f.is_integer(), f'Non-integer frame index: {ii_f}'
                    ii = int(ii_f)

                    if cfg.interpolate:
                        isbad, img1, img2, img3, img4 = read_fits_quad(
                            keys, ii, cfg.njump, cfg.segname, cfg.Ntry,
                            cfg.downsample, psf_rel, simulate_pmi_from_hmi)
                    else:
                        isbad, img1, img2 = read_fits_pair(
                            keys, ii, cfg.njump, cfg.segname, cfg.Ntry,
                            cfg.downsample, psf_rel, simulate_pmi_from_hmi)
                else:
                    ii = None

                if _MPI_AVAILABLE:
                    if cfg.interpolate:
                        isbad, img1, img2, img3, img4, ii = comm.bcast(
                            [isbad, img1, img2, img3, img4, ii], root=0)
                    else:
                        isbad, img1, img2, ii = comm.bcast(
                            [isbad, img1, img2, ii], root=0)

                if isbad:
                    continue

                # ── Patch loop (MPI distributed) ──────────────────────
                for ipatch, (clng, clat) in enumerate(xylist[lo:hi]):
                    ccf = _process_patch(
                        img1, img2, clng, clat,
                        keys, ii, dat, kernel, cfg,
                        img3=img3, img4=img4)
                    if ccf is None:
                        continue
                    save_ccf_fn(ccf, dat, clat, clng, cfg, averaged=False)
                    ccfs[ipatch] += ccf
                    nums[ipatch] += 1

                if _MPI_AVAILABLE:
                    comm.Barrier()

            except RuntimeError as e:
                if rank == 0:
                    log.error('%s: %s', dat.date(), e)
                continue

        # ── Average CCFs and extract velocities ───────────────────────
        with np.errstate(invalid='ignore'):
            ccfs_avg = np.where(
                nums[:, None, None] > 0,
                ccfs / nums[:, None, None],
                np.nan)

        ux_local = np.full(hi - lo, np.nan)
        uy_local = np.full(hi - lo, np.nan)

        for ixy, (ccf, (clng, clat)) in enumerate(zip(ccfs_avg, xylist[lo:hi])):
            save_ccf_fn(ccf, dat, clat, clng, cfg, averaged=True)
            if np.isnan(ccf).any():
                continue
            _, _, ux_local[ixy], uy_local[ixy] = get_flow_velocity(
                ccf,
                patch_size=cfg.patch_size,
                pixel_size_deg=cfg.pixel_size,
                cadence_interp=cfg.cadence_interp,
                R_sun_Mm=cfg.R_sun_Mm,
                grid_len=cfg.grid_len,
                ntry=cfg.ntry_fit,
            )

        # ── MPI gather and write ──────────────────────────────────────
        if _MPI_AVAILABLE:
            ux_all = np.empty(len(xylist), dtype='f8') if rank == 0 else None
            uy_all = np.empty(len(xylist), dtype='f8') if rank == 0 else None
            comm.Gatherv(ux_local, [ux_all, chunks], root=0)
            comm.Gatherv(uy_local, [uy_all, chunks], root=0)
        else:
            ux_all = ux_local
            uy_all = uy_local

        if rank == 0:
            write_chunk_velocities(utheta_ds, uphi_ds, it,
                                   ux_all, uy_all, ijlist)
            log.critical('Written chunk %d / %d', it + 1, nt)

    # ── Close output ──────────────────────────────────────────────────
    if rank == 0:
        h5file.close()
        log.critical('Closed: %s', outfile)
