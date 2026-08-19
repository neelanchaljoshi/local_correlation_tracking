"""
pipeline_chunk.py
------------------
Single-chunk, non-MPI processing for the LCT pipeline.

Each call handles exactly one dspan-length time chunk (e.g. one day if
dspan_hours=24, one hour if dspan_hours=1) for every patch in the grid,
sequentially in a single process. Meant to be driven by a SLURM array
where each array task is one independent chunk — embarrassingly
parallel over time instead of MPI-parallel over space. No inter-task
communication of any kind.

Reuses transform_to_fourier-adjacent per-patch physics (_process_patch)
and month/chunk time bookkeeping directly from pipeline.py so the two
entrypoints can't drift apart on that logic.
"""
from __future__ import annotations
import logging
import pathlib
from datetime import datetime
from typing import Optional

import numpy as np

from .config import Config
from .geometry import build_psfs, simulate_pmi_from_hmi
from .io import (
    load_keys_table,
    read_fits_pair,
    read_fits_quad,
    create_output_hdf5,
    write_chunk_velocities,
    save_ccf as save_ccf_fn,
)
from .lct import build_tukey_kernel, get_flow_velocity
from .mpi_utils import setup_logging
from .pipeline import _month_bounds, _count_chunks, _iter_times, _process_patch

logger = logging.getLogger(__name__)


def resolve_chunk_bounds(
    cfg: Config, year: int, month: int, chunk_index: int,
) -> Optional[tuple[datetime, datetime]]:
    """
    Return (dstart_chunk, dstop_chunk) for the chunk_index-th dspan
    window within the given month, or None if chunk_index is out of
    range for that month (e.g. a 30-day month submitted with
    --array=1-31).

    chunk_index is 0-indexed.
    """
    dstart, dstop = _month_bounds(year, month, cfg)
    return _chunk_bounds_in_range(dstart, dstop, cfg.dspan, chunk_index)


def resolve_range_chunk_bounds(
    cfg: Config, chunk_index: int,
) -> Optional[tuple[datetime, datetime]]:
    """
    Return (dstart_chunk, dstop_chunk) for the chunk_index-th dspan
    window within cfg.range_start/cfg.range_end, or None if chunk_index
    is out of range for that window.

    chunk_index is 0-indexed.

    Raises
    ------
    ValueError  if cfg.range_start/cfg.range_end are not both set
                (load_config already rejects only one being set, and
                a range spanning more than one year, so the only way
                to hit this is calling run_chunk_range on a config that
                never set a range at all).
    """
    if not cfg.has_range:
        raise ValueError(
            'range_start/range_end are not set in [job] — set both to '
            'use the explicit-range chunk pipeline, or use run_chunk() '
            'with a year/month instead')
    return _chunk_bounds_in_range(cfg.range_start, cfg.range_end, cfg.dspan,
                                   chunk_index)


def _chunk_bounds_in_range(
    dstart: datetime, dstop: datetime, dspan, chunk_index: int,
) -> Optional[tuple[datetime, datetime]]:
    """Shared bounds arithmetic for resolve_chunk_bounds/resolve_range_chunk_bounds."""
    nt = _count_chunks(dstart, dstop, dspan)
    if not (0 <= chunk_index < nt):
        return None
    dstart_chunk = dstart + chunk_index * dspan
    dstop_chunk  = min(dstart_chunk + dspan, dstop)
    return dstart_chunk, dstop_chunk


def chunk_output_filename(cfg: Config, dstart_chunk: datetime) -> pathlib.Path:
    """Return the HDF5 output path for a single chunk starting at dstart_chunk."""
    seg = 'mag' if cfg.is_magnetic else 'gran'
    dspan_h = int(cfg.dspan.total_seconds() // 3600)
    dstep_m = int(cfg.dstep.total_seconds() // 60)
    fname = (f'{dstart_chunk.strftime("%Y%m%d_%H%M")}_{seg}'
             f'_dspan{dspan_h}h_dstep{dstep_m}m'
             f'_{cfg.resolution_label}_chunk.hdf5')
    return cfg.rootdir_out / fname


# ── Main run functions ────────────────────────────────────────────────────

def run_chunk(cfg: Config, year: int, month: int, chunk_index: int,
              loglevel: int) -> None:
    """
    Run the LCT pipeline for a single time chunk within a calendar
    month, no MPI.

    Parameters
    ----------
    cfg         : validated Config object
    year        : year to process
    month       : month to process (1-12)
    chunk_index : 0-indexed chunk within the month (one dspan window)
    loglevel    : logging level
    """
    log = setup_logging(rank=0, loglevel=loglevel)

    if not cfg.validate_month(year, month):
        log.warning('Exiting cleanly — no data for %d-%02d', year, month)
        return

    bounds = resolve_chunk_bounds(cfg, year, month, chunk_index)
    if bounds is None:
        dstart, dstop = _month_bounds(year, month, cfg)
        nt = _count_chunks(dstart, dstop, cfg.dspan)
        log.warning('Chunk index %d out of range [0, %d) for %d-%02d — '
                    'nothing to do', chunk_index, nt, year, month)
        return
    dstart_chunk, dstop_chunk = bounds

    _run_chunk_body(cfg, year, dstart_chunk, dstop_chunk, log)


def run_chunk_range(cfg: Config, chunk_index: int, loglevel: int) -> None:
    """
    Run the LCT pipeline for a single time chunk within cfg's explicit
    range_start/range_end, no MPI. The year is inferred from
    range_start (load_config already guarantees range_start and
    range_end fall within the same year).

    Parameters
    ----------
    cfg         : validated Config object with range_start/range_end set
    chunk_index : 0-indexed chunk within [range_start, range_end)
    loglevel    : logging level
    """
    log = setup_logging(rank=0, loglevel=loglevel)

    if not cfg.has_range:
        raise ValueError(
            'range_start/range_end are not set in [job] — set both to '
            'use run_chunk_range(), or call run_chunk() with a year/month')

    year = cfg.range_start.year
    if not cfg.validate_range(cfg.range_start):
        log.warning('Exiting cleanly — no data for range starting %s',
                    cfg.range_start)
        return

    bounds = resolve_range_chunk_bounds(cfg, chunk_index)
    if bounds is None:
        nt = _count_chunks(cfg.range_start, cfg.range_end, cfg.dspan)
        log.warning('Chunk index %d out of range [0, %d) for %s - %s — '
                    'nothing to do', chunk_index, nt,
                    cfg.range_start, cfg.range_end)
        return
    dstart_chunk, dstop_chunk = bounds

    _run_chunk_body(cfg, year, dstart_chunk, dstop_chunk, log)


def _run_chunk_body(cfg: Config, year: int, dstart_chunk: datetime,
                     dstop_chunk: datetime, log: logging.Logger) -> None:
    """
    Shared processing for one resolved (dstart_chunk, dstop_chunk)
    window: FITS reading, the sequential (no-MPI) patch loop, CCF
    averaging, velocity extraction, and HDF5 writing. Used by both
    run_chunk() and run_chunk_range() once each has resolved its own
    chunk bounds and the year to load keys for.
    """
    # ── Startup: PSF and Tukey kernel ──────────────────────────────────
    _, _, psf_rel = build_psfs(cfg)
    kernel        = build_tukey_kernel(cfg.patch_size, cfg.alpha)

    # ── Load keys and patch grid ───────────────────────────────────────
    keys = load_keys_table(year, cfg)
    dat0 = datetime.strptime(keys['t_rec'][0], '%Y.%m.%d_%H:%M:%S_TAI')

    xylist = [(x, y) for y in cfg.clat_arr for x in cfg.clng_arr]
    ijlist = [(i, j) for j in range(len(cfg.clat_arr))
              for i in range(len(cfg.clng_arr))]

    # ── Create output (single-row: this one chunk) ────────────────────
    outfile = chunk_output_filename(cfg, dstart_chunk)
    h5file, utheta_ds, uphi_ds, tstart_ds = create_output_hdf5(outfile, 1, cfg)
    tstart_ds[0] = bytes(dstart_chunk.strftime('%Y.%m.%d_%H:%M:%S'),
                         encoding='utf-8')

    ccfs = np.zeros((len(xylist), cfg.patch_size, cfg.patch_size), dtype='f8')
    nums = np.zeros(len(xylist), dtype='i4')

    # ── Time steps within this chunk ────────────────────────────────────
    dat = dstart_chunk
    for dat in _iter_times(dstart_chunk, dstop_chunk, cfg.dstep):
        try:
            log.info('--- %s ---', dat.strftime('%Y.%m.%d_%H:%M:%S_TAI'))
            ii_f = (dat - dat0).total_seconds() / cfg.cadence_keys
            assert ii_f.is_integer(), f'Non-integer frame index: {ii_f}'
            ii = int(ii_f)

            img3 = img4 = None
            if cfg.interpolate:
                isbad, img1, img2, img3, img4 = read_fits_quad(
                    keys, ii, cfg.njump, cfg.segname, cfg.Ntry,
                    cfg.downsample, psf_rel, simulate_pmi_from_hmi)
            else:
                isbad, img1, img2 = read_fits_pair(
                    keys, ii, cfg.njump, cfg.segname, cfg.Ntry,
                    cfg.downsample, psf_rel, simulate_pmi_from_hmi)

            if isbad:
                continue

            # ── Patch loop (sequential, no MPI) ─────────────────────
            for ipatch, (clng, clat) in enumerate(xylist):
                ccf = _process_patch(
                    img1, img2, clng, clat,
                    keys, ii, dat, kernel, cfg,
                    img3=img3, img4=img4)
                if ccf is None:
                    continue
                save_ccf_fn(ccf, dat, clat, clng, cfg, averaged=False)
                ccfs[ipatch] += ccf
                nums[ipatch] += 1

        except RuntimeError as e:
            log.error('%s: %s', dat.date(), e)
            continue

    # ── Average CCFs and extract velocities ─────────────────────────────
    with np.errstate(invalid='ignore'):
        ccfs_avg = np.where(
            nums[:, None, None] > 0,
            ccfs / nums[:, None, None],
            np.nan)

    ux = np.full(len(xylist), np.nan)
    uy = np.full(len(xylist), np.nan)

    for ixy, (ccf, (clng, clat)) in enumerate(zip(ccfs_avg, xylist)):
        save_ccf_fn(ccf, dat, clat, clng, cfg, averaged=True)
        if np.isnan(ccf).any():
            continue
        _, _, ux[ixy], uy[ixy] = get_flow_velocity(
            ccf,
            patch_size=cfg.patch_size,
            pixel_size_deg=cfg.pixel_size,
            cadence_interp=cfg.cadence_interp,
            R_sun_Mm=cfg.R_sun_Mm,
            grid_len=cfg.grid_len,
            ntry=cfg.ntry_fit,
        )

    write_chunk_velocities(utheta_ds, uphi_ds, 0, ux, uy, ijlist)
    h5file.close()
    log.critical('Closed: %s', outfile)
