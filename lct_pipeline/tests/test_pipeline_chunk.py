"""
tests/test_pipeline_chunk.py
-----------------------------
Tests for lct_pipeline.pipeline_chunk — the non-MPI, one-chunk-per-task
pipeline entrypoint.

Covers the two pure functions that carry all the new logic:
  - resolve_chunk_bounds: maps a chunk index to a time window, and is
    the thing standing between "submitted --array=1-25 for a 12-month
    year" (main.py's crash on out-of-range month) and this design's
    intended behavior of exiting cleanly instead.
  - chunk_output_filename: naming for the per-chunk output file.

run_chunk() itself (FITS I/O, patch loop, HDF5 writing) is orchestration
over the same physics already covered by test_lct.py/test_geometry.py
and reuses pipeline.py's _process_patch directly (untested here, same
as pipeline.run() itself has no dedicated test — both are I/O-heavy
orchestration, not pure logic).
"""

import pathlib
import sys
import tempfile
import unittest
import warnings
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.config import load_config
from lct_pipeline.pipeline import _month_bounds, _count_chunks
from lct_pipeline.pipeline_chunk import (
    resolve_chunk_bounds,
    resolve_range_chunk_bounds,
    chunk_output_filename,
    run_chunk_range,
)


INI_TEMPLATE = """
[job]
yr_start            = 2015
yr_stop             = 2020
dspan_hours         = {dspan}
dstep_minutes       = 45
{range_lines}

[instrument]
segname             = {segname}
cadence_seconds     = 45
dataset_cadence_seconds = 45
NX                  = 4096
NY                  = 4096
Ntry                = 3
downsample          = {downsample}
interpolate         = 0

[psf]
wavelength_nm       = 617.3
aperture_hmi_m      = 0.14
aperture_pmi_m      = 0.075
pixel_scale_arcsec  = 0.5
psf_size            = 64

[grid]
clat_start          = -5.0
clat_stop           = 5.0
clat_step           = 1.0
clng_start          = -5.0
clng_stop           = 5.0
clng_step           = 1.0

[lct]
patch_size_4k       = 68
patch_size_2k       = 34
pixel_size_4k       = 0.03
pixel_size_2k       = 0.06
alpha               = 0.8
R_sun_Mm            = 695.7
grid_len            = 5
ntry_fit            = 4

[tracking]
change_track        = true
A                   = 14.034
B                   = -1.702
C                   = -2.494
CRrate              = 14.184

[b0_correction]
dI                  = -0.08
t_ref_b0            = 2010-06-07T14:17:20

[paths]
infile_fmt_4k       = /tmp/keys-%Y.fits
infile_fmt_2k       = /tmp/keys-%Y-2k.fits
rootdir_out         = {{rootdir}}
ccf_dir             = {{ccfdir}}

[output]
save_ccf            = 1
ccf_lat_threshold   = 0.1
ccf_lng_threshold   = 0.1
"""


def make_cfg(dspan=24, segname='continuum.fits', downsample=0,
             range_start=None, range_end=None):
    """Write a temp .ini with the given knobs and return the loaded Config."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    ccf = tmp / 'ccfs'
    range_lines = ''
    if range_start and range_end:
        range_lines = (f'range_start         = {range_start}\n'
                       f'range_end           = {range_end}\n')
    ini_text = INI_TEMPLATE.format(
        dspan=dspan, segname=segname, downsample=downsample,
        range_lines=range_lines,
    ).format(rootdir=tmp, ccfdir=ccf)
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    f.write(ini_text)
    f.flush()
    return load_config(f.name)


class TestResolveChunkBoundsDaily(unittest.TestCase):
    """dspan_hours=24 -> one chunk per calendar day."""

    def setUp(self):
        self.cfg = make_cfg(dspan=24)

    def test_chunk_count_matches_days_in_month(self):
        dstart, dstop = _month_bounds(2019, 6, self.cfg)
        nt = _count_chunks(dstart, dstop, self.cfg.dspan)
        self.assertEqual(nt, 30)

    def test_first_chunk_is_first_day(self):
        dstart, dstop = resolve_chunk_bounds(self.cfg, 2019, 6, 0)
        self.assertEqual(dstart, datetime(2019, 6, 1, 0, 0, 0))
        self.assertEqual(dstop, datetime(2019, 6, 2, 0, 0, 0))

    def test_last_chunk_is_last_day_clipped_to_month_end(self):
        # June has 30 days -> valid chunk indices are 0..29
        dstart, dstop = resolve_chunk_bounds(self.cfg, 2019, 6, 29)
        self.assertEqual(dstart, datetime(2019, 6, 30, 0, 0, 0))
        # Clipped to the month boundary, not a full 24h past dstart
        self.assertEqual(dstop, datetime(2019, 6, 30, 23, 59, 59))

    def test_middle_chunks_are_contiguous(self):
        _, stop_14 = resolve_chunk_bounds(self.cfg, 2019, 6, 14)
        start_15, _ = resolve_chunk_bounds(self.cfg, 2019, 6, 15)
        self.assertEqual(stop_14, start_15)

    def test_negative_index_is_out_of_range(self):
        self.assertIsNone(resolve_chunk_bounds(self.cfg, 2019, 6, -1))

    def test_index_equal_to_count_is_out_of_range(self):
        self.assertIsNone(resolve_chunk_bounds(self.cfg, 2019, 6, 30))

    def test_index_far_out_of_range_does_not_raise(self):
        """
        The chunk-level analogue of submitting --array=1-25 for a
        12-month year: an over-sized --array must not crash, only
        report nothing-to-do. This is the whole point of
        resolve_chunk_bounds returning Optional instead of raising.
        """
        self.assertIsNone(resolve_chunk_bounds(self.cfg, 2019, 6, 1000))

    def test_leap_year_february_has_29_chunks(self):
        dstart, dstop = _month_bounds(2020, 2, self.cfg)
        self.assertEqual(_count_chunks(dstart, dstop, self.cfg.dspan), 29)
        self.assertIsNotNone(resolve_chunk_bounds(self.cfg, 2020, 2, 28))
        self.assertIsNone(resolve_chunk_bounds(self.cfg, 2020, 2, 29))

    def test_non_leap_year_february_has_28_chunks(self):
        dstart, dstop = _month_bounds(2021, 2, self.cfg)
        self.assertEqual(_count_chunks(dstart, dstop, self.cfg.dspan), 28)
        self.assertIsNotNone(resolve_chunk_bounds(self.cfg, 2021, 2, 27))
        self.assertIsNone(resolve_chunk_bounds(self.cfg, 2021, 2, 28))


class TestResolveChunkBoundsHourly(unittest.TestCase):
    """dspan_hours=1 -> one chunk per hour; a different granularity entirely."""

    def setUp(self):
        self.cfg = make_cfg(dspan=1)

    def test_chunk_count_matches_hours_in_month(self):
        dstart, dstop = _month_bounds(2019, 6, self.cfg)
        nt = _count_chunks(dstart, dstop, self.cfg.dspan)
        self.assertEqual(nt, 30 * 24)

    def test_first_hourly_chunk(self):
        dstart, dstop = resolve_chunk_bounds(self.cfg, 2019, 6, 0)
        self.assertEqual(dstart, datetime(2019, 6, 1, 0, 0, 0))
        self.assertEqual(dstop, datetime(2019, 6, 1, 1, 0, 0))

    def test_last_hourly_chunk_clipped_to_month_end(self):
        last_index = 30 * 24 - 1
        dstart, dstop = resolve_chunk_bounds(self.cfg, 2019, 6, last_index)
        self.assertEqual(dstart, datetime(2019, 6, 30, 23, 0, 0))
        self.assertEqual(dstop, datetime(2019, 6, 30, 23, 59, 59))

    def test_one_past_last_hourly_chunk_is_out_of_range(self):
        self.assertIsNone(resolve_chunk_bounds(self.cfg, 2019, 6, 30 * 24))


class TestChunkOutputFilename(unittest.TestCase):

    def setUp(self):
        self.cfg = make_cfg()

    def test_daily_filename_pattern(self):
        path = chunk_output_filename(self.cfg, datetime(2019, 6, 15))
        self.assertEqual(path.parent, self.cfg.rootdir_out)
        self.assertEqual(
            path.name,
            '20190615_0000_gran_dspan24h_dstep45m_4k_chunk.hdf5')

    def test_hourly_filename_includes_hour_and_minute(self):
        cfg = make_cfg(dspan=1)
        path = chunk_output_filename(cfg, datetime(2019, 6, 15, 7))
        self.assertEqual(
            path.name,
            '20190615_0700_gran_dspan1h_dstep45m_4k_chunk.hdf5')

    def test_magnetic_config_uses_mag_segment_label(self):
        cfg = make_cfg(segname='magnetogram.fits')
        path = chunk_output_filename(cfg, datetime(2019, 6, 15))
        self.assertIn('_mag_', path.name)
        self.assertNotIn('_gran_', path.name)

    def test_downsampled_config_uses_2k_resolution_label(self):
        cfg = make_cfg(downsample=1)
        path = chunk_output_filename(cfg, datetime(2019, 6, 15))
        self.assertIn('_2k_', path.name)

    def test_full_resolution_config_uses_4k_resolution_label(self):
        path = chunk_output_filename(self.cfg, datetime(2019, 6, 15))
        self.assertIn('_4k_', path.name)

    def test_distinct_chunk_starts_get_distinct_filenames(self):
        p1 = chunk_output_filename(self.cfg, datetime(2019, 6, 15))
        p2 = chunk_output_filename(self.cfg, datetime(2019, 6, 16))
        self.assertNotEqual(p1, p2)

    def test_filename_is_marked_as_a_chunk_file(self):
        path = chunk_output_filename(self.cfg, datetime(2019, 6, 15))
        self.assertTrue(path.stem.endswith('_chunk'))

    def test_filename_distinguishable_from_monthly_pipeline_output(self):
        """
        Chunk files and the MPI pipeline's monthly files
        (Config.output_filename) may land in the same rootdir_out —
        they must never collide.
        """
        monthly = self.cfg.output_filename(2019, 6)
        chunk = chunk_output_filename(self.cfg, datetime(2019, 6, 15))
        self.assertNotEqual(monthly.name, chunk.name)


class TestResolveRangeChunkBounds(unittest.TestCase):
    """
    The motivating use case: an explicit range_start/range_end in the
    config lets --array=1-N map directly onto that range with no
    day-offset arithmetic — unlike month mode, where hour-of-day N of a
    month requires computing (day-1)*24 + N by hand.
    """

    def test_one_day_hourly_range_has_24_chunks(self):
        cfg = make_cfg(dspan=1, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-16T00:00:00')
        nt = _count_chunks(cfg.range_start, cfg.range_end, cfg.dspan)
        self.assertEqual(nt, 24)

    def test_chunk_index_maps_directly_to_hour_of_day_no_offset_needed(self):
        cfg = make_cfg(dspan=1, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-16T00:00:00')
        for hour in range(24):
            dstart, dstop = resolve_range_chunk_bounds(cfg, hour)
            self.assertEqual(dstart, datetime(2019, 6, 15, hour))
            self.assertEqual(dstop, dstart + cfg.dspan)

    def test_negative_index_is_out_of_range(self):
        cfg = make_cfg(dspan=1, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-16T00:00:00')
        self.assertIsNone(resolve_range_chunk_bounds(cfg, -1))

    def test_index_equal_to_count_is_out_of_range(self):
        cfg = make_cfg(dspan=1, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-16T00:00:00')
        self.assertIsNone(resolve_range_chunk_bounds(cfg, 24))

    def test_index_far_out_of_range_does_not_raise(self):
        cfg = make_cfg(dspan=1, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-16T00:00:00')
        self.assertIsNone(resolve_range_chunk_bounds(cfg, 1000))

    def test_last_chunk_clipped_when_dspan_does_not_evenly_divide_range(self):
        # 5h range, 2h dspan -> ceil(5/2) = 3 chunks; the last is clipped
        cfg = make_cfg(dspan=2, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-15T05:00:00')
        nt = _count_chunks(cfg.range_start, cfg.range_end, cfg.dspan)
        self.assertEqual(nt, 3)
        dstart, dstop = resolve_range_chunk_bounds(cfg, 2)
        self.assertEqual(dstart, datetime(2019, 6, 15, 4))
        self.assertEqual(dstop, datetime(2019, 6, 15, 5))  # clipped, not 06:00

    def test_raises_when_config_has_no_range(self):
        cfg = make_cfg(dspan=1)
        with self.assertRaises(ValueError):
            resolve_range_chunk_bounds(cfg, 0)


class TestRunChunkRangeValidation(unittest.TestCase):
    """
    Error/warning paths of run_chunk_range() that resolve before any
    FITS/keys I/O happens, so they're testable without real data.
    """

    def test_raises_when_config_has_no_range(self):
        cfg = make_cfg(dspan=1)
        with self.assertRaises(ValueError):
            run_chunk_range(cfg, 0, loglevel=30)

    def test_out_of_range_chunk_exits_cleanly_with_warning_not_a_crash(self):
        cfg = make_cfg(dspan=1, range_start='2019-06-15T00:00:00',
                        range_end='2019-06-16T00:00:00')
        with self.assertLogs(level='WARNING') as cm:
            run_chunk_range(cfg, 1000, loglevel=30)
        self.assertTrue(any('out of range' in msg for msg in cm.output))

    def test_range_before_data_available_exits_cleanly_with_warning(self):
        cfg = make_cfg(dspan=1, range_start='2010-01-01T00:00:00',
                        range_end='2010-01-02T00:00:00')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with self.assertLogs(level='WARNING') as cm:
                run_chunk_range(cfg, 0, loglevel=30)
        self.assertTrue(any('no data' in msg.lower() for msg in cm.output))


if __name__ == '__main__':
    unittest.main()
