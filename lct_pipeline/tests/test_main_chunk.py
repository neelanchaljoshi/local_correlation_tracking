"""
tests/test_main_chunk.py
--------------------------
CLI-level tests for main_chunk.py, run as subprocesses against a temp
.ini config (no real FITS/keys data needed for the paths tested here).

Covers the behavior that motivated this entrypoint: an out-of-range
--chunk (the chunk-granularity analogue of submitting --array=1-25 for
a 12-month year against the original MPI main.py, which crashes with
an uncaught ValueError from datetime(year, 13, 1)) must exit cleanly
here instead of crashing.
"""

import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).parents[1]
MAIN_CHUNK = REPO_ROOT / 'main_chunk.py'

INI_TEMPLATE = """
[job]
yr_start            = 2015
yr_stop             = 2020
dspan_hours         = 24
dstep_minutes       = 45

[instrument]
segname             = continuum.fits
cadence_seconds     = 45
dataset_cadence_seconds = 45
NX                  = 4096
NY                  = 4096
Ntry                = 3
downsample          = 0
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
infile_fmt_4k       = /tmp/nonexistent-keys-%Y.fits
infile_fmt_2k       = /tmp/nonexistent-keys-%Y-2k.fits
rootdir_out         = {rootdir}
ccf_dir             = {ccfdir}

[output]
save_ccf            = 0
ccf_lat_threshold   = 0.1
ccf_lng_threshold   = 0.1
"""


class TestMainChunkCLI(unittest.TestCase):

    def setUp(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.ini = tmp / 'cfg.ini'
        self.ini.write_text(
            INI_TEMPLATE.format(rootdir=tmp, ccfdir=tmp / 'ccfs'))

    def _run(self, *args):
        result = subprocess.run(
            [sys.executable, str(MAIN_CHUNK), str(self.ini), *args],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        return result.returncode, result.stdout, result.stderr

    def test_print_nchunks_reports_days_in_june(self):
        code, out, err = self._run('2019', '6', '--print-nchunks')
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), '30')

    def test_print_nchunks_reports_leap_february(self):
        code, out, err = self._run('2020', '2', '--print-nchunks')
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), '29')

    def test_invalid_month_rejected_before_touching_data(self):
        code, out, err = self._run('2019', '13', '--chunk', '1')
        self.assertNotEqual(code, 0)
        self.assertIn('month must be 1-12', err)

    def test_missing_chunk_without_print_flag_is_an_error(self):
        code, out, err = self._run('2019', '6')
        self.assertNotEqual(code, 0)
        self.assertIn('--chunk is required', err)

    def test_out_of_range_chunk_exits_cleanly_not_a_crash(self):
        """
        The direct analogue of the original --array=1-25-for-12-months
        bug, one granularity level down: an over-sized --array for this
        month's chunk count must not raise, just report nothing-to-do.
        """
        code, out, err = self._run('2019', '6', '--chunk', '1000')
        self.assertEqual(code, 0)
        self.assertNotIn('Traceback', err)
        self.assertIn('out of range', out)

    def test_negative_effective_chunk_index_exits_cleanly(self):
        # --chunk 0 -> chunk_index = -1 internally (1-indexed CLI)
        code, out, err = self._run('2019', '6', '--chunk', '0')
        self.assertEqual(code, 0)
        self.assertNotIn('Traceback', err)
        self.assertIn('out of range', out)

    def test_valid_chunk_proceeds_past_range_check_to_real_data_loading(self):
        """
        A valid --chunk should NOT be rejected by the range check; it
        should get as far as trying to load the (here, nonexistent)
        keys file and fail there with a clear error, not silently do
        nothing and not fail inside the range-checking logic itself.
        """
        code, out, err = self._run('2019', '6', '--chunk', '1')
        self.assertNotEqual(code, 0)
        self.assertNotIn('out of range', out)
        self.assertIn('Keys file not found', err)

    def test_help_runs_without_error(self):
        result = subprocess.run(
            [sys.executable, str(MAIN_CHUNK), '--help'],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        self.assertEqual(result.returncode, 0)
        self.assertIn('--chunk', result.stdout)
        self.assertIn('--print-nchunks', result.stdout)


if __name__ == '__main__':
    unittest.main()
