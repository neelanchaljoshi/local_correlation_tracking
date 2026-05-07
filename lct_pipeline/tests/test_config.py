"""tests/test_config.py — Config loading and validation."""

import pathlib
import sys
import tempfile
import unittest
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.config import load_config, DATA_AVAILABLE_FROM


VALID_INI = """
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
infile_fmt_4k       = /tmp/keys-%Y.fits
infile_fmt_2k       = /tmp/keys-%Y-2k.fits
rootdir_out         = {rootdir}
ccf_dir             = {ccfdir}

[output]
save_ccf            = 1
ccf_lat_threshold   = 0.1
ccf_lng_threshold   = 0.1
"""


def _write_ini(content: str) -> pathlib.Path:
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False)
    f.write(content)
    f.flush()
    return pathlib.Path(f.name)


class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        self.tmp  = pathlib.Path(tempfile.mkdtemp())
        self.ccf  = self.tmp / 'ccfs'
        self.ini  = _write_ini(
            VALID_INI.format(rootdir=self.tmp, ccfdir=self.ccf))

    def test_loads_successfully(self):
        cfg = load_config(self.ini)
        self.assertEqual(cfg.yr_start, 2015)
        self.assertEqual(cfg.yr_stop, 2020)

    def test_derived_cadence_no_interp(self):
        cfg = load_config(self.ini)
        self.assertEqual(cfg.cadence_interp, 45)
        self.assertEqual(cfg.njump, 1)

    def test_patch_size_4k(self):
        cfg = load_config(self.ini)
        self.assertFalse(cfg.downsample)
        self.assertEqual(cfg.patch_size, 68)
        self.assertAlmostEqual(cfg.pixel_size, 0.03)

    def test_patch_size_2k(self):
        ini = _write_ini(
            VALID_INI.format(rootdir=self.tmp, ccfdir=self.ccf)
            .replace('downsample          = 0', 'downsample          = 1'))
        cfg = load_config(ini)
        self.assertTrue(cfg.downsample)
        self.assertEqual(cfg.patch_size, 34)
        self.assertAlmostEqual(cfg.pixel_size, 0.06)

    def test_grid_arrays_correct_length(self):
        cfg = load_config(self.ini)
        self.assertEqual(len(cfg.clat_arr), 11)   # -5 to 5 step 1
        self.assertEqual(len(cfg.clng_arr), 11)

    def test_is_magnetic_false_for_continuum(self):
        cfg = load_config(self.ini)
        self.assertFalse(cfg.is_magnetic)

    def test_is_magnetic_true_for_magnetogram(self):
        ini = _write_ini(
            VALID_INI.format(rootdir=self.tmp, ccfdir=self.ccf)
            .replace('continuum.fits', 'magnetogram.fits'))
        cfg = load_config(ini)
        self.assertTrue(cfg.is_magnetic)

    def test_output_filename_format(self):
        cfg  = load_config(self.ini)
        name = cfg.output_filename(2019, 6).name
        self.assertIn('2019_06', name)
        self.assertIn('gran', name)
        self.assertIn('4k', name)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config('/nonexistent/path/config.ini')

    def test_missing_section_raises(self):
        bad = _write_ini('[job]\nyr_start=2015\n')
        with self.assertRaises(ValueError):
            load_config(bad)

    def test_yr_start_gt_yr_stop_raises(self):
        ini = _write_ini(
            VALID_INI.format(rootdir=self.tmp, ccfdir=self.ccf)
            .replace('yr_start            = 2015', 'yr_start            = 2025')
            .replace('yr_stop             = 2020', 'yr_stop             = 2015'))
        with self.assertRaises(ValueError):
            load_config(ini)

    def test_even_grid_len_raises(self):
        ini = _write_ini(
            VALID_INI.format(rootdir=self.tmp, ccfdir=self.ccf)
            .replace('grid_len            = 5', 'grid_len            = 4'))
        with self.assertRaises(ValueError):
            load_config(ini)

    def test_save_ccf_without_ccf_dir_raises(self):
        ini = _write_ini(
            VALID_INI.format(rootdir=self.tmp, ccfdir='')
            .replace('save_ccf            = 1', 'save_ccf            = 1'))
        with self.assertRaises(ValueError):
            load_config(ini)


class TestValidateMonth(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.ccf = self.tmp / 'ccfs'
        self.cfg = load_config(
            _write_ini(VALID_INI.format(rootdir=self.tmp, ccfdir=self.ccf)))

    def test_valid_month_returns_true(self):
        self.assertTrue(self.cfg.validate_month(2015, 6))

    def test_before_may_2010_returns_false_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = self.cfg.validate_month(2010, 3)
        self.assertFalse(result)
        self.assertTrue(any(issubclass(x.category, UserWarning) for x in w))

    def test_may_2010_is_valid(self):
        self.assertTrue(self.cfg.validate_month(2010, 5))

    def test_april_2010_is_invalid(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            result = self.cfg.validate_month(2010, 4)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
