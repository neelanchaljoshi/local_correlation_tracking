"""
tests/test_settings.py
------------------------
Unit tests for settings.load_config/Config -- pure parsing/path logic,
no DRMS calls, no real data. Run from get_hmi_keys/ (flat imports,
same as the rest of this stage):

    cd get_hmi_keys && python -m pytest tests/ -v
"""
import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from settings import load_config, Config, BASE_KEY_LIST


INI_TEMPLATE = """
[job]
yr_start            = {yr_start}
yr_stop             = {yr_stop}

[series]
seriesname          = {seriesname}
cadence_seconds     = {cadence_seconds}
extra_keys          = {extra_keys}

[data_availability]
data_start          = {data_start}

[quality]
qbits_pass          = {qbits_pass}

[paths]
outdir              = {outdir}
outfile_fmt         = {outfile_fmt}
"""


class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write_ini(self, **overrides):
        defaults = dict(
            yr_start=2018, yr_stop=2018,
            seriesname='hmi.v_45s', cadence_seconds=45,
            extra_keys='',
            data_start='2010-05-01',
            qbits_pass='0x00000000',
            outdir=str(self.tmp / 'out'),
            outfile_fmt='keys-%Y.fits',
        )
        defaults.update(overrides)
        path = self.tmp / 'test.ini'
        path.write_text(INI_TEMPLATE.format(**defaults))
        return path

    def test_basic_load(self):
        cfg = load_config(self._write_ini())
        self.assertEqual(cfg.seriesname, 'hmi.v_45s')
        self.assertEqual(cfg.cadence_seconds, 45)
        self.assertEqual(cfg.yr_start, 2018)
        self.assertEqual(cfg.yr_stop, 2018)
        self.assertEqual(cfg.qbits_pass, 0)
        self.assertEqual(cfg.data_start.isoformat(), '2010-05-01')

    def test_outdir_created_automatically(self):
        outdir = self.tmp / 'brand_new_dir'
        self.assertFalse(outdir.exists())
        load_config(self._write_ini(outdir=str(outdir)))
        self.assertTrue(outdir.exists())

    def test_output_path_uses_strftime_template(self):
        cfg = load_config(self._write_ini(outfile_fmt='keys_new_swan/keys-%Y.fits'))
        path = cfg.output_path(2019)
        self.assertEqual(path.name, 'keys-2019.fits')
        self.assertEqual(path.parent.name, 'keys_new_swan')
        self.assertTrue(path.parent.exists())  # auto-created

    def test_base_key_list_present_by_default(self):
        cfg = load_config(self._write_ini())
        self.assertEqual(cfg.key_list, BASE_KEY_LIST)

    def test_extra_keys_appended(self):
        cfg = load_config(self._write_ini(extra_keys='dsun_obs, car_rot'))
        names = [n for n, _ in cfg.key_list]
        self.assertIn('dsun_obs', names)
        self.assertIn('car_rot', names)
        self.assertEqual(len(cfg.key_list), len(BASE_KEY_LIST) + 2)

    def test_extra_keys_no_duplicate_of_base(self):
        cfg = load_config(self._write_ini(extra_keys='crln_obs, dsun_obs'))
        names = [n for n, _ in cfg.key_list]
        self.assertEqual(names.count('crln_obs'), 1)
        self.assertIn('dsun_obs', names)

    def test_missing_data_start_is_none(self):
        cfg = load_config(self._write_ini(data_start=''))
        self.assertIsNone(cfg.data_start)

    def test_qbits_pass_hex_and_decimal(self):
        cfg_hex = load_config(self._write_ini(qbits_pass='0xFF'))
        cfg_dec = load_config(self._write_ini(qbits_pass='255'))
        self.assertEqual(cfg_hex.qbits_pass, 255)
        self.assertEqual(cfg_dec.qbits_pass, 255)

    def test_yr_stop_before_yr_start_raises(self):
        with self.assertRaises(ValueError):
            load_config(self._write_ini(yr_start=2020, yr_stop=2019))

    def test_missing_required_field_raises(self):
        path = self.tmp / 'incomplete.ini'
        path.write_text('[job]\nyr_start = 2018\n')
        with self.assertRaises(ValueError):
            load_config(path)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_config(self.tmp / 'does_not_exist.ini')


if __name__ == '__main__':
    unittest.main()
