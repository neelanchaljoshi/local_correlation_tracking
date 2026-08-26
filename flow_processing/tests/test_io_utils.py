"""
tests/test_io_utils.py
------------------------
Tests for utils.io_utils.save_flow_array. Patches PROCESSED_DATA_DIR to
a temp directory for every test so this never touches the real
production .npy output under data/processed_data/.

Run from flow_processing/ (flat imports, same as test_flow_data.py):
    cd flow_processing && python -m pytest tests/test_io_utils.py -v
"""
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from utils import io_utils


class TestSaveFlowArray(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.patcher = mock.patch.object(io_utils, 'PROCESSED_DATA_DIR', str(self.tmp))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_explicit_suffix_used_verbatim(self):
        arr = np.arange(8, dtype='f4').reshape(2, 2, 2)
        io_utils.save_flow_array(arr, 'uphi', 'hmi.ic_45s', suffix='_scratch')
        out_path = self.tmp / 'uphi_hmi_ic_45s_scratch_processed.npy'
        self.assertTrue(out_path.exists())
        np.testing.assert_array_equal(np.load(out_path), arr)

    def test_default_suffix_for_known_legacy_dataset(self):
        arr = np.zeros((1, 1, 1), dtype='f4')
        io_utils.save_flow_array(arr, 'utheta', 'hmi.m_720s')
        expected = self.tmp / 'utheta_hmi_m_720s_dt_1h_processed.npy'
        self.assertTrue(expected.exists())

    def test_unknown_dataset_without_suffix_raises(self):
        arr = np.zeros((1, 1, 1), dtype='f4')
        with self.assertRaises(ValueError):
            io_utils.save_flow_array(arr, 'uphi', 'hmi.some_unknown_series')

    def test_creates_output_dir_if_missing(self):
        nested = self.tmp / 'does' / 'not' / 'exist' / 'yet'
        with mock.patch.object(io_utils, 'PROCESSED_DATA_DIR', str(nested)):
            io_utils.save_flow_array(np.zeros((1, 1, 1), dtype='f4'),
                                      'uphi', 'hmi.ic_45s', suffix='_scratch')
        self.assertTrue(nested.exists())


if __name__ == '__main__':
    unittest.main()
