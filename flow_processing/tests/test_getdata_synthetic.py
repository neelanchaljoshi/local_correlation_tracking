"""
tests/test_getdata_synthetic.py
--------------------------------
Fast, side-effect-free tests for FlowData.getdata()'s glob-based
concatenation, using tiny synthetic HDF5 fixtures instead of the real
15-years-of-data files test_flow_data.py needs. Covers:
  - the current lct_pipeline's per-month layout (pipeline.py)
  - the current lct_pipeline's per-chunk layout (pipeline_chunk.py)
  - out-of-order files still concatenate in time order
  - mismatched lat/lon grids across files raise a clear error
  - no matching files raises FileNotFoundError
  - an unknown which_data with no explicit pattern raises ValueError

Run from flow_processing/ (flat imports, same as test_flow_data.py):
    cd flow_processing && python -m pytest tests/test_getdata_synthetic.py -v
"""
import pathlib
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from flow_data import FlowData


def _write_hdf5(path, tstarts, nlat=3, nlng=3, seed=0):
    rng = np.random.default_rng(seed)
    nt = len(tstarts)
    lat = np.linspace(-90, 90, nlat)
    lon = np.linspace(-90, 90, nlng)
    with h5py.File(path, 'w') as f:
        f.create_dataset('tstart', data=[t.encode('utf-8') for t in tstarts])
        f.create_dataset('uphi', data=rng.normal(size=(nt, nlat, nlng)).astype('f4'))
        f.create_dataset('utheta', data=rng.normal(size=(nt, nlat, nlng)).astype('f4'))
        f.create_dataset('latitude', data=lat)
        f.create_dataset('longitude', data=lon)


class TestGetdataSyntheticLayouts(unittest.TestCase):

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_month_mode_layout(self):
        """One file per month (pipeline.py's output_filename convention)."""
        _write_hdf5(self.tmp / '2019_01_gran_dspan24h_dstep30m_4k.hdf5',
                    ['2019.01.01_00:00:00', '2019.01.01_00:30:00'])
        _write_hdf5(self.tmp / '2019_02_gran_dspan24h_dstep30m_4k.hdf5',
                    ['2019.02.01_00:00:00', '2019.02.01_00:30:00'])

        flow = FlowData('uphi', 'hmi.ic_45s')
        flow.getdata(data_root=str(self.tmp), pattern='*_gran_dspan*_4k.hdf5')

        self.assertEqual(flow.nt, 4)
        self.assertEqual((flow.nlat, flow.nlng), (3, 3))
        self.assertTrue((flow.t_array[:-1] <= flow.t_array[1:]).all())

    def test_chunk_mode_layout(self):
        """One file per chunk, one row each (pipeline_chunk.py's chunk_output_filename)."""
        _write_hdf5(self.tmp / '20190101_0000_mag_dspan6h_dstep120m_4k_chunk.hdf5',
                    ['2019.01.01_00:00:00'])
        _write_hdf5(self.tmp / '20190102_0000_mag_dspan6h_dstep120m_4k_chunk.hdf5',
                    ['2019.01.02_00:00:00'])
        _write_hdf5(self.tmp / '20190103_0000_mag_dspan6h_dstep120m_4k_chunk.hdf5',
                    ['2019.01.03_00:00:00'])

        flow = FlowData('uphi', 'hmi.m_720s')
        flow.getdata(data_root=str(self.tmp),
                      pattern='*_mag_dspan*_4k_chunk.hdf5')

        self.assertEqual(flow.nt, 3)

    def test_out_of_order_files_sorted_by_time(self):
        # Filenames deliberately out of chronological order.
        _write_hdf5(self.tmp / 'b_later.hdf5', ['2019.03.01_00:00:00'])
        _write_hdf5(self.tmp / 'a_earlier.hdf5', ['2019.01.01_00:00:00'])
        _write_hdf5(self.tmp / 'c_middle.hdf5', ['2019.02.01_00:00:00'])

        flow = FlowData('uphi', 'hmi.ic_45s')
        flow.getdata(data_root=str(self.tmp), pattern='*.hdf5')

        self.assertTrue((flow.t_array[:-1] <= flow.t_array[1:]).all())
        self.assertAlmostEqual(flow.t_array[0], 2019.0, delta=0.05)
        self.assertAlmostEqual(flow.t_array[-1], 2019.16, delta=0.05)

    def test_mismatched_grid_raises(self):
        _write_hdf5(self.tmp / 'a.hdf5', ['2019.01.01_00:00:00'], nlat=3, nlng=3)
        _write_hdf5(self.tmp / 'b.hdf5', ['2019.02.01_00:00:00'], nlat=5, nlng=5)

        flow = FlowData('uphi', 'hmi.ic_45s')
        with self.assertRaises(ValueError):
            flow.getdata(data_root=str(self.tmp), pattern='*.hdf5')

    def test_no_matching_files_raises(self):
        flow = FlowData('uphi', 'hmi.ic_45s')
        with self.assertRaises(FileNotFoundError):
            flow.getdata(data_root=str(self.tmp), pattern='*.hdf5')

    def test_unknown_which_data_without_pattern_raises(self):
        flow = FlowData('uphi', 'hmi.some_unknown_series')
        with self.assertRaises(ValueError):
            flow.getdata(data_root=str(self.tmp))


if __name__ == '__main__':
    unittest.main()
