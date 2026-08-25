"""tests/test_io.py"""

import sys
import pathlib
import tempfile
import unittest

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.io import read_fits_image


class TestReadFitsImagePathAnomalies(unittest.TestCase):
    """
    Regression tests for a SUMS-path bug: `show_info -Pq` in
    get_hmi_keys/fetch_keys.py can hand back storage-directory paths
    with a trailing newline (or other whitespace) still attached, e.g.
    '/pfs/scratch/SUMS/SUM0049/D1005681501/S00003\\n'. If that survives
    into the `path` column of keys-<year>.fits, joining it with the
    segment filename produces a directory component with an embedded
    '\\n' (e.g. '.../S00003\\n/magnetogram.fits'), which never resolves
    on disk. `read_fits_image` must tolerate such paths regardless of
    whether the anomaly lives in a freshly-fetched keys table or an
    already-written one.
    """

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.data = np.arange(9, dtype='f4').reshape(3, 3)
        self.segname = 'magnetogram.fits'
        fits.PrimaryHDU(self.data).writeto(self.tmp / self.segname, overwrite=True)

    def test_trailing_newline_is_stripped(self):
        dirty_path = f'{self.tmp}\n'
        img = read_fits_image(dirty_path, self.segname)
        np.testing.assert_array_equal(img, self.data)

    def test_trailing_whitespace_and_newline_is_stripped(self):
        dirty_path = f'{self.tmp}  \n'
        img = read_fits_image(dirty_path, self.segname)
        np.testing.assert_array_equal(img, self.data)

    def test_trailing_slash_is_still_handled(self):
        dirty_path = f'{self.tmp}/'
        img = read_fits_image(dirty_path, self.segname)
        np.testing.assert_array_equal(img, self.data)

    def test_clean_path_still_works(self):
        img = read_fits_image(str(self.tmp), self.segname)
        np.testing.assert_array_equal(img, self.data)

    def test_missing_file_still_raises(self):
        with self.assertRaises(IOError):
            read_fits_image(f'{self.tmp / "does_not_exist"}\n', self.segname)


if __name__ == '__main__':
    unittest.main()
