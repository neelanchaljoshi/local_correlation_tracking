#Created a file to check git push
import unittest
import sys
sys.path.append("..")
from base.tukey_window import tukey_twoD
from base.get_lct_map import get_lct_map
from scipy.ndimage import shift
from scipy import signal
import numpy as np

class TestModules(unittest.TestCase):

    def test_lct(self):
        kern = tukey_twoD(168, 0.6)
        patch1 = tukey_twoD(168, 1)
        patch2 = tukey_twoD(168, 1)
        patch2 = shift(patch2, 1.8, order = 3, mode = 'constant', cval = 0)
        ccf_lct, _, _ = get_lct_map(patch1, patch2, 168, kern)
        ccf_conv = signal.convolve2d(patch1, patch2, mode= 'same', boundary = 'fill', fillvalue = 0)
        diff = np.mean(ccf_lct-ccf_conv)
        print(ccf_lct-ccf_conv)
        print(diff)
        self.assertLessEqual(diff, 0.0001, 'Not close %d'%diff)
        #self.assertEqual(tukey[84, 84], 1, 'Not 1 at the highest point')



if __name__ == '__main__':
	unittest.main()