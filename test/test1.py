#Created a file to check git push
import unittest
import sys
sys.path.append("..")
from base.tukey_window import tukey_twoD

class TestModules(unittest.TestCase):

	def test_tukey(self):
		tukey = tukey_twoD(168, 0.6)
		self.assertEqual(tukey[84, 84], 1, 'Not 1 at the highest point')
		




if __name__ == '__main__':
	unittest.main()