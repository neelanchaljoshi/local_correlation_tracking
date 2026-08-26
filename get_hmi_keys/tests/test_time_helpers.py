"""
tests/test_time_helpers.py
----------------------------
Unit tests for utils.time_helpers.get_start_stop.
"""
import pathlib
import sys
import unittest
from datetime import date, datetime

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from utils.time_helpers import get_start_stop


class TestGetStartStop(unittest.TestCase):

    def test_full_year_when_no_data_start(self):
        dstart, dstop = get_start_stop(2019, data_start=None)
        self.assertEqual(dstart, datetime(2019, 1, 1))
        self.assertEqual(dstop, datetime(2020, 1, 1))

    def test_full_year_when_data_start_is_different_year(self):
        dstart, dstop = get_start_stop(2019, data_start=date(2010, 5, 1))
        self.assertEqual(dstart, datetime(2019, 1, 1))
        self.assertEqual(dstop, datetime(2020, 1, 1))

    def test_data_start_applies_in_matching_year(self):
        dstart, dstop = get_start_stop(2010, data_start=date(2010, 5, 1))
        self.assertEqual(dstart, datetime(2010, 5, 1))
        self.assertEqual(dstop, datetime(2011, 1, 1))


if __name__ == '__main__':
    unittest.main()
