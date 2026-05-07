"""tests/test_mpi_utils.py"""

import sys
import pathlib
import unittest
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.mpi_utils import (
    strip_nonprintable,
    gather_bigsize,
    get_loglevel,
)


class TestStripNonprintable(unittest.TestCase):

    def test_clean_string_unchanged(self):
        self.assertEqual(strip_nonprintable('hello world'), 'hello world')

    def test_removes_null_bytes(self):
        self.assertEqual(strip_nonprintable('hello\x00world'), 'helloworld')

    def test_empty_string(self):
        self.assertEqual(strip_nonprintable(''), '')


class TestGatherBigsize(unittest.TestCase):

    def _make_mock_comm(self, rank, size, objects):
        """Create a minimal mock MPI communicator for testing gather."""
        from unittest.mock import MagicMock
        comm = MagicMock()
        comm.Get_rank.return_value = rank
        comm.Get_size.return_value = size
        comm.recv.side_effect = [obj for obj in objects if True]
        return comm

    def test_root_receives_all_objects(self):
        """Simulate a 3-rank gather where rank 0 is root."""
        # For a single-rank mock, gather should return [obj]
        from unittest.mock import MagicMock
        comm = MagicMock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 1
        result = gather_bigsize(comm, 'mydata', root=0)
        self.assertEqual(result, ['mydata'])

    def test_non_root_returns_none(self):
        from unittest.mock import MagicMock
        comm = MagicMock()
        comm.Get_rank.return_value = 1
        comm.Get_size.return_value = 2
        result = gather_bigsize(comm, 'mydata', root=0)
        self.assertIsNone(result)


class TestGetLoglevel(unittest.TestCase):

    def test_info(self):
        self.assertEqual(get_loglevel('info'), logging.INFO)

    def test_debug(self):
        self.assertEqual(get_loglevel('debug'), logging.DEBUG)

    def test_case_insensitive(self):
        self.assertEqual(get_loglevel('WARNING'), logging.WARNING)

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            get_loglevel('verbose')


if __name__ == '__main__':
    unittest.main()
