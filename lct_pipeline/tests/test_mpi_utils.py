"""tests/test_mpi_utils.py"""

import sys
import pathlib
import unittest
import logging

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from lct_pipeline.mpi_utils import (
    strip_nonprintable,
    gather_bigsize,
    log_mpi_info,
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

    def test_root_collects_from_a_real_sender(self):
        """
        The one branch the two tests above don't reach: root (rank ==
        root) receiving from an actual non-root source via comm.recv,
        not just seeing its own local object.
        """
        from unittest.mock import MagicMock
        comm = MagicMock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 2
        comm.recv.return_value = 'from_rank_1'
        result = gather_bigsize(comm, 'from_rank_0', root=0)
        self.assertEqual(result, ['from_rank_0', 'from_rank_1'])
        comm.recv.assert_called_once_with(source=1)


class TestLogMpiInfo(unittest.TestCase):

    def test_root_logs_version_size_and_per_rank_messages(self):
        from unittest.mock import MagicMock
        comm = MagicMock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 1
        logger = logging.getLogger('test_log_mpi_info_root')
        with self.assertLogs(logger, level='DEBUG') as cm:
            log_mpi_info(comm, logger)
        joined = '\n'.join(cm.output)
        self.assertIn('Mock MPI', joined)
        self.assertIn('Size: 1 ranks', joined)
        self.assertIn('rank 0 on localhost', joined)

    def test_non_root_logs_nothing(self):
        from unittest.mock import MagicMock
        comm = MagicMock()
        comm.Get_rank.return_value = 1
        comm.Get_size.return_value = 2
        logger = logging.getLogger('test_log_mpi_info_nonroot')
        with self.assertRaises(AssertionError):
            # assertLogs itself raises AssertionError if nothing at all
            # was logged at/above the given level within the block —
            # exactly the expected behavior for a non-root rank.
            with self.assertLogs(logger, level='DEBUG'):
                log_mpi_info(comm, logger)


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
