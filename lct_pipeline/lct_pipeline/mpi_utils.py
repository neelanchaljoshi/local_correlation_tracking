"""
mpi_utils.py
------------
MPI helpers: rank-safe logging, large gather, and process info.
"""
from __future__ import annotations
import logging
import string
import sys


def strip_nonprintable(s: str) -> str:
    printable = set(string.printable)
    return ''.join(c for c in s if c in printable)


def gather_bigsize(comm, obj, root: int):
    """
    Replacement for comm.gather() that avoids hangs on large communicators
    (> ~30 ranks on some MPI implementations).

    Parameters
    ----------
    comm : MPI.COMM_WORLD
    obj  : any picklable Python object
    root : destination rank

    Returns
    -------
    list of gathered objects (on root), None on other ranks
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    out  = [] if rank == root else None
    for src in range(size):
        if src == root:
            buf = obj
        else:
            if rank == src:
                comm.send(obj, dest=root)
            elif rank == root:
                buf = comm.recv(source=src)
        if rank == root:
            out.append(buf)
    return out


def setup_logging(rank: int, loglevel: int = logging.INFO) -> logging.Logger:
    """
    Configure the root logger. Only rank 0 logs below CRITICAL by default;
    all other ranks log only CRITICAL messages to avoid log spam.

    Parameters
    ----------
    rank     : MPI rank of this process
    loglevel : logging level for rank 0 (e.g. logging.DEBUG)

    Returns
    -------
    logging.Logger
    """
    level = loglevel if rank == 0 else logging.CRITICAL
    logging.basicConfig(
        format='%(levelname)s [rank %(rank_pad)s] %(message)s',
        level=level,
        stream=sys.stdout)

    # Inject rank into all log records
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank_pad = f'{rank:04d}'
        return record
    logging.setLogRecordFactory(record_factory)

    return logging.getLogger()


def log_mpi_info(comm, logger: logging.Logger) -> None:
    """Log MPI library version and all host names (rank 0 only)."""
    from mpi4py import MPI
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = MPI.Get_processor_name()
    mpi_ver = strip_nonprintable(MPI.Get_library_version())

    msg = f'rank {rank} on {host}'
    all_msgs = gather_bigsize(comm, msg, root=0)
    if rank == 0:
        logger.critical('MPI: %s', mpi_ver)
        logger.critical('Size: %d ranks', size)
        for m in all_msgs:
            logger.debug('%s', m)


def get_loglevel(level_str: str) -> int:
    """Convert a string log level to a logging constant."""
    level = getattr(logging, level_str.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f'Unknown log level: {level_str!r}')
    return level
