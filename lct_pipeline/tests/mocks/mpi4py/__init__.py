"""Mock mpi4py for CI environments without MPI."""

class _Comm:
    def Get_rank(self): return 0
    def Get_size(self): return 1
    def bcast(self, obj, root=0): return obj
    def send(self, obj, dest): pass
    def recv(self, source): return None
    def Barrier(self): pass
    def Gatherv(self, sendbuf, recvbuf, root=0): pass
    def gather(self, obj, root=0): return [obj]

class _MPI:
    COMM_WORLD = _Comm()
    @staticmethod
    def Get_version(): return (3, 1)
    @staticmethod
    def Get_library_version(): return 'Mock MPI'
    @staticmethod
    def Get_processor_name(): return 'localhost'

MPI = _MPI()
