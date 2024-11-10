import sys


def mpi_init():
    # Initialize MPI with considering exception.
    # from https://groups.google.com/g/mpi4py/c/me2TFzHmmsQ/m/sSF99LE0t9QJ
    from mpi4py import MPI

    # Communicator
    comm = MPI.COMM_WORLD

    # Modify system exception hook to abot mpi for fatal error.
    sys_excepthook = sys.excepthook

    def mpi_excepthook(type, value, traceback):
        sys_excepthook(type, value, traceback)

        if comm.size > 1:
            sys.stderr.flush()
            comm.Abort(1)
        else:
            MPI.Finalize()            

    sys.excepthook = mpi_excepthook

    return comm
