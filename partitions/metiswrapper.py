# -*- coding: utf-8 -*-
from ctypes import POINTER, c_void_p, c_int, c_int, c_int64
from utils.ctypes import load_lib
import numpy as np

class METISWrapper:
    METIS_NOPTIONS = 40

    def __init__(self):
        # Load metis library
        lib = load_lib('metis')

        # Probe data types
        self._probe_types(lib)

        # Assign Metis functions
        self.METIS_SetDefaultOptions = lib.METIS_SetDefaultOptions
        self.METIS_SetDefaultOptions.argtypes = [c_void_p]

        self.METIS_PartMeshDual = lib.METIS_PartMeshDual
        self.METIS_PartMeshDual.argtypes = [
            POINTER(self.metis_int), POINTER(self.metis_int), c_void_p,
            c_void_p, c_void_p, c_void_p, POINTER(self.metis_int),
            POINTER(self.metis_int), c_void_p, c_void_p,
            POINTER(self.metis_int), c_void_p, c_void_p
        ]

    def _probe_types(self, lib):
        # Find integer type
        lib.METIS_SetDefaultOptions.argtypes = [c_void_p]
        opts = np.arange(0, 40, dtype=np.int64)
        err = lib.METIS_SetDefaultOptions(opts.ctypes)

        if opts[-1] != self.METIS_NOPTIONS - 1:
            self.metis_int = metis_int = c_int64
            self.metis_int_np = metis_int_np = np.int64
        else:
            self.metis_int = metis_int = c_int
            self.metis_int_np = metis_int_np = np.int32

        # Sample of part mesh nodal to find float
        opts = np.arange(0, 40, dtype=metis_int_np)
        eptr = np.array([0, 3, 6, 9], dtype=metis_int_np)
        eind = np.array([0, 1, 2, 1, 2, 3, 2, 3, 4], dtype=metis_int_np)

        nn = metis_int(5)
        ne = metis_int(3)
        ncommon = metis_int(1)
        nparts = metis_int(2)
        objval = metis_int()
        epart = np.zeros(3, dtype=metis_int_np)
        npart = np.zeros(5, dtype=metis_int_np)
        tpwgts = np.ones(2, dtype=np.float32)
        tpwgts /= np.sum(tpwgts)

        lib.METIS_PartMeshDual.argtypes = [
            POINTER(metis_int), POINTER(metis_int), c_void_p, c_void_p,
            c_void_p, c_void_p, POINTER(metis_int), POINTER(metis_int),
            c_void_p, c_void_p, POINTER(metis_int), c_void_p, c_void_p]

        err = lib.METIS_PartMeshDual(
            ne, nn, eptr.ctypes, eind.ctypes, None, None, ncommon, nparts,
            tpwgts.ctypes, None, objval, epart.ctypes, npart.ctypes
        )

        if err == 1:
            self.metis_float_np = metis_float_np = np.float32
        else:
            self.metis_float_np = metis_float_np = np.float64

    def part_mesh(self, nparts, nn, ne, eptr, eind, ncommon=1, vwts=None, opts=None, tpwgts=None):
        # Metis int type
        metis_int, metis_int_np = self.metis_int, self.metis_int_np

        # Convert integer inputs
        _nparts, _nn, _ne = metis_int(nparts), metis_int(nn), metis_int(ne)
        objval = metis_int()

        ncommon = metis_int(ncommon)

        eptr = eptr.astype(metis_int_np)
        eind = eind.astype(metis_int_np)

        # Return array
        epart = np.empty(ne, dtype=metis_int_np)
        npart = np.empty(nn, dtype=metis_int_np)

        if tpwgts is None:
            tpwgts = np.ones(nparts, dtype=self.metis_float_np)
            tpwgts /= np.sum(tpwgts)
        else:
            tpwgts = tpwgts.astype(metis_int_np)

        if opts is None:
            # Initialize default options
            opts = np.empty(40, dtype=metis_int_np)
            err = self.METIS_SetDefaultOptions(opts.ctypes)

        if vwts is not None:
            vwts.astype(metis_int_np)
        else:
            vwts = np.ones(ne, dtype=metis_int_np)

        # Run PartMeshNodal
        err = self.METIS_PartMeshDual(
            _ne, _nn, eptr.ctypes, eind.ctypes, vwts.ctypes, None, ncommon, _nparts,
            tpwgts.ctypes, opts.ctypes, objval, epart.ctypes, npart.ctypes
        )

        if err != 1:
            raise RuntimeError("METIS Error code : {}".format(err))
        else:
            return epart, npart
