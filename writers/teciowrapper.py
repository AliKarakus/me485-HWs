from ctypes import c_void_p, c_int, c_char_p, c_double, POINTER
from utils.ctypes import load_lib

import numpy as np


class TecioWrapper:
    def __init__(self):
        # Load Tecio
        self.lib = lib = load_lib('tecio')

        # tecini
        lib.tecini142.argtypes = [
            c_char_p, c_char_p, c_char_p, c_char_p,  POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int)
        ]

        # teczne
        lib.teczne142.argtypes = [
            c_char_p, POINTER(c_int), POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_double),
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int),
            POINTER(c_int), c_void_p, POINTER(c_int), POINTER(c_int),
        ]

        # tecdat
        lib.tecdat142.argtypes = [POINTER(c_int), c_void_p, POINTER(c_int)]

        # tecnod
        lib.tecnod142.argtypes = [c_void_p]

    def open(self, fname, variables, title='', cwd='.', fmt=0, ftype='full', isdouble=1):
        _ftype_map = {'full': 0, 'grid': 1, 'solution': 2}

        _variables = ' '.join(variables).encode()
        _ftype = _ftype_map[ftype]

        self.lib.tecini142(
            title.encode(), _variables, fname.encode(), cwd.encode(),
            c_int(fmt), c_int(_ftype), c_int(1), c_int(isdouble)
        )

    def zone(self, title, ztype, npts, nelm, valloc=None, t=0):
        _ztype_map = {'ordered': 0, 'line': 1, 'tri': 2,
                      'quad': 3, 'tet': 4, 'brick': 5}

        _ztype = c_int(_ztype_map[ztype])

        # Unused variable
        nface = c_int(0)
        im, jm, km = c_int(0), c_int(0), c_int(0)
        solt = c_double(t)
        stid = c_int(0)
        pz = c_int(0)
        isblk = c_int(1)
        nfcon = c_int(0)
        fnm = c_int(0)
        tnfn = c_int(0)
        ncb = c_int(0)
        tnbc = c_int(0)
        pvl = None
        shrvar = None
        shrcon = c_int(0)

        if valloc is not None:
            valloc = valloc.ctypes

        self.lib.teczne142(title.encode(), _ztype, c_int(npts), c_int(nelm), nface,
                           im, jm, km, solt, stid, pz, isblk,
                           nfcon, fnm, tnfn, ncb, tnbc, pvl, valloc, shrvar, shrcon)

    def data(self, arr):
        # Check double
        if arr.dtype == 'float64':
            isdouble = c_int(1)
        else:
            isdouble = c_int(0)

        n = c_int(len(arr))
        self.lib.tecdat142(n, arr.ctypes.data, isdouble)

    def node(self, con):
        self.lib.tecnod142(con.ctypes.data)

    def close(self):
        self.lib.tecend142()


if __name__ == "__main__":
    tecio = TecioWrapper()
    tecio.open('test', ['X', 'Y'], 'test.plt')
    tecio.zone('test', 'quad', 8, 4, 0)

    x = np.array([0.0, 1.0, 3.0, 0.0, 1.0, 3.0, 4.0, 2.0])
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0])

    tecio.data(x)
    tecio.data(y)

    con = np.array([
        [1, 2, 5, 4, ],
        [2, 3, 6, 5, ],
        [6, 7, 3, 3, ],
        [3, 2, 8, 8, ],
    ], dtype=np.int32)

    tecio.node(con)
    tecio.close()
