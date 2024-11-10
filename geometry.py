# -*- coding: utf-8 -*-
import numpy as np

from utils.misc import subclass_by_name


def get_geometry(name, *args, **kwargs):
    return subclass_by_name(BaseGeom, name)(*args, **kwargs)


class BaseFace(object):
    # Abstract object of face
    name = 'none'

    @staticmethod
    def xc(x):
        # Center points of face
        return np.average(x, axis=0)


class LineFace(BaseFace):
    name = 'line'

    @staticmethod
    def snorm(x):
        # Normal vector * length of face
        dx = x[1] - x[0]
        op = np.array([[0, -1], [1, 0]])
        return np.dot(dx, op)


class TriFace(BaseFace):
    name = 'tri'

    @staticmethod
    def snorm(x):
        # Cross product of two sides
        op = np.array([[-1, 1, 0], [-1, 0, 1]])
        dx = np.dot(op, x.swapaxes(0, 1))

        # Normal vector * area
        return 0.5*np.cross(dx[0], dx[1])


class QuadFace(BaseFace):
    name = 'quad'

    @staticmethod
    def snorm(x):
        # Cross product of diagonals
        op = np.array([[-1, 0, 1, 0], [0, -1, 0, 1]])
        dx = np.dot(op, x.swapaxes(0, 1))

        # Normal vector * area
        return 0.5*np.cross(dx[0], dx[1])


class BaseGeom(object):
    # Abstract object of element
    name = 'none'

    @property
    def nface(self):
        # Number of face
        return len(self._face)

    @property
    def fcls(self):
        # List of face objects
        fb = {}
        for ftype, fn in self._face:
            if ftype not in fb:
                fb[ftype] = subclass_by_name(BaseFace, ftype)

        return fb

    def xc(self, x):
        # Center points
        return np.average(x, axis=0)

    def xf(self, x):
        # Face center points
        xf = [self.fcls[ftype].xc(x[fn]) for ftype, fn in self._face]
        return np.array(xf)

    def dxf(self, x):
        # Displacement vector from center to face
        return self.xf(x) - self.xc(x)

    def dxv(self, x):
        # Displacement vector from center to vertex
        return x - self.xc(x)

    def snorm(self, x):
        # Face Noraml*area vector 
        snorm = [self.fcls[ftype].snorm(x[fn]) for ftype, fn in self._face]
        return np.array(snorm)

    def vol(self, x):
        # Volume of element
        ndim = x.shape[-1]
        return np.einsum('ijk,ijk->j', self.snorm(x), self.dxf(x)) / ndim


class QuadGeom(BaseGeom):
    name = 'quad'
    _face = [('line', [0, 1]), ('line', [1, 2]),
             ('line', [2, 3]), ('line', [3, 0])]
    nvertex = 4


class TriGeom(BaseGeom):
    name = 'tri'
    _face = [('line', [0, 1]), ('line', [1, 2]), ('line', [2, 0])]
    nvertex = 3


class TetGeom(BaseGeom):
    name = 'tet'
    _face = [('tri', [0, 2, 1]), ('tri', [0, 1, 3]),
             ('tri', [1, 2, 3]), ('tri', [2, 0, 3])]
    nvertex = 4


class HexGeom(BaseGeom):
    name = 'hex'
    _face = [('quad', [0, 3, 2, 1]), ('quad', [0, 1, 5, 4]), ('quad', [1, 2, 6, 5]),
             ('quad', [2, 3, 7, 6]), ('quad', [0, 4, 7, 3]), ('quad', [4, 5, 6, 7])]
    nvertex = 8


class PriGeom(BaseGeom):
    name = 'pri'
    _face = [('quad', [0, 1, 4, 3]), ('quad', [1, 2, 5, 4]), ('quad', [2, 0, 3, 5]),
             ('tri', [0, 2, 1]), ('tri', [3, 4, 5])]
    nvertex = 6


class PyrGeom(BaseGeom):
    name = 'pyr'
    _face = [('quad', [0, 3, 2, 1]),
             ('tri', [0, 1, 4]), ('tri', [1, 2, 4]), ('tri', [2, 3, 4]), ('tri', [3, 0, 4])]
    nvertex = 5
