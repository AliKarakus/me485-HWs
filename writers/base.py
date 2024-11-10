# -*- coding: utf-8 -*-

from collections import defaultdict, OrderedDict

import numpy as np
import re

from inifile import INIFile
from readers.native import NativeReader
from solvers import get_fluid


class BaseWriter(object):
    # Dimensionality of each element type
    _petype_ndim = {'tri': 2, 'quad': 2,
                    'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}

    def __init__(self, meshf, solnf, outf):
        self._outf = outf

        mesh = NativeReader(meshf)
        soln = NativeReader(solnf)

        # Check solution and mesh are compatible
        if mesh['mesh_uuid'] != soln['mesh_uuid']:
            raise RuntimeError(
                'Solution {} was not computed on mesh {}'.format(solnf, meshf))

        # Read mesh
        self._nodes = self._get_nodes(mesh)
        self._cells = self._get_cells(mesh)

        # Read solution and config
        self._cfg = cfg = INIFile()

        # Handling string for h5py 2 and 3 are different
        try:
            # h5py 2.X
            self._cfg.fromstr(soln['config'])
        except TypeError:
            # h5py 3.X
            self._cfg.fromstr(soln['config'].decode())

        self._soln, self.ndims = self._get_soln(soln, cfg)

    def _get_nodes(self, mesh):
        return mesh['nodes']

    def _get_cells(self, mesh):
        cells = defaultdict(list)
        self._ele_rank = ele_rank = []
        for k in mesh:
            m = re.match('elm_(\S+)_p(\d+)', k)
            if m:
                cells[m.group(1)].append(mesh[k])
                ele_rank.append((m.group(1), m.group(2)))

        # Convert C-style array
        if self._is_cstyle:
            off = 1
        else:
            off = 0

        for k in sorted(cells):
            v = cells[k]
            cells[k] = np.concatenate(v) - off

        return OrderedDict(cells)

    def _get_soln(self, soln, cfg):
        sol = defaultdict(list)
        aux = defaultdict(list)
        for (etype, p) in self._ele_rank:
            k = 'soln_{}_p{}'.format(etype, p)
            sol[etype].append(soln[k])

            k = 'aux_{}_p{}'.format(etype, p)
            if k in soln:
                aux[etype].append(soln[k])

        # Load Fluid Elements
        fluid_name = cfg.get('solver', 'system')
        self._elms = elms = get_fluid(fluid_name)

        elms.ndims = ndims = self._petype_ndim[etype]

        solns = [np.array(elms.conv_to_prim(np.hstack(sol[k]), cfg))
                 for k in sorted(sol)]
        solns = np.hstack(solns)
        
        if len(aux) > 0:
            auxs = np.hstack([np.hstack(aux[k]) for k in sorted(aux)])
        else:
            auxs = None

        return (solns, auxs), ndims

    def write(self):
        self._raw_write()
