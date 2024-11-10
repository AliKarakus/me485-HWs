# -*- coding: utf-8 -*-

import numpy as np

from writers.base import BaseWriter
from writers.teciowrapper import TecioWrapper


class TecplotWriter(BaseWriter):
    name = 'plt'
    _is_cstyle = False
    _ndims_map = {'tri': 2, 'quad': 2, 'tet': 3, 'pri': 3, 'pyr': 3, 'hex': 3}

    def _raw_write(self):
        # mesh data
        nodes = self._nodes.T.copy()
        nnodes = nodes.shape[1]
        cons = self._tec_cons()
        ncells = cons.shape[0]

        # Solution data
        solns = self._soln
        ndims = self._ndims_map[next(iter(self._cells.keys()))]
        self._elms.ndims = ndims

        # Variables
        variables = ['X', 'Y', 'Z'][:ndims] + self._elms.primevars

        if solns[1] is not None:
            variables += self._elms.auxvars

        try:
            # Binary writing if possible
            self._write_binary(nnodes, ncells, ndims,
                               variables, nodes, cons, *solns)
        except:
            # Fallback to ascii write
            self._write_ascii(nnodes, ncells, ndims,
                              variables, nodes, cons, *solns)

    def _tec_cons(self):
        cons = []

        # Convert to Quad and Brick type
        confmap = {
            'tri': lambda e: [e[0], e[1], e[2], e[2]],
            'quad': lambda e: e,
            'tet': lambda e: [e[0], e[1], e[2], e[2], e[3], e[3], e[3], e[3]],
            'pyr': lambda e: [e[0], e[1], e[2], e[3], e[4], e[4], e[4], e[4]],
            'pri': lambda e: [e[0], e[1], e[2], e[2], e[3], e[4], e[5], e[5]],
            'hex': lambda e: e
        }

        for k, v in self._cells.items():
            conf = confmap[k]
            cons.append(np.array([conf(e) for e in v]))

        return np.vstack(cons)

    def _write_binary(self, nnodes, ncells, ndims, variables, nodes, cons, soln, aux):
        # Construct tecio wrapper
        tecio = TecioWrapper()
        tecio.open(self._outf, variables)

        # Zone type and variable locations
        _zone_etype = {2: 'quad', 3: 'brick'}
        etype = _zone_etype[ndims]

        valloc = np.zeros(len(variables), dtype=np.int32)
        valloc[:ndims] = 1

        # Initialize zone
        tecio.zone(self._outf.split('.')[0],
                   etype, nnodes, ncells, valloc=valloc)

        # Write node and solutions
        tecio.data(nodes[:ndims].ravel())
        tecio.data(soln.ravel())

        if aux is not None:
            tecio.data(aux.ravel())

        # Write cons
        tecio.node(cons)

        # close
        tecio.close()

    def _write_ascii(self, nnodes, ncells, ndims, variables, nodes, cons, soln, aux):
        varlists = " ".join("\"{}\"".format(e) for e in variables)
        centerloc = ','.join(str(e+1) for e in range(ndims, len(variables)))

        _zone_type = {2: 'FEQUADRILATERAL', 3: 'FEBRICK'}
        zonet = _zone_type[ndims]

        # Write
        with open(self._outf, 'w') as fp:
            fp.write("VARIABLES = {}\n".format(varlists))
            fp.write("ZONE NODES={}, ELEMENTS={}, DATAPACKING=BLOCK, ZONETYPE={}\nVARLOCATION=([{}]=CELLCENTERED)\n".format(
                nnodes, ncells, zonet, centerloc))

            np.savetxt(fp, nodes[:ndims], fmt="%lf", delimiter='\n')

            np.savetxt(fp, soln, fmt="%E", delimiter='\n')

            np.savetxt(fp, aux, fmt="%E", delimiter='\n')

            np.savetxt(fp, cons, fmt="%d", delimiter='\n')
