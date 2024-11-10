# -*- coding: utf-8 -*-
# Original code
# https://github.com/PyFR/PyFR/blob/develop/pyfr/readers/gmsh.py
# Modified by jspark
# 
from collections import defaultdict
import re

import numpy as np
from readers.base import BaseReader, ConsAssembler, NodesAssembler


def msh_section(mshit, section):
    endln = f'$End{section}\n'
    endix = int(next(mshit))

    for i, l in enumerate(mshit, start=1):
        if l == endln:
            raise ValueError(f'Unexpected end of section ${section}')

        yield l.strip()

        if i == endix:
            break
    else:
        raise ValueError('Unexpected EOF')

    if next(mshit) != endln:
        raise ValueError(f'Expected $End{section}')


class GMSHReader(BaseReader):
    name = 'gmsh'
    extn = ['.msh']

    # Gmsh element types to petype and node counts
    _etype_map = {
        1: ('line', 2), 2: ('tri', 3), 3: ('quad', 4), 
        4: ('tet', 4), 5: ('hex', 8), 6: ('pri', 6), 7 : ('pyr', 5), 
    }

    # Node numbers associated with each element face
    _petype_fnmap = {
        'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
        'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
        'tet': {'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
        'hex': {'quad': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6],
                         [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
        'pri': {'quad': [[0, 1, 3, 4], [1, 2, 4, 5], [0, 2, 3, 5]],
                'tri': [[0, 1, 2], [3, 4, 5]]},
        'pyr': {'quad': [[0, 1, 2, 3]],
                'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]}
    }

    def __init__(self, msh, scale):
        self._scale = scale

        if isinstance(msh, str):
            msh = open(msh)

        # Get an iterator over the lines of the mesh
        mshit = iter(msh)

        # Section readers
        sect_map = {
            'MeshFormat': self._read_mesh_format,
            'PhysicalNames': self._read_phys_names,
            'Entities': self._read_entities,
            'Nodes': self._read_nodes,
            'Elements': self._read_eles
        }

        for l in filter(lambda l: l != '\n', mshit):
            # Ensure we have encountered a section
            if not l.startswith('$'):
                raise ValueError('Expected a mesh section')

            # Strip the '$' and '\n' to get the section name
            sect = l[1:-1]

            # Try to read the section
            try:
                sect_map[sect](mshit)
            # Else skip over it
            except KeyError:
                endsect = f'$End{sect}\n'

                for el in mshit:
                    if el == endsect:
                        break
                else:
                    raise ValueError(f'Expected $End{sect}')
    
    def _read_mesh_format(self, mshit):
        ver, ftype, dsize = next(mshit).split()

        if ver == '2.2':
            self._read_nodes_impl = self._read_nodes_impl_v2
            self._read_eles_impl = self._read_eles_impl_v2
        elif ver == '4.1':
            self._read_nodes_impl = self._read_nodes_impl_v41
            self._read_eles_impl = self._read_eles_impl_v41
        else:
            raise ValueError('Invalid mesh version')

        if ftype != '0':
            raise ValueError('Invalid file type')
        if dsize != '8':
            raise ValueError('Invalid data size')

        if next(mshit) != '$EndMeshFormat\n':
            raise ValueError('Expected $EndMeshFormat')

    def _read_phys_names(self, mshit):
        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        #  - periodic faces
        self._felespent = None
        self._bfacespents = {}
        self._pfacespents = defaultdict(list)

        # Seen physical names
        seen = set()

        # Extract the physical names
        for l in msh_section(mshit, 'PhysicalNames'):
            m = re.match(r'(\d+) (\d+) "((?:[^"\\]|\\.)*)"$', l)
            if not m:
                raise ValueError('Malformed physical entity')

            pent, name = int(m[2]), m[3].lower()

            # print(f'{pent:10} ==> {name:10}')

            # Ensure we have not seen this name before
            if name in seen:
                raise ValueError(f'Duplicate physical name: {name}')

            # Fluid elements
            if name == 'fluid':
                self._felespent = pent
            # Periodic boundary faces
            elif name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                self._pfacespents[p[1]].append(pent)
            # Other boundary faces
            else:
                self._bfacespents[name] = pent

            seen.add(name)

        if self._felespent is None:
            raise ValueError('No fluid elements in mesh')

        if any(len(pf) != 2 for pf in self._pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

    def _read_entities(self, mshit):
        self._tagpents = tagpents = {}

        # Obtain the entity counts
        npts, *ents = (int(i) for i in next(mshit).split())

        # Skip over the point entities
        for i in range(npts):
            next(mshit)

        # Iterate through the curves, surfaces, and volume entities
        for ndim, nent in enumerate(ents, start=1):
            for j in range(nent):
                ent = next(mshit).split()
                etag, enphys = int(ent[0]), int(ent[7])

                if enphys == 0:
                    continue
                elif enphys == 1:
                    tagpents[ndim, etag] = abs(int(ent[8]))
                else:
                    raise ValueError('Invalid physical tag count for entity')

        if next(mshit) != '$EndEntities\n':
            raise ValueError('Expected $EndEntities')

    def _read_nodes(self, mshit):
        nodes = self._read_nodes_impl(mshit)

        self._nodepts = nodepts = np.empty((max(nodes.keys())+1, 3))
        for k, v in nodes.items():
            nodepts[k] = v

        pass

    def _read_nodes_impl_v2(self, mshit):
        nodepts = {}

        # Read in the nodes as a dict
        for l in msh_section(mshit, 'Nodes'):
            nv = l.split()
            nodepts[int(nv[0])] = np.array([float(x) for x in nv[1:]])

        return nodepts

    def _read_nodes_impl_v41(self, mshit):
        # Entity count, node count, minimum and maximum node numbers
        ne, nn, ixl, ixu = (int(i) for i in next(mshit).split())

        nodepts = {}

        for i in range(ne):
            nen = int(next(mshit).split()[-1])
            nix = [int(next(mshit)[:-1]) for _ in range(nen)]

            for j in nix:
                nodepts[j] = np.array([float(x) for x in next(mshit).split()])

        if next(mshit) != '$EndNodes\n':
            raise ValueError('Expected $EndNodes')
        
        return nodepts

    def _read_eles(self, mshit):
        self._read_eles_impl(mshit)

    def _read_eles_impl_v2(self, mshit):
        elenodes = defaultdict(list)

        for l in msh_section(mshit, 'Elements'):
            # Extract the raw element data
            elei = [int(i) for i in l.split()]
            enum, etype, entags = elei[:3]
            etags, enodes = elei[3:3 + entags], elei[3 + entags:]

            if etype not in self._etype_map:
                raise ValueError(f'Unsupported element type {etype}')

            # Physical entity type (used for BCs)
            epent = etags[0]

            elenodes[etype, epent].append(enodes)

        self._elenodes = {k: np.array(v) for k, v in elenodes.items()}

    def _read_eles_impl_v41(self, mshit):
        elenodes = defaultdict(list)

        # Block and total element count
        nb, ne = (int(i) for i in next(mshit).split()[:2])

        for i in range(nb):
            edim, etag, etype, ecount = (int(j) for j in next(mshit).split())

            if etype not in self._etype_map:
                raise ValueError(f'Unsupported element type {etype}')

            # Physical entity type (used for BCs)
            epent = self._tagpents.get((edim, etag), -1)
            append = elenodes[etype, epent].append

            for j in range(ecount):
                append([int(k) for k in next(mshit).split()[1:]])

        if ne != sum(len(v) for v in elenodes.values()):
            raise ValueError('Invalid element count')

        if next(mshit) != '$EndElements\n':
            raise ValueError('Expected $EndElements')

        self._elenodes = {k: np.array(v) for k, v in elenodes.items()}

    def _to_raw_pbm(self):
        pents = self._felespent, self._bfacespents, self._pfacespents
        maps = self._etype_map, self._petype_fnmap
        elenodes, nodepts = self._elenodes, self._nodepts
        self._cons = ConsAssembler(elenodes, pents, maps, nodepts)
        self._nodes = NodesAssembler(
            nodepts, elenodes, self._felespent, self._bfacespents, 
            self._etype_map, self._scale
        )

        rawm = {}
        # print(type(self._cons.get_connectivity()))
        # print('\n \n', self._cons.get_vtx_connectivity())
        # print('\n \n', self._nodes.get_nodes())

        rawm.update(self._cons.get_connectivity())
        rawm.update(self._cons.get_vtx_connectivity())
        rawm.update(self._nodes.get_nodes())

        return rawm