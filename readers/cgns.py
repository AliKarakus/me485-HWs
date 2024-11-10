# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
import re

from .cgnswrapper import CGNSWrapper
from readers.base import BaseReader, ConsAssembler, NodesAssembler


class CGNSZoneReader(object):
    # CGNS element types to petype and node counts
    cgns_map = {
        3: ('line', 2), 5: ('tri', 3), 7: ('quad', 4), 10: ('tet', 4),
        12: ('pyr', 5), 14: ('pri', 6), 17: ('hex', 8),
    }

    def __init__(self, cgns, base, idx):
        self._cgns = cgns
        zone = cgns.zone_read(base, idx)

        # Read nodes
        self.nodepts = self._read_nodepts(zone)

        # Read bc
        bc = self._read_bc(zone)

        # Read elements
        self.elenodes = elenodes = {}
        self.pents = pents = {}

        # Construct elenodes and physical entity
        for idx in range(cgns.nsections(zone)):
            elerng, elenode = self._read_element(zone, idx)

            for jdx, (bcname, (bcrng, bclist)) in enumerate(bc.items()):
                if ((elerng[0] <= bcrng[0]) and (bcrng[0] <= elerng[1])) or ((elerng[0] <= bcrng[1]) and (bcrng[1] <= elerng[1])):
                    name = bcname
                    bclist = np.array(bclist)
                    picks = bclist[(bclist >= elerng[0]) & (bclist < elerng[1] + 1)] - elerng[0]
                    break
            else:
                name = 'fluid'
                picks = Ellipsis
                jdx = -1

            pent = pents.setdefault(name, jdx+1)

            elenodes.update({(k, pent): v[picks] for k, v in elenode.items()})

    def _read_nodepts(self, zone):
        nnode = zone['size'][0]
        ndim = zone['base']['PhysDim']
        nodepts = np.zeros((3, nnode))

        for i, x in enumerate('XYZ'[:ndim]):
            self._cgns.coord_read(zone, 'Coordinate{}'.format(x), nodepts[i])

        return nodepts

    def _read_bc(self, zone):
        nbc = self._cgns.nbocos(zone)
        bc = {}

        for idx_bc in range(nbc):
            boco = self._cgns.boco_read(zone, idx_bc)
            name = boco['name'].lower()
            name = re.sub('\s+', '_', name)
            bc[name] = boco['range'], boco['list']

        return bc

    def _read_element(self, zone, idx):
        s = self._cgns.section_read(zone, idx)

        elerng = s['range']
        conn = np.zeros(s['dim'], dtype=self._cgns.int_np)
        self._cgns.elements_read(s, conn)

        cgns_type = s['etype']
        elenode = {}

        spts = self.cgns_map[cgns_type][1]
        elenode[cgns_type] = conn.reshape(-1, spts)

        return elerng, elenode


class CGNSReader(BaseReader):
    # Supported file types and extensions
    name = 'cgns'
    extn = ['.cgns']

    # CGNS element types to petype and node counts
    _etype_map = CGNSZoneReader.cgns_map

    # Node numbers associated with each element face
    _petype_fnmap = {
        'tri': {'line': [[0, 1], [1, 2], [2, 0]]},
        'quad': {'line': [[0, 1], [1, 2], [2, 3], [3, 0]]},
        'tet': {'tri': [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]},
        'hex': {'quad': [[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5],
                         [2, 3, 7, 6], [0, 4, 7, 3], [4, 5, 6, 7]]},
        'pri': {'quad': [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]],
                'tri': [[0, 2, 1], [3, 4, 5]]},
        'pyr': {'quad': [[0, 3, 2, 1]],
                'tri': [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]}
    }

    def __init__(self, msh, scale):
        # Load and wrap CGNS
        self._cgns = cgns = CGNSWrapper()

        # Read CGNS mesh file
        self._file = file = cgns.open(msh)
        base = cgns.base_read(file, 0)

        # Read zones and stack nodepts, pents and elenodes
        offset = 0
        pent = 0
        pents = {}
        elenodes = {}
        for idx in  range(cgns.nzones(base)):
            # read zone
            zone = CGNSZoneReader(cgns, base, idx)
            
            ndims, nn = zone.nodepts.shape

            if idx == 0:
                # Add 1st row to start node number as 1
                nodepts = np.zeros(ndims)[:, None]

            # Stack nodes
            nodepts = np.hstack([nodepts, zone.nodepts])

            # Collect pents and local mapping in each zone
            pmap = {}
            for k, v in zone.pents.items():
                if k not in pents:
                    pents[k] = pent
                    pent += 1

                pmap[v] = pents[k]

            # Collect elenodes
            for k, v in zone.elenodes.items():
                # Keys as (petype and pent)
                new = k[0], pmap[k[1]]

                # Add offset for global node numbering
                v += offset

                # Stack elenodes
                if new in elenodes:
                    elenodes[new] = np.vstack([elenodes[new], v])
                else:
                    elenodes[new] = v

            # Update offset of elenode for next zone
            offset += nn

        # Transpose nodepts
        nodepts = nodepts.T

        # Physical entities can be divided up into:
        #  - fluid elements ('the mesh')
        #  - boundary faces
        felespent = pents.pop('fluid')
        bfacespents = {}
        pfacespents = defaultdict(list)

        for name, pent in pents.items():
            if name.startswith('periodic'):
                p = re.match(r'periodic[ _-]([a-z0-9]+)[ _-](l|r)$', name)
                if not p:
                    raise ValueError('Invalid periodic boundary condition')

                pfacespents[p.group(1)].append(pent)
            # Other boundary faces
            else:
                bfacespents[name] = pent

        if any(len(pf) != 2 for pf in pfacespents.values()):
            raise ValueError('Unpaired periodic boundary in mesh')

        # Construct node db
        pents = felespent, bfacespents, pfacespents
        maps = self._etype_map, self._petype_fnmap
        self._cons = ConsAssembler(elenodes, pents, maps, nodepts)
        self._nodes = NodesAssembler(
            nodepts, elenodes, felespent, bfacespents, self._etype_map, scale)

    def __del__(self):
        if hasattr(self, '_file'):
            self._cgns.close(self._file)

    def _to_raw_pbm(self):
        rawm = {}

        rawm.update(self._cons.get_connectivity())
        rawm.update(self._cons.get_vtx_connectivity())
        rawm.update(self._nodes.get_nodes())

        return rawm
