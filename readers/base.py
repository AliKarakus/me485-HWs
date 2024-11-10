# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import chain
import uuid

import numpy as np

from utils.np import fuzzysort


class BaseReader(object, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw_pbm(self):
        pass

    def to_pbm(self):
        mesh = self._to_raw_pbm()

        # Add metadata
        mesh['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')

        return mesh


class ConsAssembler(object):
    # Face numberings for each element type
    _petype_fnums = {
        'tri': {'line': [0, 1, 2]},
        'quad': {'line': [0, 1, 2, 3]},
        'tet': {'tri': [0, 1, 2, 3]},
        'hex': {'quad': [0, 1, 2, 3, 4, 5]},
        'pri': {'quad': [0, 1, 2], 'tri': [3, 4]},
        'pyr': {'quad': [0], 'tri': [1, 2, 3, 4]}
    }

    def __init__(self, elenodes, pents, maps, nodepts):
        self._elenodes, self._pents = elenodes, pents
        self._etype_map, self._petype_fnmap = maps
        self._nodepts = nodepts

    def _extract_fluid(self, elenodes, felespent):
        elemap = defaultdict(dict)

        for (etype, pent), eles in elenodes.items():
            petype = self._etype_map[etype][0]
            elemap[pent][petype] = eles

        return elemap.pop(felespent), elemap

    def _extract_faces(self, fpart):
        faces = defaultdict(list)
        for petype, eles in fpart.items():
            for pftype, fnmap in self._petype_fnmap[petype].items():
                fnums = self._petype_fnums[petype][pftype]
                con = [(petype, i, j, 0)
                       for i in range(len(eles)) for j in fnums]
                nodes = np.sort(eles[:, fnmap]).reshape(len(con), -1)
                faces[pftype].append((con, nodes))

        return faces

    def _pair_fluid_faces(self, faces):
        pairs = defaultdict(list)
        resid = {}

        for pftype, face in faces.items():
            for f, n in chain.from_iterable(zip(f, n) for f, n in face):
                sn = tuple(n)

                if sn in resid:
                    pairs[pftype].append([resid.pop(sn), f])
                else:
                    resid[sn] = f

        return pairs, resid

    def _pair_periodic_fluid_faces(self, bparts, resid, pfacespents):
        # paired faces (same as faces)
        pfaces = defaultdict(list)

        # paired bfaces (same as boundary)
        pbfaces = defaultdict(list)

        nodepts = self._nodepts

        for lpent, rpent in pfacespents.values():
            for pftype in bparts[lpent]:
                lfnodes = bparts[lpent][pftype]
                rfnodes = bparts[rpent][pftype]

                lfpts = np.array([[nodepts[n] for n in fn] for fn in lfnodes])
                rfpts = np.array([[nodepts[n] for n in fn] for fn in rfnodes])

                lfidx = fuzzysort(lfpts.mean(axis=1).T, range(len(lfnodes)))
                rfidx = fuzzysort(rfpts.mean(axis=1).T, range(len(rfnodes)))

                for lfn, rfn in zip(lfnodes[lfidx], rfnodes[rfidx]):
                    lf = resid.pop(tuple(sorted(lfn)))
                    rf = resid.pop(tuple(sorted(rfn)))

                    pfaces[pftype].append([lf, rf])
                    pbfaces[lpent].append(lf)
                    pbfaces[rpent].append(rf)

        return pfaces, pbfaces

    def _identify_boundary_faces(self, bparts, resid, bfacespents):
        bfaces = defaultdict(list)

        bpents = set(bfacespents.values())

        for pent, fnodes in bparts.items():
            if pent in bpents:
                for fn in chain.from_iterable(fnodes.values()):
                    bfaces[pent].append(resid.pop(tuple(sorted(fn))))

        return bfaces

    def get_connectivity(self):
        felespent, bfacespents, pfacespents = self._pents

        # Extract fluid
        fpart, bparts = self._extract_fluid(self._elenodes, felespent)

        # Extract faces
        faces = self._extract_faces(fpart)

        # Pair faces
        pairs, resid = self._pair_fluid_faces(faces)

        # Periodic faces
        ppairs, pbfaces = self._pair_periodic_fluid_faces(bparts, resid, pfacespents)

        # Identify boundary faces
        bfaces = self._identify_boundary_faces(bparts, resid, bfacespents)

        if any(resid.values()):
            raise ValueError('Unpaired faces in mesh')

        # Flatten pairs
        pairs = chain(chain.from_iterable(pairs.values()),
                      chain.from_iterable(ppairs.values()))

        # Connectivity
        con = list(pairs)

        # Boundary connectivity
        bcon = {}
        for name, pent in bfacespents.items():
            bcon[name] = bfaces[pent]

        # Virtual boundary connectivity
        for name, (lpent, rpent) in pfacespents.items():
            bcon['_virtual_'+name+'_l'] = pbfaces[lpent]
            bcon['_virtual_'+name+'_r'] = pbfaces[rpent]

        # Output
        ret = {'con_p0': np.array(con, dtype='S4,i4,i1,i1').T}

        for k, v in bcon.items():
            ret['bcon_{0}_p0'.format(k)] = np.array(v, dtype='S4,i4,i1,i1')

        return ret

    def _extract_vtx_con(self, elenodes, felespent):
        vcon = defaultdict(set)
        for etype, pent in elenodes:
            if pent != felespent:
                continue

            # Elements and type information
            petype, nnode = self._etype_map[etype]

            eles = elenodes[etype, pent]

            for i, ele in enumerate(eles):
                for j, n in enumerate(ele):
                    vcon[n].update({(petype, i, j, 0)})

        return vcon

    def _extract_pair_vtx_con(self, elenodes, pfacespents, vcon):
        nodepts = self._nodepts
        pairs = []

        for etype, pent in elenodes:
            for lpent, rpent in pfacespents.values():
                if lpent == pent:
                    pairs.append([(etype, lpent), (etype, rpent)])

        for lk, rk in pairs:
            lnodes = np.unique(elenodes[lk])
            rnodes = np.unique(elenodes[rk])

            lpts = np.array([nodepts[i] for i in lnodes])
            rpts = np.array([nodepts[i] for i in rnodes])

            lidx = fuzzysort(lpts.T, range(len(lpts)))
            ridx = fuzzysort(rpts.T, range(len(rpts)))

            for li, ri in zip(lnodes[lidx], rnodes[ridx]):
                if li != ri:
                    # Prevent duplicated vcon for periodic faces
                    vcon[li].update(vcon[li], vcon[ri])
                    vcon[ri] = {}
                pass

    def get_vtx_connectivity(self):
        felespent, pfacespents = self._pents[0], self._pents[-1]

        # Extract vertex
        vcon = self._extract_vtx_con(self._elenodes, felespent)

        self._extract_pair_vtx_con(self._elenodes, pfacespents, vcon)

        # Flatten vtx
        vtx = chain.from_iterable([vcon[k] for k in sorted(vcon)])
        vtx = np.array(list(vtx), dtype='S4,i4,i1,i1')

        # Get address in terms of vertex connectivity
        ivtx = np.cumsum([0] + [len(vcon[k]) for k in sorted(vcon) if len(vcon[k]) > 0])

        # Output
        ret = {'vtx_p0': vtx, 'ivtx_p0': ivtx}

        return ret


class NodesAssembler(object):
    # Dimensionality of each element type
    _petype_ndim = {'tri': 2, 'quad': 2,
                    'tet': 3, 'hex': 3, 'pri': 3, 'pyr': 3}
    
    def __init__(self, nodepts, elenodes, felespent, bfacespents, etype_map, scale):
        self._nodepts, self._elenodes = nodepts, elenodes
        self._bfacespents = {v: k for k, v in bfacespents.items()}
        self._felespent = felespent
        self._etype_map = etype_map
        self._scale = scale

    def _fluid_elm(self):
        elm = {}
        spt = {}
        bnode = defaultdict(list)
        for (etype, pent), ele in self._elenodes.items():
            petype = self._etype_map[etype][0]

            if pent == self._felespent:
                elm['elm_{}_p0'.format(petype)] = ele
                spt['spt_{}_p0'.format(petype)] = self._get_spt_ele(petype, ele)
            elif pent in self._bfacespents:
                bname = self._bfacespents[pent]
                bnode[bname].append(np.unique(ele.ravel()))

        bnode = {k : np.concatenate(v) for k, v in bnode.items()}

        return elm, spt, bnode

    def get_nodes(self):
        vals = self._nodepts[1:]*self._scale

        ret = {'nodes': vals}
        elm, spt, bnode = self._fluid_elm()
        ret.update(elm)
        ret.update(spt)
        ret.update(self._extract_bnodes(bnode))

        return ret

    def _get_spt_ele(self, petype, ele):
        ndim = self._petype_ndim[petype]
        nodepts = self._nodepts

        # Get nodes and sort them
        arr = nodepts[ele].swapaxes(0, 1)
        return arr[..., :ndim]

    def _extract_bnodes(self, bnode):
        return {'bnode_' + k : self._nodepts[v] for k, v in bnode.items()}
