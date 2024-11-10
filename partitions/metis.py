import h5py
import numpy as np
import uuid

from collections import defaultdict
from itertools import combinations

from partitions.metiswrapper import METISWrapper


class METISPartition:
    _wmap = {'quad': 3, 'tri': 2, 'tet': 4, 'pri': 5, 'pyr': 5, 'hex': 6}

    def __init__(self, msh, out, npart):
        # TODO: merge mesh
        # TODO: partitioning solution

        # Partitioning elements and mapping
        newm, eidx_g2l = self.partition_mesh(msh, npart)

        # Update elements, connectivities, vertex and nodes
        newm.update(self.partition_elm(msh, eidx_g2l))
        newm.update(self.partition_spt(msh, eidx_g2l))
        newm.update(self.partition_cons(msh, eidx_g2l))
        newm.update(self.partition_bcons(msh, eidx_g2l))
        newm.update(self.partition_vtx(msh, eidx_g2l))
        self.copy_nodes(msh, newm)

        # Assign new UUID
        newm['mesh_uuid'] = np.array(str(uuid.uuid4()), dtype='S')

        # Save new mesh
        with h5py.File(out, 'w') as f:
            for k, v in newm.items():
                f[k] = v

    def partition_mesh(self, msh, npart):
         # list of elements type
        etypes = [n.split('_')[1] for n in msh if n.startswith('elm')]

        # number of elements
        nele = {t: msh['elm_{}_p0'.format(t)].shape[0] for t in etypes}

        # List of element connectivity
        elms = []
        for t in etypes:
            elms += msh['elm_{}_p0'.format(t)].tolist()

        # Do metis Partition
        epart = self._metis_part(npart, etypes, nele, elms)

        # Make new mesh
        newmesh = defaultdict(list)

        # Global address
        egidx = [(n, i) for n in etypes for i in range(nele[n])]

        # local address counter
        etype_rank = [(n, p) for n in etypes for p in range(npart)]
        lcounter = dict(zip(etype_rank, [0]*len(etype_rank)))

        # eidx_g2l (etype, g) -> (rank, l)
        eidx_g2l = {}
        for (t, e), p in zip(egidx, epart):
            i = lcounter[t, p]
            eidx_g2l[(t, e)] = (p, i)
            lcounter[t, p] += 1

        return newmesh, eidx_g2l

    def _metis_part(self, npart, etypes, nele, elms):
        # Linked list of elms
        eind = np.concatenate(elms) - 1
        eptr = np.cumsum([0] + [len(e) for e in elms])

        # Weights
        vwgt = []
        for t in etypes:
            vwgt += [self._wmap[t]]*nele[t]
        vwgt = np.array(vwgt)

        ncommon = 2

        # Partitioning with METIS
        ne = sum([nele[t] for t in etypes])
        nn = eind.max() + 1

        metis = METISWrapper()
        epart, _ = metis.part_mesh(npart, nn, ne, eptr, eind, ncommon, vwgt)

        return epart

    def partition_elm(self, msh, eidx_g2l):
        elms = {
            n.split('_')[1]: msh[n] for n in msh if n.startswith('elm')
        }

        # Sort elements per rank
        newelm = defaultdict(list)
        for (t, g), (p, l) in eidx_g2l.items():
            ele = elms[t][g]
            newelm['elm_{}_p{}'.format(t, p)].append(ele)

        return {k: np.array(v) for k, v in newelm.items()}
    
    def partition_spt(self, msh, eidx_g2l):
        spts = {
            n.split('_')[1]: msh[n] for n in msh if n.startswith('spt')
        }

        # Sort spt per rank
        newelm = defaultdict(list)
        for (t, g), (p, _) in eidx_g2l.items():
            spt = spts[t][:, g]
            newelm['spt_{}_p{}'.format(t, p)].append(spt)

        arr = {k: np.array(v).swapaxes(0, 1) for k, v in newelm.items()}
        return arr

    def partition_cons(self, msh, eidx_g2l):
        lhs, rhs = msh['con_p0'].astype('U4,i4,i1,i1').tolist()

        # Partition cons
        cons = defaultdict(list)
        for (lt, le, lf, lz), (rt, re, rf, rz) in zip(lhs, rhs):
            pl, lel = eidx_g2l[(lt, le)]
            pr, rel = eidx_g2l[(rt, re)]

            if pl == pr:
                # Save internal connectivity
                cons['con_p{}'.format(pl)].append(
                    [(lt, lel, lf, lz), (rt, rel, rf, rz)])
            else:
                # Save parallel connectivity
                cons['con_p{}p{}'.format(pl, pr)].append((lt, lel, lf, lz))
                cons['con_p{}p{}'.format(pr, pl)].append((rt, rel, rf, rz))

        return {k: np.array(v, dtype='S4,i4,i1,i1').T for k, v in cons.items()}

    def partition_bcons(self, msh, eidx_g2l):
        # Partitioning bcons
        bcons = defaultdict(list)
        for k in msh:
            if k.startswith('bcon'):
                bctype = '_'.join(k.split('_')[1:-1])
                bc = msh[k].astype('U4,i4,i1,i1').tolist()

                for (t, e, f, z) in bc:
                    p, el = eidx_g2l[(t, e)]

                    # Save boundary connectivity per rank
                    bcons['bcon_{}_p{}'.format(bctype, p)].append(
                        (t, el, f, z))

        return {k: np.array(v, dtype='S4,i4,i1,i1') for k, v in bcons.items()}

    def partition_vtx(self, msh, eidx_g2l):
        new = defaultdict(list)

        # Partitioning vtx
        vtx = msh['vtx_p0'].astype('U4,i4,i1,i1').tolist()
        ivtx = msh['ivtx_p0']

        for i1, i2 in zip(ivtx[:-1], ivtx[1:]):
            # Convert local vtx for rank
            lvtx = defaultdict(list)
            for i in range(i1, i2):
                t, e, f, z = vtx[i]
                p, el = eidx_g2l[(t, e)]
                lvtx[p].append((t, el, f, z))

            for p in lvtx:
                new['vtx_p{}'.format(p)].extend(lvtx[p])
                new['ivtx_p{}'.format(p)].append(len(lvtx[p]))

            if len(lvtx) > 1:
                for (p1, p2) in combinations(lvtx, 2):
                    nvtx1 = len(new['ivtx_p{}'.format(p1)]) - 1
                    nvtx2 = len(new['ivtx_p{}'.format(p2)]) - 1

                    new['nvtx_p{}p{}'.format(p1, p2)].append(nvtx1)
                    new['nvtx_p{}p{}'.format(p2, p1)].append(nvtx2)

        # Make numpy array
        for k, v in new.items():
            if k.startswith('vtx'):
                new[k] = np.array(v, dtype='S4,i4,i1,i1')
            elif k.startswith('ivtx'):
                new[k] = np.cumsum([0] + v, dtype='i4')
            elif k.startswith('nvtx'):
                new[k] = np.array(v, dtype='i4')

        return new

    def copy_nodes(self, msh, newm):
        # Copy nodes
        newm['nodes'] = msh['nodes']

        # Copy bnode
        for k in msh:
            if k.startswith('bnode'):
                newm[k] = msh[k]