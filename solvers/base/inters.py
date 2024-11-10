# -*- coding: utf-8 -*-
import functools as fc
import numpy as np


class BaseInters:
    name = 'base'

    def __init__(self, be, cfg, elemap, lhs):
        # Save arguments
        self.be = be
        self.cfg = cfg

        # Dimensions
        self.nfpts = len(lhs)
        self.ele0 = ele0 = elemap[next(iter(elemap))]
        self.ndims, self.nvars, self.nfvars = ele0.ndims, ele0.nvars, ele0.nfvars

        # Primitive variables
        self.primevars = ele0.primevars

        # Collect constants
        self._const = cfg.items('constants')

        # Order of spatial discretization
        self.order = cfg.getint('solver', 'order', 1)

        # Normal Vector of face
        self._mag_snorm = self._get_fpts('_mag_snorm_fpts', elemap, lhs)[0]
        self._vec_snorm = self._get_fpts('_vec_snorm_fpts', elemap, lhs)

    def _get_fpts(self, meth, elemap, lhs):
        # Get element property at face and sort it
        arr = [getattr(elemap[t], meth)[f, e] for t, e, f, z in lhs]
        arr = np.vstack(arr).T
        return arr.copy()

    def _get_upts(self, meth, elemap, lhs):
        # Get element property and sort it
        arr = [getattr(elemap[t], meth)[e] for t, e, f, z in lhs]
        arr = np.vstack(arr).T
        return arr.copy()

    def _get_index(self, elemap, lhs):
        # Parse index of elements and make index of each face point
        cell_nums = {c: i for i, c in enumerate(elemap)}
        return np.array([[cell_nums[t], e, f] for t, e, f, z in lhs]).T.copy()
    
    @property
    @fc.lru_cache()
    def _rcp_dx(self):
        return 1/np.linalg.norm(self._dx_adj, axis=0)


class BaseIntInters(BaseInters):
    def __init__(self, be, cfg, elemap, lhs, rhs):
        super().__init__(be, cfg, elemap, lhs)

        self._lidx = self._get_index(elemap, lhs)
        self._ridx = self._get_index(elemap, rhs)

        if self.order > 1:
            # Delx = xc2 - xc1 across face
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

        # Construct neighboring element within current Elements
        self._construct_ele_graph(elemap, lhs, rhs)

    def _compute_dxc(self, *dx):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        def compute_dxc(i_begin, i_end, dx_adj, *dxc):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(ndims):
                    xl = dxc[lti][lfi, lei, jdx]
                    xr = dxc[rti][rfi, rei, jdx]

                    dx = xr - xl
                    dx_adj[jdx, idx] = dx

                    dxc[lti][lfi, lei, jdx] = dx
                    dxc[rti][rfi, rei, jdx] = -dx

        # Compute dx_adj
        self.be.make_loop(nface, compute_dxc)(self._dx_adj, *dx)

    def _construct_ele_graph(self, elemap, lhs, rhs):
        # Convert lhs, rhs list to numpy array
        lhs = np.array(lhs, dtype='U4,i4,i1,i1')
        rhs = np.array(rhs, dtype='U4,i4,i1,i1')

        # Construct connectivity (fact to ele)
        con = np.hstack([[lhs, rhs], [rhs, lhs]])[['f0', 'f1', 'f2']]

        for t, ele in elemap.items():
            mask = (con['f0'][0] == t) & (con['f0'][1] == t)

            # Default nei_ele
            ele.nei_ele = nei_ele = np.tile(
                np.arange(ele.neles, dtype=int), ele.nface
                ).reshape(ele.nface,-1)
    
            if np.any(mask):    
                # Get local connectiviy for each element
                lcon = con[:, mask]
                
                # Reorder w.r.t. left
                idx = np.lexsort([lcon['f2'][0], lcon['f1'][0]])
                l, r = lcon[:, idx]

                # Get offset (address array)
                tab = np.where(l['f1'][1:] != l['f1'][:-1])[0]
                off = np.concatenate([[0], tab + 1, [len(l)]])
                eidx = np.concatenate([
                    [l['f1'][i1]]*(i2-i1) for (i1, i2) in zip(off[:-1], off[1:])
                    ])
                
                # data
                data = r['f1'].copy()

                # Assign neighboring element array to each element
                nei_ele[l['f2'], eidx] = data

                # Rearrange indptr
                ind = np.zeros(ele.neles, dtype=int)
                ind[l['f1'][off[:-1]]] = np.diff(off)
                indptr = np.concatenate([[0], np.cumsum(ind)])
            else:                
                # Null graph
                indptr = np.zeros(ele.neles+1, dtype=int)
                data = np.array([], dtype=int)

            # Save as graph
            ele.graph = {'indptr' : indptr, 'indices' : data}


class BaseBCInters(BaseInters):
    _reqs = None

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs)
        self.bctype = bctype

        self._lidx = self._get_index(elemap, lhs)

        if self.order > 1:
            # Delx across face
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

        # Compute face center at boundary
        self.xf = self._get_fpts('xf', elemap, lhs)

    def _compute_dxc(self, *dx):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx

        nf = self._vec_snorm

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        def compute_dxc(i_begin, i_end, dx_adj, *dxc):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                # Compute normal component of (xf - xc) as dxn
                dxn = 0
                for jdx in range(ndims):
                    dxn += -dxc[lti][lfi, lei, jdx]*nf[jdx, idx]

                for jdx in range(ndims):
                    dx = 2*dxn*nf[jdx, idx]
                    dxc[lti][lfi, lei, jdx] = dx
                    dx_adj[jdx, idx] = dx

        # Compute dx_adj
        self.be.make_loop(nface, compute_dxc)(self._dx_adj, *dx)


class BaseVRInters(BaseInters):
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs)
        self.bctype = bctype

        self._lidx = self._get_index(elemap, lhs)

        # Compute face center at boundary
        self.xf = self._get_fpts('xf', elemap, lhs)


class BaseMPIInters(BaseInters):
    def __init__(self, be, cfg, elemap, lhs, dest):
        super().__init__(be, cfg, elemap, lhs)
        self._dest = dest

        self._lidx = self._get_index(elemap, lhs)

        if self.order > 1:
            # Delx = xc2 - xc1 across face
            dxc = [cell.dxc for cell in elemap.values()]
            self._compute_dxc(*dxc)

    def _compute_dxc(self, *dx):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        buf = np.empty((nface, ndims), dtype=np.float64)

        # Connecting vector from adjacent elements
        self._dx_adj = np.empty((ndims, nface))

        def pack(i_begin, i_end, buf, *dxc):
            # Save dxc to buf for communication
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(ndims):
                    buf[idx, jdx] = dxc[lti][lfi, lei, jdx]

        def compute_dxc(i_begin, i_end, dx_adj, buf, *dxc):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(ndims):
                    xl = dxc[lti][lfi, lei, jdx]
                    xr = buf[idx, jdx]

                    dx = xr - xl

                    dxc[lti][lfi, lei, jdx] = dx
                    dx_adj[jdx, idx] = dx

        # Pack dx
        self.be.make_loop(nface, pack)(buf, *dx)

        # Exchange halo
        comm.Sendrecv_replace(buf, dest=self._dest, source=self._dest)

        # Compute dxc
        self.be.make_loop(nface, compute_dxc)(self._dx_adj, buf, *dx)
