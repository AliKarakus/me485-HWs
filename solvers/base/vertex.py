# -*- coding: utf-8 -*-
import numpy as np


class BaseVertex:
    def __init__(self, be, cfg, elemap, vtx, ivtx, neivtx):
        # Save arguments
        self.be = be
        self.cfg = cfg

        # Number of vertex
        self.nvtx = len(ivtx) - 1
        
        # Dimensions from element
        ele0 = elemap[next(iter(elemap))]
        self.ndims, self.nvars = ele0.ndims, ele0.nvars
        self.primevars = ele0.primevars

        # Get index of vertex
        self._idx = self._get_index(elemap, vtx)
        self._ivtx = ivtx
        self._neivtx = neivtx

        # Construct element-vertex connectivity
        self._construct_vcon(elemap, vtx, ivtx)
        # print(eles.vcon)

    def _construct_vcon(self, elemap, vtx, ivtx):
        # Read vertex
        vtx = np.array(vtx, dtype='U4,i4,i1,i1')

        # Sort vertex w.r.t (etype, vidx, eidx)
        idx = np.lexsort([vtx['f2'], vtx['f1'], vtx['f0']])
        # print(self.nvtx)
        # print(vtx, idx)


        # Sort element type
        etypes = vtx['f0'][idx]
        # print(etypes)

        # Break index for element
        bidx = np.where(etypes[:-1] != etypes[1:])[0] + 1
        bidx = np.array([0, *bidx, len(etypes)])

        # Assign local vertex index and sort
        vi = np.concatenate([
            [e]*(ivtx[i+1] - ivtx[i]) for e, i in enumerate(range(self.nvtx))
            ])
        vi = vi[idx]
        
        for i in range(len(bidx) - 1):
            etype = etypes[bidx[i]]
            vcon = vi[bidx[i]:bidx[i+1]]

            # Assign element-vertex connectivity 
            # for each elements: e1 - (v1, v2, v3,...)
            ele = elemap[etype]
            ele._vcon = vcon.reshape(-1, ele.geom.nvertex)
            # print(ele._vcon)

    def _get_index(self, elemap, vtx):
        # Parse index of vertex
        cell_nums = {c: i for i, c in enumerate(elemap)}
        return np.array([[cell_nums[t], e, v] for t, e, v, z in vtx]).T.copy()
