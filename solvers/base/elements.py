# -*- coding: utf-8 -*-
import functools as fc
import numpy as np

from geometry import get_geometry
from utils.np import chop, npeval


class BaseElements:
    name = 'base'

    def __init__(self, be, cfg, name, eles):
        # Argument save
        self.be = be
        self.cfg = cfg
        self.name = name
        self.eles = eles


        # Dimensions
        self.nvtx, self.neles, self.ndims = self.eles.shape
        self.geom = get_geometry(name)

        # print(self.geom.fcls)
        self.nface = nface = self.geom.nface

        # Order of spatial discretization
        self.order = order = cfg.getint('solver', 'order', 1)

        if order > 1:
            self.dxc = self.xc - self.xf
        
        # print(self.xc.shape, self.xf.shape, self.dxf.shape, self.dxv.shape)
        # Gradient method
        self._grad_method = cfg.get('solver', 'gradient', 'hybrid').lower()

    def coloring(self):
        try:
            color = self._coloring_nx()
        except:
            color = self._coloring_greedy()

         # Save colors as linked-list
        ncolor = np.cumsum([sum(color==i) for i in range(color.max()+1)])
        icolor = np.argsort(color)

        return ncolor, icolor, color

    def _coloring_nx(self):
        from scipy import sparse
        import networkx as nx

        # Multi-Coloring (greedy)
        graph = self.graph
        indptr  = graph['indptr']
        indices = graph['indices']

        # Build adjacent array (CSR)
        n = len(indptr) - 1
        adj =sparse.csr_array((np.ones_like(indices), indices, indptr), shape=(n,n))

        # Build graph
        G = nx.from_scipy_sparse_array(adj)

        # Greedy coloring
        strategy = self.cfg.get('solver-time-integrator', 'coloring', 'largest_first')
        col_dict = nx.greedy_color(G, strategy=strategy)
        color = np.array([col_dict[k] for k in sorted(col_dict)]) + 1

        return color        

    def _coloring_greedy(self):
        # Multi-Coloring (greedy)
        #TODO: Check computing cost (pure python implementation)
        graph = self.graph
        indptr  = graph['indptr']
        indices = graph['indices']

        degrees = np.diff(indptr)
        xn = np.sum(self.xc, axis=1)

        color = np.zeros(self.neles, dtype=int)
        avail_colors = set(range(1, max(degrees)+2))
        nei_colors = np.empty(max(degrees)+1, dtype=int)

        # Search Coloring (Search along hyperplane and max degrees)
        for idx in np.lexsort([xn, -degrees]):
            # Seach colors of neighboring cells
            n = 0
            for jdx in range(indptr[idx], indptr[idx+1]):                
                nei = indices[jdx]
                nei_color = color[nei]

                if nei_color > 0:
                    nei_colors[n] = nei_color
                    n += 1

            # Find current color (greedy)
            c = min(avail_colors - set(nei_colors[:n]))
            color[idx] = c

        return color

    def reordering(self):
        try:
            # Use Scipy sparse packages
            from scipy import sparse
            from scipy.sparse.csgraph import reverse_cuthill_mckee

            # Convert graph to csr sparse matrix    
            graph = self.graph
            mtx = sparse.csr_matrix(
                (np.ones_like(graph['indices']), graph['indices'], graph['indptr'])
            )

            # reverse Cuthill MacKee reordering
            mapping = reverse_cuthill_mckee(mtx)
            unmapping = np.argsort(mapping)

        except:
            # If Scipy is not existed
            mapping = np.arange(self.neles, dtype=int)
            unmapping = np.arange(self.neles, dtype=int)

        return mapping, unmapping

    def set_ics_from_cfg(self):
        xc = self.geom.xc(self.eles).T

        # Parse initial condition from expressions
        subs = dict(zip('xyz', xc))       
        ics = [npeval(self.cfg.getexpr('soln-ics', v, self._const), subs)
               for v in self.primevars]
        ics = self.prim_to_conv(ics, self.cfg)

        # Allocate numpy array and copy parsed values
        self._ics = np.empty((self.nvars, self.neles))
        for i in range(self.nvars):
            self._ics[i] = ics[i]

    def set_ics_from_sol(self, sol):
        # Just copy provided solution array
        self._ics = sol.astype(float)

    @property
    @fc.lru_cache()
    def _vol(self):
        # Volume of element
        return np.abs(self.geom.vol(self.eles))

    @property
    @fc.lru_cache()
    def tot_vol(self):
        # Sum of element volumes
        return np.sum(self._vol)

    @property
    @fc.lru_cache()
    def rcp_vol(self):
        # recipropal of volume of element
        return 1/np.abs(self._vol)

    @fc.lru_cache()
    def _gen_snorm_fpts(self):
        # Check the direction of mesh (right hand side or left hand side)
        sign = np.sign(self.geom.vol(self.eles))[..., None]

        # Compute surface normal vector
        snorm = self.geom.snorm(self.eles)

        # Split snorm as magnitude and direction vector
        mag = np.einsum('...i,...i', snorm, snorm)
        mag = np.sqrt(mag)
        vec = snorm / mag[..., None]*sign
        return mag, vec

    @property
    def _mag_snorm_fpts(self):
        # Save magnitude of surface normal vector at each face point
        return self._gen_snorm_fpts()[0]

    @property
    def _vec_snorm_fpts(self):
        # Save direction vector of surface normal vector at each face point
        return self._gen_snorm_fpts()[1]

    @property
    def mag_fnorm(self):
        # Public property of magnitude of surface normal vector
        return self._mag_snorm_fpts

    @property
    @fc.lru_cache()
    def vec_fnorm(self):
        # Public proporty of direction vector of surface noraml
        return self._vec_snorm_fpts.swapaxes(1, 2).copy()

    @property
    def _perimeter(self):
        # Perimeter (or sum of all area) of element
        return np.sum(self._mag_snorm_fpts, axis=0)

    @property
    @fc.lru_cache()
    def le(self):
        # Characteristic length of cell : vol / sum(S)
        return 1/(self.rcp_vol * self._perimeter)

    @property
    @fc.lru_cache()
    def xc(self):
        # Cell center point
        return self.geom.xc(self.eles)

    @property
    @fc.lru_cache()
    def xf(self):
        # Face center point
        return self.geom.xf(self.eles)

    @property
    @fc.lru_cache()
    @chop
    def _prelsq(self):
        # Difference of displacement vector (cell to cell)
        dxc = np.rollaxis(self.dxc, 2)
        distance = np.linalg.norm(dxc, axis=0)

        # Normal vector and volume
        snorm_mag = self._mag_snorm_fpts
        snorm_vec = np.rollaxis(self._vec_snorm_fpts, 2)
        vol = self._vol

        if self._grad_method == 'least-square':
            beta, w = 1.0, 1.0
        elif self._grad_method == 'weighted-least-square':
            # Invserse distance weight
            beta, w = 1.0, 1/distance**2
        elif self._grad_method == 'green-gauss':
            beta, w = 0.0, 1.0
        elif self._grad_method == 'hybrid':
            # Shima et al., Green–Gauss/Weighted-Least-Squares
            # Hybrid Gradient Reconstruction for
            # Arbitrary Polyhedra Unstructured Grids, AIAA J., 2013
            # WLSQ(G)
            dxf = self.dxf.swapaxes(0, 1)

            dxcn = np.einsum('ijk,ijk->jk', dxc, snorm_vec)
            dxfn = np.einsum('ijk,ijk->jk', dxf, snorm_vec)

            w =  (2*dxfn/dxcn)**2*snorm_mag / distance

            # Compute blending function (GLSQ)
            ar = 2*np.linalg.norm(self.dxf, axis=1).max(axis=0)*snorm_mag.max(axis=0)/vol
            beta = np.minimum(1, 2/ar)
        else:
            raise ValueError("Invalid gradient method : ", self._grad_method)

        # Scaled dxc vector
        dxcs = dxc*np.sqrt(w)

        # Least square matrix [dx*dy] and its inverse
        lsq = np.array([[np.einsum('ij,ij->j', x, y)
                         for y in dxcs] for x in dxcs])

        # Hybrid type of Ax=b
        A = beta*lsq + 2*(1-beta)*vol*np.eye(self.ndims)[:,:,None]
        b = beta*dxc*w + 2*(1-beta)*0.5*snorm_vec*snorm_mag

        # Solve Ax=b
        op = np.linalg.solve(np.rollaxis(A, axis=2), np.rollaxis(b, axis=2)).transpose(1,2,0)

        return op

    @property
    @fc.lru_cache()
    def dxf(self):
        # Displacement vector of face center from cell center
        return self.geom.dxf(self.eles).swapaxes(1, 2)

    @property
    @fc.lru_cache()
    def dxv(self):
        # Displacement vector of vertex from cell center
        return self.geom.dxv(self.eles).swapaxes(1, 2)
