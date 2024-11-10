# -*- coding: utf-8 -*-
import numpy as np
import re

from solvers.base import BaseElements
from backends.types import ArrayBank, Kernel, NullKernel
from utils.np import eps


class gradFluidElements:
    @property
    def primevars(self):
        # Primitive variables
        return ['q'] 
    @property
    def conservars(self):
        # Conservative variables
        pri = self.primevars

        # rho,rhou,rhov,rhow,E
        return [pri[0]]

    def prim_to_conv(self, pri, cfg):
        return [pri[0]]

    def conv_to_prim(self, con, cfg):
        return [con[0]]

    def fix_nonPys_container(self):
        # Constants and dimensions
        # gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def fix_nonPhy(u):
            u[0] = u[0]
        # Compile the function
        return self.be.compile(fix_nonPhy)

class GradElements(BaseElements,  gradFluidElements):
    nreg = 1
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars
        self._const = cfg.items('constants')
        self._grad_method = cfg.get('solver', 'gradient')
        # print(self.dxv)

    def construct_kernels(self, vertex, nreg, impl_op=0):
        self.vertex = vertex

        # Upts : Solution vector
        self.upts = upts = [self._ics.copy() for i in range(nreg)]
        del(self._ics)

        # Solution vector bank and assign upts index
        self.upts_in = upts_in = ArrayBank(upts, 0)
        self.upts_out = upts_out = ArrayBank(upts, 1)

        # Construct arrays for flux points, dt and derivatives of source term
        self.fpts = fpts = np.empty((self.nface, self.nvars, self.neles))
        self.grad = grad = np.zeros((self.ndims, self.nvars, self.neles))

        lim = np.ones((self.nvars, self.neles))

        limiter = self.cfg.get('solver', 'limiter', 'none')
        # Prepare vertex array
        vpts = vertex.make_array(limiter)

        # Build kernels
        # Kernel to compute flux points
        if(self._grad_method == 'green-gauss-node'):
            # vpts_in = [upts_in for ele in elemap.values()]
            self.compute_avgv = Kernel(self._make_compute_avgv(), vpts, upts_in)            
            self.compute_fpts = Kernel(self._make_compute_fpts_ggn(),self.geom._face, self._vcon.T, vpts, fpts)
        else:
            self.compute_fpts = Kernel(self._make_compute_fpts(), upts_in, fpts)

        if(self._grad_method=='least-square'):
            self.compute_grad = Kernel(self._make_grad_ls(), fpts, grad)
        elif(self._grad_method == 'green-gauss-cell' or self._grad_method == 'green-gauss-node'):
            self.compute_grad = Kernel(self._make_grad_gg(), fpts, grad)
        else:
            self.compute_grad = Kernel(self._make_grad_ls(), fpts, grad)

        
        # Kernel for linear reconstruction
        self.compute_recon = Kernel(self._make_recon(), upts_in, grad, lim, fpts)
        
        # Kernel to compute residuals
        # self.compute_residual = Kernel(self._make_compute_residual(), self.upts_out)

        if limiter != 'none':
            # Kenerl to compute slope limiter (MLP-u)
            self.compute_mlp_u = Kernel(self._make_mlp_u(limiter), upts_in, grad, vpts, lim)
        else:
            self.compute_mlp_u = NullKernel

        # Kernel to post-process
        self.post = Kernel(self._make_post(), upts_in)

#-------------------------------------------------------------------------------#
    def compute_residual(self):
        nvars, nface, ndims = self.nvars, self.nface, self.ndims
        vol = self._vol
        resid = np.empty([ndims,nvars])
        for k in range(nvars):
            for j in range(ndims):
                rsum = 0
                for i in range(self.neles):
                    rsum += self.grad[j, k, i]**2*vol[i]
                resid[j,k] = rsum        
        return resid
#-------------------------------------------------------------------------------#
    # Assign cell centers values to face centers
    def _make_compute_fpts(self):
        nvars, nface = self.nvars, self.nface

        def _compute_fpts(i_begin, i_end, upts, fpts):
            # Copy upts to fpts
            for idx in range(i_begin, i_end):
                for j in range(nvars):
                    tmp = upts[j, idx]
                    for k in range(nface):
                        fpts[k, j, idx] = tmp
        
        return self.be.make_loop(self.neles, _compute_fpts)

#-------------------------------------------------------------------------------#
    # Assign cell centers values to face centers
    def _make_compute_fpts_ggn(self):
        nvars, nface = self.nvars, self.nface
        def _compute_fpts(i_begin, i_end, face, etov, vpts, fpts):
            for idx in range(i_begin, i_end):
                for j in range(nvars):
                    for k in range(nface):
                        fnodes = face[k][1]
                        fpts[k, j, idx] = 0.0
                        for l in range(len(fnodes)): 
                            vid = etov[fnodes[l], idx]
                            fpts[k, j, idx] += vpts[0, j, vid]
                        
                        fpts[k, j, idx] /= len(fnodes)

        return self.be.make_loop(self.neles, _compute_fpts)

#-------------------------------------------------------------------------------#
    def _make_compute_avgv(self):
        # vtx  = self.vertex._vtx
        ivtx     = self.vertex._ivtx
        t, e, v  = self.vertex._idx
        nvars    = self.nvars
        neivtx   = self.vertex._neivtx
        #inverse distance
        idist    = 1.0/np.linalg.norm(self.dxv, axis=1)

        def cal_avgv(i_begin, i_end, vavg, upts):
            for i in range(i_begin, i_end):
                wt = 0
                vavg[:, :, i] = 0.0
                for idx in range(ivtx[i], ivtx[i+1]):
                    # element/local vertex id connected to global vertex i
                    ei = e[idx]
                    vi = v[idx]
                    # inverse distance weights
                    wi = idist[vi, ei]
                    # sum of the weights
                    wt += wi
                    for jdx in range(nvars):
                        vavg[0, jdx, i] += wi*upts[jdx, ei]

                #get wighted distance for the vertex            
                vavg[0, :, i] /= wt                    
        return self.be.make_loop(self.vertex.nvtx, cal_avgv)
#-------------------------------------------------------------------------------#
    def _make_grad_ls(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars

        # Gradient operator 
        op = self._grad_operator

        def _cal_grad(i_begin, i_end, fpts, grad):
            # Elementwise dot product
            # TODO: Reduce accesing global array
            for i in range(i_begin, i_end):
                for l in range(nvars):
                    for k in range(ndims):
                        tmp = 0
                        for j in range(nface):
                            tmp += op[k, j, i]*fpts[j, l, i]
                        grad[k, l, i] = tmp

        # Compile the function
        return self.be.make_loop(self.neles, _cal_grad)    

    def _make_grad_gg(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
         # Normal vector and volume
        snorm_mag = self._mag_snorm_fpts
        snorm_vec = np.rollaxis(self._vec_snorm_fpts, 2)
        vol       = self._vol


        def _cal_grad(i_begin, i_end, fpts, grad):
            # Elementwise dot product
            for i in range(i_begin, i_end):
                evol = vol[i]; 
                for l in range(nvars):
                    for k in range(ndims):
                        tmp = 0
                        for j in range(nface):
                            tmp +=  snorm_vec[k, j, i]*snorm_mag[j,i]*fpts[j, l, i]
                        grad[k, l, i] = tmp / evol

        # Compile the function
        return self.be.make_loop(self.neles, _cal_grad)      



    def _make_recon(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars

        # Displacement vector
        op = self.dxf

        def _cal_recon(i_begin, i_end, upts, grad, lim, fpts):
            # Elementwise dot product and scale with limiter
            # TODO: Reduce accesing global array
            for i in range(i_begin, i_end):
                for l in range(nvars):
                    for k in range(nface):
                        tmp = 0
                        for j in range(ndims):
                            tmp += op[k, j, i]*grad[j, l, i]
                                                    
                        fpts[k, l, i] = upts[l, i] + lim[l, i]*tmp

        return self.be.make_loop(self.neles, _cal_recon)



    def _make_mlp_u(self, limiter):
        nvtx, ndims, nvars = self.nvtx, self.ndims, self.nvars

        dx = self.dxv
        cons = self._vcon.T

        def u1(dup, dum, ee2):
            # u1 function
            return min(1.0, dup/dum)

        def u2(dup, dum, ee2):
            # u2 function
            dup2 = dup**2
            dum2 = dum**2
            dupm = dup*dum
            return ((dup2 + ee2)*dum + 2*dum2*dup)/(dup2 + 2*dum2 + dupm + ee2)/dum

        # x_i^1.5 : Characteristic length for u2 function
        le32 = self.le**1.5

        if limiter == 'mlp-u2':
            is_u2 = True
            u2k = self.cfg.getfloat('solver', 'u2k', 5.0)

            # Don't use ee2 for very small u2k
            if u2k < eps:
                is_u2 = False

            limf = self.be.compile(u2)
        else:
            is_u2 = False
            u2k = 0.0
            limf = self.be.compile(u1)

        def _cal_mlp_u(i_begin, i_end, upts, grad, vext, lim):
            for i in range(i_begin, i_end):
                for j in range(nvtx):
                    vi = cons[j, i]
                    for k in range(nvars):
                        duv = 0

                        if is_u2:
                            # parameter for u2 
                            dvv = vext[0, k, vi] - vext[1, k, vi]
                            ee = dvv / le32[i] / u2k
                            ee2 = u2k*dvv**2/(ee + 1.0)
                        else:
                            ee2 = 0.0

                        # Difference of values between vertex and cell-center
                        for l in range(ndims):
                            duv += dx[j, l, i]*grad[l, k, i]

                        # MLP-u slope limiter
                        if duv > eps:
                            limj = limf(
                                (vext[0, k, vi] - upts[k, i]), duv, ee2)
                        elif duv < -eps:
                            limj = limf(
                                (vext[1, k, vi] - upts[k, i]), duv, ee2)
                        else:
                            limj = 1.0

                        if j == 0:
                            lim[k, i] = limj
                        else:
                            lim[k, i] = min(lim[k, i], limj)

        return self.be.make_loop(self.neles, _cal_mlp_u)

    @property
    # @fc.lru_cache()
    # @chop
    def _grad_operator(self):
        # Difference of displacement vector (cell to cell)
        # (Nfaces, Nelements, dim) -> (dim, Nfaces, Nelements)
        dxc = np.rollaxis(self.dxc, 2)

        # (Nfaces, Nelements)
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

    def _make_post(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        grad = self.grad
        # # Get post-process function
        # _fix_nonPys = self.fix_nonPys_container()

        def post(i_begin, i_end, upts):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                # for j in range(ndims):
                print(grad[0, 0, idx],grad[1, 0, idx])

        return self.be.make_loop(self.neles, post)
