# -*- coding: utf-8 -*-
import numpy as np
import re

from solvers.base import BaseElements
from backends.types import ArrayBank, Kernel, NullKernel
from utils.np import eps, chop, npeval


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
        
        self.compute_fpts = Kernel(self._make_compute_fpts(), upts_in, fpts)

       
        if( (self._grad_method=='least-square') or (self._grad_method=='weighted-least-square')):
            self.compute_grad = Kernel(self._make_grad_ls(), fpts, grad)       
        elif(self._grad_method == 'green-gauss-cell'):
            self.compute_grad = Kernel(self._make_grad_gg(), fpts, grad)
        else:
            self.compute_grad = Kernel(self._make_grad_ls(), fpts, grad)

        
        # Kernel for linear reconstruction
        self.compute_recon = Kernel(self._make_recon(), upts_in, grad, lim, fpts)
        
        # Kernel to post-process
        self.post = Kernel(self._make_post(), upts_in)
 #-------------------------------------------------------------------------------#
    def compute_L2_norm(self):
        nvars, nface, ndims = self.nvars, self.nface, self.ndims
        vol                 = self._vol
        # **************************#

        #Complete

        # **************************#

        return resid

#-------------------------------------------------------------------------------#
    # Assign cell centers values to face centers
    def _make_compute_fpts(self):
        nvars, nface = self.nvars, self.nface

        def _compute_fpts(i_begin, i_end, upts, fpts):
            # Copy upts to fpts
            for idx in range(i_begin, i_end):
                for j in range(nvars):
                    # **************************#

                    #Complete

                    # **************************#
        
        return self.be.make_loop(self.neles, _compute_fpts)   
#-------------------------------------------------------------------------------#
    def _make_grad_ls(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        # Gradient operator 
        op = self._grad_operator

        def _cal_grad(i_begin, i_end, fpts, grad):
            # Elementwiseloop
            for i in range(i_begin, i_end):
                # **************************#

                #Complete

                # **************************#

        # Compile the function
        return self.be.make_loop(self.neles, _cal_grad)    
#-------------------------------------------------------------------------------#
    def _make_grad_gg(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        Normal vector and volume
        snorm_mag = self._mag_snorm_fpts
        snorm_vec = np.rollaxis(self._vec_snorm_fpts, 2)
        vol       = self._vol

        def _cal_grad(i_begin, i_end, fpts, grad):
            # Elementwise loop starts here
            for i in range(i_begin, i_end):
               
                # **************************#
                #Complete
                # **************************#

        # Compile the function
        return self.be.make_loop(self.neles, _cal_grad)      

#-------------------------------------------------------------------------------#
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



#-------------------------------------------------------------------------------#
    @property
    # @fc.lru_cache()
    # @chop
    def _grad_operator(self):
        # Difference of displacement vector (cell to cell)
        # (Nfaces, Nelements, dim) -> (dim, Nfaces, Nelements)
        dxc = np.rollaxis(self.dxc, 2)
        # (Nfaces, Nelements)
        distance = np.linalg.norm(dxc, axis=0)


        # **************************#
        #Complete
        # **************************#
        return op

    def _make_post(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        grad = self.grad
        xc = self.xc.T        
        def post(i_begin, i_end, upts):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):          
                print(idx, grad[0, 0, idx], grad[1, 0, idx])

        return self.be.make_loop(self.neles, post)
