# -*- coding: utf-8 -*-
import numpy as np
import re

from solvers.base import BaseElements
from backends.types import ArrayBank, Kernel, NullKernel
from utils.np import eps
import functools as fc

class ParabolicFluidElements:
    @property
    def auxvars(self):
        return ['mu']

    @fc.lru_cache()
    def mu_container(self):
        mu = self._const['mu']
        def compute_mu(*args):
            return mu
        return self.be.compile(compute_mu)

    @property
    def primevars(self):
        # Primitive variables
        return ['q'] 
    @property
    def conservars(self):
        # Conservative variables
        pri = self.primevars
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

class ParabolicElements(BaseElements, ParabolicFluidElements):
#-------------------------------------------------------------------------------#
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars
        self._const = cfg.items('constants')
        self._correction = cfg.get('solver', 'correction', 'minimum')
        self._disc_method = cfg.get('solver', 'parabolic')

#-------------------------------------------------------------------------------#   
    def construct_kernels(self, vertex, nreg, impl_op):
        self.vertex = vertex
        # Upts : Solution vector
        self.upts = upts = [self._ics.copy() for i in range(nreg)]
        del(self._ics)

        # Solution vector bank and assign upts index
        self.upts_in = upts_in = ArrayBank(upts, 0)
        self.upts_out = upts_out = ArrayBank(upts, 1)

        # Construct arrays for flux points, dt and derivatives of source term
        self.fpts = fpts = np.empty((self.nface, self.nvars, self.neles))
        self.dt = np.empty(self.neles)
        self.dsrc = np.zeros((self.nvars, self.neles))

        if self.order > 1:
            # Array for gradient and limiter
            self.grad = grad = np.zeros((self.ndims, self.nvars, self.neles))
            lim = np.ones((self.nvars, self.neles))
            limiter = self.cfg.get('solver', 'limiter', 'none')

            # Prepare vertex array
            vpts = vertex.make_array(limiter)

        # Build kernels

        # Kernel to compute flux points
        self.compute_fpts = Kernel(self._make_compute_fpts(), upts_in, fpts)

        # Kernel to compute divergence of solution
        self.div_upts = Kernel(self._make_div_upts(), upts_out, fpts)

        # Kernel to compute residuals
        self.compute_norm = Kernel(self._make_compute_norm(), self.upts_out)

        if self.order > 1:
            # Kernel to compute gradient
            self.compute_grad = Kernel(self._make_grad(), fpts, grad)
        else:
            self.compute_grad = NullKernel

        # Kernel to post-process
        self.post = Kernel(self._make_post(), upts_in)
        # Kernel to compute timestep
        self.timestep = Kernel(self._make_timestep(), self.upts_in, self.dt)

#-------------------------------------------------------------------------------#
    def _make_compute_norm(self):
        #*************************# 
        # get required data here




        #*************************# 
        def run(upts):
        #*************************# 
        # compute L2 norm in this function
        # upts is the solution field




        #*************************# 

            return norm

        return self.be.compile(run, outer=True)

#-------------------------------------------------------------------------------#
    def _make_compute_fpts(self):
        #*************************# 
        # get required data here
        #*************************# 
        def _compute_fpts(i_begin, i_end, upts, fpts):
        #*************************# 
        # Complete function
        # upts: array holding cell center values
        # fpts: array holding face values




        #*************************# 
            
        
        return self.be.make_loop(self.neles, _compute_fpts)

#-------------------------------------------------------------------------------#
    def _make_grad(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        # Gradient operator 
        op = self._prelsq
        def _cal_grad(i_begin, i_end, fpts, grad):
        #*************************# 
        # Complete function
        # grad: array holding cell center gradient values
        # fpts: array holding face values

        #*************************# 


        # Compile the numba function
        return self.be.make_loop(self.neles, _cal_grad)   


#-------------------------------------------------------------------------------#
    def _make_div_upts(self):
        # Global variables for compile
        gvars = {"np": np, "rcp_vol": self.rcp_vol}

        # Position, constants and numerical functions
        subs = {x: 'xc[{0}, idx]'.format(i)
                for i, x in enumerate('xyz'[:self.ndims])}
        subs.update(self._const)
        subs.update({'sin': 'np.sin', 'cos': 'np.cos',
                     'exp': 'np.exp', 'tanh': 'np.tanh'})

        # Parase source term
        src = [self.cfg.getexpr('solver-source-terms', k, subs, default=0.0)
               for k in self.conservars]

        # Parse xc in source term
        if any([re.search(r'xc\[.*?\]', s) for s in src]):
            gvars.update({"xc": self.xc.T})

        # Construct function text
        f_txt = (
            f"def _div_upts(i_begin, i_end, rhs, fpts, t=0):\n"
            f"    for idx in range(i_begin, i_end): \n"
            f"        rcp_voli = rcp_vol[idx]\n"
        )
        for j, s in enumerate(src):
            subtxt = "+".join("fpts[{},{},idx]".format(i, j)
                              for i in range(self.nface))
            f_txt += "        rhs[{}, idx] = -rcp_voli*({}) + {}\n".format(
                j, subtxt, s)

        # Execute python function and save in lvars
        lvars = {}
        exec(f_txt, gvars, lvars)

        # Compile the funtion
        return self.be.make_loop(self.neles, lvars["_div_upts"])


#-------------------------------------------------------------------------------#
    def _make_timestep(self):
        # Dimensions
        ndims, nface = self.ndims, self.nface
        dxf = np.linalg.norm(self.dxf, axis=1)

        # Static variables
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()

        # # Constants
        mu = self._const['mu']
        def timestep(i_begin, i_end, u, dt, cfl):
            for idx in range(i_begin, i_end):
                lamdf = 1000.0
                Ve = vol[idx]
                for jdx in range(nface):
                    Af = smag[jdx, idx]
                    df = dxf[jdx, idx]
                    lc = Af*Af/(2*df)
                    lamdf = min(Ve/lc,lamdf)
                # Time step : 
                dt[idx] = cfl * lamdf*lamdf/(2*ndims*mu)

        return self.be.make_loop(self.neles, timestep)
        
#-------------------------------------------------------------------------------#
    def _make_post(self):
        # Get post-process function
        _fix_nonPys = self.fix_nonPys_container()

        def post(i_begin, i_end, upts):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])

        return self.be.make_loop(self.neles, post)
