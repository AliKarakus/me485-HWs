# -*- coding: utf-8 -*-
from utils.np import chop, npeval
from solvers.base import BaseElements
from backends.types import ArrayBank, Kernel, NullKernel

import numpy as np
import re
from utils.np import eps
import functools as fc
from utils.nb import dot

class AdvectionFluidElements:
    @property
    def auxvars(self):
        return ['v']

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

    @fc.lru_cache()
    def flux_container(self):
        ndims, nfvars = self.ndims, self.nfvars
        def flux(u, v, nf, f):
            # Compute normal component of flux
            for i in range(ndims):
                for j in range(nfvars):
                    f[j] += v[i]*nf[i]*u[j]
        return self.be.compile(flux)

    def fix_nonPys_container(self):
        # Constants and dimensions
        ndims, nfvars = self.ndims, self.nfvars

        def fix_nonPhy(u):
            u[0] = u[0]
        # Compile the function
        return self.be.compile(fix_nonPhy)

class AdvectionElements(BaseElements, AdvectionFluidElements):
#-------------------------------------------------------------------------------#
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars
        self._const = cfg.items('constants')
        self._flux = cfg.get('solver', 'flux', 'upwind')
        self._disc_method = cfg.get('solver', 'advection')
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
        self.fpts    = fpts    = np.empty((self.nface, self.nvars, self.neles))
        self.fext    = fext    = np.empty((2, self.nface, self.nvars, self.neles))
        self.velpts  = velpts  = np.empty((self.nface, self.ndims, self.neles))
        self.dt = np.empty(self.neles)

        if self.order > 1:
            # Array for gradient and limiter
            self.grad = grad = np.zeros((self.ndims, self.nvars, self.neles))

            # Prepare limiter
            lim = np.ones((self.nvars, self.neles))
            self.limiter = self.cfg.get('solver', 'limiter', 'none')
            # Prepare vertex array
            vpts = vertex.make_array(self.limiter)

            # Kernel for linear reconstruction
            self.compute_recon = Kernel( self._make_recon(), upts_in, grad, lim, fpts)

            if self.limiter != 'none':
                if(self.limiter=='mlp-u1' or self.limiter=='mlp-u2'):
                    # Kenerl to compute slope limiter (MLP-u)
                    self.compute_limiter = Kernel( self._make_mlp_u(self.limiter), upts_in, grad, vpts, lim)
                elif(self.limiter=='barth-jespersen' or self.limiter=='venkatarishnan'):
                    self.compute_limiter = Kernel( self._make_barth_jespersen(self.limiter), upts_in, grad, fext, lim)      
            else:
                self.compute_limiter = NullKernel
        else:
            self.compute_grad  = NullKernel
            self.compute_recon = NullKernel
            self.compute_mlp_u = NullKernel
        # Build kernels
        self.compute_vel  = Kernel(self._make_compute_vel(), velpts)
        # Kernel to compute flux points
        self.compute_fpts = Kernel(self._make_compute_fpts(), upts_in, fpts)

        # Kernel to compute divergence of solution
        self.div_upts = Kernel(self._make_div_upts(), upts_out, fpts)
        if self.order > 1:
            # Kernel to compute gradient
            self.compute_grad = Kernel(self._make_grad(), fpts, grad)
        else:
            self.compute_grad = NullKernel
        # Kernel to post-process
        self.post = Kernel(self._make_post(), upts_in)        
        # Kernel to compute timestep
        self.timestep = Kernel(self._make_timestep(), self.velpts, self.dt)
        # Kernel to compute residuals
        self.compute_norm = Kernel(self._make_compute_norm(), self.upts)

        self.compute_vel()


#-------------------------------------------------------------------------------#
    def _make_barth_jespersen(self, limiter):
        nface, ndims,  nvars = self.nface, self.ndims, self.nvars
        dxf = self.dxf
        # Compiler arguments
        array = self.be.local_array()
        # fext: max(0)/ min(1) of the cell center values on face 
        # size of [2,nface,nvars, nelem]
        
        # lim: limiter array: result of limiter function, size of [nvars, nelem] 
        # upts: solution at cell centers, size of [nvars, nelem] 
        # grad: gradient at cell centers, size of [ndims, nvars, nelem] 
        def _cal_barth_jespersen(i_begin, i_end, upts, grad, fext, lim):
            for i in range(i_begin, i_end):
               # complete the function















        return self.be.make_loop(self.neles, _cal_barth_jespersen)

#-------------------------------------------------------------------------------#
    def _make_compute_norm(self):
        import numba as nb
        vol = self._vol
        neles, nvars, ndims = self.neles, self.nvars, self.ndims
        xc = self.xc.T   
        def run(upts):
           # complete the function







        return self.be.compile(run, outer=True)


#-------------------------------------------------------------------------------#
    def _make_compute_vel(self):
        nvars, nface, ndims = self.nvars, self.nface, self.ndims
        xf = self.geom.xf(self.eles).swapaxes(1,2)
        xf = xf.swapaxes(0,1)

        # Parse velocity from expressions
        components = ['ux', 'uy', 'uz']
        subs = dict(zip('xyz', xf))
        vcs = [npeval(self.cfg.getexpr('soln-velocity', v, self._const), subs)
               for v in components[0:self.ndims]]
        vcs = np.array(vcs)

        full = 1 if vcs.shape == xf.shape else 0

        def _compute_vel(i_begin, i_end, velpts):
            # Copy upts to fpts
            for idx in range(i_begin, i_end):
                for k in range(nface):
                    for dim in range(ndims):
                        if(full):
                            velpts[k, dim, idx] = vcs[dim, k, idx]
                        else:
                            velpts[k, dim, idx] = vcs[dim]
        
        return self.be.make_loop(self.neles, _compute_vel)


#-------------------------------------------------------------------------------#
    def _make_compute_fpts(self):
        nvars, nface = self.nvars, self.nface
        def _compute_fpts(i_begin, i_end, upts, fpts):
            # Copy upts to fpts
            for idx in range(i_begin, i_end):
                # complete the function








        return self.be.make_loop(self.neles, _compute_fpts)

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
    def _make_grad(self):
        nface, ndims, nvars = self.nface, self.ndims, self.nvars
        # Gradient operator 
        op = self._prelsq
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

#-------------------------------------------------------------------------------#
    def _make_timestep(self):
        # Dimensions
        ndims, nface = self.ndims, self.nface
        dxf = np.linalg.norm(self.dxf, axis=1)
         # Characteristic length for u2 function
        h = (self.le)
        # Static variables
        smag, svec = self._gen_snorm_fpts()
        def timestep(i_begin, i_end, u, dt, cfl):
            for idx in range(i_begin, i_end):
                dx = h[idx]
                vel_mag = 0.0
                for f in range(nface):
                    vel_mag = max(np.sqrt(dot(u[f, :, idx], u[f, :, idx], ndims)), vel_mag)
                # Time step : 
                dt[idx] = cfl*dx/vel_mag

        return self.be.make_loop(self.neles, timestep)

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

#-------------------------------------------------------------------------------#
    def _make_post(self):
        # Get post-process function
        _fix_nonPys = self.fix_nonPys_container()

        def post(i_begin, i_end, upts):
            # Apply the function over eleemnts
            for idx in range(i_begin, i_end):
                _fix_nonPys(upts[:, idx])

        return self.be.make_loop(self.neles, post)
