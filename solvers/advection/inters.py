# -*- coding: utf-8 -*-
from solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from backends.types import Kernel, NullKernel
from solvers.advection.bcs import get_bc
from solvers.advection.fsolvers import get_fsolver

from utils.np import npeval
from utils.nb import dot
import functools as fc
import numpy as np
import re

#-------------------------------------------------------------------------------#    
class AdvectionIntInters(BaseIntInters):
    def construct_kernels(self, elemap, impl_op):
        # View of elemenet array
        self._fpts   = fpts   = [cell.fpts for cell in elemap.values()]
        self._velpts = velpts = [cell.velpts for cell in elemap.values()]
        self._fext   = fext   = [cell.fext for cell in elemap.values()]

        # Array for gradient at face
        self._gradf  = gradf   = np.empty((self.ndims, self.nvars, self.nfpts))
        if self.order > 1:
            self.compute_delu   = Kernel(self._make_delu(),   *fpts)
            self.compute_minmax = Kernel(self._make_minmax(), fpts, *fext)
        else:
            self.compute_delu   = NullKernel
            self.compute_minmax = NullKernel

        muf = np.empty(self.nfpts)
        self.compute_flux = Kernel(self._make_flux(), self._velpts, *fpts)

#-------------------------------------------------------------------------------#    
    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm
        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux_func' : self.ele0.flux_container(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'flux')
        flux = get_fsolver(scheme, self.be, cplargs)

        def comm_flux(i_begin, i_end, vf, *uf):
            for idx in range(i_begin, i_end):
                # flux function to be filled
                fn = array(nfvars)
                 #---------------------------------#  
                # complete the function
                #---------------------------------#  

                # call the numerical flux function here : i.e. upwind or rusanov
                #flux(ul, ur, vl, vr, nfi, fn)







                for jdx in range(nfvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] =  fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)
#-------------------------------------------------------------------------------#    
    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = uf[rti][rfi, jdx, rei]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] =  du
                    uf[rti][rfi, jdx, rei] = -du

        return self.be.make_loop(self.nfpts, compute_delu)

    #-------------------------------------------------------------------------------#    
    def _make_minmax(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        def compute_minmax(i_begin, i_end, uf, *uext):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                #---------------------------------#  
                # complete the function
                #---------------------------------#  




        return self.be.make_loop(self.nfpts, compute_minmax)
#-------------------------------------------------------------------------------#    
class AdvectionMPIInters(BaseMPIInters):
    _tag = 1234

    def construct_kernels(self, elemap, impl_op):
        # Buffers
        lhs = np.empty((self.nvars, self.nfpts))
        self._rhs = rhs = np.empty((self.nvars, self.nfpts))

        # View of element array
        self._fpts   = fpts = [cell.fpts for cell in elemap.values()]
        self._fext   = fext = [cell.fext for cell in elemap.values()]
        self._velpts = velpts = [cell.velpts for cell in elemap.values()]

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(self._make_delu(), rhs, *fpts)

        self.compute_flux = Kernel(self._make_flux(), rhs, self._velpts, *fpts)

        # Kernel for pack, send, receive
        self.pack = Kernel(self._make_pack(), lhs, *fpts)
        self.send, self.sreq = self._make_send(lhs)
        self.recv, self.rreq = self._make_recv(rhs)
#-------------------------------------------------------------------------------#    
    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # mu = self.ele0._const['mu']

       # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux_func' : self.ele0.flux_container(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'flux')
        flux = get_fsolver(scheme, self.be, cplargs)

        def comm_flux(i_begin, i_end, rhs, vel, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)            
                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                ur = rhs[:, idx]

                vl = vf[lti][lfi, :, lei]
                # assume uniform velocity
                vr = vl

                 # Compute approixmate Riemann solver
                flux(ul, ur, vl, vr, nfi, fn)

                for jdx in range(nfvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] =  fn[jdx]*sf[idx]


        return self.be.make_loop(self.nfpts, comm_flux)

#-------------------------------------------------------------------------------#    
    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx

        def compute_delu(i_begin, i_end, rhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = rhs[jdx, idx]
                    du = ur - ul
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)

#-------------------------------------------------------------------------------#    
    def _make_pack(self):
        nvars = self.nvars
        lt, le, lf = self._lidx

        def pack(i_begin, i_end, lhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    lhs[jdx, idx] = uf[lti][lfi, jdx, lei]

        return self.be.make_loop(self.nfpts, pack)

#-------------------------------------------------------------------------------#    
    def _sendrecv(self, mpifn, arr):
        # MPI Send or Receive init
        req = mpifn(arr, self._dest, self._tag)

        def start(q):
            # Function to save request in queue and start Send/Receive
            q.register(req)
            return req.Start()

        # Return Non-blocking send/recive and request (for finalise)
        return start, req

#-------------------------------------------------------------------------------#    
    def _make_send(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Send_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req

#-------------------------------------------------------------------------------#    
    def _make_recv(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Recv_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req

#-------------------------------------------------------------------------------#    
class AdvectionBCInters(BaseBCInters):
    _get_bc = get_bc
    def construct_bc(self):
        # Parse BC function name
        bcf = re.sub('-', '_', self.name)

        # Constants for BC function
        if self._reqs:
            bcsect = 'soln-bcs-{}'.format(self.bctype)
            bcc = {k: npeval(self.cfg.getexpr(bcsect, k, self._const))
                   for k in self._reqs}
        else:
            bcc = {}

        bcc['ndims'], bcc['nvars'], bcc['nfvars'] = self.ndims, self.nvars, self.nfvars

        bcc.update(self._const)
        # print(bcc)

        # Get bc from `bcs.py` and compile them
        self.bc = self._get_bc(self.be, bcf, bcc)

#-------------------------------------------------------------------------------#    
    def construct_kernels(self, elemap, impl_op):
        self.construct_bc()

        # View of elemenet array
        self._fpts   = fpts   = [cell.fpts for cell in elemap.values()]
        self._velpts = velpts = [cell.velpts for cell in elemap.values()]
        self._fext   = fext   = [cell.fext for cell in elemap.values()]


        # dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

       # # Gradient at face
       #  self._gradf = gradf = np.empty((self.ndims, self.nvars, self.nfpts))

        if self.order > 1:
            self.compute_delu = Kernel(self._make_delu(), *fpts)
            self.compute_minmax = Kernel(self._make_minmax(), fpts, *fext)
        else:
            self.compute_delu   = NullKernel
            self.compute_minmax = NullKernel


        # Kernel to compute flux
        self.compute_flux = Kernel(self._make_flux(nele), self._velpts, *fpts)

#-------------------------------------------------------------------------------#    
    def _make_flux(self, nele):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm
        
        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux_func' : self.ele0.flux_container(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'flux')
        flux = get_fsolver(scheme, self.be, cplargs)
        # Get bc function 
        bc = self.bc

        def comm_flux(i_begin, i_end, vf, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)
                #---------------------------------#  
                # complete the function
                #---------------------------------#  











                for jdx in range(nfvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)
#-------------------------------------------------------------------------------#    
    def _make_delu(self):
        nvars = self.nvars
        ndims = self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc
        array = self.be.local_array()

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                ur = array(nvars)
                vr = array(ndims)
                vl = array(ndims)
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                bc(ul, ur, vl, vr, nfi)
                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)

#-------------------------------------------------------------------------------#    
    def _make_minmax(self):
        nvars = self.nvars
        ndims = self.ndims
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc
        array = self.be.local_array()

        def compute_minmax(i_begin, i_end, uf, *uext):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ur = array(nvars)
                vr = array(ndims)
                vl = array(ndims)
                #---------------------------------#  
                # complete the function
                #---------------------------------#  








        return self.be.make_loop(self.nfpts, compute_minmax)
#-------------------------------------------------------------------------------#    
class AdvectionDrichletBCInters(AdvectionBCInters):
    name = 'drichlet'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        self._reqs = self.primevars


#-------------------------------------------------------------------------------#    
class AdvectionNeumannBCInters(AdvectionBCInters):
    name = 'neumann'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        self._reqs = self.primevars
