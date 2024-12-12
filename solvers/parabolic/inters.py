# -*- coding: utf-8 -*-
from solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from backends.types import Kernel, NullKernel
from solvers.parabolic.bcs import get_bc
# from solvers.parabolic.visflux import make_visflux

from utils.np import npeval
from utils.nb import dot
import functools as fc
import numpy as np
import re

#-------------------------------------------------------------------------------#    
class ParabolicIntInters(BaseIntInters):
    def construct_kernels(self, elemap, impl_op):
        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

        self._correction = self.cfg.get('solver', 'correction', 'minimum')
       
        # Array for gradient at face
        self._gradf  = gradf   = np.empty((self.ndims, self.nvars, self.nfpts))
        self._weight = weight  = np.empty((self.nfpts))

        if self.order > 1:
            self.compute_delu = Kernel(self._make_delu(), *fpts)
            dxf = [cell.dxf for cell in elemap.values()]        
            self.compute_weight(dxf)
        else:
            self.compute_delu = NullKernel

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at_face = Kernel(
                                    self._make_grad_at_face(nele), gradf, *fpts, *dfpts)

        muf = np.empty(self.nfpts)
        self.compute_flux = Kernel(self._make_flux(nele), muf, gradf, *fpts)

#-------------------------------------------------------------------------------#    
    def compute_weight(self, dxf):
        nface, ndims = self.nfpts, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        for idx in range(self.nfpts):
            lti, lfi, lei = lt[idx], lf[idx], le[idx]
            rti, rfi, rei = rt[idx], rf[idx], re[idx]

            dl = 0.0
            dr = 0.0
            for jdx in range(ndims):
                xjl = dxf[lti][lfi, jdx, lei]
                xjr = dxf[rti][rfi, jdx, rei]
                dl += xjl*xjl
                dr += xjr*xjr

            dl  = np.sqrt(dl)
            dr  = np.sqrt(dr)

            self._weight[idx] = dr/(dl+dr)

#-------------------------------------------------------------------------------#    
    def _make_flux(self, nele):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Inverse distance between the elements
        inv_ef    = self._rcp_dx
        # unit vector connecting cell centers 
        ef = self._dx_adj * inv_ef

        correction = self._correction

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        # visflux = make_visflux(self.be, cplargs)

        def comm_flux(i_begin, i_end, muf, gradf, *uf):
            # Parse element views (fpts, grad)
            du    = uf[:nele]
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                fn = array(nfvars)
                Tf = array(ndims)
                # Normal vector
                nfi = nf[:, idx]
                # Gradient and solution at face
                gf = gradf[:,:, idx]
                # Compute viscosity and viscous flux
                muf[idx] = mu = compute_mu()
                
                inv_efi  = inv_ef[idx]
                efi = ef[:, idx]
                sfi = sf[idx]

                if(correction=='minimum'):
                    # Minimum Correction Approach
                    alpha = dot(efi, nfi, ndims)
                    Ef = sfi*alpha
                    for dim in range(ndims):
                        Tf[dim] = sfi*(nfi[dim] - alpha*efi[dim]) 
                elif(correction=='orthogonal'):
                    # Orthogonal Correction Approach
                    alpha = 1.0
                    Ef = sfi*alpha
                    for dim in range(ndims):
                        Tf[dim] = sfi*(nfi[dim] - alpha*efi[dim]) 
                elif(correction=='over_relaxed'):
                    # Over Relaxed Correction Approach
                    alpha = dot(efi, nfi, ndims)
                    Ef = sfi*1.0/alpha
                    for dim in range(ndims):
                        Tf[dim] = sfi*(nfi[dim] - alpha*efi[dim]) 

                for jdx in range(nfvars):
                    fn[jdx] = -mu*du[lti][lfi, jdx, lei]*inv_efi*Ef
                    for dim in range(ndims):
                        # tfi = sfi*(nfi[dim] - alpha*efi[dim])
                        fn[jdx] -= mu*gf[dim][jdx]*Tf[dim]

                    uf[lti][lfi, jdx, lei] =  fn[jdx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]

        return self.be.make_loop(self.nfpts, comm_flux)


#-------------------------------------------------------------------------------#    
    def _make_grad_at_face(self, nele):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx    
        # Inverse distance between the cell center
        weight    = self._weight
        # Stack-allocated array
        array = self.be.local_array()

        def grad_at(i_begin, i_end, gradf, *uf):
            # Parse element views (fpts, grad)
            du    = uf[:nele]
            gradu = uf[nele:]

            for idx in range(i_begin, i_end):
                gf = array(ndims)
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                # Compute the average of gradient at face
                for jdx in range(nvars):
                    gfl = gradu[lti][:, jdx, lei]
                    gfr = gradu[rti][:, jdx, rei]
                    # Compute gradient with jump term
                    for dim in range(ndims):
                        gradf[dim, jdx, idx] =  weight[idx]*gfl[dim] + (1.0-weight[idx])*gfr[dim]

        return self.be.make_loop(self.nfpts, grad_at)

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
class ParabolicMPIInters(BaseMPIInters):
    _tag = 1234

    def construct_kernels(self, elemap, impl_op):
        # Buffers
        lhs = np.empty((self.nvars, self.nfpts))
        self._rhs = rhs = np.empty((self.nvars, self.nfpts))

        self._correction = self.cfg.get('solver', 'correction', 'minimum')
        # Gradient at face and buffer
        self._gradf = gradf = np.empty((self.ndims, self.nvars, self.nfpts))
        grad_rhs = np.empty((self.ndims, self.nvars, self.nfpts))

        # View of element array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(self._make_delu(), rhs, *fpts)

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at = Kernel(
            self._make_grad_at(), gradf, grad_rhs, *fpts
        )

        muf = np.empty(self.nfpts)
        self.compute_flux = Kernel(self._make_flux(), muf, gradf, rhs, *fpts)

        # Kernel for pack, send, receive
        self.pack = Kernel(self._make_pack(), lhs, *fpts)
        self.send, self.sreq = self._make_send(lhs)
        self.recv, self.rreq = self._make_recv(rhs)

        self.pack_grad = Kernel(self._make_pack_grad(), gradf, *dfpts)
        self.send_grad, self.sgreq = self._make_send(gradf)
        self.recv_grad, self.rgreq = self._make_recv(grad_rhs)


#-------------------------------------------------------------------------------#    
    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        # mu = self.ele0._const['mu']

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        # scheme = self.cfg.get('solver', 'riemann-solver')
        # flux = get_rsolver(scheme, self.be, cplargs)

        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        # visflux = make_visflux(self.be, cplargs)

        def comm_flux(i_begin, i_end, muf, gradf, rhs, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)
                um = array(nfvars)
                
                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                ur = rhs[:, idx]

                # Gradient and solution at face
                gf = gradf[:,:, idx]

                for jdx in range(nfvars):
                    um[jdx] = 0.5*(ul[jdx] + ur[jdx])

                # Compute viscosity and viscous flux
                muf[idx] = mu = compute_mu()
                # visflux(um, gf, nfi, mu, fn)
                for jdx in range(nfvars):
                    fn[jdx] = 0.0
                    for dim in range(ndims):
                        fn[jdx] -= mu*nfi[dim]*gf[dim][jdx]

                for jdx in range(nfvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] =  fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)

#-------------------------------------------------------------------------------#    
    def _make_grad_at(self):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx

        # Mangitude and direction of the connecting vector
        inv_tf = self._rcp_dx
        tf = self._dx_adj * inv_tf
        avec = self._vec_snorm/np.einsum('ij,ij->j', tf, self._vec_snorm)

        # Stack-allocated array
        array = self.be.local_array()

        def grad_at(i_begin, i_end, gradf, grad_rhs, *du):
            for idx in range(i_begin, i_end):
                gf = array(ndims)

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gf[kdx] = 0.5*(gradf[kdx, jdx, idx] +
                                       grad_rhs[kdx, jdx, idx])

                    gft = dot(gf, tfi, ndims)

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gf[kdx] -= (gft - du[lti][lfi, jdx, lei]
                                    * inv_tfi)*aveci[kdx]

                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at)

#-------------------------------------------------------------------------------#    
    def _make_pack_grad(self):
        ndims, nvars = self.ndims, self.nvars
        lt, le, _ = self._lidx

        def pack(i_begin, i_end, lhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lei = lt[idx], le[idx]

                for jdx in range(nvars):
                    for kdx in range(ndims):
                        lhs[kdx, jdx, idx] = uf[lti][kdx, jdx, lei]

        return self.be.make_loop(self.nfpts, pack)

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
class ParabolicBCInters(BaseBCInters):
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

        self._correction = self.cfg.get('solver', 'correction', 'minimum')
        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]
        dfpts = [cell.grad for cell in elemap.values()]
        nele = len(fpts)

       # Gradient at face
        self._gradf = gradf = np.empty((self.ndims, self.nvars, self.nfpts))

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(self._make_delu(), *fpts)

        # Kernel to compute gradient at face (Averaging gradient)
        self.compute_grad_at_face = Kernel(
            self._make_grad_at_face(nele), gradf, *fpts, *dfpts
        )
  
        # Save viscosity on face (for implicit operator)
        muf = np.empty(self.nfpts)

        # Kernel to compute flux
        self.compute_flux = Kernel(self._make_flux(nele), muf, gradf, *fpts)

#-------------------------------------------------------------------------------#    
    def _make_flux(self, nele):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Mangitude and direction of the connecting vector
        inv_ef = self._rcp_dx
        ef = self._dx_adj * inv_ef
        # avec = self._vec_snorm/np.einsum('ij,ij->j', ef, self._vec_snorm)

        correction = self._correction
        # Compiler arguments
        array = self.be.local_array()
        # mu = self.ele0._const['mu']
        cplargs = {
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }
        # Get compiled function of viscosity and viscous flux
        compute_mu = self.ele0.mu_container()
        # Get bc function 
        bc = self.bc

        def comm_flux(i_begin, i_end, muf, gradf, *uf):
            # Parse element views (fpts, grad)
            du    = uf[:nele]
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                fn = array(nfvars)
                Tf = array(ndims)
                # Normal vector
                nfi = nf[:, idx]
                # Gradient and solution at face
                gf = gradf[:,:, idx]                
                # Viscosity at face
                muf[idx] = mu = compute_mu()

                inv_efi  = inv_ef[idx]
                efi = ef[:, idx]
                sfi = sf[idx]

                if(correction=='minimum'):
                    # Minimum Correction Approach
                    alpha = dot(efi, nfi, ndims)
                    Ef = sfi*alpha
                    for dim in range(ndims):
                        Tf[dim] = sfi*(nfi[dim] - alpha*efi[dim]) 
                elif(correction=='orthogonal'):
                    # Orthogonal Correction Approach
                    alpha = 1.0
                    Ef = sfi*alpha
                    for dim in range(ndims):
                        Tf[dim] = sfi*(nfi[dim] - alpha*efi[dim]) 
                elif(correction=='over_relaxed'):
                    # Over Relaxed Correction Approach
                    alpha = dot(efi, nfi, ndims)
                    Ef = sfi*1.0/alpha
                    for dim in range(ndims):
                        Tf[dim] = sfi*(nfi[dim] - alpha*efi[dim]) 

                for jdx in range(nfvars):
                    fn[jdx] = -mu*du[lti][lfi, jdx, lei]*inv_efi*Ef
                    for dim in range(ndims):
                        fn[jdx] -= mu*gf[dim][jdx]*Tf[dim]

                    uf[lti][lfi, jdx, lei] =  fn[jdx]

        return self.be.make_loop(self.nfpts, comm_flux)

#-------------------------------------------------------------------------------#    
    def _make_grad_at_face(self, nele):
        nvars, ndims = self.nvars, self.ndims
        lt, le, lf = self._lidx

        # Mangitude and direction of the connecting vector
        inv_tf = self._rcp_dx
        tf = self._dx_adj * inv_tf
        avec = self._vec_snorm/np.einsum('ij,ij->j', tf, self._vec_snorm)

        # Stack-allocated array
        array = self.be.local_array()

        def grad_at(i_begin, i_end, gradf, *uf):
            # Parse element views (fpts, grad)
            du = uf[:nele]
            gradu = uf[nele:]

            for idx in range(i_begin, i_end):
                gf = array(ndims)

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                tfi = tf[:, idx]
                inv_tfi = inv_tf[idx]
                aveci = avec[:, idx]

                # Compute the average of gradient at face
                for jdx in range(nvars):
                    for kdx in range(ndims):
                        gf[kdx] = gradu[lti][kdx, jdx, lei]

                    # Compute gradient with jump term
                    for kdx in range(ndims):
                        gradf[kdx, jdx, idx] = gf[kdx]

        return self.be.make_loop(self.nfpts, grad_at)

#-------------------------------------------------------------------------------#    
    def _make_delu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc
        array = self.be.local_array()

        def compute_delu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                ur = array(nvars)
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                bc(ul, ur, nfi)
                for jdx in range(nvars):
                    du = ur[jdx] - ul[jdx]
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_delu)


#-------------------------------------------------------------------------------#    
class ParabolicDrichletBCInters(ParabolicBCInters):
    name = 'drichlet'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        self._reqs = self.primevars


#-------------------------------------------------------------------------------#    
class ParabolicNeumannBCInters(ParabolicBCInters):
    name = 'neumann'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        self._reqs = self.primevars
