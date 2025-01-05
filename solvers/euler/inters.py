# -*- coding: utf-8 -*-
from solvers.baseadvec import BaseAdvecIntInters, BaseAdvecBCInters, BaseAdvecMPIInters
from backends.types import Kernel
from solvers.euler.rsolvers import get_rsolver
from solvers.euler.bcs import get_bc

import numpy as np


class EulerIntInters(BaseAdvecIntInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)        

        # Kernel to compute flux
        fpts = self._fpts
        self.compute_flux = Kernel(self._make_flux(), *fpts)

        if impl_op == 'spectral-radius':
            # Kernel to compute Spectral radius
            nele = len(fpts)
            fspr = [cell.fspr for cell in elemap.values()]
            self.compute_spec_rad = Kernel(self._make_spec_rad(nele), *fpts, *fspr)
        elif impl_op == 'approx-jacobian':
            # Kernel to compute Jacobian matrices
            nele = len(fpts)
            fjmat = [cell.jmat for cell in elemap.values()]
            self.compute_aprx_jac = Kernel(self._make_aprx_jac(nele), *fpts, *fjmat)

    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        def comm_flux(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)

                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)

                for jdx in range(nfvars):
                    # Save it at left and right solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]
                    uf[rti][rfi, jdx, rei] = -fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)


    def _make_spec_rad(self, nele):
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf = self._vec_snorm

        # Get wave speed function
        wave_speed = self.ele0.make_wave_speed()

        def comm_spr(i_begin, i_end, *ufl):
            uf, lam = ufl[:nele], ufl[nele:]

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]

                # Compute wave speed on both cell
                laml = wave_speed(ul, nfi)
                lamr = wave_speed(ur, nfi)

                # Compute spectral radius on face
                lami = max(laml, lamr)
                lam[lti][lfi, lei] = lami
                lam[rti][rfi, rei] = lami

        return self.be.make_loop(self.nfpts, comm_spr)
    

    def _make_aprx_jac(self, nele):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian

        nfvars = self.nfvars

        lt, le, lf = self._lidx
        rt, re, rf = self._ridx
        nf = self._vec_snorm

        cplargs = {
            'ndims': self.ndims,
            'gamma': self.ele0._const['gamma'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        # Get Jacobian functions
        pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')
        neg_jacobian = make_convective_jacobian(self.be, cplargs, 'negative')

        # Temporal matrix
        matrix = self.be.local_matrix()

        def comm_apj(i_begin, i_end, *ufj):
            uf, jmats = ufj[:nele], ufj[nele:]

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]

                # Jacobian matrix
                ap = matrix(nfvars*nfvars, (nfvars, nfvars))
                am = matrix(nfvars*nfvars, (nfvars, nfvars))

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]
                ul = uf[lti][lfi, :, lei]
                ur = uf[rti][rfi, :, rei]

                # Compute Jacobian matrix on surface
                # based on left/right cell
                pos_jacobian(ul, nfi, ap)
                neg_jacobian(ur, nfi, am)

                # Compute approximate Jacobian on face
                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = ap[row][col]
                        jmats[lti][1, row, col, lfi, lei] = -am[row][col]
                        jmats[rti][0, row, col, rfi, rei] = -am[row][col]
                        jmats[rti][1, row, col, rfi, rei] = ap[row][col]

        return self.be.make_loop(self.nfpts, comm_apj)


class EulerMPIInters(BaseAdvecMPIInters):
    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)        

        # Kernel to compute flux
        fpts, rhs = self._fpts, self._rhs
        self.compute_flux = Kernel(self._make_flux(), rhs, *fpts)

        if impl_op == 'spectral-radius':
            # Kernel to compute Spectral radius
            nele = len(fpts)
            fspr = [cell.fspr for cell in elemap.values()]
            self.compute_spec_rad = Kernel(self._make_spec_rad(nele), *fpts, *fspr)
        elif impl_op == 'approx-jacobian':
            # Kernel to compute Jacobian matrices
            nele = len(fpts)
            fjmat = [cell.jmat for cell in elemap.values()]
            self.compute_aprx_jac = Kernel(self._make_aprx_jac(nele), *fpts, *fjmat)

    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars
        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        def comm_flux(i_begin, i_end, rhs, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)

                # Normal vector
                nfi = nf[:, idx]

                # Left and right solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]
                ur = rhs[:, idx]

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)

                for jdx in range(nfvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, comm_flux)

    def _make_spec_rad(self, nele):
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        # Get wave speed function
        wave_speed = self.ele0.make_wave_speed()

        def comm_spr(i_begin, i_end, *ufl):
            uf, lam = ufl[:nele], ufl[nele:]

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]

                # Left solution
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Compute spectral radius on face
                lami = wave_speed(ul, nfi)
                lam[lti][lfi, lei] = lami

        return self.be.make_loop(self.nfpts, comm_spr)

    def _make_aprx_jac(self, nele):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian
        
        nfvars = self.nfvars

        lt, le, lf = self._lidx
        nf = self._vec_snorm

        cplargs = {
            'ndims': self.ndims,
            'gamma': self.ele0._const['gamma'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        # Get Jacobian functions
        com_aprx_jac = make_convective_jacobian(self.be, cplargs, 'positive')

        # Temporal matrix
        matrix = self.be.local_matrix()

        def comm_apj(i_begin, i_end, *ufj):
            uf, jmats = ufj[:nele], ufj[nele:]

            A = matrix(nfvars*nfvars, (nfvars, nfvars))

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]

                # Left solution
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Compute Jacobian matrix on face
                com_aprx_jac(ul, nfi, A)
                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = A[row][col]

        return self.be.make_loop(self.nfpts, comm_apj)


class EulerBCInters(BaseAdvecBCInters):
    _get_bc = get_bc

    def construct_kernels(self, elemap, impl_op):
        super().construct_kernels(elemap)
        
        # Kernel to compute flux
        fpts = self._fpts
        self.compute_flux = Kernel(self._make_flux(), *fpts)

        if impl_op == 'spectral-radius':
            # Kernel to compute Spectral radius
            nele = len(fpts)
            fspr = [cell.fspr for cell in elemap.values()]
            self.compute_spec_rad = Kernel(self._make_spec_rad(nele), *fpts, *fspr)
        elif impl_op == 'approx-jacobian':
            # Kernel to compute Jacobian matrices
            nele = len(fpts)
            fjmat = [cell.jmat for cell in elemap.values()]
            self.compute_aprx_jac = Kernel(self._make_aprx_jac(nele), *fpts, *fjmat)

    def _make_flux(self):
        ndims, nfvars = self.ndims, self.nfvars

        # Compiler arguments
        array = self.be.local_array()
        cplargs = {
            'flux' : self.ele0.flux_container(),
            'to_primevars' : self.ele0.to_flow_primevars(),
            'ndims' : ndims,
            'nfvars' : nfvars,
            'array' : array,
            **self._const
        }

        # Get numerical schems from `rsolvers.py`
        scheme = self.cfg.get('solver', 'riemann-solver')
        flux = get_rsolver(scheme, self.be, cplargs)

        # Get bc function (`self.bc` was defined at `baseadvec.inters`)
        bc = self.bc

        lt, le, lf = self._lidx
        nf, sf = self._vec_snorm, self._mag_snorm,

        def bc_flux(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                fn = array(nfvars)
                ur = array(nfvars)

                # Normal vector
                nfi = nf[:, idx]

                # Left solutions
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Compute BC
                bc(ul, ur, nfi)

                # Compute approixmate Riemann solver
                flux(ul, ur, nfi, fn)

                for jdx in range(nfvars):
                    # Save it at left solution array
                    uf[lti][lfi, jdx, lei] = fn[jdx]*sf[idx]

        return self.be.make_loop(self.nfpts, bc_flux)

    def _make_spec_rad(self, nele):
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        # Get wave speed function
        wave_speed = self.ele0.make_wave_speed()

        def comm_spr(i_begin, i_end, *ufl):
            uf, lam = ufl[:nele], ufl[nele:]

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]

                # Left solution
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Compute spectral radius on face
                lami = wave_speed(ul, nfi)
                lam[lti][lfi, lei] = lami

        return self.be.make_loop(self.nfpts, comm_spr)
    
    def _make_aprx_jac(self, nele):
        from pybaram.solvers.euler.jacobian import make_convective_jacobian
        
        nfvars = self.nfvars

        lt, le, lf = self._lidx
        nf = self._vec_snorm

        cplargs = {
            'ndims': self.ndims,
            'gamma': self.ele0._const['gamma'],
            'to_prim': self.ele0.to_flow_primevars()
        }

        # Get Jacobian functions
        pos_jacobian = make_convective_jacobian(self.be, cplargs, 'positive')

        # Temporal matrix
        matrix = self.be.local_matrix()

        def comm_apj(i_begin, i_end, *ufj):
            uf, jmats = ufj[:nele], ufj[nele:]

            A = matrix(nfvars*nfvars, (nfvars, nfvars))

            for idx in range(i_begin, i_end):
                # Normal vector
                nfi = nf[:, idx]

                # Left solution
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                ul = uf[lti][lfi, :, lei]

                # Compute Jacobian matrix on face
                pos_jacobian(ul, nfi, A)
                for row in range(nfvars):
                    for col in range(nfvars):
                        jmats[lti][0, row, col, lfi, lei] = A[row][col]

        return self.be.make_loop(self.nfpts, comm_apj)


class EulerSupOutBCInters(EulerBCInters):
    name = 'sup-out'


class EulerSlipWallBCInters(EulerBCInters):
    name = 'slip-wall'


class EulerSupInBCInters(EulerBCInters):
    name = 'sup-in'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        # require all primitive variables at the boundary
        self._reqs = self.primevars


class EulerFarInBCInters(EulerBCInters):
    name = 'far'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
        # require all primitive variables at the boundary
        self._reqs = self.primevars


class EulerSubOutPBCInters(EulerBCInters):
    name = 'sub-outp'
    # require only pressure
    _reqs = ['p']


class EulerSubInvBCInters(EulerBCInters):
    name = 'sub-inv'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['rho'] + ['u', 'v', 'w'][:self.ndims]


class EulerSubInpttBCInters(EulerBCInters):
    name = 'sub-inptt'

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

        self._reqs = ['p0', 'cpt0', 'dir']


class EulerSubOutMdotBCInters(EulerBCInters):
    name = 'sub-outmdot'
    _reqs = ['mdot', 'dir']

    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
