# -*- coding: utf-8 -*-
from solvers.base import BaseIntInters, BaseBCInters, BaseMPIInters
from backends.types import Kernel, NullKernel
from solvers.grad.bcs import get_bc
from utils.np import npeval

import numpy as np
import re


class GradIntInters(BaseIntInters):
    def construct_kernels(self, elemap,impl_op=0):
        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]
         # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(self._make_delu(), *fpts)
        self.compute_avgu = Kernel(self._make_avgu(), *fpts)

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

    def _make_avgu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        rt, re, rf = self._ridx

        def compute_avgu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]
                rti, rfi, rei = rt[idx], rf[idx], re[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = uf[rti][rfi, jdx, rei]
                    au = 0.5*(ur + ul)
                    uf[lti][lfi, jdx, lei] = au
                    uf[rti][rfi, jdx, rei] = au

        return self.be.make_loop(self.nfpts, compute_avgu)

class GradMPIInters(BaseMPIInters):
    _tag = 1234

    def construct_kernels(self, elemap, impl_op=0):
        # Buffers
        lhs = np.empty((self.nvars, self.nfpts))
        self._rhs = rhs = np.empty((self.nvars, self.nfpts))

        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]

        if self.order > 1:
            # Kernel to compute differnce of solution at face
            self.compute_delu = Kernel(self._make_delu(), rhs, *fpts)
            self.compute_avgu = Kernel(self._make_avgu(), rhs, *fpts)
        else:
            self.compute_delu = NullKernel
            self.compute_avgu = NullKernel

        # Kernel for pack, send, receive
        self.pack = Kernel(self._make_pack(), lhs, *fpts)
        self.send, self.sreq = self._make_send(lhs)
        self.recv, self.rreq = self._make_recv(rhs)

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

    def _make_avgu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx

        def compute_avgu(i_begin, i_end, rhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    ul = uf[lti][lfi, jdx, lei]
                    ur = rhs[jdx, idx]
                    du = 0.5*(ur + ul)
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_avgu)

    def _make_pack(self):
        nvars = self.nvars
        lt, le, lf = self._lidx

        def pack(i_begin, i_end, lhs, *uf):
            for idx in range(i_begin, i_end):
                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                for jdx in range(nvars):
                    lhs[jdx, idx] = uf[lti][lfi, jdx, lei]

        return self.be.make_loop(self.nfpts, pack)

    def _sendrecv(self, mpifn, arr):
        # MPI Send or Receive init
        req = mpifn(arr, self._dest, self._tag)

        def start(q):
            # Function to save request in queue and start Send/Receive
            q.register(req)
            return req.Start()

        # Return Non-blocking send/recive and request (for finalise)
        return start, req

    def _make_send(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Send_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req

    def _make_recv(self, arr):
        from mpi4py import MPI

        mpifn = MPI.COMM_WORLD.Recv_init
        start, req = self._sendrecv(mpifn, arr)

        return start, req


class GradBCInters(BaseBCInters):
    _get_bc = get_bc
    def construct_bc(self):
        # print('here in the bcs')
        # # Parse BC function name
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

        # Get bc from `bcs.py` 
        self.bc = self._get_bc(self.be, bcf, bcc)

    def construct_kernels(self, elemap, impl_op=0):
        self.construct_bc()

        # View of elemenet array
        self._fpts = fpts = [cell.fpts for cell in elemap.values()]

        # Kernel to compute differnce of solution at face
        self.compute_delu = Kernel(self._make_delu(), *fpts)
        self.compute_avgu = Kernel(self._make_avgu(), *fpts)

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


    def _make_avgu(self):
        nvars = self.nvars
        lt, le, lf = self._lidx
        nf = self._vec_snorm

        bc = self.bc
        array = self.be.local_array()

        def compute_avgu(i_begin, i_end, *uf):
            for idx in range(i_begin, i_end):
                ur = array(nvars)
                nfi = nf[:, idx]

                lti, lfi, lei = lt[idx], lf[idx], le[idx]

                ul = uf[lti][lfi, :, lei]
                bc(ul, ur, nfi)

                for jdx in range(nvars):
                    du = 0.5*(ur[jdx] + ul[jdx])
                    uf[lti][lfi, jdx, lei] = du

        return self.be.make_loop(self.nfpts, compute_avgu)

class GradWallBCInters(GradBCInters):
    name = 'drichlet'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)

class GradNeumannBCInters(GradBCInters):
    name = 'neumann'
    def __init__(self, be, cfg, elemap, lhs, bctype):
        super().__init__(be, cfg, elemap, lhs, bctype)
