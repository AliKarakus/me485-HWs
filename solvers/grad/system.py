# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.grad import GradElements, GradIntInters, GradMPIInters, GradBCInters, GradVertex
import numpy as np
import re


class GradSystem(BaseSystem):
    name = 'grad'
    _elements_cls  = GradElements
    _intinters_cls = GradIntInters
    _bcinters_cls  = GradBCInters
    _mpiinters_cls = GradMPIInters
    _vertex_cls    = GradVertex
    def rhside(self, idx_in=0, idx_out=1, t=0, is_norm=True):
        # Adjust Banks
        self.eles.upts_in.idx = idx_in
        self.eles.upts_out.idx = idx_out

        # Queue for MPI
        q = self._queue
        
        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        if self.mpiint:
            # Start MPI communication for Inters
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        # if least squares, compute jumps at faces
        if( self._grad_method=='least-square'   or  self._grad_method=='weighted-least-square'):
            # Compute Difference of solution at Inters
            self.iint.compute_delu()
            self.bint.compute_delu()
            if self.mpiint:
                # Finalize MPI communication
                q.sync()
                # Compute Difference of solution at MPI Inters
                self.mpiint.compute_delu()
        # if green-gauss, compute averages at faces
        else:
            # Compute Difference of solution at Inters
            self.iint.compute_avgu()
            self.bint.compute_avgu()
            if self.mpiint:
                # Finalize MPI communication
                q.sync()

                # Compute Difference of solution at MPI Inters
                self.mpiint.compute_avgu()

        # Compute gradient
        self.eles.compute_grad()

        res = self.L2Norm()
        print('L2 norm of gradient:', res)

        # self.post()


        # # Compute extreme values at vertex
        # self.vertex.compute_extv()

        # if self.vertex.mpi:
        #     # Start MPI communication for Vertex
        #     self.vertex.pack()
        #     self.vertex.send(q)
        #     self.vertex.recv(q)


        # if self.vertex.mpi:
        #     # Finalize MPI communication
        #     q.sync()

        #     # Unpack (Sort vetex extremes)
        #     self.vertex.unpack()

        # Compute slope limiter and reconstruction
        # self.eles.compute_mlp_u()
        # self.eles.compute_recon()




        # if is_norm:
        #     # Compute residual if requested
        #     resid = sum(self.eles.compute_residual())
        #     print(resid)
        #     return resid
        # else:
        #     return 'none'

    # Post-process
    def post(self, idx_in=0):
        self.eles.upts_in.idx = idx_in
        self.eles.post()

     # Post-process
    def L2Norm(self, idx_in=0):
        self.eles.upts_in.idx = idx_in
        L2 = self.eles.compute_L2_norm()
        return L2 

