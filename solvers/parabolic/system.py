# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.parabolic import ParabolicElements, ParabolicIntInters, ParabolicMPIInters, ParabolicBCInters, ParabolicVertex
import sys

class ParabolicSystem(BaseSystem):
    name = 'parabolic'
    _elements_cls = ParabolicElements
    _intinters_cls = ParabolicIntInters
    _bcinters_cls = ParabolicBCInters
    _mpiinters_cls = ParabolicMPIInters
    _vertex_cls = ParabolicVertex

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

        # Compute Difference of solution at Inters
        self.iint.compute_delu()
        self.bint.compute_delu()


        # if self.mpiint:
        #     # Finalize MPI communication
        #     q.sync()

        #     # Compute Difference of solution at MPI Inters
        #     self.mpiint.compute_delu()

        # Compute extreme values at vertex
        # self.vertex.compute_extv()

        # if self.vertex.mpi:
        #     # Start MPI communication for Vertex
        #     self.vertex.pack()
        #     self.vertex.send(q)
        #     self.vertex.recv(q)

        # Compute gradient
        self.eles.compute_grad()

        # if self.vertex.mpi:
        #     # Finalize MPI communication
        #     q.sync()

        #     # Unpack (Sort vetex extremes)
        #     self.vertex.unpack()

        # Compute gradient at face
        self.iint.compute_grad_at_face()
        self.bint.compute_grad_at_face()

        # print(self.iint._gradf)
        # print(self.bint._gradf)

        # if self.mpiint:
        #     # Start MPI communication for gradient at Inters
        #     self.mpiint.pack_grad()
        #     self.mpiint.send_grad(q)
        #     self.mpiint.recv_grad(q)

        # Compute slope limiter
        # self.eles.compute_mlp_u()

        # if self.mpiint:
        #     # Finalize MPI communication
        #     q.sync()

        #     # Compute gradient at MPI Inters
        #     self.mpiint.compute_grad_at()

        # Compute reconstruction
        # self.eles.compute_recon()

        # if self._is_recon and self.mpiint:
        #     # Start MPI communication to exchange reconstructed values at face
        #     self.mpiint.pack()
        #     self.mpiint.send(q)
        #     self.mpiint.recv(q)

        # # Compute flux
        self.iint.compute_flux()
        self.bint.compute_flux()

        # if self.mpiint:
        #     # Finalize MPI communication
        #     q.sync()

        #     # Compute flux at MPI Inters
        #     self.mpiint.compute_flux()

        # Compute divergence 
        self.eles.div_upts(t)


        # sys.exit()

        # print(self.iint._fpts)
        # print(self.bint._fpts)

        if is_norm:
            # Compute residual if requested
            resid = sum(self.eles.compute_norm())
            # print(resid)
            return resid
        else:
            return 'none'

        sys.exit()
        
#-------------------------------------------------------------------------------#    
    def timestep(self, cfl, idx_in=0):
        # Compute time step with the given CFL number
        self.eles.upts_in.idx = idx_in
        self.eles.timestep(cfl)

#-------------------------------------------------------------------------------#
    def post(self, idx_in=0):
        # Post-process
        self.eles.upts_in.idx = idx_in
        self.eles.post()
