# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.baseadvec import BaseAdvecElements, BaseAdvecIntInters, BaseAdvecMPIInters, BaseAdvecBCInters, BaseAdvecVertex


class BaseAdvecSystem(BaseSystem):
    name = 'baseadvec'
    _elements_cls = BaseAdvecElements
    _intinters_cls = BaseAdvecIntInters
    _bcinters_cls = BaseAdvecBCInters
    _mpiinters_cls = BaseAdvecMPIInters
    _vertex_cls = BaseAdvecVertex

    def rhside(self, idx_in=0, idx_out=1, t=0, is_norm=False):
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

        if self.mpiint:
            # Finalize MPI communication
            q.sync()

            # Compute Difference of solution at MPI Inters
            self.mpiint.compute_delu()

        # Compute extreme values at vertex
        self.vertex.compute_extv()

        if self.vertex.mpi:
            # Start MPI communication for Vertex
            self.vertex.pack()
            self.vertex.send(q)
            self.vertex.recv(q)

        # Compute gradient
        self.eles.compute_grad()

        if self.vertex.mpi:
            # Finalize MPI communication
            q.sync()

            # Unpack (Sort vetex extremes)
            self.vertex.unpack()

        # Compute slope limiter and reconstruction
        self.eles.compute_mlp_u()
        self.eles.compute_recon()

        if self._is_recon and self.mpiint:
            # Start MPI communication to exchange reconstructed values at face
            self.mpiint.pack()
            self.mpiint.send(q)
            self.mpiint.recv(q)

        # Compute flux
        self.iint.compute_flux()
        self.bint.compute_flux()

        if self.mpiint:
            # Finalize MPI communication
            q.sync()

            # Compute flux at MPI Inters
            self.mpiint.compute_flux()

        # Compute divergence
        self.eles.div_upts(t)

        if is_norm:
            # Compute residual if requested
            resid = sum(self.eles.compute_resid())
            return resid
        else:
            return 'none'

    def spec_rad(self):
        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        # Compute spectral radius on faces
        self.iint.compute_spec_rad()
        self.bint.compute_spec_rad()

        if self.mpiint:
            self.mpiint.compute_spec_rad()

    def approx_jac(self):
        # Compute solution at flux point (face center)
        self.eles.compute_fpts()

        # Compute approximate Jacobian matrix on faces
        self.iint.compute_aprx_jac()
        self.bint.compute_aprx_jac()

        if self.mpiint:
            self.mpiint.compute_aprx_jac()

    def timestep(self, cfl, idx_in=0):
        # Compute time step with the given CFL number
        self.eles.upts_in.idx = idx_in
        self.eles.timestep(cfl)

    def post(self, idx_in=0):
        # Post-process
        self.eles.upts_in.idx = idx_in
        self.eles.post()
