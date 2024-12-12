# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.parabolic import ParabolicElements, ParabolicIntInters, ParabolicBCInters, ParabolicVertex
import sys

class ParabolicSystem(BaseSystem):
    name = 'parabolic'
    _elements_cls = ParabolicElements
    _intinters_cls = ParabolicIntInters
    _bcinters_cls = ParabolicBCInters
    _vertex_cls = ParabolicVertex

    def rhside(self, idx_in=0, idx_out=1, t=0, is_norm=True):
         # Adjust Banks
        self.eles.upts_in.idx = idx_in
        self.eles.upts_out.idx = idx_out

        # Queue for MPI
        q = self._queue

        # Compute solution at flux point (face center)
        self.eles.compute_fpts()


        # Compute Difference of solution at Inters
        self.iint.compute_delu()
        self.bint.compute_delu()

        # Compute gradient
        self.eles.compute_grad()


        # Compute gradient at face
        self.iint.compute_grad_at_face()
        self.bint.compute_grad_at_face()


        # # Compute flux
        self.iint.compute_flux()
        self.bint.compute_flux()

        # Compute divergence 
        self.eles.div_upts(t)

        if is_norm:
            # Compute residual if requested
            resid = sum(self.eles.compute_norm())
            # print(resid)
            return resid
        else:
            return 'none'
        
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
