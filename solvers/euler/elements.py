# -*- coding: utf-8 -*-
import functools as fc
import numpy as np

from solvers.baseadvec import BaseAdvecElements
from backends.types import Kernel
from utils.nb import dot
from utils.np import eps


class FluidElements:   
    @property
    def primevars(self):
        # Primitive variables
        return ['rho', 'p'] + [k for k in 'uvw'[:self.ndims]]

    @property
    def conservars(self):
        # Conservative variables
        pri = self.primevars

        # rho,rhou,rhov,rhow,E
        return [pri[0]] + [pri[0] + v for v in pri[2:self.ndims+2]] + ['E']

    def prim_to_conv(self, pri, cfg):
        # Convert primitives to conservatives
        (rho, p), v = pri[:2], pri[2:self.ndims+2]
        rhov = [rho * u for u in v]
        gamma = cfg.getfloat('constants', 'gamma')
        et = p / (gamma - 1) + 0.5 * rho * sum(u * u for u in v)
        return [rho] + rhov + [et]

    def conv_to_prim(self, con, cfg):
        # Convert conservatives to primitives
        rho, et = con[0], con[self.ndims+1]
        v = [rhov / rho for rhov in con[1:self.ndims+1]]
        gamma = cfg.getfloat('constants', 'gamma')
        p = (gamma - 1) * (et - 0.5 * rho * sum(u * u for u in v))
        return [rho, p] + v

    @fc.lru_cache()
    def flux_container(self):
        # Constants and dimensions
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def flux(u, nf, f):
            # Compute normal component of flux
            rho, et = u[0], u[nfvars-1]

            contrav = dot(u, nf, ndims, 1)/rho

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)
            if p < pmin:
                p = pmin
                u[nfvars - 1] = et = p/(gamma-1) + 0.5 * \
                    dot(u, u, ndims, 1, 1)/rho

            ht = et + p

            f[0] = rho*contrav
            for i in range(ndims):
                f[i + 1] = u[i + 1]*contrav + nf[i]*p
            f[nfvars-1] = ht*contrav

            return p, contrav

        # Compile the function
        return self.be.compile(flux)

    @fc.lru_cache()
    def to_flow_primevars(self):
        # Constants and dimensions
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def to_primevars(u, v):
            # Compute primitives
            rho, et = u[0], u[nfvars-1]

            for i in range(ndims):
                v[i] = u[i + 1] / rho

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)
            if p < pmin:
                p = pmin
                u[nfvars - 1] = p/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho

            return p

        # Compile the function
        return self.be.compile(to_primevars)

    def fix_nonPys_container(self):
        # Constants and dimensions
        gamma, pmin = self._const['gamma'], self._const['pmin']
        ndims, nfvars = self.ndims, self.nfvars

        def fix_nonPhy(u):
            # Fix non-physical solution (negative density, pressure)
            rho, et = u[0], u[nfvars-1]
            if rho < 0:
                u[0] = rho = eps

            p = (gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho)

            if p < pmin:
                u[nfvars - 1] = pmin/(gamma-1) + 0.5*dot(u, u, ndims, 1, 1)/rho

        # Compile the function
        return self.be.compile(fix_nonPhy)


class EulerElements(BaseAdvecElements, FluidElements):
    def __init__(self, be, cfg, name, eles):
        super().__init__(be, cfg, name, eles)
        self.nvars = len(self.primevars)
        self.nfvars = self.nvars

        # Get constants
        cfg.get('constants', 'pmin', '1e-15')
        self._const = cfg.items('constants')

    def construct_kernels(self, vertex, nreg, impl_op):
        # Call paraent method
        super().construct_kernels(vertex, nreg)

        if impl_op == 'spectral-radius':
            # Spectral radius on face
            self.fspr = np.empty((self.nface, self.neles))
        elif impl_op == 'approx-jacobian':
            # Jacobian matrix on face
            self.jmat = np.empty((2, self.nfvars, self.nfvars, \
                                  self.nface, self.neles))

        # Kernel to compute timestep
        self.timestep = Kernel(self._make_timestep(),
                               self.upts_in, self.dt)

    def _make_timestep(self):
        # Dimensions
        ndims, nface = self.ndims, self.nface

        # Static variables
        vol = self._vol
        smag, svec = self._gen_snorm_fpts()

        # Constants
        gamma, pmin = self._const['gamma'], self._const['pmin']

        def timestep(i_begin, i_end, u, dt, cfl):
            for idx in range(i_begin, i_end):
                rho = u[0, idx]
                et = u[-1, idx]
                rv2 = dot(u[:, idx], u[:, idx], ndims, 1, 1)/rho

                p = max((gamma - 1)*(et - 0.5*rv2), pmin)
                c = np.sqrt(gamma*p/rho)

                # Sum of Wave speed * surface area
                sum_lamdf = 0.0
                for jdx in range(nface):
                    lamdf = abs(dot(u[:, idx], svec[jdx, idx], ndims, 1))/rho + c
                    sum_lamdf += lamdf*smag[jdx, idx]

                # Time step : CFL * vol / sum(lambda_f S_f)
                dt[idx] = cfl*vol[idx] / sum_lamdf

        return self.be.make_loop(self.neles, timestep)

    def make_wave_speed(self):
        # Dimensions and constants
        ndims, nfvars = self.ndims, self.nfvars
        gamma, pmin = self._const['gamma'], self._const['pmin']

        def _lambdaf(u, nf, *args):
            rho, et = u[0], u[nfvars-1]

            contra = dot(u, nf, ndims, 1)/rho
            p = max((gamma - 1)*(et - 0.5*dot(u, u, ndims, 1, 1)/rho), pmin)
            c = np.sqrt(gamma*p/rho)

            # Wave speed : abs(Vn) + c
            return abs(contra) + c

        return self.be.compile(_lambdaf)
