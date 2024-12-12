# -*- coding: utf-8 -*-
from mpi4py import MPI
from backends.types import Kernel
from inifile import INIFile
from integrators.base import BaseIntegrator
from utils.misc import ProxyList

import numpy as np


class BaseUnsteadyIntegrator(BaseIntegrator):
    """
    Explict Time integrator for
    - EulerExplicit
    - Runge-Kutta
    """
    mode = 'unsteady'
    impl_op = 'none'

    def __init__(self, be, cfg, msh, soln, comm):
        # get MPI_COMM_WORLD
        self._comm = comm

        # get list of physical time to run
        self.tlist = eval(cfg.get('solver-time-integrator', 'time'))

        if soln:
            # Get stats and current time from restarted solution
            stats = INIFile()
            stats.fromstr(soln['stats'])
            self.tcurr = stats.getfloat('solver-time-integrator', 'tcurr')
            self.iter = stats.getint('solver-time-integrator', 'iter', 0)
        else:
            # Initialize current time and iteration
            self.tcurr = self.tlist[0]
            self.iter = 0

        super().__init__(be, cfg, msh, soln, comm)

        # Construct stages (for RK schemes)
        self.construct_stages()

        # Configure time step method
        controller = cfg.get('solver-time-integrator', 'controller', 'cfl')
        if controller == 'cfl':
            # Time step is computed using CFL
            self.cfl = cfg.getfloat('solver-time-integrator', 'cfl')
            self._timestep = self._dt_cfl
        else:
            # Fixed time step
            dt = cfg.getfloat('solver-time-integrator', 'dt')
            self._timestep = lambda ttag: min(dt, ttag - self.tcurr)

    def add_tlist(self, dt):
        # Add intermediate time in physcal time list with stride
        tlist = self.tlist
        tmp = np.arange(tlist[0], tlist[-1], dt)
        self.tlist = np.sort(np.unique(np.concatenate([tlist, tmp])))

    def _make_stages(self, out, *args):
        # Generate formulation of each RK stage 
        eq_str = '+'.join('{}*upts[{}][j, idx]'.format(a, i) for a, i in zip(args[::2], args[1::2]))

        # Generate Python function for each RK stage
        f_txt =(
            f"def stage(i_begin, i_end, dt, *upts):\n"
            f"  for idx in range(i_begin, i_end):\n"
            f"      for j in range(nvars):\n"
            f"          upts[{out}][j, idx] = {eq_str}"
        )

        kernels = []
        for ele in self.sys.eles:
            # Initiate Python function of RK stage for each element
            gvars = {'nvars' : ele.nvars}
            lvars = {}
            exec(f_txt, gvars, lvars)

            # Generate JIT kernel by looping RK stage function
            _stage = self.be.make_loop(ele.neles, lvars['stage'])
            kernels.append(Kernel(_stage, *ele.upts, arg_trans_pos=True))
        
        # Collect RK stage kernels for elements
        return ProxyList(kernels)

    def run(self):
        for t in self.tlist:
            self.advance_to(t)

    def _dt_cfl(self, ttag):
        # Compute timestep of each cell using CFL
        self.sys.timestep(self.cfl, self._curr_idx)

        # Get minimum over whole cells
        dt = min(self.sys.eles.dt.min())
        dtmin = self._comm.allreduce(dt, op=MPI.MIN)

        # Adjust time step for target time
        return min(ttag - self.tcurr, dtmin)

    def advance_to(self, ttag):
        while self.tcurr < ttag:
            # Compute dt
            self.dt = dt = self._timestep(ttag)

            # Compute one RK step
            self._curr_idx = self.step(dt, self.tcurr)

            # Post actions after iteration
            self.tcurr += dt
            self.iter += 1
            self.completed_handler(self)


class EulerExplicit(BaseUnsteadyIntegrator):
    name = 'eulerexplicit'
    nreg = 2

    def construct_stages(self):
        self._stages = stages = []
        stages.append(self._make_stages(0, 1, 0, 'dt', 1))

    def step(self, dt,t):
        sys = self.sys
        stages = self._stages

        sys.rhside()
        stages[0](dt)

        self.sys.post(0)

        return 0


class TVDRK3(BaseUnsteadyIntegrator):
    name = 'tvd-rk3'
    nreg = 3

    def construct_stages(self):
        self._stages = stages = []
        stages.append(self._make_stages(2, 1, 0, 'dt', 1))
        stages.append(self._make_stages(2, 3/4, 0, 1/4, 2, 'dt/4', 1))
        stages.append(self._make_stages(0, 1/3, 0, 2/3, 2, '2*dt/3', 1))

    def step(self, dt, t):
        sys = self.sys
        stages = self._stages

        sys.rhside(t=t)
        stages[0](dt)
        self.sys.post(2)

        sys.rhside(2, 1, t=t)
        stages[1](dt)
        self.sys.post(2)

        sys.rhside(2, 1, t=t)
        stages[2](dt)        
        self.sys.post(0)

        return 0