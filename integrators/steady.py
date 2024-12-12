# -*- coding: utf-8 -*-
from mpi4py import MPI
from backends.types import Kernel, MetaKernel
from inifile import INIFile
from integrators.base import BaseIntegrator
from utils.misc import ProxyList
from utils.np import eps

import numpy as np
import re


class BaseSteadyIntegrator(BaseIntegrator):
    mode = 'steady'
    impl_op = 'none'

    def __init__(self, be, cfg, msh, soln, comm):
        # get MPI_COMM_WORLD
        self._comm = comm

        # Get configurations for iterators
        self.itermax = cfg.getint('solver-time-integrator', 'max-iter')
        self.tol = cfg.getfloat('solver-time-integrator', 'tolerance')

        # Set current iteration
        if soln:
            # Get current iteration from previous result
            stats = INIFile()
            stats.fromstr(soln['stats'])
            self.iter = stats.getint('solver-time-integrator', 'iter', 0)
            if self.iter > 0:
                self.resid0 = np.array(stats.getlist(
                    'solver-time-integrator', 'resid0'))
        else:
            # Initialize iteration
            self.iter = 0

        # Indicator if solution is converted or not
        self.isconv = False

        super().__init__(be, cfg, msh, soln, comm)

        # Get CFL number 
        self._cfl0 = cfg.getfloat('solver-time-integrator', 'cfl', 1.0)

        # Get configuration of CFL linear ramp 
        self._cfl_iter0 = cfg.getint('solver-cfl-ramp', 'iter0', self.itermax)
        self._cfl_itermax = cfg.getint('solver-cfl-ramp', 'max-iter', self.itermax)
        self._cflmax = cfg.getfloat('solver-cfl-ramp', 'max-cfl', self._cfl0)
        
        # Caculate increment of cfl for CFL ramp
        if self._cfl_itermax > self._cfl_iter0:
            self._dcfl = (self._cflmax - self._cfl0) / (self._cfl_itermax - self._cfl_iter0)
        else:
            self._dcfl = 0

        # For turbulent simulation
        if self.sys.name.startswith('rans'):
            self._is_turb = True

            # Get turbulent CFL factor
            self._tcfl_fac = cfg.getfloat('solver-time-integrator', 'turb-cfl-factor', 1.0)
        else:
            self._is_turb = False

        # Specify residual variable for monitoring
        ele = next(iter(self.sys.eles))
        self.conservars = conservars = ele.conservars
        rvar = cfg.get('solver-time-integrator', 'res-var', 'rho')
        self._res_idx = [i for i, e in enumerate(conservars) if e == rvar][0]

        # Get total volume
        voli = sum(self.sys.eles.tot_vol)
        self.vol = comm.allreduce(voli, op=MPI.SUM)

        # Construct kernels
        self.cfg = cfg
        self.construct_stages()

    @property
    def _cfl(self):
        # Return CFL considering CFL ramp
        if self.iter < self._cfl_iter0:
            return self._cfl0
        elif self.iter > self._cfl_itermax:
            return self._cflmax
        else:
            return self._cfl0 + self._dcfl*(self.iter - self._cfl_iter0)

    def run(self):
        # Run integerator until max iteration
        while self.iter < self.itermax:
            # Compute one iteration
            self.advance_to()

            # Check if tolerance is satisfied
            residual = (self.resid / self.resid0)
            if residual[self._res_idx] < self.tol:
                break

        # Fire off plugins
        self.isconv = True
        self.completed_handler(self)
        self.print_res(residual)

    def complete_step(self, resid):
        self.resid = resid

        # Check if reference residual (resid0) is existed or not
        if not hasattr(self, 'resid0'):
            # Avoid zero in resid0
            self.resid0 = [r if r != 0 else eps for r in self.resid]

        self.iter += 1
        self.completed_handler(self)

    def _make_stages(self, out, *args):
        # Generate formulation of each RK stage 
        eq_str = '+'.join('{}*upts[{}][j, idx]'.format(a, i) for a, i in zip(args[::2], args[1::2]))

        if self._is_turb:
            # Substitute 'dt' string as dt array
            eqf_str = re.sub('dt', 'dt[idx]', eq_str)
            eqt_str = re.sub('dt', '{}*dt[idx]'.format(self._tcfl_fac), eq_str)

            # Generate Python function for each RK stage
            f_txt =(
                f"def stage(i_begin, i_end, dt, *upts):\n"
                f"  for idx in range(i_begin, i_end):\n"
                f"      for j in range(nfvars):\n"
                f"          upts[{out}][j, idx] = {eqf_str}\n"
                f"      for j in range(nfvars, nvars):\n"
                f"          upts[{out}][j, idx] = {eqt_str}"
            )
        else:
            # Substitute 'dt' string as dt array
            eq_str = re.sub('dt', 'dt[idx]', eq_str)

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
            kernels.append(Kernel(_stage, ele.dt, *ele.upts))
        
        # Collect RK stage kernels for elements
        return MetaKernel(kernels)

    def _local_dt(self):
        # Compute timestep of each cell using CFL
        self.sys.timestep(self._cfl, self._curr_idx)

    def advance_to(self):
        # Compute dt
        self._local_dt()

        # Compute one RK step
        self._curr_idx, resid = self.step()

        # Post actions after iteration
        self.complete_step(resid)

    def rhs(self, idx_in=0, idx_out=1, is_norm=False):
        # Compute right hand side
        residi = self.sys.rhside(idx_in, idx_out, is_norm=is_norm)

        # Compute L2 norm residual
        if is_norm:
            resid = self._comm.allreduce(residi, op=MPI.SUM)
            return np.sqrt(resid) / self.vol

    def print_res(self, residual):
        # Print residual result
        idx = self._res_idx
        res = residual[idx]
        if res < self.tol:
            print("Converged : Residual of {} = {:05g} <= {:05g}".format(
                self.conservars[idx], res, self.tol))
        else:
            print("Not converged : Residual of {} = {:05g} > {:05g}".format(
                self.conservars[idx], res, self.tol))


class EulerExplicit(BaseSteadyIntegrator):
    name = 'eulerexplicit'
    nreg = 2

    def construct_stages(self):
        self._stages = [self._make_stages(0, 1, 0, 'dt', 1)]

    def step(self):
        stages = self._stages

        resid = self.rhs(0, 1, is_norm=True)
        stages[0]()

        self.sys.post(0)

        return 0, resid


class TVDRK3(BaseSteadyIntegrator):
    name = 'tvd-rk3'
    nreg = 3

    def construct_stages(self):
        self._stages = [
            self._make_stages(2, 1, 0, 'dt', 1),
            self._make_stages(2, 3/4, 0, 1/4, 2, 'dt/4', 1),
            self._make_stages(0, 1/3, 0, 2/3, 2, '2*dt/3', 1),
        ]

    def step(self):
        stages = self._stages

        self.rhs()
        stages[0]()
        self.sys.post(2)

        self.rhs(2, 1)
        stages[1]()
        self.sys.post(2)

        resid = self.rhs(2, 1, is_norm=True)
        stages[2]()
        self.sys.post(0)

        return 0, resid


class FiveStageRK(BaseSteadyIntegrator):
    """
    Jameson Multistage scheme
    ref : Blazek book 6.1.1 (Table 6.1)
    """
    name = 'rk5'
    nreg = 3

    def construct_stages(self):
        self._stages = [
            self._make_stages(2, 1, 0, '0.0533*dt', 1),
            self._make_stages(2, 1, 0, '0.1263*dt', 1),
            self._make_stages(2, 1, 0, '0.2375*dt', 1),
            self._make_stages(2, 1, 0, '0.4414*dt', 1),
            self._make_stages(0, 1, 0, 'dt', 1)
        ]

    def step(self):
        stages = self._stages

        self.rhs()
        stages[0]
        self.sys.post(2)
        
        self.rhs(2, 1)
        stages[1]()
        self.sys.post(2)

        self.rhs(2, 1)
        stages[2]()
        self.sys.post(2)

        self.rhs(2, 1)
        stages[3]()
        self.sys.post(2)

        resid = self.rhs(2, 1, is_norm=True)
        stages[4]()
        self.sys.post(0)

        return 0, resid
 

class LUSGS(BaseSteadyIntegrator):
    name = 'lu-sgs'
    nreg = 3
    impl_op = 'spectral-radius'

    def construct_stages(self):
        from integrators.lusgs import make_lusgs_common, make_lusgs_update, make_serial_lusgs

        be = self.be

        # LU-SGS for each elements
        for ele in self.sys.eles:      
            # Get reordering result
            mapping, unmapping = ele.reordering()

            # diagonal and lambda array
            diag = np.empty(ele.neles)

            # Get Python functions of flux and wave speed
            _flux = ele.flux_container()
            nv = (0, ele.nfvars)

            # Compile LU-SGS functions
            _update = make_lusgs_update(ele)
            _pre_lusgs = make_lusgs_common(ele, factor=1.0)
            _lsweep, _usweep = make_serial_lusgs(
                be, ele, nv, mapping, unmapping, _flux
            )

            # Initiate LU-SGS kernel objects
            pre_lusgs = Kernel(
                be.make_loop(ele.neles, _pre_lusgs), 
                ele.dt, diag, ele.fspr
            )            
            lsweeps = Kernel(be.make_loop(ele.neles, func=_lsweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.fspr) 

            usweeps = Kernel(be.make_loop(ele.neles, func=_usweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.fspr)

            kernels = [pre_lusgs, lsweeps, usweeps]

            # LU-SGS for turbulent variables
            if self._is_turb:
                # Get Python function of flux and wave speed for turbulent variables
                _tflux = ele.tflux_container()
                tnv = (ele.nfvars, ele.nvars)

                # Compile LU-SGS functions for turbulent variables
                _pre_tlusgs = make_lusgs_common(ele, factor=self._tcfl_fac)
                _tlsweep, _tusweep = make_serial_lusgs(
                    be, ele, tnv, mapping, unmapping, _tflux
                )

                # Initiate LU-SGS kernel objects for turbulent variables
                pre_tlusgs = Kernel(
                    be.make_loop(ele.neles, _pre_tlusgs), 
                    ele.dt, diag, ele.tfspr
                )                
                tlsweeps = Kernel(be.make_loop(ele.neles, func=_tlsweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.tfspr) 

                tusweeps = Kernel(be.make_loop(ele.neles, func=_tusweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.tfspr)    

                kernels += [pre_tlusgs, tlsweeps, tusweeps]             

            # Collect kernels and make meta kernels
            ele.lusgs = MetaKernel(kernels)

            # Update kernel
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                ele.upts[0], ele.upts[1]
            )
    
    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.spec_rad()
        self.sys.eles.lusgs()
        self.sys.eles.update()

        self.sys.post(0)

        return 0, resid


class ColoredLUSGS(BaseSteadyIntegrator):
    name = 'colored-lu-sgs'
    nreg = 3
    impl_op = 'spectral-radius'

    def construct_stages(self):
        from integrators.lusgs import  make_lusgs_common, make_lusgs_update, make_colored_lusgs

        be = self.be

        # colored-LU-SGS for each elements
        for ele in self.sys.eles:
            # Get Coloring result
            ncolor, icolor, lev_color = ele.coloring()
             
            # diagonal and lambda array
            diag = np.empty(ele.neles)

            # Get Python functions of flux and wave speed
            _flux = ele.flux_container()
            nv = (0, ele.nfvars)

            # Compile LU-SGS functions
            _update = make_lusgs_update(ele)
            _pre_lusgs = make_lusgs_common(ele, factor=1.0)
            _lsweep, _usweep = make_colored_lusgs(
                be, ele, nv, icolor, lev_color, _flux
            )

            # Initiate LU-SGS kernel objects
            pre_lusgs = Kernel(
                be.make_loop(ele.neles, _pre_lusgs), 
                ele.dt, diag, ele.fspr
            )
            
            lsweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_lsweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.fspr
                ) 
                for n0, ne in zip(ncolor[:-1], ncolor[1:])
            ]

            usweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_usweep),
                ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.fspr
                ) 
                for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
            ]

            kernels = [pre_lusgs, *lsweeps, *usweeps]

            # LU-SGS for turbulent variables
            if self._is_turb:
                # Get Python function of flux and wave speed for turbulent variables
                _tflux = ele.tflux_container()
                tnv = (ele.nfvars, ele.nvars)

                # Compile LU-SGS functions for turbulent variables
                _pre_tlusgs = make_lusgs_common(ele, factor=self._tcfl_fac)
                _tlsweep, _tusweep = make_colored_lusgs(
                    be, ele, tnv, icolor, lev_color, _tflux
                )

                # Initiate LU-SGS kernel objects for turbulent variables
                pre_tlusgs = Kernel(
                    be.make_loop(ele.neles, _pre_tlusgs), 
                    ele.dt, diag, ele.tfspr
                )                

                tlsweeps = [
                    Kernel(be.make_loop(n0=n0, ne=ne, func=_tlsweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.tfspr
                    ) 
                    for n0, ne in zip(ncolor[:-1], ncolor[1:])
                ]

                tusweeps = [
                    Kernel(be.make_loop(n0=n0, ne=ne, func=_tusweep),
                    ele.upts[0], ele.upts[1], ele.upts[2], diag, ele.dsrc, ele.tfspr
                    ) 
                    for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
                ] 

                kernels += [pre_tlusgs, *tlsweeps, *tusweeps]  

            # Collect kernels and make meta kernels
            ele.lusgs = MetaKernel(kernels)

            # Update kernel
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                ele.upts[0], ele.upts[1]
            )
    
    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.spec_rad()
        self.sys.eles.lusgs()
        self.sys.eles.update()

        self.sys.post(0)

        return 0, resid


class BlockJacobi(BaseSteadyIntegrator):
    name = 'jacobi'
    nreg = 4
    impl_op = 'approx-jacobian'

    def construct_stages(self):
        from integrators.jacobi import make_jacobi_update, make_jacobi_sweep, \
                                            make_pre_jacobi, make_tpre_jacobi

        # Constants for Jacobi method
        self.subiter = self.cfg.getint('solver-time-integrator', 'sub-iter', 10)
        self.subtol = self.cfg.getfloat('solver-time-integrator', 'sub-tol', 0.05)

        be = self.be

        for ele in self.sys.eles:
            # Temporal arrays
            diag = np.empty((ele.nfvars, ele.nfvars, ele.neles))
            ele.subres = np.zeros((ele.neles,), dtype=np.float64)
            ele.norm = np.empty((ele.neles,))

            nv = (0, ele.nfvars)

            # Diagonal matrix computation kernel
            _pre_jacobi = make_pre_jacobi(ele, nv)
            pre_jacobi = Kernel(be.make_loop(ele.neles, _pre_jacobi),
                                ele.dt, diag, ele.jmat)
            
            # Sweep and compute subiteration step solution
            _sweep, _compute = make_jacobi_sweep(be, ele, nv, res_idx=self._res_idx)
            sweep = Kernel(be.make_loop(ele.neles, _sweep),
                               ele.upts[1], ele.upts[2], ele.upts[3], ele.jmat)
            compute = Kernel(be.make_loop(ele.neles, _compute),
                                 ele.upts[2], ele.upts[3], diag, ele.subres, ele.norm)
            
            main_kernels = [sweep, compute]
            
            if self._is_turb:
                tdiag = np.empty((ele.nturbvars, ele.nturbvars, ele.neles))
                tnv = (ele.nfvars, ele.nvars)

                _srcjacobian = ele.make_source_jacobian()

                _pre_tjacobi = make_tpre_jacobi(ele, tnv, _srcjacobian, self._tcfl_fac)
                pre_tjacobi = Kernel(be.make_loop(ele.neles, _pre_tjacobi),
                                     ele.upts[0], ele.dt, tdiag, ele.tjmat)
                
                # Turbulent sweeps and compute kernel
                _tsweep, _tcompute = make_jacobi_sweep(be, ele, tnv, fdx=0)
                tsweep = Kernel(be.make_loop(ele.neles, _tsweep),
                                ele.upts[1], ele.upts[2], ele.upts[3], ele.tjmat)
                tcompute = Kernel(be.make_loop(ele.neles, _tcompute),
                                  ele.upts[2], ele.upts[3], tdiag)
                
                main_kernels += [tsweep, tcompute]
                pre_kernels = [pre_jacobi, pre_tjacobi]
                ele.pre_jacobi = MetaKernel(pre_kernels)
            else:
                ele.pre_jacobi = pre_jacobi
            
            # Collect kernels and make meta kernels
            ele.jacobi_sweep = MetaKernel(main_kernels)

            # Update kernel
            _update = make_jacobi_update(ele)
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                                ele.upts[0], ele.upts[2], ele.subres)
            
            # Initialize dub array
            ele.upts[2][:] = 0.0

    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.approx_jac()
        self.sys.eles.pre_jacobi()

        # Sub-iteration for Jacobi method
        for subiter in range(self.subiter):
            # Jacobi sweep
            self.sys.eles.jacobi_sweep()

            # Collect error from entire domain
            drhoi = sum(self.sys.eles.norm.sum())
            drho = self._comm.allreduce(drhoi, op=MPI.SUM)
            drho = np.sqrt(drho)

            if subiter == 0:
                drho1 = drho
            else:
                if drho/drho1 < self.subtol:
                    break

        self.sys.eles.update()
        self.sys.post(0)
        
        return 0, resid


class BlockLUSGS(BaseSteadyIntegrator):
    name = 'blu-sgs'
    nreg = 3
    impl_op = 'approx-jacobian'

    def construct_stages(self):
        from integrators.blusgs import make_pre_blusgs, make_tpre_blusgs, \
                                            make_serial_blusgs, make_blusgs_update

        # Constants for Block LU-SGS subiteration
        self.subiter = self.cfg.getint('solver-time-integrator', 'sub-iter', 10)
        self.subtol = self.cfg.getfloat('solver-time-integrator', 'sub-tol', 0.1)

        be = self.be

        # Block LU-SGS method for each element
        for ele in self.sys.eles:
            # Get reordering result
            mapping, unmapping = ele.reordering()

            # Temporal array and matrix
            diag = np.empty((ele.nfvars, ele.nfvars, ele.neles))
            ele.subres = np.zeros((ele.neles,), dtype=np.float64)
            ele.norm = np.empty(1)

            nv = (0, ele.nfvars)
            
            # Compile Block LU-SGS functions
            _update = make_blusgs_update(ele)
            _pre_blusgs = make_pre_blusgs(be, ele, nv)
            _lower, _upper = make_serial_blusgs(be, ele, nv, mapping, unmapping, \
                                                res_idx=self._res_idx)

            # Initiate Block LU-SGS kernel objects
            pre_blusgs = Kernel(be.make_loop(ele.neles, _pre_blusgs),
                                ele.dt, diag, ele.jmat)
            
            # sweep kernels
            lsweeps = Kernel(be.make_loop(ele.neles, func=_lower),
                ele.upts[1], ele.upts[2], diag, ele.jmat)
            
            usweeps = Kernel(be.make_loop(ele.neles, func=_upper),
                ele.upts[1], ele.upts[2], diag, ele.jmat, ele.subres, ele.norm)
            
            sweep_kernels = [lsweeps, usweeps]

            # Block LU-SGS for turbulent model
            if self._is_turb:
                tdiag = np.empty((ele.nturbvars, ele.nturbvars, ele.neles))
                tnv = (ele.nfvars, ele.nvars)

                _srcjacobian = ele.make_source_jacobian()

                _pre_tblusgs = make_tpre_blusgs(be, ele, tnv, _srcjacobian, self._tcfl_fac)
                pre_tblusgs = Kernel(be.make_loop(ele.neles, _pre_tblusgs),
                                     ele.upts[0], ele.dt, tdiag, ele.tjmat)
                
                _tlsweep, _tusweep = make_serial_blusgs(be, ele, tnv, mapping, unmapping, \
                                                        fdx=0)
                tlsweep = Kernel(be.make_loop(ele.neles, _tlsweep),
                                 ele.upts[1], ele.upts[2], tdiag, ele.tjmat)
                tusweep = Kernel(be.make_loop(ele.neles, _tusweep),
                                 ele.upts[1], ele.upts[2], tdiag, ele.tjmat)

                sweep_kernels += [tlsweep, tusweep]
                pre_kernels = [pre_blusgs, pre_tblusgs]
                ele.pre_blusgs = MetaKernel(pre_kernels)
            else:
                ele.pre_blusgs = pre_blusgs

            # Collect all kernels
            ele.blusgs_sweep = MetaKernel(sweep_kernels)

            # Update kernel
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                                ele.upts[0], ele.upts[2], ele.subres)

            # Initialize dub array
            ele.upts[2][:] = 0.0

    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.approx_jac()

        # Compute diagonal matrix
        self.sys.eles.pre_blusgs()

        # Subiteration for Block LU-SGS
        for subiter in range(self.subiter):
            # Block LU-SGS sweep
            self.sys.eles.blusgs_sweep()

            # Collect L2-norm for all domain
            drhoi = sum(self.sys.eles.norm[0])
            drho = self._comm.allreduce(drhoi, op=MPI.SUM)
            drho = np.sqrt(drho)

            # Check sub-convergence
            if subiter == 0:
                drho1 = drho
            else:
                if drho/drho1 < self.subtol:
                    break

        self.sys.eles.update()
        self.sys.post(0)

        return 0, resid


class ColoredBlockLUSGS(BaseSteadyIntegrator):
    name = 'colored-blu-sgs'
    nreg = 3
    impl_op = 'approx-jacobian'

    def construct_stages(self):
        from integrators.blusgs import make_pre_blusgs, make_tpre_blusgs, \
                                            make_colored_blusgs, make_blusgs_update
        
        # Constants for Block LU-SGS subiteration
        self.subiter = self.cfg.getint('solver-time-integrator', 'sub-iter', 10)
        self.subtol = self.cfg.getfloat('solver-time-integrator', 'sub-tol', 0.1)

        be = self.be

        # Colored Block LU-SGS for each elements
        for ele in self.sys.eles:
            # Get coloring result
            ncolor, icolor, lev_color = ele.coloring()

            # Temporal array and matrix
            diag = np.empty((ele.nfvars, ele.nfvars, ele.neles))
            ele.subres = np.zeros((ele.neles,), dtype=np.float64)
            ele.norm = np.empty((ele.neles,))

            nv = (0, ele.nfvars)

            # Compile Block LU-SGS functions
            _update = make_blusgs_update(ele)
            _pre_blusgs = make_pre_blusgs(be, ele, nv)
            _lower, _upper = make_colored_blusgs(be, ele, nv, icolor, lev_color, \
                                                 res_idx=self._res_idx)

            # Initiate Block LU-SGS kernel objects
            pre_blusgs = Kernel(
                be.make_loop(ele.neles, _pre_blusgs),
                ele.dt, diag, ele.jmat
            )

            lsweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_lower),
                ele.upts[1], ele.upts[2], diag, ele.jmat)
                for n0, ne in zip(ncolor[:-1], ncolor[1:])
            ]

            usweeps = [
                Kernel(be.make_loop(n0=n0, ne=ne, func=_upper),
                ele.upts[1], ele.upts[2], diag, ele.jmat, ele.subres, ele.norm)
                for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
            ]

            sweep_kernels = [*lsweeps, *usweeps]

            # Colored Block LU-SGS for turbulence model
            if self._is_turb:
                # digonal matrix for turbulence model
                tdiag = np.empty((ele.nturbvars, ele.nturbvars, ele.neles))
                tnv = (ele.nfvars, ele.nvars)

                # Source term Jacobian
                _srcjacobian = ele.make_source_jacobian()

                _pre_tblusgs = make_tpre_blusgs(be, ele, tnv, _srcjacobian, self._tcfl_fac)
                pre_tblusgs = Kernel(be.make_loop(ele.neles, _pre_tblusgs),
                                     ele.upts[0], ele.dt, tdiag, ele.tjmat)
                
                _tlsweep, _tusweep = make_colored_blusgs(be, ele, tnv, icolor, lev_color, fdx=0)

                tlsweeps = [
                    Kernel(be.make_loop(n0=n0, ne=ne, func=_tlsweep),
                    ele.upts[1], ele.upts[2], tdiag, ele.tjmat)
                    for n0, ne in zip(ncolor[:-1], ncolor[1:])
                ]

                tusweeps = [
                    Kernel(be.make_loop(n0=n0, ne=ne, func=_tusweep),
                    ele.upts[1], ele.upts[2], tdiag, ele.tjmat)
                    for n0, ne in zip(ncolor[::-1][1:], ncolor[::-1][:-1])
                ]

                sweep_kernels += [*tlsweeps, *tusweeps]
                pre_kernels = [pre_blusgs, pre_tblusgs]
                ele.pre_blusgs = MetaKernel(pre_kernels)
            else:
                ele.pre_blusgs = pre_blusgs

            # Collect all kernels
            ele.blusgs_sweep = MetaKernel(sweep_kernels)

            # Update kernel
            ele.update = Kernel(be.make_loop(ele.neles, _update),
                                ele.upts[0], ele.upts[2], ele.subres)
            
            # Initialize dub array
            ele.upts[2][:] = 0.0

    def step(self):
        resid = self.rhs(0, 1, is_norm=True)
        self.sys.approx_jac()

        # Compute diagonal matrix
        self.sys.eles.pre_blusgs()

        # Subiteration for Block LU-SGS
        for subiter in range(self.subiter):
            # Block LU-SGS sweep
            self.sys.eles.blusgs_sweep()

            # Collect L2-norm for all domain
            drhoi = sum(self.sys.eles.norm.sum())
            drho = self._comm.allreduce(drhoi, op=MPI.SUM)
            drho = np.sqrt(drho)

            # Check sub-convergence
            if subiter == 0:
                drho1 = drho
            else:
                if drho/drho1 < self.subtol:
                    break

        self.sys.eles.update()
        self.sys.post(0)

        return 0, resid

