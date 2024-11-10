# -*- coding: utf-8 -*-
from mpi4py import MPI


import numpy as np

from plugins.base import BasePlugin, csv_write
from utils.np import npeval


class SurfIntPlugin(BasePlugin):
    # Compute integrated and/or average value at boundary
    name = 'surface'

    def __init__(self, intg, cfg, suffix):
        self.cfg = cfg
        sect = 'soln-plugin-{}-{}'.format(self.name, suffix)

        # get MPI_COMM_WORLD and rank
        self._comm = comm = MPI.COMM_WORLD
        self._rank = rank = comm.rank

        # Get constants and expressions of variable
        self._const = cfg.items('constants')
        items = [e.strip() for e in cfg.get(sect, 'items', 'area').split(',')]
        self._exprs = [cfg.get(sect, item, 1) for item in items]

        # Parse normal vector name
        self.ndims = ndims = intg.sys.ndims
        self._nvec = ['n{}'.format(e) for e in 'xyz'[:ndims]]

        # Map as {BC type : Boundary interface objects and Virtual interface objects}
        bcmap = {bc.bctype: bc for bc in intg.sys.bint}
        bcmap.update({vr.bctype: vr for vr in intg.sys.vint})

        self._bcinfo = bcinfo = {}
        area = 0.0

        if suffix in bcmap:
            # Get normal vector and element index for bc
            bc = bcmap[suffix]
            t, e, _ = bc._lidx
            mag, vec = bc._mag_snorm, bc._vec_snorm

            for i in np.unique(t):
                mask = (t == i)
                eidx = e[mask]
                nvec, nmag = vec[:, mask], mag[mask]
                bcinfo[i] = (eidx, nvec, nmag)
                area += np.sum(nmag)

        # Compute surface area at boundary
        area = np.array(area)
        if self._rank != 0:
            self._comm.Reduce(area, None, op=MPI.SUM, root=0)
        else:
            self._comm.Reduce(MPI.IN_PLACE, area, op=MPI.SUM, root=0)
        
        self._area = area

        # Check integratro mode (steady | unsteady) and frequency to compute the plugin
        self.mode = intg.mode
        if self.mode == 'steady':
            self.itout = cfg.getint(sect, 'iter-out', 100)
            lead = ['iter']
        else:
            self.dtout = cfg.getfloat(sect, 'dt-out')
            self.tout_next = intg.tcurr + self.dtout
            intg.add_tlist(self.dtout)
            lead = ['t']

        # Out file name and header
        if rank == 0:
            header = lead + ['sum_{}'.format(e) for e in items] + ['avg_{}'.format(e) for e in items]
            fname = "surface_{}.csv".format(suffix)
            self.outf = csv_write(fname, header)

    def __call__(self, intg):
        # Check if force is computed or not at this iteration or time
        if self.mode == 'steady':
            if not intg.isconv and intg.iter % self.itout:
                return
            txt = [intg.iter]

        else:
            if abs(intg.tcurr - self.tout_next) > 1e-6:
                return

            self.tout_next += self.dtout
            
            txt = [intg.tcurr]

        # Convert elements and solutions as list
        eles = list(intg.sys.eles)
        solns = list(intg.curr_soln)

        dist = []
        for i, (eidx, nvec, nmag) in self._bcinfo.items():
            # Convert Primevars
            soln = solns[i]
            pn = eles[i].primevars
            pv = eles[i].conv_to_prim(soln[:, eidx], self.cfg)

            # Compute variables using expressions
            subs = {n : v for n,v in zip(pn, pv)}
            subs.update({n : v for n, v in zip(self._nvec, nvec)})
            subs.update(self._const)
            var_at = np.array([npeval(expr, subs) for expr in self._exprs])
            dist.append(var_at*nmag)

        if dist:      
            # If dist is compuated in this rank, pack data
            if len(var_at.shape) > 1:
                # Integrate vector variable 
                integ_var = np.sum(np.hstack(dist), axis=1)
            else:
                # Not computed int this rank, give zero vector
                integ_var = np.array([np.sum(dist)])
        else:
            integ_var = np.zeros(len(self._exprs))

        # Collect integerated variables over all ranks
        if self._rank != 0:
            self._comm.Reduce(integ_var, None, op=MPI.SUM, root=0)
        else:
            self._comm.Reduce(MPI.IN_PLACE, integ_var, op=MPI.SUM, root=0)

        if self._rank == 0:
            # Write
            row = txt + integ_var.tolist() + (integ_var/self._area).tolist()
            print(','.join(str(r) for r in row), file=self.outf)

            # Flush to disk
            self.outf.flush()