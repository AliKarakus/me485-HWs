# -*- coding: utf-8 -*-
from plugins import get_plugin
from solvers import get_system
from utils.misc import ProxyList


class BaseIntegrator:
    def __init__(self, be, cfg, msh, soln, comm):
        self.be = be
        self.mesh = msh
        
        # Get system of equations
        self.sys = get_system(be, cfg, msh, soln, comm, self.nreg, self.impl_op)

        # Current index for pointing current array
        self._curr_idx = 0

        # Check aux array (turbulence variables or others for post processing)
        try:
            self.curr_aux
            self.is_aux=True
        except:
            self.is_aux=False

        # Store plugins in the handler
        self.completed_handler = plugins = ProxyList()
        for sect in cfg.sections():
            if sect.startswith('soln-plugin'):
                # Extract plugin name
                name = sect.split('-')[2:]

                # Check plugin has suffix
                if len(name) > 1:
                    name, suffix = name
                else:
                    name, suffix = name[0], None

                # Initiate plugin object and save it to handler
                plugins.append(get_plugin(name, self, cfg, suffix))

    @property
    def curr_soln(self):
        # Return current solution array
        return self.sys.eles.upts[self._curr_idx]

    @property
    def curr_aux(self):
        # Return current aux variable array
        return self.sys.eles.aux

    @property
    def curr_mu(self):
        # Get visocisty variable (mu) vector
        mu = self.sys.eles.mu

        if hasattr(self.sys.eles, 'mut'):
            # If turbulent viscosity (mu_t) is defined, return mu + mu_t
            mu = ProxyList([m1 + m2 for m1, m2 in zip(mu, self.sys.eles.mut)])
        
        return mu