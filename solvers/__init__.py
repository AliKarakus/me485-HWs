# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.euler.system import EulerSystem
from solvers.euler.elements import FluidElements
from utils.misc import subclass_by_name


# Choose system class for the integrators
def get_system(be, cfg, msh, soln, comm, nreg, impl_op):
    name = cfg.get('solver', 'system')
    return subclass_by_name(BaseSystem, name)(be, cfg, msh, soln, comm, nreg, impl_op)


def get_fluid(name):
    if name in ['euler']:
        return FluidElements()
    else:
        return subclass_by_name(FluidElements, name)()        