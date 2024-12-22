# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.advection.system import AdvectionSystem
from solvers.advection.elements import AdvectionFluidElements

from utils.misc import subclass_by_name

# Choose system class for the integrators
def get_system(be, cfg, msh, soln, comm, nreg, impl_op):
    name = cfg.get('solver', 'system')
    return subclass_by_name(BaseSystem, name)(be, cfg, msh, soln, comm, nreg, impl_op)


def get_fluid(name):
    if name in ['euler']:
        return FluidElements()
    elif name in ['grad']:
        return gradFluidElements()
    elif name in ['parabolic']:
        return ParabolicFluidElements()
    elif name in ['advection']:
        return AdvectionFluidElements()
    else:
        print(name)
        return subclass_by_name(FluidElements, name)()        