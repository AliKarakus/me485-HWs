# -*- coding: utf-8 -*-
from solvers.base.system import BaseSystem
from solvers.grad.system import GradSystem
from solvers.grad.elements import gradFluidElements
# from solvers.euler.system import EulerSystem
# from solvers.navierstokes.system import NavierStokeSystem
# from solvers.ranssa.system import RANSSASystem
# from solvers.ranskwsst.system import RANSKWSSTSystem
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
    else:
        print(name)
        return subclass_by_name(FluidElements, name)()        