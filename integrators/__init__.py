# -*- coding: utf-8 -*-
from integrators.base import BaseIntegrator
from integrators.unsteady import BaseUnsteadyIntegrator
from integrators.steady import BaseSteadyIntegrator
from integrators.onestep import OneStepIntegrator
from utils.misc import subclass_by_name


def get_integrator(be, cfg, msh, soln, comm):
  mode = cfg.get('solver-time-integrator', 'mode', 'unsteady')
  stepper = cfg.get('solver-time-integrator', 'stepper', 'tvd-rk3')

  if mode == 'unsteady':
    intg = subclass_by_name(BaseUnsteadyIntegrator, stepper)
  elif mode == 'steady':
    intg = subclass_by_name(BaseSteadyIntegrator, stepper)
  else :
    intg = OneStepIntegrator
    
  return intg(be, cfg, msh, soln, comm)