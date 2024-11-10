# -*- coding: utf-8 -*-
from mpi4py import MPI
from backends.types import Kernel
from inifile import INIFile
from integrators.base import BaseIntegrator
from utils.misc import ProxyList
import numpy as np


class OneStepIntegrator(BaseIntegrator):
    mode = 'oneestep'
    impl_op = 'none'
    nreg = 1
    iter = 1 
    isconv = 1 

    def __init__(self, be, cfg, msh, soln, comm):
        # get MPI_COMM_WORLD
        self._comm = comm
        self.cfg = cfg
        super().__init__(be, cfg, msh, soln, comm)
        # Construct kernels

    def run(self):
        self.sys.rhside()  
        self.completed_handler(self)
