# -*- coding: utf-8 -*-
from utils.nb import dot
import numpy as np


def get_bc(self, be, name, bcargs):
    bc = eval('make_bc_'+name)
    return be.compile(bc(bcargs))


def make_bc_drichlet(bcargs):
    nvars = bcargs['nfvars']
    ndims = bcargs['ndims']
    ub    = np.empty(nvars)
    ub[0] = bcargs['q'] 
    def bc(ul, ur, vl, vr,*args):
        ur[0] = ul[0]
        for i in range(ndims):
            vr[i] = vl[i]
    return bc