import scipy as sp
import numpy as np
import mesh as mesh
import argparse
import math as mth
import matplotlib.pyplot as plt


from src import base as base
from src import grad as grad
from src import diff as diff
from src import euler as euler

from src import timeStepper as timeStepper


# uniform flow, rho =1.4, u=3.0, rE = 8.8
# give conservative variables
def initialCond(time,  x):
	qi = np.zeros(4)
	qi[0] = 1.4
	qi[1] = 3.0*1.4
	qi[2] = 0.0
	qi[3] = 8.8
	return qi


def boundaryCond(bc, time, x, n, qM):
	qi = np.zeros(4)
	# wall boundary, slip, reflective
	if(bc==1):
		rho = qM[0]
		u  = qM[1]/rho
		v  = qM[2]/rho
		re = qM[3]
		qi[0] = rho
		qi[1] = rho*(u - (u*n[0] + v*n[1]))
		qi[2] = rho*(v - (u*n[0] + v*n[1]))
		qi[3] = re
	# Inflow, hypersonic, free-stream		
	if(bc==2):
		qi[0] = 1.4
		qi[1] = 3.0*1.4
		qi[2] = 0.0
		qi[3] = 8.8
	# outflow, hypersonic, all from interior
	if(bc==3):
		qi[0] = qM[0]
		qi[1] = qM[1]
		qi[2] = qM[2]
		qi[3] = qM[3]		
	return qi

# exact solution if it is available
def exactSolution(time, x):
	exct = mth.sin(mth.pi*x[0])*mth.sin(mth.pi*x[1])
	return exct


#-----PROBLEM SETUP:
parser = argparse.ArgumentParser(prog='Euler',description='Solver Euler Equations using FVM',
                    epilog='--------------------------------')

parser.add_argument('--meshFile', default ='data/ffsTri.msh', help='mesh file to be read')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the problem')

parser.add_argument('--limiter',  type=str, default='BARTH-JESPERSEN', 
					choices=['BARTH-JESPERSEN', 'VENKATAKRISHNAN'])

parser.add_argument('--flux',  type=str, default='LLF', choices=['LLF', 'HLLC'])


parser.add_argument('--IC', type = initialCond, default = initialCond, help='Initial condition function')
parser.add_argument('--BC', type = boundaryCond, default = boundaryCond, help='Boundary condition function')


parser.add_argument('--timeMethod',  type=str, default='LSERK', choices=['LSRK4'])
parser.add_argument('--tstart',      type=float, default = 0.0,    help='initial time')
parser.add_argument('--tend',        type=float, default = 4.0,    help='final time')
parser.add_argument('--dt',          type=float, default = 0.001,   help='time step size')
parser.add_argument('--Noutput',     type=int,   default = 100,     help='output frequency')

args = parser.parse_args()


# Read mesh file and setup geometry and connections
msh = base(args.meshFile)
# create solver class and set arguments
solver = euler(msh); solver.set(args)

# create initial conditions: primitive and conservative variables
solver.qp = msh.createEfield(solver.Nfields)
solver.qc = msh.createEfield(solver.Nfields)
for elm,info in msh.Element.items():
	x = info['ecenter']
	solver.qc[elm,:] = args.IC(0.0, x)

solver.qb = solver.grd.createFaceBfield(solver.qc)

# Integrate 
solver.timeStepper.run(solver.qc, solver.qb)
