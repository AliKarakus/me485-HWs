import scipy as sp
import numpy as np
import mesh as mesh
import argparse
import math as mth
import matplotlib.pyplot as plt


from src import base as base
from src import grad as grad
from src import diff as diff

from src import timeStepper as timeStepper


def initialCond(time,  x):
	qi = 0.0
	return qi

def boundaryCond(bc, time, x, qM):
	qi = 0
	if(bc==1):
		qi = mth.sin(mth.pi*x[0])*mth.sin(mth.pi*x[1])
	return qi

def diffusionCoeff(time, x):
	coeff = 1
	return coeff

def diffusionSource(time, x):
	source = 2.0*mth.pi**2*mth.sin(mth.pi*x[0])*mth.sin(mth.pi*x[1])
	return source

def exactSolution(time, x):
	exct = mth.sin(mth.pi*x[0])*mth.sin(mth.pi*x[1])
	return exct


#-----PROBLEM SETUP:
parser = argparse.ArgumentParser(prog='Parabolic',description='Compute Diffusion Equation using FVM',
                    epilog='--------------------------------')
parser.add_argument('--Nfields', type=int, default=1, help='Number of fields')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the problem')

parser.add_argument('--timeMethod',  type=str, default='ADAMS-BASHFORTH', 
					choices=['LSRK4', 'FORWARD-EULER', 'ADAMS-BASHFORTH', 'BACKWARD-EULER', 'BDF'])
parser.add_argument('--timeOrder',  type=int, default=2, 
	                choices=[2, 3], help='Order for AB and BDF methods')
parser.add_argument('--meshFile', default ='data/cavityTri.msh', help='mesh file to be read')
parser.add_argument('--method', type=str, default='NO-CORRECTION', \
					choices=['MINIMUM-CORRECTION', 'NO-CORRECTION'], help='Diffusion formulation')

parser.add_argument('--IC', type = initialCond, default = initialCond, help='Initial condition function')
parser.add_argument('--BC', type = boundaryCond, default = boundaryCond, help='Boundary condition function')
parser.add_argument('--DC', type = diffusionCoeff, default = diffusionCoeff, help='Diffusion coefficient function')
parser.add_argument('--DS', type = diffusionSource, default = diffusionSource, help='Diffusion source function')

parser.add_argument('--tstart',      type=float, default = 0.0,    help='initial time')
parser.add_argument('--tend',        type=float, default = 1.0,    help='final time')
parser.add_argument('--dt',          type=float, default = 0.001,   help='time step size')
parser.add_argument('--Noutput',     type=int,   default = 100,     help='output frequency')

parser.add_argument('--linSolver', type=str, default='CG', choices=['DIRECT','CG','GMRES'], help='Linear solver')
parser.add_argument('--linTolerance', type=float, default=1e-8, help='Linear solver tolerance')
parser.add_argument('--linPrecond', type=str, default='AMG', choices=['JACOBI', 'ILU', 'AMG'])

args = parser.parse_args()
Nsteps = int(np.ceil((args.tend - args.tstart)/args.dt)); 
dt = (args.tend - args.tstart)/Nsteps

# Read mesh file and setup geometry and connections
msh = base(args.meshFile)

#Create intitial condition, boundary condition and source term
Te = msh.createEfield(args.Nfields)
Ts = msh.createEfield(args.Nfields)
Tc = msh.createEfield(args.Nfields)
for elm,info in msh.Element.items():
	x = info['ecenter']
	Te[elm][:] = args.IC(0.0, x)
	Tc[elm][:] = args.DC(0.0, x)
	Ts[elm][:] = args.DS(0.0, x)

#COMPUTE DIFFUSION
dff  = diff(msh);  dff.set(args)
Tbf  = dff.createFaceBfield(Te)

timeStepper = timeStepper(msh,dff)
timeStepper.set(args)

# Integrate 
timeStepper.run(Te, Tbf)

#--POSTPROCESS DIFFUSION: 
Tbv  = dff.createVertexBfield(Te)
Tv = msh.cell2Node(Te,  Tbv, 'average')
msh.plotVTU("diffusion.vtu", Tv)

# Compute Infinity Norm of Error
linf = 0.0; l2 = 0.0
for elm, info in msh.Element.items():
	x = info['ecenter']
	vol = info['volume']
	exct = exactSolution(0.0, x)
	soln = Te[elm]
	err =  abs(exct -soln)
	l2 = l2 + vol*err**2
	if err >linf: linf = err;

print("Infinity Norm of error: %.8e" %linf)
print("L2 Norm of error: %.8e" %l2)

# print(msh.Face[10])