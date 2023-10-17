import scipy as sp
import numpy as np
import mesh as mesh
import argparse

from src import base as base
from src import grad as grad
from src import diff as diff


def initialCond(time,  x):
	qi = 0.0
	return qi

def boundaryCond(bc, time, x, qM):
	qi = 0
	if(bc==1):
		qi = 1 + x[1]
	elif(bc==2):
		qi = 0
	elif(bc==3):
		qi = 0
	return qi

def diffusionCoeff(time, x):
	coeff =1
	return coeff

def diffusionSource(time, x):
	source = 10 + x[0]**2 + x[1]**2
	return source

#-----PROBLEM SETUP:
# Create parser 
parser = argparse.ArgumentParser(prog='Diffusion',description='Compute Diffusion Equation using FVM',
                    epilog='--------------------------------')

# Set default values for gradient computation
# parser.add_argument('--meshFile', default ='data/heat.msh', help='mesh file to be read')
parser.add_argument('--meshFile', default ='data/text.msh', help='mesh file to be read')

parser.add_argument('--method', type=str, default='FALSE', choices=['MINIMUM-CORRECTION', 'FALSE'], help='Diffusion formulation')



parser.add_argument('--Nfields', type=int, default=1, help='Number of fields')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the problem')

parser.add_argument('--IC', type = initialCond, default = initialCond, help='Initial condition function')
parser.add_argument('--BC', type = boundaryCond, default = boundaryCond, help='Boundary condition function')
parser.add_argument('--DC', type = diffusionCoeff, default = diffusionCoeff, help='Diffusion coefficient function')
parser.add_argument('--DS', type = diffusionSource, default = diffusionSource, help='Diffusion source function')


parser.add_argument('--linSolver', type=str, default='CG', choices=['DIRECT', 'CG'], help='Linear solver')
parser.add_argument('--linTolerance', type=float, default=1e-5, help='Linear solver tolerance')
parser.add_argument('--linPrecond', type=str, default='JACOBI', choices=['JACOBI', 'ILU'])

args = parser.parse_args()
time = 0.0

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

# #--COMPUTE DIFFUSION
dff  = diff(msh); dff.set(args)
Tb   = dff.createBfield(Te)

dff.assemble(Te, Tb, Ts, Tc)
T = dff.solve(args)


Tb  = dff.extractBfield(T)


#--POSTPROCESS DIFFUSION: 
Tv = msh.cell2Node(T,  Tb, 'average')
msh.plotVTU("diffusion.vtu", Tv)
