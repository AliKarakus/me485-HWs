# This is to solve  a model problem
#
# 	div( k \cdot \grad q) + Qs = 0
# 	where
# 		k 	: diffusion coefficient
# 		Q 	: source term
# 	Corresbonding discrete scheme will be
# 		A*q + b  = 0
# 		where 
# 		A 	: system matrix including terms including q
# 		b     : all explicit terms which are not function of q
# then A q = -b  solve for q

#------------------------------------------------------------------------------#
import scipy as sp
import numpy as np
import mesh as mesh
import argparse
import math as mth
import matplotlib.pyplot as plt


from src import base as base
from src import grad as grad
from src import diff as diff

#------------------------------------------------------------------------------#
def initialCond(time,  x):
	qi = x[0] + x[1]
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

#------------------------------------------------------------------------------#
#-----PROBLEM SETUP:
# Create parser 
parser = argparse.ArgumentParser(prog='Diffusion',description='Compute Diffusion Equation using FVM',
                    epilog='--------------------------------')

parser.add_argument('--meshFile', default ='data/cavityTri.msh', help='mesh file to be read')

parser.add_argument('--method', type=str, default='NO-CORRECTION', 
					choices=['MINIMUM-CORRECTION', 'ORTHOGONAL-CORRECTION', 'OVER-RELAXED-CORRECTION', 'NO-CORRECTION'], 
					help='Diffusion formulation')

parser.add_argument('--Nfields', type=int, default=1, help='Number of fields')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the problem')

parser.add_argument('--IC', type = initialCond, default = initialCond, help='Initial condition function')
parser.add_argument('--BC', type = boundaryCond, default = boundaryCond, help='Boundary condition function')
parser.add_argument('--DC', type = diffusionCoeff, default = diffusionCoeff, help='Diffusion coefficient function')
parser.add_argument('--DS', type = diffusionSource, default = diffusionSource, help='Diffusion source function')


parser.add_argument('--linSolver', type=str, default='CG', choices=['DIRECT','CG','GMRES'], help='Linear solver')
parser.add_argument('--linTolerance', type=float, default=1e-8, help='Linear solver tolerance')
parser.add_argument('--linPrecond', type=str, default='AMG', choices=['JACOBI', 'ILU', 'AMG'])

args = parser.parse_args()
time = 0.0
#----------------------------------------------------------------#
# Read mesh file and setup geometry and connections
msh = base(args.meshFile)

#Create intitial condition, boundary condition and source term
Qe = msh.createEfield(args.Nfields)
Qc = msh.createEfield(args.Nfields)
Qs = msh.createEfield(args.Nfields)
for elm,info in msh.Element.items():
	x = info['ecenter']
	Qe[elm][:] = args.IC(0.0, x)

#----------------------------------------------------------------#
#COMPUTE DIFFUSION
dff  = diff(msh);  dff.set(args)
Qbf  = dff.createFaceBfield(Qe)

# forms the system A*q + b = 0
A, b  = dff.assemble(Qe, Qbf)
Qe     = dff.solve(args, A, -b)
#----------------------------------------------------------------#
#POSTPROCESS DIFFUSION: 
Qbv  = dff.createVertexBfield(Qe)
Qv = msh.cell2Node(Qe,  Qbv, 'average')
msh.plotVTU("diffusion.vtu", Qv)

#----------------------------------------------------------------#
# Compute Infinity Norm of Error
linf = 0.0; l2 = 0.0
for elm, info in msh.Element.items():
	x = info['ecenter']
	vol = info['volume']
	exct = exactSolution(0.0, x)
	soln = Qe[elm]
	err =  abs(exct -soln)

	l2 = l2 + vol*err**2
	if err >linf: linf = err;

print("Infinity Norm of error: %.8e" %linf)
print("L2 Norm of error: %.8e" %l2)

