import scipy as sp
import numpy as np
import mesh as mesh
import argparse
import matplotlib.pyplot as plt

from src import base as base
from src import grad as grad

# Define initial and boundary conditions
def initialCond(time, x):
	Nfields = 1
	qi = np.zeros(Nfields, float)
	qi = x[0]**2 + x[1]**2
	# qi = 1
	return qi


def boundaryCond(bc, time, x, qM):
	Nfields = 1
	qi = np.zeros(Nfields, float)
	if(bc==1 or bc ==2):
		qi = x[0]**2 + x[1]**2
	elif(bc==3):
		qi = qM
	return qi

def exactSoln(time, x):
	exctx = 2.0*x[0]
	excty = 2.0*x[1]
	return (exctx, excty)

#-----PROBLEM SETUP:
# Create parser 
parser = argparse.ArgumentParser(prog='Gradient',description='Compute Gradient using FVM',
                    epilog='--------------------------------')

# Set default values for gradient computation
parser.add_argument('--meshFile', default ='data/cavityTri.msh', help='mesh file to be read')

parser.add_argument('--method', type=str, default='GREEN-GAUSS-NODE', \
	choices=['GREEN-GAUSS-CELL', 'GREEN-GAUSS-NODE', 'LEAST-SQUARES', 'WEIGHTED-LEAST-SQUARES'],\
	help='Gradient Reconstruction algorithm')

parser.add_argument('--Correct', type=bool, default=True, help='Correction for GGCB reconstruction')
parser.add_argument('--Nfields', type=int, default=1, help='Number of fields')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the proble ')
parser.add_argument('--IC', type = initialCond, default = initialCond, help='Initial condition function')
parser.add_argument('--BC', type = boundaryCond, default = boundaryCond, help='Boundary condition function')
args = parser.parse_args()


# print(vars(args))
# Create Boundary Field Defined on Boundary Faces 
time = 0.0; 

#-----CREATE MESH and CONNECTIVITY
# Create mesh class and read mesh file
# Read mesh file and setup geometry and connections
msh = base(args.meshFile)

# Create an element field and assign initial condition
Qe = msh.createEfield(args.Nfields)
for elm,info in msh.Element.items():
	x = info['ecenter']
	Qe[elm, :] = args.IC(time, x)

#--COMPUTE GRADIENT
# Instantinate gradient class
grd = grad(msh);  
# Set gradient options
grd.set(args); 
# Obtain boundary data using BC function
Qb   = grd.createFaceBfield(Qe)
# Qbv   = grd.createVertexBfield(Qe)
# Compute the gradient
gradQ = grd.compute(Qe, Qb)


#--POSTPROCESS GRADIENT: 
# Interpolate gradient field to faces
gradQf = grd.interpolateToFace(Qe, Qb, gradQ)
# Get gradient at boundaries
gradQb = grd.extractBoundaryFromFace(gradQf)
# Get node values using inverse distance averaging
gradQv = msh.cell2Node(gradQ,  gradQb, 'average')
# Plot field
msh.plotVTU("grad.vtu", gradQv)

# Compute Infinity Norm of Error
linfX = 0.0; linfY = 0.0; vol = 0.0
l2x = 0.0; l2y = 0.0; 
for elm, info in msh.Element.items():
	x = info['ecenter']
	vol = info['volume']
	exctx, excty = exactSoln(0.0, x)
	solnx = gradQ[elm, :, 0]
	solny = gradQ[elm, :, 1]

	errx =  abs(exctx -solnx)
	erry =  abs(excty -solny)

	l2x = l2x + vol*errx**2
	l2y = l2y + vol*erry**2

	if errx >linfX: linfX = errx;
	if erry>linfY: linfY = erry;

print("Infinity Norm of X Component: ", linfX)
print("Infinity Norm of Y Component: ", linfY)
print("L2 Norm of X Component: ", l2x)
print("L2 Norm of Y Component: ", l2y)

