import scipy as sp
import numpy as np
import mesh as mesh
import argparse
from src import base as base

#-----PROBLEM SETUP:
# Create parser to read input 
parser = argparse.ArgumentParser(prog='Test',description='Test for mesh info',
	       epilog='--------------------------------')

# Set default values for testing topology
parser.add_argument('--meshFile', default ='data/test.msh2', help='mesh file to be read')
parser.add_argument('--Nfields', type=int, default=1, help='Number of fields')
parser.add_argument('--dim', type=int, default=2, choices= [2,3], help='Dimension of the problem')
args = parser.parse_args()

# Read mesh file and setup geometry and connections
msh = base(args.meshFile)

# for elmM, info in msh.Element.items():
# 	# obtain type of element i.e. triangle, quad etc..
# 	etype   = info['elementType']
# 	# number of elements for this element type
# 	nfaces  = msh.elmInfo[etype]['nfaces'] 	
# 	# reach geometric information, i.e. cell center coordinates 
# 	xc = info['ecenter']
# 	# volume of this element
# 	vol = info['volume']
# 	# loop over faces of this element
# 	for face in range(nfaces):
# 		# get boundary id for the face 
# 		bc = info['boundary'][face]
# 		# neighbor element of the face
# 		elmP = info['neighElement'][face]
# 		# get normal of the face
# 		nf = info['normal'][face]
# 		# center of the face
# 		xf = info['fcenter'][face]
# 		# area of the face
# 		af = info['area'][face] 

# eid = 100
# etype = msh.Element[eid]['elementType']
# print('element type=', etype)
# print('elem Info = ', msh.elmInfo[etype])

# for fM, info in msh.Face.items():
# 	# owner and neighbor elements
# 	eM = info['owner'] 
# 	eP = info['neigh']
# 	# check the boundary for this face
# 	bc     = info['boundary']
# 	# check the boundary id of the face
# 	bcid = info['bcid']
# 	# normal of the face pointed from owner to neighbor
# 	normal = info['normal']
# 	# weight for weighted average 
# 	weight = info['weight']
# 	# area of face
# 	area   = info['area']
# 	# get cell center coordinates of owner cell
#     xM = msh.Element[eM]['ecenter']
# 	# get cell center coordinates of neighbor cell
#     xP = msh.Element[eP]['ecenter']
#     # center coordinates of the cell
#     xF = info['center']


for vrtx, info in self.Node.items():
    # list of elements connected to vertex
    elements = info['element']
    # check if this vertex is on the boundary of the domain
    bc       = info['boundary']
    # coordinates of the vertex
    xv       = info['coord']

    # go over the lements connected to this vertex
    for elm in range(len(elements)):
    	# element id
        eid = elements[elm]
        




# print(, msh.elmInfo[msh.Element[250]['elementType']], msh.Element[250]['neighElement'])
# print(msh.Element[150]['neighElement'])



# Post process result
# Create a field defined on element centers, initialized with zero
Te = msh.createEfield(args.Nfields)
Tf = msh.createFfield(args.Nfields, msh.dim)
Tv = msh.createVfield(args.Nfields, msh.dim)

msh.plotVTU("test.vtu", Tv)
