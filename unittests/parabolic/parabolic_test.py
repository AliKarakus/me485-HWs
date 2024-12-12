import sys
import os
sys.path.append("../../") # Adds higher directory to python modules path.

# Import all required modules
from unittests.modules import *
#----------------------------------------------------------------------------------#
def convertVTU():
	extension = '.pbrs'
	for root, _, filenames in os.walk(os.getcwd()):
	   for file in filenames:
	      if extension is None or file.endswith(extension):
	      	name = os.path.splitext(file)[0]
	      	export_soln(meshoutfile, name+'.pbrs', name+'.vtu')
#----------------------------------------------------------------------------------#
# Give the name of the INI file
inifile = "parabolic.ini"

# Prepare configure object from input file
cfg   = INIFile(inifile)

# Read mesh file name from INI file
meshinfile  = cfg.get('mesh', 'in_name')

# Read the name of the internal mesh file
meshoutfile = cfg.get('mesh', 'out_name')

#soln_file = cfg.get('soln-plugin-writer', 'name')

#get beckend object
backend     = get_backend('cpu', cfg)

# MPI Communicator
comm        = mpi_init()

# Import mesh and convert to native format to be used
import_mesh(meshinfile, meshoutfile)

# Create mesh object from native reader  
msh = NativeReader(meshoutfile)

# Run the simulation
run(msh, cfg)

# Convert internal pbrs files to paraview vtu files
convertVTU()




