# Import all required modules
from modules import *
#----------------------------------------------------------------------------------#
# Give the name of the INI file
inifile = "grad.ini"

# Prepare configure object from input file
cfg   = INIFile(inifile)

# Read mesh file name from INI file
meshinfile  = cfg.get('mesh', 'in_name')

# Read the name of the internal mesh file
meshoutfile = cfg.get('mesh', 'out_name')

# get output file to export solution for visualization
soln_file = cfg.get('soln-plugin-writer', 'name')

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

# export solution file in vtu format to open in paraview
export_soln(meshoutfile, soln_file+'.pbrs', soln_file+'.vtu')




