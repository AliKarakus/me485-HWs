import sys
sys.path.append("../") # Adds higher directory to python modules path.

from argparse import ArgumentParser, FileType
from api.io import import_mesh, partition_mesh, export_soln
from api.simulation import run, restart
from inifile import INIFile
from readers.native import NativeReader
from solvers.grad import GradElements, GradIntInters, GradMPIInters, GradBCInters, GradVertex
from geometry import get_geometry
from solvers import get_system

from backends import get_backend
from integrators import get_integrator
from utils.mpi import mpi_init

# Give the name of the INI file
inifile = "test.ini"
# Prepare configure object from input file
cfg   = INIFile(inifile)
# Read mesh file name from INI file
meshinfile  = cfg.get('mesh', 'in_name')
# Read the name of the internal mesh file
meshoutfile = cfg.get('mesh', 'out_name')

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

# Export visualization file
soln_file = cfg.get('soln-plugin-writer', 'name')
export_soln(meshoutfile, soln_file+'.pbrs', soln_file+'.vtu')




