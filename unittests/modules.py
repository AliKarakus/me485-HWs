#----------------------------------------------------------------------------------#
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
from utils.np import chop, npeval