# -*- coding: utf-8 -*-
from backends import get_backend
from integrators import get_integrator
from utils.mpi import mpi_init

from tqdm import tqdm


def run(mesh, cfg, be='none', comm='none'):
    """
    Fresh run from mesh and configuration files.

    :param mesh: mefvm NativeReader object
    :type mesh: mefvm mesh
    :param cfg: mefvm INIFile object
    :type cfg: config
    :param be: mefvm backend object
    :type be: Backend
    :param comm: mpi4py comm object
    :type comm: MPI communicator
    """
    # Run common
    _common(mesh, None, cfg, be, comm)


def restart(mesh, soln, cfg, be='none', comm='none'):
    """
    Restarted run from mesh and configuration files.


    :param mesh: mefvm NativeReader object
    :type mesh: mefvm mesh
    :param soln: mefvm NativeReader object
    :type soln: mefvm solution
    :param cfg: mefvm INIFile object
    :type cfg: config
    :param be: mefvm backend object
    :type be: Backend
    :param comm: mpi4py comm object
    :type comm: MPI communicator
    """
    # Check mesh and solution file
    if mesh['mesh_uuid'] != soln['mesh_uuid']:
        raise RuntimeError('Solution is not computed by the mesh')

    # Run common
    _common(mesh, soln, cfg, be, comm)


def _common(msh, soln, cfg, backend, comm):
    if comm == 'none':        
        # Initiate MPI comm world
        comm = mpi_init()

    # Get backend
    if backend == 'none':
        backend = get_backend('cpu', cfg)

    # Get integrator

    integrator = get_integrator(backend, cfg, msh, soln, comm)

    # Add progress bar
    if comm.rank == 0:
        if integrator.mode == 'unsteady':
            pb = tqdm( total=integrator.tlist[-1], initial=integrator.tcurr,
                       unit_scale=True)
            def callb(intg): return pb.update(intg.dt)
            integrator.completed_handler.append(callb)
        elif integrator.mode =='steady':
            pb = tqdm(total=integrator.itermax, initial=integrator.iter)

            def callb(intg): return pb.update(1)
            integrator.completed_handler.append(callb)
        else:
            print("no integrator is choosen: just a single step will be performed")

    integrator.run()
