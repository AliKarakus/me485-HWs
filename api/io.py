# -*- coding: utf-8 -*-
from readers import get_reader
from partitions import get_partition
from readers.native import NativeReader
from writers import get_writer


import h5py
import os


def import_mesh(inmesh, outmesh, scale=1.0):
    """
    Import genreated mesh to pyBaram.

    :param string inmesh: Original mesh from generator (CGNS, Gmsh)
    :param string outmesh: Converted pyBaram mesh (.pbrm)
    :param float scale: Geometric scale factor 
    """
    # Split ext
    extn = os.path.splitext(inmesh)[1]

    # Get reader
    reader = get_reader(extn, inmesh, scale)

    # Get mesh in the pbm format
    mesh = reader.to_pbm()

    # Save to disk
    with h5py.File(outmesh, 'w') as f:
        for k, v in mesh.items():
            f[k] = v


def partition_mesh(inmesh, outmesh, npart):
    """
    Paritioning pyBarm mesh

    :param string inmesh: path and name of unspliited pyBaram mesh
    :param string outmesh: path and name of patitioned mesh
    :param int npart: number of partition
    """

    # mesh
    msh = NativeReader(inmesh)

    npart = int(npart)

    get_partition(msh, outmesh, npart)


def export_soln(mesh, soln, out):
    """
    Export solution to visualization file

    :param string mesh: pyBaram mesh file
    :param string soln: pyBaram solution file
    :param string out: exported file for visualization
    """
    # Get writer
    writer = get_writer(mesh, soln, out)

    writer.write()