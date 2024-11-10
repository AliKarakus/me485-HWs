# -*- coding: utf-8 -*-
from utils.misc import subclass_by_name
from writers.base import BaseWriter
from writers.vtk import VTKWriter
from writers.tecplot import TecplotWriter


def get_writer(meshf, solnf, outf, **kwargs):
    suffix = outf.split('.')[-1]
    writer = subclass_by_name(BaseWriter, suffix)

    return writer(meshf, solnf, outf, **kwargs)
