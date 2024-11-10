# -*- coding: utf-8 -*-

from readers.base import BaseReader
from readers.cgns import CGNSReader
from readers.gmsh import GMSHReader

from utils.misc import subclass_by_name, subclasses


def get_reader(extn, *args, **kwargs):
    reader_map = {ex : cls for cls in subclasses(BaseReader) for ex in cls.extn}

    return reader_map[extn](*args, **kwargs)
