# -*- coding: utf-8 -*-
import h5py


class NativeReader(object):
    def __init__(self, meshf):
        self._file = h5py.File(meshf, 'r')

    def __getitem__(self, item):
        return self._file[item][()]

    def __iter__(self):
        return self._file.__iter__()
