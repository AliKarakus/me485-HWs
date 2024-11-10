# -*- coding: utf-8 -*-
from backends.base import Backend
from backends.cpu.backend import CPUBackend
from utils.misc import subclass_by_name


def get_backend(name, *args, **kwargs):
    return subclass_by_name(Backend, name)(*args, **kwargs)
