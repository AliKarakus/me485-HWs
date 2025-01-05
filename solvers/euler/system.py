# -*- coding: utf-8 -*-
from solvers.baseadvec.system import BaseAdvecSystem
from solvers.euler import EulerElements, EulerBCInters, EulerIntInters, EulerMPIInters


class EulerSystem(BaseAdvecSystem):
    name = 'euler'
    _elements_cls = EulerElements
    _intinters_cls = EulerIntInters
    _bcinters_cls = EulerBCInters
    _mpiinters_cls = EulerMPIInters
