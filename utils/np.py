# -*- coding: utf-8 -*-


import functools as ft
import numpy as np


eps = np.finfo(np.float64).eps


def chop(fn):
    @ft.wraps(fn)
    def newfn(*args, **kwargs):
        arr = fn(*args, **kwargs)

        # Determine a tolerance and flush
        arr[abs(arr) < np.finfo(arr.dtype).resolution] = 0

        return arr
    return newfn


def fuzzysort(arr, idx, dim=0, tol=1e-6):
    # Extract our dimension and argsort
    arrd = arr[dim]
    srtdidx = sorted(idx, key=arrd.__getitem__)

    i, ix = 0, srtdidx[0]
    for j, jx in enumerate(srtdidx[1:], start=1):
        if arrd[jx] - arrd[ix] >= tol:
            if j - i > 1:
                srtdidx[i:j] = fuzzysort(arr, srtdidx[i:j], dim + 1, tol)
            i, ix = j, jx

    if i != j:
        srtdidx[i:] = fuzzysort(arr, srtdidx[i:], dim + 1, tol)

    return srtdidx


_np_syms = {'sin': np.sin, 'cos': np.cos,
            'exp': np.exp, 'tanh': np.tanh, 'pi': np.pi,
            'sqrt': np.sqrt}


def npeval(expr, subs={}):
    return eval(expr, _np_syms, subs)
