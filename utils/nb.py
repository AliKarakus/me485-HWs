import numba as nb


@nb.jit(nopython=True, fastmath=True)
def dot(a, b, n, ofa=0, ofb=0):
    # Dotting a and b
    v = 0
    for i in range(n):
        v += a[ofa + i]*b[ofb + i]

    return v
