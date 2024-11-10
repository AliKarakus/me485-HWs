import numpy as np


def make_diff_flux(nvars, dnv, fluxf, array):
    # Difference of flux vectors
    def _diff_flux(u, du, df, nf):
        f = array(dnv)
        for i in range(nvars):
            du[i] += u[i]

        fluxf(u, nf, f)
        fluxf(du, nf, df)

        for i in range(dnv):
            df[i] -= f[i]

    return _diff_flux


def make_lusgs_update(ele):
    # Number of variables
    nvars = ele.nvars

    def _update(i_begin, i_end, uptsb, rhsb):
        # Update solution by adding residual
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                uptsb[kdx, idx] += rhsb[kdx, idx]

    return _update


def make_lusgs_common(ele, factor=1.0):
    # Number of faces
    nface = ele.nface

    # Normal vectors for faces and displacement from cell center to neighbor cells
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    def _pre_lusgs(i_begin, i_end, dt, diag, lambdaf):
        # Construct Matrices for LU-SGS
        for idx in range(i_begin, i_end):
            # Diagonals of implicit operator
            diag[idx] = 1 / (dt[idx]*factor)

            for jdx in range(nface):               
                # Diffusive margin of wave speed at face
                lamf = lambdaf[jdx, idx]*1.01

                # Save spectral radius
                lambdaf[jdx, idx] = lamf

                # Add portion of lower and upper spectral radius
                diag[idx] += 0.5*lamf*fnorm_vol[jdx, idx]

    return _pre_lusgs


def make_serial_lusgs(be, ele, nv, mapping, unmapping, _flux):
    # dimensions for variable and face
    nvars, nface = ele.nvars, ele.nface
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Local array function
    array = be.local_array()

    # Pre-compile function to compute difference of flux vector
    _diff_flux = be.compile(make_diff_flux(nvars, dnv, _flux, array))

    def _lower_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        # Lower sweep via mapping
        for _idx in range(i_begin, i_end):
            idx = mapping[_idx]

            du = array(nvars)
            dfj = array(dnv)
            df = array(dnv)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                # Compute lower portion of off-diagonal
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if unmapping[neib] < _idx:
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0

                    for kdx in range(nv[0], nv[1]):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv[0], neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Siedel Update residual with lower portion
                dub[kdx+nv[0], idx] = (rhsb[kdx+nv[0], idx] -
                                       0.5*df[kdx])/(diag[idx] + dsrc[kdx+nv[0], idx])

    def _upper_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        for _idx in range(i_end-1, i_begin-1, -1):
            # Upper sweep via mapping (reverse order)
            idx = mapping[_idx]

            du = array(nvars)
            dfj = array(dnv)
            df = array(dnv)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if unmapping[neib] > _idx:
                    # Compute upper portion of off-diagonal
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0
                        
                    for kdx in range(nv[0], nv[1]):
                        du[kdx] = rhsb[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * rhsb[kdx+nv[0], neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Siedel Update residual with upper portion
                rhsb[kdx+nv[0], idx] = dub[kdx+nv[0], idx] - \
                    0.5*df[kdx]/(diag[idx] + dsrc[kdx+nv[0], idx])

    return _lower_sweep, _upper_sweep


def make_colored_lusgs(be, ele, nv, icolor, lcolor, _flux):
    # dimensions for variable and face
    nvars, nface = ele.nvars, ele.nface
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol, vec_fnorm = ele.mag_fnorm * ele.rcp_vol, ele.vec_fnorm

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Local array function
    array = be.local_array()

    # Pre-compile function to compute difference of flux vector
    _diff_flux = be.compile(make_diff_flux(nvars, dnv, _flux, array))

    def _lower_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        for _idx in range(i_begin, i_end):
            # Lower sweep with coloring
            idx = icolor[_idx]
            curr_level = lcolor[idx]

            du = array(nvars)
            dfj = array(dnv)
            df = array(dnv)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                # Compute lower portion of off-diagonal
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if lcolor[neib] < curr_level:
                #if neib < idx:
                    u = uptsb[:, neib]

                    for kdx in range(nvars):
                        du[kdx] = 0.0
                        
                    for kdx in range(nv[0], nv[1]):
                        du[kdx] = dub[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * dub[kdx+nv[0], neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Siedel Update residual with lower portion
                dub[kdx+nv[0], idx] = (rhsb[kdx+nv[0], idx] -
                                       0.5*df[kdx])/(diag[idx] + dsrc[kdx+nv[0], idx])

    def _upper_sweep(i_begin, i_end, uptsb, rhsb, dub, diag, dsrc, lambdaf):
        #for _idx in range(i_end-1, i_begin-1, -1):
        for _idx in range(i_begin, i_end):
            # Upper sweep via coloring (reverse level of coloring)
            idx = icolor[_idx]
            curr_level = lcolor[idx]

            du = array(nvars)
            dfj = array(dnv)
            df = array(dnv)

            for kdx in range(dnv):
                df[kdx] = 0.0

            for jdx in range(nface):
                # Compute upper portion of off-diagonal
                nf = vec_fnorm[jdx, :, idx]

                neib = nei_ele[jdx, idx]
                if lcolor[neib] > curr_level:
                    u = uptsb[:, neib]
                    
                    for kdx in range(nvars):
                        du[kdx] = 0.0
                        
                    for kdx in range(nv[0], nv[1]):
                        du[kdx] = rhsb[kdx, neib]

                    _diff_flux(u, du, dfj, nf)

                    for kdx in range(dnv):
                        df[kdx] += (dfj[kdx] - lambdaf[jdx, idx]
                                    * rhsb[kdx+nv[0], neib])*fnorm_vol[jdx, idx]

            for kdx in range(dnv):
                # Gauss-Siedel Update residual with upper portion
                rhsb[kdx+nv[0], idx] = dub[kdx+nv[0], idx] - \
                    0.5*df[kdx]/(diag[idx] + dsrc[kdx+nv[0], idx])

    return _lower_sweep, _upper_sweep
