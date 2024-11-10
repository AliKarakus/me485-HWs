from utils.nb import dot
import numpy as np


def make_jacobi_update(ele):
    # Number of variables
    nvars = ele.nvars

    # Update next time step solution
    def _update(i_begin, i_end, uptsb, dub, subres):
        for idx in range(i_begin, i_end):
            for kdx in range(nvars):
                # Update solution by adding residual
                uptsb[kdx, idx] += dub[kdx, idx]

                # Initialize dub array
                dub[kdx, idx] = 0.0
            
            # Initialize subres array
            subres[idx] = 0.0

    return _update


def make_pre_jacobi(ele, nv, factor=1.0):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    def _pre_diag(i_begin, i_end, dt, diag, fjmat):
        # Compute diagonal matrix
        for idx in range(i_begin, i_end):
            diag[:, :, idx] = 0.0

            for jdx in range(nface):
                diag[:, :, idx] += fjmat[0, :, :, jdx, idx]*fnorm_vol[jdx, idx]
            
            for kdx in range(dnv):
                diag[kdx, kdx, idx] += 1/(dt[idx]*factor)
            
            diag[:, :, idx] = np.linalg.inv(diag[:, :, idx])

    return _pre_diag


def make_tpre_jacobi(ele, nv, dsrc, factor):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol
    
    def _pre_tdiag(i_begin, i_end, uptsb, dt, tdiag, tfjmat):
        for idx in range(i_begin, i_end):
            tdiag[:, :, idx] = 0.0
            u = uptsb[:, idx]

            for jdx in range(nface):
                tdiag[:, :, idx] += tfjmat[0, :, :, jdx, idx]*fnorm_vol[jdx, idx]
            
            # Source term Jacobian
            dsrc(u, tdiag[:, :, idx], idx)

            for kdx in range(dnv):
                tdiag[kdx, kdx, idx] += 1/(dt[idx]*factor)

            tdiag[:, :, idx] = np.linalg.inv(tdiag[:, :, idx])

    return _pre_tdiag


def make_jacobi_sweep(be, ele, nv, fdx=1, res_idx=0):
    # Make local array
    array = be.local_array()

    # Get element attributes
    nface = ele.nface
    dnv = nv[1] - nv[0]

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Normal vectors at faces
    fnorm_vol= ele.mag_fnorm * ele.rcp_vol

    def _jacobi_sweep(i_begin, i_end, rhsb, dub, rod, fjmat):
        # Compute R-(L+U)x
        for idx in range(i_begin, i_end):
            rhs = array(dnv)

            # Initialize rhs array with RHS
            for kdx in range(dnv):
                rhs[kdx] = rhsb[kdx+nv[0], idx]

            # Computes Jacobian matrix based on neighbor cells
            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if neib != idx:
                    neimat = fjmat[fdx, :, :, jdx, idx]

                    for kdx in range(dnv):
                        rhs[kdx] += dot(neimat[kdx, :], dub[:, neib], dnv, 0, nv[0]) \
                                    * fnorm_vol[jdx, idx]

            # Allocates to each rod array
            for kdx in range(dnv):
                rod[kdx+nv[0], idx] = rhs[kdx]
        
    def _jacobi_compute(i_begin, i_end, dub, rod, diag, subres=None, norm=None):
        # Compute Ax = b
        for idx in range(i_begin, i_end):
            rhs = array(dnv)

            # Reallocate rod element value to rhs array
            for kdx in range(dnv):
                rhs[kdx] = rod[kdx+nv[0], idx]
            
            # Inner-update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = dot(diag[kdx, :, idx], rhs, dnv)

            # Save error
            if subres is not None:
                norm[idx] = (dub[res_idx, idx] - subres[idx])**2

                # Save sub-residual of previous step
                subres[idx] = dub[res_idx, idx]

    return _jacobi_sweep, _jacobi_compute


