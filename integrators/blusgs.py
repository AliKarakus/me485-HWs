from utils.inverse import make_lu_dcmp, make_substitution
from utils.nb import dot


def make_blusgs_update(ele):
    # Number of variables
    nvars = ele.nvars

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


def make_pre_blusgs(be, ele, nv, factor=1.0):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    # LU decompose function
    dcmp_func = make_lu_dcmp(be, dnv)

    def _pre_blusgs(i_begin, i_end, dt, diag, fjmat):
        # Compute digonal matrix
        for idx in range(i_begin, i_end):
            diag[:, :, idx] = 0.0

            # Computes diagonal matrix based on neighbor cells
            for jdx in range(nface):
                diag[:, :, idx] += fjmat[0, :, :, jdx, idx]*fnorm_vol[jdx, idx]
            
            # Complete implicit operator
            for kdx in range(dnv):
                diag[kdx, kdx, idx] += 1/(dt[idx]*factor)
            
            # LU decomposition for inverse process
            dcmp_func(diag[:, :, idx])

    return _pre_blusgs


def make_tpre_blusgs(be, ele, nv, dsrc, factor):
    # Number of faces
    nface = ele.nface

    # Number of variables
    dnv = nv[1] - nv[0]

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    # LU decompose function
    dcmp_func = make_lu_dcmp(be, dnv)

    def _pre_tblusgs(i_begin, i_end, uptsb, dt, tdiag, tfjmat):
        # Compute digonal matrix
        for idx in range(i_begin, i_end):
            tdiag[:, :, idx] = 0.0
            u = uptsb[:, idx]

            # Computes diagonal matrix based on neighbor cells
            for jdx in range(nface):
                tdiag[:, :, idx] += tfjmat[0, :, :, jdx, idx]*fnorm_vol[jdx, idx]
            
            # Source term Jacobian
            dsrc(u, tdiag[:, :, idx], idx)
            
            # Complete implicit operator
            for kdx in range(dnv):
                tdiag[kdx, kdx, idx] += 1/(dt[idx]*factor)
            
            # LU decomposition for inverse process
            dcmp_func(tdiag[:, :, idx])

    return _pre_tblusgs


def make_serial_blusgs(be, ele, nv, mapping, unmapping, fdx=1, res_idx=0):
    # Make local array
    array = be.local_array()

    # Get element attributes
    nface = ele.nface
    dnv = nv[1] - nv[0]

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    # Matrix inverse - vector multiplication
    sub_func = make_substitution(be, dnv)

    def _lower_sweep(i_begin, i_end, rhsb, dub, diag, fjmat):
        # Lower (Forward) sweep
        for _idx in range(i_begin, i_end):
            idx = mapping[_idx]
            rhs = array(dnv)

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv[0], idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if unmapping[neib] != _idx:
                    neimat = fjmat[fdx, :, :, jdx, idx]

                    for kdx in range(dnv):
                        rhs[kdx] += dot(neimat[kdx, :], dub[:, neib], dnv, 0, nv[0]) \
                                    * fnorm_vol[jdx, idx]

            # Compute inverse of diagonal matrix multiplication
            sub_func(diag[:, :, idx], rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = rhs[kdx]

    def _upper_sweep(i_begin, i_end, rhsb, dub, diag, fjmat, subres=None, norm=None):
        # Upper (Backward) sweep
        for _idx in range(i_end-1, i_begin-1, -1):
            idx = mapping[_idx]
            rhs = array(dnv)

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv[0], idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if unmapping[neib] != _idx:
                    neimat = fjmat[fdx, :, :, jdx, idx]

                    for kdx in range(dnv):
                        rhs[kdx] += dot(neimat[kdx, :], dub[:, neib], dnv, 0, nv[0]) \
                                    * fnorm_vol[jdx, idx]

            # Compute inverse of diagonal matrix multiplication
            sub_func(diag[:, :, idx], rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = rhs[kdx]

            # Compute sub-residual and L2-norm of rho error
            if subres is not None:
                # Initialize
                if _idx == i_end-1:
                    norm[0] = 0.0

                norm[0] += (dub[res_idx, idx] - subres[idx])**2

                # Save sub-residual of previous step
                subres[idx] = dub[res_idx, idx]

    return _lower_sweep, _upper_sweep


def make_colored_blusgs(be, ele, nv, icolor, lcolor, fdx=1, res_idx=0):
    # Make local array
    array = be.local_array()

    # Get element attributes
    nface = ele.nface
    dnv = nv[1] - nv[0]

    # Get index array for neihboring cells
    nei_ele = ele.nei_ele

    # Normal vectors at faces
    fnorm_vol = ele.mag_fnorm * ele.rcp_vol

    # Matrix inverse - vector multiplication
    sub_func = make_substitution(be, dnv)

    def _lower_sweep(i_begin, i_end, rhsb, dub, diag, fjmat):
        for _idx in range(i_begin, i_end):
            # Lower sweep with coloring
            idx = icolor[_idx]
            curr_level = lcolor[idx]
            rhs = array(dnv)

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv[0], idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if lcolor[neib] != curr_level:
                    neimat = fjmat[fdx, :, :, jdx, idx]

                    for kdx in range(dnv):
                        rhs[kdx] += dot(neimat[kdx, :], dub[:, neib], dnv, 0, nv[0]) \
                                    * fnorm_vol[jdx, idx]

            # Compute inverse of diagonal matrix multiplication
            sub_func(diag[:, :, idx], rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = rhs[kdx]

    def _upper_sweep(i_begin, i_end, rhsb, dub, diag, fjmat, subres=None, norm=None):
        for _idx in range(i_begin, i_end):
            # Upper sweep with coloring (reverse level)
            idx = icolor[_idx]
            curr_level = lcolor[idx]
            rhs = array(dnv)

            # Initialize rhs array with RHS
            for k in range(dnv):
                rhs[k] = rhsb[k+nv[0], idx]

            for jdx in range(nface):
                neib = nei_ele[jdx, idx]

                if lcolor[neib] != curr_level:
                    neimat = fjmat[fdx, :, :, jdx, idx]

                    for kdx in range(dnv):
                        rhs[kdx] += dot(neimat[kdx, :], dub[:, neib], dnv, 0, nv[0]) \
                                    * fnorm_vol[jdx, idx]

            # Compute inverse of diagonal matrix multiplication
            sub_func(diag[:, :, idx], rhs)

            # Update dub array
            for kdx in range(dnv):
                dub[kdx+nv[0], idx] = rhs[kdx]

            # Compute sub-residual and L2-norm of rho error
            if subres is not None:
                norm[idx] = (dub[res_idx, idx] - subres[idx])**2

                # Save sub-residual of previous step
                subres[idx] = dub[res_idx, idx]

    return _lower_sweep, _upper_sweep
