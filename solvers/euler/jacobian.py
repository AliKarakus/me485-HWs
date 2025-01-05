from utils.nb import dot
import numpy as np


def make_convective_jacobian(be, cplargs, sign):
    
    gamma = cplargs['gamma']
    ndims = cplargs['ndims']
    to_prim = cplargs['to_prim']
    array = be.local_array()

    # Constants
    gam1 = gamma - 1.0
    gamp1 = gamma + 1.0
    gamc1 = gamma*gam1
    gampc = gamma/gamp1

    # convective Jacobian function
    _inviscid = make_inviscid_jacobian(be, cplargs)

    def vl_positive_jacobian(uf, nf, ap):
        """ Computes Van Leer FVS Jacobian matrix (Positive part)

        Parameters
        ----------
        uf : array
            Conservative variables
        nf : array
            Surface normal vector
        ap : array
            Positive flux Jacobian matrix
        """ 

        # Computes primitive variables
        # from conservative variables.
        rho = uf[0]
        v = array(ndims)
        p = to_prim(uf, v)
        e = uf[ndims+1]/rho

        cv = dot(v, nf, ndims)
        v2 = dot(v, v, ndims)
        cc = np.sqrt(gamma*p/rho)
        M = cv/cc

        # Computes Van Leer FVS Jacobian
        if M >= 1.0:
            # Computes convective flux Jacobian
            _inviscid(uf, nf, ap)
        elif M <= -1.0:
            # ap[:] = 0.0
            for i in range(ndims+2):
                for j in range(ndims+2):
                    ap[i][j] = 0.0
        else:
            # Computes Van Leer FVS Jacobian
            f2 = array(ndims)
            ind = array(ndims)
            rc = rho*cc
            gr = gamma*rho

            # Positive flux
            f1 = 0.25*rc*(M+1.0)**2
            for i in range(ndims):
                f2[i] = f1*(nf[i]*(-cv+2.0*cc)/gamma + v[i])
            f3 = f1*((gam1*cv*(2.0*cc-cv)+2.0*cc**2)/(gamma**2-1.0)+0.5*v2)

            c1 = gamc1/8.0/cc*(1.0-M**2)
            c2 = 0.5*(1.0+M)
            ap[0][0] = 0.25*(1.-M**2)*(cc+0.5*gamc1/cc*(v2-e))
            for i in range(ndims):
                ap[0][i+1] = -c1*v[i] + c2*nf[i]
            ap[0][ndims+1] = c1

            for i in range(ndims):
                for k in range(ndims):
                    ind[k] = 1.0 if k == i else 0.0
                c1 = f2[i]/f1
                c2 = gam1*nf[i]/rc
                ap[i+1][0] = c1*ap[0][0] + f1*(c2*(v2-e) + (nf[i]*cv-gamma*v[i])/gr)
                for k in range(ndims):
                    ap[i+1][k+1] = c1*ap[0][k+1] + f1*(-c2*v[k] - (nf[i]*nf[k])/gr + ind[k]/rho)
                ap[i+1][ndims+1] = c1*ap[0][ndims+1] + f1*c2
            
            c1 = f3/f1
            c2 = gampc/cc * (2.0*cc + gam1*cv)
            c3 = 2.0*(-cv+cc)/gamp1
            c4 = f1/rho
            ap[ndims+1][0] = c1*ap[0][0] + c4*(c2*(v2-e) - c3*cv - v2)
            for i in range(ndims):
                ap[ndims+1][i+1] = c1*ap[0][i+1] + c4*(-c2*v[i] + c3*nf[i] + v[i])
            ap[ndims+1][ndims+1] = c1*ap[0][ndims+1] + c4*c2

    def vl_negative_jacobian(uf, nf, am):
        """ Computes Van Leer FVS Jacobian matrix (Negative part)

        Parameters
        ----------
        uf : array
            Conservative variables
        nf : array
            Surface normal vector
        ap : array
            Positive flux Jacobian matrix
        """ 

        # Computes primitive variables
        # from conservative variables.
        v = array(ndims)
        rho = uf[0]
        p = to_prim(uf, v)
        e = uf[ndims+1]/rho

        cv = dot(v, nf, ndims)
        v2 = dot(v, v, ndims)
        cc = np.sqrt(gamma*p/rho)
        M = cv/cc

        # Computes negative flux Jacobian matrix
        if M >= 1.0:
            # am[:] = 0.0
            for i in range(ndims+2):
                for j in range(ndims+2):
                    am[i][j] = 0.0
        elif M <= -1.0:
            # Computes convective flux Jacobian
            _inviscid(uf, nf, am)
        else:
            # Computes Van Leer FVS Jacobian
            f2 = array(ndims)
            ind = array(ndims)
            rc = rho*cc
            gr = gamma*rho

            f1 = -0.25*rc*(M-1.)**2
            for i in range(ndims):
                f2[i] = f1*(nf[i]*(-cv-2.0*cc)/gamma + v[i])
            f3 = f1*((gam1*cv*(-cv-2.*cc)+2.*cc**2)/(gamma**2 - 1.)+0.5*v2)

            c1 = gamc1/8.0/cc*(1.0-M**2)
            c2 = 0.5*(1.0-M)
            am[0][0] = -0.25*(1.-M**2)*(cc+0.5*gamc1/cc*(v2-e))
            for i in range(ndims):
                am[0][i+1] = c1*v[i] + c2*nf[i]
            am[0][ndims+1] = -c1

            for i in range(ndims):
                for k in range(ndims):
                    ind[k] = 1.0 if k==i else 0.0
                c1 = f2[i]/f1
                c2 = gam1*nf[i]/rc
                am[i+1][0] = c1*am[0][0] - f1*(c2*(v2-e) - (nf[i]*cv-gamma*v[i])/gr)
                for k in range(ndims):
                    am[i+1][k+1] = c1*am[0][k+1] - f1*(-c2*v[k] + (nf[i]*nf[k])/gr - ind[k]/rho)
                am[i+1][ndims+1] = c1*am[0][ndims+1] - f1*c2
            
            c1 = f3/f1
            c2 = gampc/cc * (2.0*cc - gam1*cv)
            c3 = 2.0*(-cv-cc)/gamp1
            c4 = f1/rho
            am[ndims+1][0] = c1*am[0][0] + c4*(c2*(v2-e) - c3*cv-v2)
            for i in range(ndims):
                am[ndims+1][i+1] = c1*am[0][i+1] + c4*(-c2*v[i] + c3*nf[i] + v[i])
            am[ndims+1][ndims+1] = c1*am[0][ndims+1] + c4*c2

    if sign == 'positive':
        return be.compile(vl_positive_jacobian)
    elif sign == 'negative':
        return be.compile(vl_negative_jacobian)


def make_inviscid_jacobian(be, cplargs):

    gamma = cplargs['gamma']
    to_prim = cplargs['to_prim']
    ndims = cplargs['ndims']
    array = be.local_array()

    gam1 = gamma - 1.0

    def inviscid_jacobian(uf, nf, A):
        """ Computes convective flux Jacobian
        Ref : Katate Masatsuka. (2013). I do like CFD, vol.1. CRADLE.

        Parameters
        ----------
        uf : array
            Conservative variables
        nf : array
            Surface normal vector
        ap : array
            Positive flux Jacobian matrix
        """ 

        # Computes primitive variables
        # from convservative variables.
        rho = uf[0]
        v = array(ndims)
        p = to_prim(uf, v)
        c = np.sqrt(gamma*p/rho)
        q2 = dot(v, v, ndims)
        qn = dot(v, nf, ndims)
        H = c**2 / gam1 + 0.5 * q2

        # Computes each Jacobian matrix element.
        A[0][0] = 0.0
        A[0][1:ndims+1] = nf
        A[0][ndims+1] = 0.0

        for i in range(ndims):
            A[i+1][0] = 0.5*gam1*q2*nf[i] - v[i]*qn
            for k in range(ndims):
                A[i+1][k+1] = v[i]*nf[k] - gam1*v[k]*nf[i]
            A[i+1][i+1] += qn
            A[i+1][ndims+1] = gam1*nf[i]
        
        A[ndims+1][0] = qn*(0.5*gam1*q2 - H)
        for i in range(ndims):
            A[ndims+1][i+1] = H*nf[i] - gam1*v[i]*qn
        A[ndims+1][ndims+1] = gamma*qn

    return be.compile(inviscid_jacobian)

