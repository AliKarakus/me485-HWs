import numpy as np
import scipy as sp
import argparse
from mesh import mesh
from .base import base
from .grad import grad
import inspect

class diff():

#Solves -\nabla k \nabla phi = Qc 


#-------------------------------------------------------------------------------------------------#
    def __init__(self, mesh):
       
        self.mesh = mesh
        self.grd = grad(mesh)
#-------------------------------------------------------------------------------------------------#
    def set(self, args):
        self.Nfields = args.Nfields
        self.method  = args.method

        parser = argparse.ArgumentParser(prog='Gradient',description='Compute Gradient using FVM',
                    epilog='--------------------------------')

        # # Set default values for gradient computation
        # parser.add_argument('--meshFile', default ='data/cavityQuad.msh', help='mesh file to be read')

        parser.add_argument('--method', type=str, default='WEIGHTED-LEAST-SQUARES')
        parser.add_argument('--Correct', type=bool, default=True)
        parser.add_argument('--Nfields', type=int, default=args.Nfields)
        parser.add_argument('--dim', type=int, default=args.dim)
        parser.add_argument('--IC', type = args.IC, default = args.IC)
        parser.add_argument('--BC', type = args.BC, default = args.BC)
        gargs = parser.parse_args()

        self.grd.set(gargs)
        self.bcFunc = args.BC
#-------------------------------------------------------------------------------------------------#
    def assemble(self, Qe, Qb, Qc, K):
        #no orthogonal correction
        if(self.method=='FALSE'):
            self.assembleOrthogonal(Qe, Qb, Qc, K)
        
        elif(self.method=='MINIMUM-CORRECTION'):
            self.assembleMinCorrect(Qe, Qb, Qc, K)
        
        elif(self.method=='ORTHOGONAL-CORRECTION'):
            self.assembleOrthoCorrect(Qe, Qb, Qc, K)
        
        elif(self.method=='OVER-RELAXED'):
            self.assembleOverCorrect(Qe, Qb, Qc, K)


 #-------------------------------------------------------------------------------------------------#
    def createBfield(self, Qe):
        BCField =  np.zeros((self.mesh.NBFaces, self.Nfields), float)
        for face, info in self.mesh.Face.items():
            bc    = info['boundary']
            coord = info['center']
            if(bc !=0):
                bcid = info['bcid']
                eM   = info['owner']
                BCField[bcid]   = self.bcFunc(bc, 0.0, coord, Qe[eM])
        return BCField

 #-------------------------------------------------------------------------------------------------#
    def extractBfield(self, Qe):
        BCField  = np.zeros((self.mesh.NBFaces, self.Nfields), float)
        for face, info in self.mesh.Face.items():
            bc    = info['boundary']
            coord = info['center']
            if(bc !=0):
                bcid = info['bcid']
                eM   = info['owner']
                # BCField[bcid]   = self.bcFunc(bc, 0.0, coord, Qe[eM])
                if(self.mesh.BCMap[bc]['dtype'] == 'NEUMANN'):
                    BCField[bcid] = Qe[eM]
                else:
                    BCField[bcid]  = self.bcFunc(bc, 0.0, coord, Qe[eM])
        return BCField
#-------------------------------------------------------------------------------------------------#
    def assembleOrthogonal(self, Qe, Qb, Qc, K):
        msh = self.mesh
        rows = np.zeros((5*msh.Nelements), int) -1
        cols = np.zeros((5*msh.Nelements), int) -1
        vals = np.zeros((5*msh.Nelements), float) -1
        rhs  = np.zeros((msh.Nelements,1), float)

        sk = 0
        for elm, info in msh.Element.items():
            eM = elm
            etype   = info['elementType']
            nfaces  = msh.elmInfo[etype]['nfaces'] 
            xM      = info['ecenter']
            vol     = info['volume']
            rhs[eM] = Qc[eM]*vol

            kM       = K[eM, :]

            valM = 0.0
            for f in range(nfaces):
                bc     = info['boundary'][f]
                weight = info['weight'][f]
                area   = info['area'][f]
                eP     = info['neighElement'][f]
                xP       = msh.Element[eP]['ecenter']
                
                # if boundary convert xP to face center
                if(bc !=0):
                    xP           = info['fcenter'][f]

                dxMP         = np.linalg.norm(xP-xM)

                kP  = K[eP, :] 
                kf  = weight*kM + (1-weight)*kP

                if(bc !=0):
                    qb           = Qb[info['bcid'][f]]
                    if(msh.BCMap[bc]['dtype'] == 'NEUMANN'):
                        rhs[eM]      = rhs[eM]  - (qb*area)
                    elif(msh.BCMap[bc]['dtype'] == 'DRICHLET'):
                        rhs[eM]      = rhs[eM]  - ( - kf*area/dxMP*qb)
                        valM         = valM     + kf*area/dxMP 
                else:
                    rows[sk] = eM
                    cols[sk] = eP
                    vals[sk] = -kf*area/dxMP
                    valM     = valM + kf*area/dxMP 
                    sk       = sk+1

            rows[sk] = eM 
            cols[sk] = eM 
            vals[sk] = valM 
            sk = sk+1

        rows = rows[0:sk]
        cols = cols[0:sk]
        vals = vals[0:sk]

        self.rhs = rhs
        self.A   = sp.sparse.coo_matrix((vals[:], (rows[:], cols[:])), shape=(msh.Nelements, msh.Nelements), dtype=float)


#-------------------------------------------------------------------------------------------------#
    def assembleMinCorrect(self, Qe, Qb, Qc, K):
        msh = self.mesh

        rows = np.zeros((5*msh.Nelements), int)   -1
        cols = np.zeros((5*msh.Nelements), int)   -1
        vals = np.zeros((5*msh.Nelements), float) -1
        rhs  = np.zeros((msh.Nelements,1), float)


        gQb    = self.grd.createBfield( Qe) 
        # Compute Gradient and interpolate to face
        gradQ  = self.grd.compute(Qe, gQb)
        gradQf = self.grd.interpolateToFace(Qe, gQb, gradQ)

        sk = 0
        for elm, info in msh.Element.items():
            eM = elm
            etype   = info['elementType']
            nfaces  = msh.elmInfo[etype]['nfaces'] 
            xM      = info['ecenter']
            vol     = info['volume']
            rhs[eM] = Qc[eM]*vol

            kM       = K[eM, :]

            valM = 0.0
            for f in range(nfaces):
                bc     = info['boundary'][f]
                weight = info['weight'][f]
                area   = info['area'][f]
                normal   = info['normal'][f]
                eP     = info['neighElement'][f]
                xP       = msh.Element[eP]['ecenter']
                
                # if boundary convert xP to face center
                if(bc !=0):
                    xP           = info['fcenter'][f]

                kP  = K[eP, :] 
                kf  = weight*kM + (1-weight)*kP
                

                dxMP         = np.linalg.norm(xP-xM)

                nMP   = (xP- xM)/np.linalg.norm(dxMP)
                areaO = np.dot(nMP, normal)*area

                tempT = area*normal - areaO*nMP
                areaT = np.linalg.norm(tempT)
                # check fully orthogonal grid
                if(areaT > 1e-8):
                    nT = tempT / np.linalg.norm(tempT)
                else:
                    nT = 0*tempT
                # read gradient at face from face storage 
                faceid = info['facemap'][f]
                gQ = gradQf[faceid, 0, :]

                # add non-orthogonal difussion
                rhs[eM] = rhs[eM] - (-kf*(gQ[0]*nT[0] + gQ[1]*nT[1])*areaT) 

                if(bc !=0):
                    qb           = Qb[info['bcid'][f]]
                    if(msh.BCMap[bc]['dtype'] == 'NEUMANN'):
                        rhs[eM]      = rhs[eM]  - (qb*areaO)
                    elif(msh.BCMap[bc]['dtype'] == 'DRICHLET'):
                        rhs[eM]      = rhs[eM]  - ( - kf*areaO/dxMP*qb)
                        valM         = valM   + kf*areaO/dxMP 
                else:
                    rows[sk] = eM
                    cols[sk] = eP
                    vals[sk] = -kf*areaO/dxMP
                    valM     = valM + kf*areaO/dxMP 
                    sk       = sk+1

            rows[sk] = eM 
            cols[sk] = eM 
            vals[sk] = valM 
            sk = sk+1

        rows = rows[0:sk]
        cols = cols[0:sk]
        vals = vals[0:sk]

        self.rhs = rhs
        self.A   = sp.sparse.coo_matrix((vals[:], (rows[:], cols[:])), \
            shape=(msh.Nelements, msh.Nelements), dtype=float)
#-------------------------------------------------------------------------------------------------#
    def solve(self, args):


        X = np.zeros((self.mesh.Nelements,1), float)
        if(args.linSolver=='DIRECT'):
            X[:,0] = sp.sparse.linalg.spsolve(self.A.tocsr(), self.rhs)
        elif(args.linSolver == 'CG'):
            iters = 0
            TOL = args.linTolerance

            res= []
            def report(xk):
                nonlocal iters
                frame = inspect.currentframe().f_back
                res.append(frame.f_locals['resid'])
                iters = iters+1

            if(args.linPrecond == 'JACOBI'):
                M = sp.sparse.spdiags(1 / (self.A.diagonal()), 0, self.A.shape[0], self.A.shape[0])
            elif(args.linPrecond=='ILU'):
                sA_iLU = sp.sparse.linalg.spilu(self.A)
                M = sp.sparse.linalg.LinearOperator(self.A.shape, sA_iLU.solve)
            else:
                M = None
           

            # X[:,0], info = sp.sparse.linalg.cg(self.A.tocsr(), self.rhs, tol = TOL, callback=report)
            X[:,0], info = sp.sparse.linalg.cg(self.A.tocsr(), self.rhs, M=M, tol = TOL, callback=report)

            print("PCG is converged in", iters, ' in the tolerance of ', TOL)

            # print(iters, res)

        return X

