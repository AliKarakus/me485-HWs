import numpy as np
import scipy as sp
import argparse
from mesh import mesh
from .base import base
from .grad import grad

import matplotlib.pyplot as plt


import inspect
import pyamg

class diff():

#Solves -\nabla k \nabla phi = Qc 


#-------------------------------------------------------------------------------------------------#
    def __init__(self, mesh):
       
        self.mesh = mesh
        self.grd = grad(mesh)
#-------------------------------------------------------------------------------------------------#
    def set(self, args):
        self.args = args;
        self.Nfields = args.Nfields
        self.method  = args.method

        parser = argparse.ArgumentParser(prog='Gradient',
                    description='Compute Gradient using FVM',
                    epilog='--------------------------------')

        parser.add_argument('--method', type=str, default='WEIGHTED-LEAST-SQUARES')
        parser.add_argument('--Correct', type=bool, default=False)
        parser.add_argument('--Nfields', type=int, default=args.Nfields)
        parser.add_argument('--dim', type=int, default=args.dim)
        parser.add_argument('--IC', type = args.IC, default = args.IC)
        parser.add_argument('--BC', type = args.BC, default = args.BC)
        gargs = parser.parse_args()

        self.grd.set(gargs)
        self.bcFunc = args.BC

        self.Ke = self.setFromFunction(args.DC)
        self.Qs = self.setFromFunction(args.DS)
#-------------------------------------------------------------------------------------------------#
    def rhsQ(self,Qe, Qb):
        msh = self.mesh
        rhsq  = np.zeros((msh.Nelements, self.Nfields), float)
        # # Create boundary field for gradient computation
        # gQb    = self.grd.createFaceBfield(Qe) 
        # # Compute Gradient and interpolate to face
        # gradQ  = self.grd.compute(Qe, gQb)
        for elm, info in msh.Element.items():
            eM = elm
            # get element type and number of faces
            etype   = info['elementType']
            nfaces  = msh.elmInfo[etype]['nfaces'] 
            # element center 
            xM      = info['ecenter']
            vol     = info['volume']


            rhsq[eM] = self.Qs[eM]*vol

            kM       = self.Ke[eM, :]

            qM = Qe[eM]; 

            valM = 0.0
            for f in range(nfaces):
                bc     = info['boundary'][f]
                weight = info['weight'][f]
                area   = info['area'][f]
                normal = info['normal'][f]
                eP     = info['neighElement'][f]
                xP     = msh.Element[eP]['ecenter']

                qP = Qe[eP]; 
                
                # if boundary convert xP to face center
                if(bc !=0):
                    xP = info['fcenter'][f]

                kP  = self.Ke[eP, :] 
                kf  = weight*kM + (1-weight)*kP

                dxMP = 0.0; nMP = 0.0; ne =0.0; nt = 0.0;

                dx   = xP-xM;  
                dxMP = np.linalg.norm(dx)      
                # eMP = dx/dxMP
                # angle = np.dot(normal, eMP)
                # if(self.method=='MINIMUM-CORRECTION'):                        
                #     # Note that ne and nt are not unite vectors for now
                #     ne  = angle*eMP
                #     nt  = normal - ne
                # elif(self.method=='ORTHOGONAL-CORRECTION'):
                #     ne  = eMP
                #     nt  = normal -ne
                # elif(self.method=='OVER-RELAXED-CORRECTION'):
                #     ne = (1.0/angle) *eMP
                #     nt = normal - ne
                # else:
                #     nt = 0.0*normal
                #     ne = normal

                # norm_ne = np.linalg.norm(ne)
                # norm_nt = np.linalg.norm(nt)

                # # make them unite vectors to match with lecture notes
                # if(norm_nt > 1e-10 and norm_nt < 1e-1):
                #     # activated +=1
                #     nt = nt/norm_nt
                # else:
                #     # notactivated +=1
                #     nt      = 0.0*nt
                #     norm_nt = 0.0

                # Tf = norm_nt*area
                # Ef = norm_ne*area
                # ne = ne/norm_ne
               
                Tf = 0.0; Ef = area; 

                # # read gradient at face from face storage 
                # faceid = info['facemap'][f]
                # gQ = weight*gradQ[eM, 0,:] + (1.0 - weight)*gradQ[eP, 0, :]
                # rhsq[eM] += (kf*(gQ[0]*nt[0] + gQ[1]*nt[1])*Tf) 

                if(bc !=0):
                    qb           = Qb[info['bcid'][f]]
                    if(msh.BCMap[bc]['dtype'] == 'NEUMANN'):
                        rhsq[eM]      += Ef*qb   # qb = k*gradq*n
                    elif(msh.BCMap[bc]['dtype'] == 'DRICHLET'):
                        rhsq[eM]      += kf*(qb-qM)*Ef/dxMP
                else:
                    rhsq[eM] += kf*Ef*(qP - qM)/ dxMP 


            rhsq[eM] = rhsq[eM]/vol
        return rhsq

#-------------------------------------------------------------------------------------------------#
    def assemble(self, Qe, Qb):
        #no orthogonal correction
        if(self.method=='NO-CORRECTION'):
            A, b = self.assembleOrthogonal(Qe, Qb)
        else:
            A, b = self.assembleCorrection(Qe, Qb)  
        return A, b  
 #-------------------------------------------------------------------------------------------------#
    def createFaceBfield(self, Qe):
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
    def createVertexBfield(self, Qe):
        BCField =  np.zeros((self.mesh.NBVertices, self.Nfields), float)
        for vrtx, info in self.mesh.Node.items():
            bc    = info['boundary']
            coord = info['coord']
            if(bc):
                bcid = info['bcid']
                BCField[bcid]   = self.bcFunc(bc, 0.0, coord, 0.0)
        return BCField
 
 #-------------------------------------------------------------------------------------------------#
    def setFromFunction(self, func):
        val = np.zeros((self.mesh.Nelements, self.Nfields), float)
        for elm, info in self.mesh.Element.items():
            xM      = info['ecenter']
            val[elm][:] = func(0.0, xM)

        return val

 #-------------------------------------------------------------------------------------------------#
    def extractBfield(self, Qe):
        BCField  = np.zeros((self.mesh.NBFaces, self.Nfields), float)
        for face, info in self.mesh.Face.items():
            bc    = info['boundary']
            coord = info['center']
            if(bc !=0):
                bcid = info['bcid']
                eM   = info['owner']
                # correct this later AK!
                if(self.mesh.BCMap[bc]['dtype'] == 'NEUMANN'):
                    BCField[bcid] = Qe[eM]
                else:
                    BCField[bcid]  = self.bcFunc(bc, 0.0, coord, Qe[eM])
        return BCField
#-------------------------------------------------------------------------------------------------#
    def assembleOrthogonal(self, Qe, Qb):
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
            
            rhs[eM] = self.Qs[eM]*vol

            kM      = self.Ke[eM, :]

            valM = 0.0
            for f in range(nfaces):
                bc     = info['boundary'][f]
                weight = info['weight'][f]
                area   = info['area'][f]
                eP     = info['neighElement'][f]
                xP     = msh.Element[eP]['ecenter']
                
                # if boundary convert xP to face center
                if(bc !=0):
                    xP = info['fcenter'][f]

                dxMP = np.linalg.norm(xP-xM)

                kP  = self.Ke[eP, :] 
                kf  = weight*kM + (1-weight)*kP

                if(bc !=0):
                    qb           = Qb[info['bcid'][f]]
                    if(msh.BCMap[bc]['dtype'] == 'NEUMANN'):
                        rhs[eM]      = rhs[eM]  - (area*qb)
                    elif(msh.BCMap[bc]['dtype'] == 'DRICHLET'):
                        rhs[eM]      = rhs[eM] + kf*area/dxMP*qb
                        valM         = valM    - kf*area/dxMP 
                else:
                    rows[sk] = eM
                    cols[sk] = eP
                    vals[sk] = kf*area/dxMP
                    valM     = valM - kf*area/dxMP 
                    sk       = sk+1

            rows[sk] = eM 
            cols[sk] = eM 
            vals[sk] = valM 
            sk = sk+1

        rows = rows[0:sk]
        cols = cols[0:sk]
        vals = vals[0:sk]

        A   = sp.sparse.coo_matrix((vals[:], (rows[:], cols[:])), shape=(msh.Nelements, msh.Nelements), dtype=float)
        b   = rhs
        return A, b; 
#-------------------------------------------------------------------------------------------------#
    def assembleCorrection(self, Qe, Qb):
        msh = self.mesh

        # Initialize the system matrices i.e. A x = b with -1 
        # A is in coordinate form A(i, j) = val
        # rows holds i, cols hold j and vals holds values
        rows = np.zeros((5*msh.Nelements), int)   -1
        cols = np.zeros((5*msh.Nelements), int)   -1
        vals = np.zeros((5*msh.Nelements), float) -1
        rhs  = np.zeros((msh.Nelements,1), float)

        # Create boundary field for gradient computation
        gQb    = self.grd.createFaceBfield(Qe) 
        # Compute Gradient and interpolate to face
        gradQ  = self.grd.compute(Qe, gQb)

        sk = 0
        activated = 0
        notactivated = 0
        for elm, info in msh.Element.items():
            eM = elm
            # get element type and number of faces
            etype   = info['elementType']
            nfaces  = msh.elmInfo[etype]['nfaces'] 
            
            # element center 
            xM      = info['ecenter']
            vol     = info['volume']
            rhs[eM] = self.Qs[eM]*vol

            kM       = self.Ke[eM, :]

            valM = 0.0
            for f in range(nfaces):
                bc     = info['boundary'][f]
                weight = info['weight'][f]
                area   = info['area'][f]
                normal = info['normal'][f]
                eP     = info['neighElement'][f]
                xP     = msh.Element[eP]['ecenter']
                
                # if boundary convert xP to face center
                if(bc !=0):
                    xP = info['fcenter'][f]

                kP  = self.Ke[eP, :] 
                kf  = weight*kM + (1-weight)*kP

                dxMP = 0.0; nMP = 0.0; ne =0.0; nt = 0.0;

                dx   = xP-xM;  dxMP = np.linalg.norm(dx)      
                eMP = dx/dxMP
                angle = np.dot(normal, eMP)
                if(self.method=='MINIMUM-CORRECTION'):                        
                    # Note that ne and nt are not unite vectors for now
                    ne  = angle*eMP
                    nt  = normal - ne
                elif(self.method=='ORTHOGONAL-CORRECTION'):
                    ne  = eMP
                    nt  = normal -ne
                elif(self.method=='OVER-RELAXED-CORRECTION'):
                    ne = (1.0/angle) *eMP
                    nt = normal - ne
                else:
                    nt = 0.0*normal
                    ne = normal

                norm_ne = np.linalg.norm(ne)
                norm_nt = np.linalg.norm(nt)

                # make them unite vectors to match with lecture notes
                if(norm_nt > 1e-10 and norm_nt < 1e-1):
                    activated +=1
                    nt = nt/norm_nt
                else:
                    notactivated +=1
                    nt      = 0.0*nt
                    norm_nt = 0.0

                Tf = norm_nt*area
                Ef = norm_ne*area
                ne = ne/norm_ne

                # read gradient at face from face storage 
                faceid = info['facemap'][f]
                # gQ = gradQf[faceid, 0, :]

                gQ = weight*gradQ[eM, 0,:] + (1.0 - weight)*gradQ[eP, 0, :]
                rhs[eM] -= (-kf*(gQ[0]*nt[0] + gQ[1]*nt[1])*Tf) 
                if(bc !=0):
                    qb           = Qb[info['bcid'][f]]
                    if(msh.BCMap[bc]['dtype'] == 'NEUMANN'):
                        rhs[eM]      = rhs[eM]  + (kf*qb*areaO)
                    elif(msh.BCMap[bc]['dtype'] == 'DRICHLET'):
                        rhs[eM]      = rhs[eM] + kf*Ef/dxMP*qb
                        valM         = valM   - kf*Ef/dxMP 
                else:
                 # add cross-difussion
                    rows[sk] = eM
                    cols[sk] = eP
                    vals[sk] = kf*Ef/dxMP
                    valM     = valM - kf*Ef/dxMP 
                    sk       = sk+1

            rows[sk] = eM 
            cols[sk] = eM 
            vals[sk] = valM 
            sk = sk+1

        print('Number of elements with correction: ', activated, 'and the others: ', notactivated)
        rows = rows[0:sk]
        cols = cols[0:sk]
        vals = vals[0:sk]

        b = rhs
        A   = sp.sparse.coo_matrix((vals[:], (rows[:], cols[:])), \
            shape=(msh.Nelements, msh.Nelements), dtype=float)

        return A,b

#-------------------------------------------------------------------------------------------------#
    def solve(self, args, A, b):
        # Define a callback function for solvers so that we can keep resid
        res= []; iters = 0; 
        def report(xk):
            nonlocal iters
            frame = inspect.currentframe().f_back
            res.append(frame.f_locals['resid'])
            iters = iters+1

        msh = self.mesh

        # Define Preconditioners for the solvers
        if(args.linPrecond == 'JACOBI'):
            A = sp.sparse.csr_matrix(A)
            M = sp.sparse.spdiags(1 / (A.diagonal()), 0, A.shape[0], A.shape[0])
        elif(args.linPrecond=='ILU'):
            A = sp.sparse.csc_matrix(A)
            sA_iLU = sp.sparse.linalg.spilu(A)
            M      = sp.sparse.linalg.LinearOperator((msh.Nelements, msh.Nelements), sA_iLU.solve)
        elif(args.linPrecond=='AMG'):
            A = sp.sparse.csr_matrix(A)
            Ml = pyamg.aggregation.smoothed_aggregation_solver(A)
            M =  Ml.aspreconditioner(cycle='V')        
        else:
            M = None

        TOL = args.linTolerance
        X = np.zeros((self.mesh.Nelements,1), float)
        
        # Solve the problems
        if(args.linSolver=='DIRECT'):
            X[:,0] = sp.sparse.linalg.spsolve(A, b)
        else:
            if(args.linSolver == 'CG'):   
                X[:,0], info = sp.sparse.linalg.cg(A, b, M=M, tol = TOL, callback=report)
            elif(args.linSolver == 'GMRES'):
                X[:,0], info = sp.sparse.linalg.gmres(A, b, M=M, tol = TOL, callback=report)

            print(args.linSolver, 'is converged in', iters,'iterations with tolerance of', TOL, 
                'using', args.linPrecond, 'preconditioner')

        
            # plt.semilogy(res, color='k', linestyle ='dashed', marker='o', markeredgecolor='r', 
            #     markerfacecolor='b', linewidth= 2); 
            # plt.xlabel('Iteration Number'); 
            # plt.ylabel('Residual') 
            # plt.title(args.linSolver + ' with '+ args.linPrecond + ' Preconditioner') 
            # plt.show()
        return X

