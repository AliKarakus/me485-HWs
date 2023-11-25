import numpy as np
import os
from mesh import mesh
from .base import base

class grad():
#-------------------------------------------------------------------------------------------------#
    def __init__(self, mesh):
        self.mesh = mesh

#-------------------------------------------------------------------------------------------------#
    # def set(self, Nfields, method,  correct, bcFunc):
    def set(self, args):
        self.Nfields = args.Nfields
        self.method  = args.method
        self.correct = args.Correct
        self.bcFunc  = args.BC

    def printInfo(self):
        print("----------------------- Reporting Options for Gradient ----------------------------"
            .center(os.get_terminal_size().columns))

        print('{0:<40} :'.format("Gradient Methods Implemented"))
        print('{0:<40}'.format("GREEN-GAUSS-CELL"))
        print('{0:<40}'.format("GREEN-GAUSS-NODE"))
        print('{0:<40}'.format("LEAST-SQUARES"))
        print('{0:<40}'.format("WEIGHTED-LEAST-SQUARES"))

        print('{0:<40}'.format("-------------------------------------------------"))
        print('{0:<40} :'.format("Implemented methods (functions) in the class:"))
        
        method_list = [methods for methods in dir(grad) if methods.startswith('__') is False]
        print(method_list)

        print('{0:<40}'.format("-------------------------------------------------"))
        print('{0:<40} :'.format("To set options:"))
        print('{0:<40} '.format("Usage : set(time, Nfields)"))
        print('{0:<40}'.format("-------------------------------------------------"))
        print('{0:<40} :'.format("To create boundary field:"))
        print('{0:<40} '.format("Usage : Qb  = createBfield(time, Nfields)"))
        print('{0:<40}'.format("-------------------------------------------------"))
        print('{0:<40} :'.format("To Compute Gradient:"))
        print('{0:<40} '.format("Usage: gradQe = compute(Qe, Qb)"))
        print('{0:<40}'.format("-------------------------------------------------"))
        print('{0:<40} :'.format("Interpolate Gradient To Face:"))
        print('{0:<40} '.format("gradQf = interpolateToFace(Qe, Qb, gradQe)"))
        print("------------------------------ DONE ------------------------------------"
            .center(os.get_terminal_size().columns))

#-------------------------------------------------------------------------------------------------#
    def compute(self, Qe, Qb):
        if(Qb.shape[1] != Qe.shape[1]):
            print(Qb.shape, Qe.shape)
            print('wrong dimension in boundary and field data')
            exit(-1)
        
        if(self.method == 'GREEN-GAUSS-CELL'):
            gradQ = self.greenGaussCell(Qe, Qb)
            if(self.correct):
                for i in range(2):
                    gradQ = self.correctGrad(Qe, Qb, gradQ)
        elif(self.method == 'GREEN-GAUSS-NODE'):
            gradQ = self.greenGaussNode(Qe, Qb)
        elif(self.method == 'LEAST-SQUARES'):
            gradQ = self.leastSquares(Qe, Qb)
        elif(self.method == 'WEIGHTED-LEAST-SQUARES'):
            gradQ = self.weightedLeastSquares(Qe, Qb)
        else:
            print('the gradient method -- %s --  is not implemented' %self.method)

        return gradQ

#-------------------------------------------------------------------------------------------------#
    def createFaceBfield(self, Qe):
        BCField =  np.zeros((self.mesh.NBFaces, self.Nfields), float)
        for face, info in self.mesh.Face.items():
            bc    = info['boundary']
            coord = info['center']
            if(bc !=0):
                bcid = info['bcid']
                eM   = info['owner']
                BCField[bcid]   = self.bcFunc(bc, 0.0, coord, Qe[eM,:])
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
    def extractBoundaryFromFace(self, Qf):
        msh = self.mesh
        Nfields  = Qf.shape[1]
        Ngrad    = Qf.shape[2]
        gQb      = np.zeros((msh.NBFaces, Nfields, Ngrad), float)

        for face, info in msh.Face.items():
            bc = info['boundary']
            if(bc != 0):
                bcid = info['bcid']
                bctype = self.mesh.BCMap[bc]['gtype']
                if(bctype == 'DRICHLET'):
                    gQb[bcid] = Qf[face]
                else:
                    gQb[bcid] = Qf[face]
        return gQb

#-------------------------------------------------------------------------------------------------#
    def greenGaussCell(self, Qe, Qb):
        msh = self.mesh
        Nfields    = Qe.shape[1]
        gradQ      = np.zeros((msh.Nelements, Nfields, msh.dim), float)
        self.QF    = np.zeros((msh.NFaces, Nfields), float)

        bcid = 0
        for fM, info in msh.Face.items():
            # Get element ids of owner and neighbor 
            eM = info['owner']; qM =  Qe[eM,:]; 
            eP = info['neigh']; qP =  Qe[eP,:]; 

            # Get boundary info, and geometric entries
            bc     = info['boundary']
            normal = info['normal']
            weight = info['weight']
            area   = info['area']
            
            qf = 0.0
            
            if(self.correct):
                weight = 0.5 

            #integrate boundary faces
            if(bc>0):
                qb = Qb[info['bcid']]
                qP = (qb - weight*qM)/(1.0-weight) 
                qf = weight*qM + (1.0 - weight)*qP
                gradQ[eM, :, 0] += qf[:]*area*normal[0]
                gradQ[eM, :, 1] += qf[:]*area*normal[1]
                if(msh.dim == 3):
                    gradQ[eM, :, 2] += qf[:]*area*normal[2]
                  
            #integrate internal faces
            else:
                qf = weight*qM + (1.0 - weight)*qP
                gradQ[eM,:, 0] += qf[:]*area*normal[0]
                gradQ[eM,:, 1] += qf[:]*area*normal[1]

                gradQ[eP,:, 0] -= qf[:]*area*normal[0]
                gradQ[eP,:, 1] -= qf[:]*area*normal[1]

                if(msh.dim == 3):
                    gradQ[eM, :, 2] +=qf[:]*area*normal[2]
                    gradQ[eP, :, 2] -=qf[:]*area*normal[2]
                
            self.QF[fM] = qf

        for elm, info in msh.Element.items():
            vol = info['volume']
            gradQ[elm,:,:] = gradQ[elm,:,:]/vol

        return gradQ
#-------------------------------------------------------------------------------------------------#
    def correctGrad(self, Qe, Qb, gQ):

        msh = self.mesh
        Nfields = Qe.shape[1]
        gradQ      = np.zeros((msh.Nelements, Nfields, msh.dim), float)

        for fM, info in msh.Face.items():
            eM = info['owner'];  eP = info['neigh']
            normal = info['normal']
            area   = info['area']
            bc     = info['boundary']
            
            # element and face center coordinates
            xM = msh.Element[eM]['ecenter']
            xP = msh.Element[eP]['ecenter']
            # Position vector from face center to midway of eM and eP
            xF = info['center']
            dx  =  xF - 0.5*(xP + xM)
            
            # Average gradient at the face
            gQA  = 0.5*(gQ[eM,:,:] + gQ[eP,:,:])
            # Average face value
            qFA  = 0.5*(Qe[eM,:] + Qe[eP,:])

            # Correct face value using previous gradient
            qf  = qFA + 0.5*( gQA[:, 0]* dx[0] + gQA[:, 1]* dx[1])
            if(msh.dim == 3):
                qf = qf + 0.5*gQA[:, 2]* dx[2]

            # Recompute the gradient using corrected face values
            if(bc>0 ):
                qf = Qb[info['bcid']]                
                gradQ[eM,:, 0] += qf[:]*area*normal[0]
                gradQ[eM,:, 1] += qf[:]*area*normal[1]
                if(msh.dim == 3):
                    gradQ[eM, :, 2] += qf[:]*area*normal[2]
            else:    
                gradQ[eM,:, 0] += qf[:]*area*normal[0]
                gradQ[eM,:, 1] += qf[:]*area*normal[1]
                
                gradQ[eP,:, 0] -= qf[:]*area*normal[0]
                gradQ[eP,:, 1] -= qf[:]*area*normal[1]

                if(msh.dim == 3):
                    gradQ[eM, :, 2] += qf[:]*area*normal[2]
                    gradQ[eP, :, 2] -= qf[:]*area*normal[2]

        # Devide by volume of element                    
        for elm, info in msh.Element.items():
            vol = info['volume']
            gradQ[elm,:,:] = gradQ[elm,:,:]/vol

        return gradQ
#-------------------------------------------------------------------------------------------------#
    def greenGaussNode(self, Qe, Qb):
        msh = self.mesh
        Nfields = Qe.shape[1]
        gradQ = np.zeros((msh.Nelements, Nfields, msh.dim), float)       
        Qbv   = self.createVertexBfield(Qe)

        Qv = msh.cell2Node(Qe, Qbv, 'average')

        for fM, info in msh.Face.items():
            eM      = info['owner']
            eP      = info['neigh']

            bc     = info['boundary']
            normal = info['normal']
            area   = info['area']

            verts   = info['nodes']
            
            qf      = np.zeros((Nfields),float)
            qf[:]   = np.sum(Qv[verts])/np.size(Qv[verts])
            #integrate boundary faces
            if(bc != 0):
                qf   =  Qb[info['bcid']]
                gradQ[eM, :, 0] += qf[:]*area*normal[0]
                gradQ[eM, :, 1] += qf[:]*area*normal[1]
                if(msh.dim == 3):
                    gradQ[eM,:,2] += qf[:]*area*normal[2]
         
            #integrate internal faces
            else:
                gradQ[eM,:, 0] += qf[:]*area*normal[0]
                gradQ[eM,:, 1] += qf[:]*area*normal[1]
                if(msh.dim == 3):
                    gradQ[eM,:, 2] += qf[:]*area*normal[2]
            
                gradQ[eP,:, 0] -= qf[:]*area*normal[0]
                gradQ[eP,:, 1] -= qf[:]*area*normal[1]
                if(msh.dim == 3):      
                    gradQ[eP,:, 2] -= qf[:]*area*normal[2]


        for eM in msh.Element.keys():
            vol = msh.Element[eM]['volume']
            gradQ[eM,:,:] = gradQ[eM,:,:]/vol
            
        return gradQ

#-------------------------------------------------------------------------------------------------#
    def leastSquares(self, Qe, Qb):

        msh = self.mesh
        Nfields = Qe.shape[1]
        gradQ = np.zeros((msh.Nelements, Nfields, msh.dim), float)

        for elm, info in msh.Element.items():
            # get element type, quad, tri, tet, etc.
            etype   = info['elementType']
            # read abstract info about the element type
            nfaces  = msh.elmInfo[etype]['nfaces'] 

            # element center and solution 
            xM = info['ecenter']; qM = Qe[elm]

            # Initialize system matirces and loop over neighbor elements
            A = np.zeros((nfaces,  msh.dim), float)
            b = np.zeros((Nfields, nfaces), float)
            for face in range(nfaces):
                bc = info['boundary'][face]
                eP = info['neighElement'][face]
                xP = msh.Element[eP]['ecenter']
                qP = Qe[eP]

                # if this is a bc face, use face value and coordinates
                if(bc != 0):
                    qP = Qb[info['bcid'][face]]
                    xP = info['fcenter'][face]
                
                # form the matrix and rhs vector b
                A[face,:]   = xP[0:msh.dim] - xM[0:msh.dim]
                b[:, face]  = qP - qM

            #solve the system for all faces
            for field in range(Nfields):
                gradQ[elm, field, :] = np.linalg.pinv(A)@np.transpose(b[field, :])
                
        return gradQ


#-------------------------------------------------------------------------------------------------#
    def weightedLeastSquares(self, Qe, Qb):

        msh = self.mesh
        Nfields = Qe.shape[1]
        gradQ = np.zeros((msh.Nelements, Nfields, msh.dim), float)

        for elm, info in msh.Element.items():
            etype   = info['elementType']
            nfaces  = msh.elmInfo[etype]['nfaces'] 

            xM = info['ecenter']
            qM = Qe[elm]

            A = np.zeros((nfaces,  msh.dim), float)
            b = np.zeros((Nfields, nfaces), float)
            for face in range(nfaces):
                bc = info['boundary'][face]
                eP = info['neighElement'][face]
                xP = msh.Element[eP]['ecenter']
                qP = Qe[eP]

                if(bc != 0):
                    qP = Qb[info['bcid'][face]] 
                    xP = info['fcenter'][face]
                    # print(qP)

                wf = 1.0/ (np.linalg.norm(xP-xM))**2

                A[face,:]   = wf*(xP[0:msh.dim] - xM[0:msh.dim])
                b[:, face]  = wf*(qP - qM)

            for field in range(Nfields):
                gradQ[elm, field, :] = np.linalg.pinv(A)@np.transpose(b[field, :])
                
        return gradQ

#-------------------------------------------------------------------------------------------------#
    def interpolateToFace(self, Qe, Qb, gQ):
        msh = self.mesh
        Nfields = Qe.shape[1]
        Ngrad   = gQ.shape[2]
        # gQf = msh.createFfield(Nfields, Ngrad)
        gQf = np.zeros((self.mesh.NFaces, Nfields, Ngrad), float)

        for fM, info in msh.Face.items():
            # get owner element id
            eM = info['owner'];qM =  Qe[eM];
            # get neighbor element id
            eP = info['neigh'];qP =  Qe[eP]

            # read geometric info
            bc     = info['boundary']
            normal = info['normal']
            weight = info['weight']

            # Obtain weighted average of gradient at the face
            gQM =  gQ[eM]; gQP =  gQ[eP]
            gQA = weight*gQM + (1.0-weight)*gQP

            # center coordinates of elements
            xp = msh.Element[eP]['ecenter']
            xm = msh.Element[eM]['ecenter']
            
            # if the face sits on the boundary replace neigh info
            if(bc != 0):
                xp = info['center']
                qP = Qb[info['bcid']]
                if(msh.BCMap[bc]['gtype'] == 'NEUMANN'):
                    qP = Qe[eM,:]

            # Position vector from Owner to Neigh
            dMP = np.linalg.norm(xp - xm) # distance
            nMP = (xp - xm)/dMP           # unit vector

            # for all fields correct average gradient
            for f in range(Nfields):
                # Normal gradent
                normalGradQ = gQA[f][0]*nMP[0] + gQA[f][1]*nMP[1]
                if(msh.dim==3):
                    normalGradQ = normalGradQ + gQA[f][2]*nMP[2]

                # Average gradient from M to P
                avgGradQ = (qP[f] - qM[f])/dMP 

                # Correct average gradient
                gQf[fM][f][0] = gQA[f][0] + (-normalGradQ + avgGradQ )*nMP[0]
                gQf[fM][f][1] = gQA[f][1] + (-normalGradQ + avgGradQ )*nMP[1]
                if(msh.dim==3):
                    gQf[fM][f][2] = gQA[f][2] - (-normalGradQ + avgGradQ )*nMP[2]
        return gQf
