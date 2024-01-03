import numpy as np
import scipy as sp
import argparse
from mesh import mesh
from .base import base
from .grad import grad
from .timeStepper import timeStepper

import matplotlib.pyplot as plt

class euler():
#-------------------------------------------------------------------------------------------------#
    def __init__(self, _mesh):
        self.mesh = _mesh
        self.grd = grad(self.mesh)
#-------------------------------------------------------------------------------------------------#
    def set(self, args):
      self.args     = args;
      self.dim      = args.dim
      if self.dim==2:
        self.Nfields = 4
      else:
        self.Nfields =5


      self.limiter  = args.limiter
      self.flux     = args.flux
      #------------------------------------------------------------#
      # set gradient methods using weighted least-squares
      parser = argparse.ArgumentParser(prog='Gradient',
                  description='Compute Gradient using FVM',
                  epilog='--------------------------------')
      parser.add_argument('--method', type=str, default='WEIGHTED-LEAST-SQUARES')
      parser.add_argument('--Correct', type=bool, default=False)
      parser.add_argument('--Nfields', type=int, default=self.Nfields)
      parser.add_argument('--dim', type=int, default=self.dim)
      parser.add_argument('--BC', type = args.BC, default = args.BC)
      gargs = parser.parse_args()
      self.grd.set(gargs)
      #------------------------------------------------------------#
      # set time stepper method as LSERK
      parser = argparse.ArgumentParser(prog='Time Stepper', epilog='----------')

      parser.add_argument('--timeMethod', type=str, default='LSRK4')
      parser.add_argument('--Nfields', type=int, default=self.Nfields)
      parser.add_argument('--Noutput', type=int, default=self.args.Noutput)
      parser.add_argument('--tstart', type=float, default=self.args.tstart)
      parser.add_argument('--tend', type=float, default=self.args.tend)
      parser.add_argument('--dt', type=float, default=self.args.dt)
      parser.add_argument('--dim', type=int, default=self.dim)
      parser.add_argument('--BC', type = args.BC, default = args.BC)
      targs = parser.parse_args()

      self.timeStepper = timeStepper(self.mesh, self)
      self.timeStepper.set(targs)

      self.bcFunc = args.BC

      self.gamma = 1.4

      self.qp  = np.zeros((self.mesh.Nelements, self.Nfields), float);
      self.qc  = np.zeros((self.mesh.Nelements, self.Nfields), float);
      self.dqc = np.zeros((self.mesh.Nelements, self.Nfields, self.dim), float);
      self.dqp = np.zeros((self.mesh.Nelements, self.Nfields, self.dim), float);
#-------------------------------------------------------------------------------------------------#
    def rhsQ(self, Qc, Qb):
      msh = self.mesh
      
      # storage for the rhs computations
      rhsq  = np.zeros((msh.Nelements, self.Nfields), float)
      
      # create boundary conditions at faces
      self.qb = self.grd.createFaceBfield(self.qc)
      
      # compute gradient of conservative variables
      self.dqc  = self.grd.compute(self.qc, self.qb)
      
      # compute limited face values
      Qf = self.reconstructFaceValues(self.qc, self.dqc, self.qb)

      for eM, info in msh.Element.items():
        etype   = info['elementType']
        nfaces  = msh.elmInfo[etype]['nfaces']
        vol     = info['volume']
        #go over every face of elements add contributions
        for fM in range(nfaces):
          bc     = info['boundary'][fM]
          eP     = info['neighElement'][fM]
          fP     = info['neighFace'][fM]
          area   = info['area'][fM]
          xf     = info['fcenter'][fM]
          normal = info['normal'][fM]

          qfM = Qf[eM, :, fM]; 
          qfP = Qf[eP, :, fP];

          if( bc >0 ):
            qfP = self.bcFunc(bc, 0.0, xf, normal, qfM)
            # set field value to boundary state to enforce flux at boundary
            qfM = qfP

          # get flux function
          flux  = self.getFlux(normal, qfM, qfP)

          rhsq[eM, :]  -=  area*flux/vol

      return rhsq

#-------------------------------------------------------------------------------------------------#
    def getFlux(self, normal, qcM, qcP):

      fxM = np.zeros(self.Nfields, float)
      fxP = np.zeros(self.Nfields, float)

      fyM = np.zeros(self.Nfields, float)
      fyP = np.zeros(self.Nfields, float)

      # compute flux terms and primitive variables
      if self.flux=="LLF" :
        rM  = qcM[0]; rP  = qcP[0]
        ruM = qcM[1]; ruP = qcP[1]
        rvM = qcM[2]; rvP = qcP[2]
        reM = qcM[3]; reP = qcP[3]
        # get primitive varibales
        uM = ruM/rM; vM = rvM/rM
        pM = (self.gamma-1)*(reM - 0.5*(ruM*uM + rvM*vM));
        # 
        uP = ruP/rP; vP = rvP/rP
        pP = (self.gamma-1)*(reP - 0.5*(ruP*uP + rvP*vP));

        fxM[0] = ruM; fxM[1] = ruM*uM + pM; fxM[2] = rvM*uM;      fxM[3] = uM*(reM + pM); 
        fyM[0] = rvM; fyM[1] = ruM*vM;      fyM[2] = rvM*vM + pM; fyM[3] = vM*(reM + pM); 

        fxP[0] = ruP; fxP[1] = ruP*uP + pP; fxP[2] = rvP*uP;      fxP[3] = uP*(reP + pP); 
        fyP[0] = rvP; fyP[1] = ruP*vP;      fyP[2] = rvP*vP + pP; fyP[3] = vP*(reP + pP); 

        # approximate maximum wave speed at the face
        velM = np.sqrt(uM**2 + vM**2) + np.sqrt(abs(self.gamma*pM/rM))
        velP = np.sqrt(uP**2 + vP**2) + np.sqrt(abs(self.gamma*pP/rP))

        maxvel = np.maximum(velM, velP)

        flux = 0.5*(normal[0]*(fxP +fxM) + normal[1]*(fyP + fyM) + maxvel*(qcM - qcP))

      # elif self.flux == "HLLC" :


      # else:



      return flux




#-------------------------------------------------------------------------------------------------#
    def minmaxBJ(self, dqf, dqmin, dqmax):
      lfunc = np.zeros(self.Nfields, float)
      for fld in range(self.Nfields):
        if( dqf[fld] > 0):
          lfunc[fld] = min(1,dqmax[fld]/dqf[fld])
        elif(dqf[fld] < 0):
          lfunc[fld] = min(1,dqmin[fld]/dqf[fld])
        else:
          lfunc[fld] = 1.0

      return lfunc

#-------------------------------------------------------------------------------------------------#
    def reconstructFaceValues(self, qc, dqc, qb):
      # allocate memory for the face values
      msh = self.mesh
      qmax = np.zeros((msh.Nelements,self.Nfields, 2),float) 
      Qf   = np.zeros((msh.Nelements, self.Nfields, 5), float) - 1.0
      if(self.limiter=="BARTH-JESPERSEN"):
        # find the maximum and minumum of the stencils
        for elm, info in msh.Element.items():
          qemax = self.qc[elm,:] 
          qemin = self.qc[elm,:] 
          etype   = info['elementType']
          nfaces  = msh.elmInfo[etype]['nfaces']
          for face in range(nfaces):
            bc     = info['boundary'][face]
            ep     = info['neighElement'][face]
            qp     = qc[ep, :]
            if(bc !=0):
              qp           = qb[info['bcid'][face], :]
            # element maximum and minum on the stencils
            qemax = np.maximum(qemax, qp)
            qemin = np.minimum(qemin, qp)


          # after finding stencil maximum, now limit face values
          dqmax = qemax -qc[elm,:]
          dqmin = qemin -qc[elm,:]

          lfunc = np.ones(self.Nfields,float) 

          # find the minimum of the limiter functions
          for face in range(nfaces):
            bc     = info['boundary'][face]
            eP     = info['neighElement'][face]
            xP     = msh.Element[eP]['ecenter']
            xM     = info['ecenter']
            dxMP   = xP - xM

            dqe = dqc[elm, :,:]

            dqf = dqe[:,0]*dxMP[0] + dqe[:,1]*dxMP[1]
            if(self.dim==3):
              dqf += dqe[:,2]*dxMP[2]

            lfunc = np.minimum(lfunc, self.minmaxBJ(dqf, dqmin, dqmax))

          # lfuncmin = np.min(lfunc)
          lfuncmin = lfunc
          # now we can recontruct face values
          for face in range(nfaces):
            bc     = info['boundary'][face]
            xf     = info['fcenter'][face]
            xe     = info['ecenter']
            dxf    = xf - xe

            qe     = qc[elm, :]
            dqe    = dqc[elm, :, :]

            if(bc !=0):
              qf   = qb[info['bcid'][face], :]
            else:
              qf   =  qe + lfuncmin*(dqe[:, 0]*dxf[0] +dqe[:, 1]*dxf[1] )
              if(self.dim==3):
                qf += lfuncmin*(dqe[:, 2]*dxf[2])

            Qf[elm, :, face] = qf[:]
      
      elif(self.limiter=="VENKATAKRISHNAN"):
        print("not implemented yet\n")

        
      # no limiting
      else:
         # find the maximum and minumum of the stencils
        for elm, info in msh.Element.items():
          qemax = self.qc[elm,:] 
          qemin = self.qc[elm,:] 
          etype   = info['elementType']
          nfaces  = msh.elmInfo[etype]['nfaces']
          # now we can recontruct face values
          for face in range(nfaces):
            bc     = info['boundary'][face]
            xf     = info['fcenter'][face]
            xe     = info['ecenter']
            dxf    = xf - xe
            # face values and 
            qe     = qc[elm, :]
            dqe    = dqc[elm, :, :]

            if(bc !=0):
              qf   = qb[info['bcid'][face], :]
            else:
              qf   =  qe + (dqe[:, 0]*dxf[0] +dqe[:, 1]*dxf[1] )
            if(self.dim==3):
              qf += (dqe[:, 2]*dxf[2])

          Qf[elm, :, face] = qf[:]

      

      return Qf

#-------------------------------------------------------------------------------------------------#
    def primitiveToConservative(self, qp, qc):
      qc[:,0]  = qp[:,0]
      qc[:,1]  = qp[:,1]*qp[:,0]
      qc[:,2]  = qp[:,2]*qp[:,0]
      if(self.dim==3):
        qc[:,3]  = qp[:,3]*qp[:,0]
        qc[:,4]  = qp[:,4]/(self.gamma -1.0) + 0.5*qp[:,0]*(qp[:,1]*qp[:,1] +qp[:,2]*qp[:,2] + qp[:,3]*qp[:,3])
      else:
        qc[:,3]  = qp[:,3]/(self.gamma -1.0) + 0.5*qp[:,0]*(qp[:,1]*qp[:,1] +qp[:,2]*qp[:,2])
#-------------------------------------------------------------------------------------------------#
    def conservativeToPrimitive(self, qc, qp):
      qp[:,0] = qc[:,0]
      qp[:,1]  = qc[:,1]/qp[:,0]
      qp[:,2]  = qc[:,2]/qp[:,0]
      if(self.dim==3):
        qp[:,3]  = qc[:,3]/qc[:,0]
        qp[:,4]  = (self.gamma -1.0)*(qc[:,4] - 0.5*qp[:,0]*(qp[:,1]*qp[:,1] +qp[:,2]*qp[:,2] + qp[:,3]*qp[:,3]))
      else:
        qp[:,3]  = (self.gamma -1.0)*(qc[:,3] - 0.5*qp[:,0]*(qp[:,1]*qp[:,1] +qp[:,2]*qp[:,2] ))







