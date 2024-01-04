import numpy as np
import scipy as sp
from .base import base
import os
from mesh import mesh

class timeStepper():
#-------------------------------------------------------------------------------------------------#
    def __init__(self, _mesh, _solver):
       
        self.mesh = _mesh
        self.N    = _mesh.Nelements
        self.solver =_solver
#-------------------------------------------------------------------------------------------------#
    # def set(self, Nfields, method,  correct, bcFunc):
    def set(self, args):
        self.Nfields      = args.Nfields
        self.method       = args.timeMethod
        self.Noutput      = args.Noutput
        self.frame        = 0
        self.Nfields      = args.Nfields

        if(self.method == 'LSRK4'):
            self.setLSRK4(args)
        elif(self.method == 'FORWARD-EULER'):
            self.setForwardEuler(args)
        elif(self.method == 'ADAMS-BASHFORTH'):
            self.setAdamsBashforth(args)
        elif(self.method == 'BACKWARD-EULER'):
            self.setBackwardEuler(args)
        elif(self.method == 'BDF'):
            self.setBDF(args)
        else:
            print('the time method -- %s --  is not implemented' %self.method)
#-------------------------------------------------------------------------------------------------#
    # def set(self, Nfields, method,  correct, bcFunc):
    def run(self, Qe, Qb):
        if(self.method == 'LSRK4'):
            self.runLSRK4(Qe, Qb)
        elif(self.method == 'FORWARD-EULER'):
            self.runForwardEuler(Qe, Qb)
        elif(self.method == 'ADAMS-BASHFORTH'):
            self.runAdamsBashforth(Qe, Qb)
        elif(self.method == 'BACKWARD-EULER'):
            self.runBackwardEuler(Qe, Qb)
        elif(self.method == 'BDF'):
            self.runBDF(Qe, Qb)
        else:
            print('the time method -- %s --  is not implemented' %self.method)
#-------------------------------------------------------------------------------------------------#
    def setLSRK4(self,args):
        self.Nsteps = int(np.ceil((args.tend - args.tstart)/args.dt)); 
        self.dt = (args.tend - args.tstart)/self.Nsteps
        self.time = args.tstart

        self.Nstage = 5
        self.resq = np.zeros((self.N, self.Nfields), dtype=float)
        self.rhsq = np.zeros((self.N, self.Nfields), dtype=float)
        self.rk4a = np.zeros(self.Nstage, dtype=float)
        self.rk4b = np.zeros(self.Nstage, dtype=float)
        self.rk4c = np.zeros(self.Nstage, dtype=float)

        self.rk4a[0] = 0.0
        self.rk4a[1] = -567301805773.0/1357537059087.0
        self.rk4a[2] = -2404267990393.0/2016746695238.0
        self.rk4a[3] = -3550918686646.0/2091501179385.0
        self.rk4a[4] = -1275806237668.0/842570457699.0

        self.rk4b[0] = 1432997174477.0/9575080441755.0
        self.rk4b[1] = 5161836677717.0/13612068292357.0
        self.rk4b[2] = 1720146321549.0/2090206949498.0
        self.rk4b[3] = 3134564353537.0/4481467310338.0
        self.rk4b[4] = 2277821191437.0/14882151754819.0

        self.rk4c[0] = 0.0
        self.rk4c[1] = 1432997174477.0/9575080441755.0
        self.rk4c[2] = 2526269341429.0/6820363962896.0
        self.rk4c[3] = 2006345519317.0/3224310063776.0
        self.rk4c[4] = 2802321613138.0/2924317926251.0
#-------------------------------------------------------------------------------------------------#
    def setForwardEuler(self,args):
        self.Nsteps = int(np.ceil((args.tend - args.tstart)/args.dt)); 
        self.dt = (args.tend - args.tstart)/self.Nsteps
        self.time = args.tstart
        self.step = 0
#-------------------------------------------------------------------------------------------------#
    def setAdamsBashforth(self,args):
        self.max_order = args.timeOrder
        self.Nsteps = int(np.ceil((args.tend - args.tstart)/args.dt)); 
        self.dt = (args.tend - args.tstart)/self.Nsteps
        self.time = args.tstart
        #storage for the history i.e. t - dt
        self.qm1 = np.zeros((self.N, self.Nfields), dtype=float)
        self.qm2 = np.zeros((self.N, self.Nfields), dtype=float)
        self.coeff = {1: [1.0], 2:[1.5 , -0.5], 3:[23.0/12.0, -16.0/12.0, 5.0/12.0]}
        self.step = 0
#-------------------------------------------------------------------------------------------------#
    def setBackwardEuler(self,args):
        self.Nsteps = int(np.ceil((args.tend - args.tstart)/args.dt)); 
        self.dt = (args.tend - args.tstart)/self.Nsteps
        self.time = args.tstart
        self.step = 0
#-------------------------------------------------------------------------------------------------#
    def setBDF(self,args):
        self.max_order = args.timeOrder
        self.Nsteps = int(np.ceil((args.tend - args.tstart)/args.dt)); 
        self.dt = (args.tend - args.tstart)/self.Nsteps
        self.time = args.tstart
        #storage for the history i.e. t - dt
        self.qm1 = np.zeros((self.N, self.Nfields), dtype=float)
        self.qm2 = np.zeros((self.N, self.Nfields), dtype=float)
        self.coeff = {1: [1.0], 2:[4.0/3.0 , -1.0/3.0], 3:[18/11.0, -9.0/11.0, 2.0/11.0]}
        self.step = 0
#-------------------------------------------------------------------------------------------------#
    def runLSRK4(self, Qe, Qb):
        self.step = 0
        self.report(Qe)
        #for every time step        
        for step in range(self.Nsteps):
            # For every stage
            for stage in range(self.Nstage):
                # call integration function
                self.rhsq = self.solver.rhsQ(Qe, Qb)
                # update resudual
                self.resq = self.rk4a[stage]*self.resq + self.dt*self.rhsq
                Qe       += self.rk4b[stage]*self.resq

            #increase time
            self.time += self.dt
            self.step = step+1

            #dump out an output file
            if self.step % self.Noutput == 0:
                self.report(Qe)
#-------------------------------------------------------------------------------------------------#
    def runForwardEuler(self, Qe, Qb):
        for step in range(self.Nsteps):
            # Call integration function
            Qe  += self.dt*self.solver.rhsQ(Qe, Qb)
            self.time += self.dt
            self.step = step

            #produce an output file for postprocessing
            if step % self.Noutput == 0:
                self.report(Qe)
#-------------------------------------------------------------------------------------------------#
    def runAdamsBashforth(self, Qe, Qb):
        #-----------------------------------------------------#
        self.report(Qe)
        #-----------------------------------------------------#
        if(self.max_order == 3):
            self.qm2 = Qe; self.qm1 = Qe

        if(self.max_order==2):
            self.qm1 = Qe
        #-----------------------------------------------------#
        # Second order implementation
        order = 1;
        for step in range(self.Nsteps):            
            # Call integration function
            Qnew  =  self.coeff[order][0]*Qe + self.dt*self.solver.rhsQ(Qe, Qb)
            if(order>1): 
                Qnew += self.coeff[order][1]*self.qm1
            elif(order>2):
                Qnew += self.coeff[order][2]*self.qm2

            # second-order step if step>=1
            if(step==0 and self.max_order>1): order +=1;  
            # move to third order step if step>=2
            if(step==1 and self.max_order>2): order +=1;

            # Update history
            if(order>2): self.qm2 = self.qm1;
            if(order>1): self.qm1 = Qe;
            Qe = Qnew

            # Increase time
            self.time += self.dt
            self.step  = step

            #produce an output file for postprocessing
            if step % self.Noutput==0 and step != 0:
                self.report(Qe)
#-------------------------------------------------------------------------------------------------#
    def runBackwardEuler(self, Qe, Qb):
        #-----------------------------------------------------#
        self.report(Qe)
        #-----------------------------------------------------#
        # First form the sytem matrices for integration
        A, b  = self.solver.assemble(Qe, Qb)
        
        #-----------------------------------------------------#
        vals = np.zeros(self.mesh.Nelements, float);
        vol = vals.reshape((self.mesh.Nelements,1))
        for elm, info in self.mesh.Element.items():
            vol[elm] = info['volume']

        Ap = sp.sparse.spdiags(np.transpose(vol),0, m=self.mesh.Nelements, n=self.mesh.Nelements,
         format=A.getformat())
        #-----------------------------------------------------#
        # (qn1 - qn0)*V/ dt = L(q) = [A(qn1) + b]
        # [qn1*V - dt*A(qn1)] = qn0*V + dt*b
        # Ap (qn1) = rhs
        Ap = Ap - self.dt*A
        rhs = self.dt*b  + np.multiply(Qe, vol) 
        #-----------------------------------------------------#
        for step in range(self.Nsteps):
            # Call integration function
            Qe[:] = self.solver.solve(self.solver.args, Ap, rhs);
            rhs = self.dt*b  + np.multiply(Qe, vol) 
            self.time += self.dt
            self.step = step

            if step % self.Noutput == 0:
                self.report(Qe)

#-------------------------------------------------------------------------------------------------#
    def runBDF(self, Qe, Qb):
        #-----------------------------------------------------#
        self.report(Qe)
        #-----------------------------------------------------#
        # First form the sytem matrices for integration
        A, b  = self.solver.assemble(Qe, Qb)
        
        #-----------------------------------------------------#
        vals = np.zeros(self.mesh.Nelements, float);
        vol = vals.reshape((self.mesh.Nelements,1))
        for elm, info in self.mesh.Element.items():
            vol[elm] = info['volume']

        Ap = sp.sparse.spdiags(np.transpose(vol),0, m=self.mesh.Nelements, n=self.mesh.Nelements,
         format=A.getformat())
        #-----------------------------------------------------#
        # (qn1 - qn0)*V/ dt = L(q) = [A(qn1) + b]
        # [qn1*V - dt*A(qn1)] = qn0*V + dt*b
        # Ap (qn1) = rhs
        Ap = Ap - self.dt*A
        #-----------------------------------------------------#
        order = 1
        for step in range(self.Nsteps):
            if(order==1):
               rhs = self.dt*b  + self.coeff[order][0]*np.multiply(Qe, vol)
               self.qm1 = Qe
               if(self.max_order>1): order += 1; 
            elif(order==2):
                rhs = self.dt*b  + np.multiply(self.coeff[order][0]*Qe\
                                              + self.coeff[order][1]*self.qm1, vol) 
                self.qm2 = self.qm1
                self.qm1 = Qe
                if(self.max_order>2): order += 1;                 
            elif(order==3):
                rhs = self.dt*b  + np.multiply(self.coeff[order][0]*Qe\
                                              + self.coeff[order][1]*self.qm1\
                                              + self.coeff[order][2]*self.qm2, vol)
                self.qm2 = self.qm1
                self.qm1 = Qe 

            # Call integration function
            Qe[:] = self.solver.solve(self.solver.args, Ap, rhs);
            self.time += self.dt
            self.step  = step

            if step % self.Noutput == 0:
                self.report(Qe)

#-------------------------------------------------------------------------------------------------#
    def report(self, Pe):
      Pv = self.solver.mesh.cell2Node(Pe, 'average')
      fname = "output_{:04d}.vtu".format(self.frame)
      self.solver.mesh.plotVTU(fname, Pv)
      print('time: %.4e tstep: %d' %(self.time, self.step))
      self.frame += 1


