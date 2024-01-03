import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------------#
class grid1d(object):

    def __init__(self, nx, ng, xmin=0.0, xmax=1.0):

        self.ng = ng
        self.nx = nx

        self.xmin = xmin
        self.xmax = xmax

        # Make easy intergers to know where the real data lives
        # excluding ghost cellls
        self.ids = ng
        self.ide = ng+nx-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.x  = xmin + (np.arange(nx+2*ng)-ng+0.5)*self.dx
        self.xe = xmin + (np.arange(nx+2*ng)-ng)*self.dx

        self.xf = np.zeros((self.xe.shape[0], 2), dtype=np.float64)
        self.xf[:,0] = self.xe[:]
        self.xf[:,1] = self.xe[:] + self.dx

        # storage for the solution
        self.q = np.zeros((nx+2*ng), dtype=np.float64)


    def scratch_array(self,ndim=1):
        # return a scratch array dimensioned for our grid
        return np.zeros((self.nx+2*self.ng), dtype=np.float64)


    def applyBCs(self):
        # fill all single ghostcell with periodic boundary conditions
        for n in range(self.ng):
            # left boundary
            self.q[self.ids-1-n] = self.q[self.ide-n]

            # right boundary
            self.q[self.ide+1+n] = self.q[self.ids+n]

    def norm(self, e):
        # return the norm of quantity e which lives on the grid
        if len(e) != 2*self.ng + self.nx:
            return None

        #return np.sqrt(self.dx*np.sum(e[self.ids:self.ide+1]**2))
        return np.max(abs(e[self.ids:self.ide+1]))


# ------------------------------------------------------------------------------------#
class solver(object):

    def __init__(self, grid, u, CFL=0.8, method="upwind"):
        self.grid    = grid
        self.t       = 0.0 # solver time
        self.u       = u   # the constant advective velocity
        self.CFL     = CFL   # CFL number
        self.method  = method


    def init_cond(self, type="tophat"):
        """ initialize the data """
        if type == "tophat":
            self.grid.q[:] = 0.0
            self.grid.q[np.logical_and(self.grid.x >= 0.333,
                                       self.grid.x <= 0.666)] = 1.0

        elif type == "sine":
            self.grid.q[:] = np.sin(2.0*np.pi*self.grid.x/(self.grid.xmax-self.grid.xmin))

        elif type == "gaussian":
            ql = 1.0 + np.exp(-60.0*(self.grid.xf[:,0] - 0.5)**2)
            qr = 1.0 + np.exp(-60.0*(self.grid.xf[:,1] - 0.5)**2)
            qc = 1.0 + np.exp(-60.0*(self.grid.x - 0.5)**2)
            
            self.grid.q[:] = (1./6.)*(ql + 4*qc + qr)

    def timestep(self):
        # return the advective timestep
        return self.CFL*self.grid.dx/self.u


    def period(self):
        # return the period for advection with velocity u
        return (self.grid.xmax - self.grid.xmin)/self.u


    # Fill this function to complete the assignment
    def states(self, dt):
        """ compute the left and right interface states """

        # compute the piecewise linear slopes
        grd   = self.grid
       	psi    = np.zeros(((self.grid.nx+2*self.grid.ng),2), dtype=np.float64)

       	# machine zero to prevent division by zero
       	eps = 1e-10; 
        if self.method == "upwind":
            # piecewise constant = 0 slopes
            # first face
            psi[:,0] = 0.0
            #second face
            psi[:,1] = 0.0
        elif self.method == "minmod":
            # minmod limited slope
            for i in range(grd.ids-1, grd.ide+2):
              # fill this part
              # 
              # 
        elif self.method == "osher":
            # minmod limited slope
            for i in range(grd.ids-1, grd.ide+2):
              # fill this part
              # 
              # 
        elif self.method == "muscl":
            # minmod limited slope
            for i in range(grd.ids-1, grd.ide+2):
              # fill this part
              # 
              # 
        elif self.method == "superbee":
            # minmod limited slope
            for i in range(grd.ids-1, grd.ide+2):
              # fill this part
              # 
              # 
        elif self.method == "vanleer":
            # minmod limited slope
            for i in range(grd.ids-1, grd.ide+2):
              # fill this part
              # 
              # 

        # find face values for Riemann solver
        qf  =  np.zeros(((self.grid.nx+2*self.grid.ng),2), dtype=np.float64)

        for i in range(grd.ids, grd.ide+2):
	        # assumes advection velocity is always in +x direction
	        qf[i,0] = grd.q[i-1] + 0.5*psi[i,0]*(grd.q[i] - grd.q[i-1]) 
	        qf[i,1] = grd.q[i]   + 0.5*psi[i,1]*(grd.q[i+1] - grd.q[i]) 
        return qf


    def riemann(self, qf):
        # Riemann problem for advection -- this is simply upwinding, but we return the flux

        if self.u > 0.0:
            return self.u*qf
        else:
            print("error: advection velocity should be in +x direction")
            return -1


    def update(self, dt, flux):
    		# update the solver
        grd = self.grid

        qnew = grd.scratch_array()

        # update solution in time
        qnew[grd.ids:grd.ide+1] = grd.q[grd.ids:grd.ide+1] - \
                                  dt/grd.dx * (flux[grd.ids:grd.ide+1,1] - flux[grd.ids:grd.ide+1,0])

        return qnew

    # evolve the linear advection equation """
    def evolve(self, num_periods=1):
        self.t = 0.0
        grd = self.grid

        tmax = num_periods*self.period()
        # main evolution loop
        while self.t < tmax:
            # fill the boundary conditions
            # periodic boundaries for thus problem
            grd.applyBCs()

            # get the timestep
            dt = self.timestep()
            if self.t + dt > tmax:
                dt = tmax - self.t
            
            # get the interface states
            qf = self.states(dt)
            
            # solve the Riemann problem at all interfaces
            # for this problem iti is just upwind flux
            flux = self.riemann(qf)
            
            # do the conservative update
            qnew = self.update(dt, flux)
            grd.q[:] = qnew[:]
            self.t += dt

# ------------------------------------------------------------------------------------#
# compare flux limiting
if __name__ == "__main__":

    xmin = 0.0; xmax = 1.0; nx = 40; ng = 2
    # create 1D grid with nx element and 2*ng ghost elements 
    grd = grid1d(nx, ng, xmin=xmin, xmax=xmax)
    # constant advection field
    u = 1.0

    # advc.init_cond("sine")
    # advc.init_cond("gaussian")

    advc = solver(grd, u, CFL=0.5, method="upwind")
    advc.init_cond("tophat")
    qinit = advc.grid.q.copy()

    # plot exact solution
    plt.plot(grd.x[grd.ids:grd.ide+1], qinit[grd.ids:grd.ide+1],
             ls="--", label="exact")

    # evolve the solver # period of time
    advc.evolve(num_periods=3)
    # plot solution 
    plt.plot(grd.x[grd.ids:grd.ide+1], advc.grid.q[grd.ids:grd.ide+1],
             label="upwind")

    plt.legend(frameon=False, loc="best")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$a$")
    plt.savefig("fv_flux_limiting.pdf")