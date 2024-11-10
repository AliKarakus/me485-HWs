import gmsh
import sys
import math

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize(sys.argv)

# Next add a new model named "cavity" 
gmsh.model.add("test")
lc = 4e-1

# The first type of `elementary entity' in Gmsh is a `Point'. To create a point
# with the built-in CAD kernel, the Python API function is gmsh.model.geo.addPoint():
# - the first 3 arguments are the point coordinates (x, y, z)
# - the next (optional) argument is the target mesh size close to the point
p1 = gmsh.model.geo.addPoint( 0.50,  0.00, 0, lc);
p2 = gmsh.model.geo.addPoint( 0.70,  0.30, 0, lc);
p3 = gmsh.model.geo.addPoint( 0.30,  0.70, 0, lc);
p4 = gmsh.model.geo.addPoint( 0.00,  0.60, 0, lc);
p5 = gmsh.model.geo.addPoint(-1.00,  0.25, 0, lc);
p6 = gmsh.model.geo.addPoint(-1.50,  0.00, 0, lc);
p7 = gmsh.model.geo.addPoint(-1.25, -0.50, 0, lc);
p8 = gmsh.model.geo.addPoint(-0.25, -1.00, 0, lc);
p9 = gmsh.model.geo.addPoint( 0.25, -0.20, 0, lc);

#As a general rule, elementary entity tags in Gmsh have to be unique per geometrical dimension.
l1 = gmsh.model.geo.addBSpline([p1, p2, p3, p4, p5])
l2 = gmsh.model.geo.addBSpline([p5, p6, p7, p8, p9, p1])

# The third elementary entity is the surface. In order to define a simple
# rectangular surface from the four curves defined above, a curve loop has first
# to be defined. A curve loop is defined by an ordered list of connected curves,
# a sign being associated with each curve (depending on the orientation of the
# curve to form a loop). The API function to create curve loops takes a list
# of tags as first argument.
c1 = gmsh.model.geo.addCurveLoop([l1, l2])
s1 = gmsh.model.geo.addPlaneSurface([c1])

# Here we define a physical curve with name "dirichlet" and "neumann" 
# in two groups (with prescribed tags 1 and 2); and a physical surface with name
# "fluid"  containing the geometrical surface 1:
gmsh.model.addPhysicalGroup(1, [l1], 1 , name="dirichlet")
gmsh.model.addPhysicalGroup(1, [l2], 2 , name="neumann")
gmsh.model.addPhysicalGroup(2, [s1], 9 , name="fluid")

gmsh.model.geo.synchronize()

# Save it to disk
gmsh.model.mesh.generate(2)
gmsh.write("mymesh.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# This should be called when you are done using the Gmsh Python API:
gmsh.finalize()