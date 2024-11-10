import gmsh
import sys
import math

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize(sys.argv)

# Next add a new model named "cavity" 
gmsh.model.add("cavity")
lc = 5e-1

# The first type of `elementary entity' in Gmsh is a `Point'. To create a point
# with the built-in CAD kernel, the Python API function is gmsh.model.geo.addPoint():
# - the first 3 arguments are the point coordinates (x, y, z)
# - the next (optional) argument is the target mesh size close to the point
p1 = gmsh.model.geo.addPoint(-1.0,-1.0, 0.0, lc)
p2 = gmsh.model.geo.addPoint( 1.0,-1.0, 0.0, lc)
p3 = gmsh.model.geo.addPoint( 1.0, 1.0, 0.0, lc)
p4 = gmsh.model.geo.addPoint(-1.0, 1.0, 0.0, lc)

#As a general rule, elementary entity tags in Gmsh have to be unique per geometrical dimension.
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# The third elementary entity is the surface. In order to define a simple
# rectangular surface from the four curves defined above, a curve loop has first
# to be defined. A curve loop is defined by an ordered list of connected curves,
# a sign being associated with each curve (depending on the orientation of the
# curve to form a loop). The API function to create curve loops takes a list
# of tags as first argument.
c1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
s1 = gmsh.model.geo.addPlaneSurface([c1])

gmsh.model.geo.mesh.setTransfiniteCurve(l1, 3)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, 3)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, 3)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, 3)

gmsh.model.geo.mesh.setTransfiniteSurface(s1)
gmsh.model.geo.mesh.setRecombine(2, s1)
# Here we define a physical curve with name "wall" that groups the left, bottom, top and right curves
# in a single group (with prescribed tag 1); and a physical surface with name
# "fluid" (with an automatic tag) containing the geometrical surface 1:
gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], 1 , name="drichlet")
gmsh.model.addPhysicalGroup(2, [s1], 9 , name="fluid")

# Before they can be meshed (and, more generally, before they can be used by API
# functions outside of the built-in CAD kernel functions), the CAD entities must
# be synchronized with the Gmsh model, which will create the relevant Gmsh data
# structures. 
gmsh.model.geo.synchronize()

# Save it to disk
gmsh.model.mesh.generate(2)
gmsh.write("cavity.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# This should be called when you are done using the Gmsh Python API:
gmsh.finalize()



