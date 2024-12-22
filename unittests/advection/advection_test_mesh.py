import gmsh
import sys
import math

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize(sys.argv)

# Next add a new model named "cavity" 
gmsh.model.add("mixed")
lc = 0.025

#Points
p1  = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc);
p2  = gmsh.model.geo.addPoint(0.5, 0.0, 0.0, lc);
p3  = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc);
p4  = gmsh.model.geo.addPoint(1.0, 1.0, 0.0, lc);
p5  = gmsh.model.geo.addPoint(0.5, 1.0, 0.0, lc);
p6  = gmsh.model.geo.addPoint(0.0, 1.0, 0.0, lc);

l1  = gmsh.model.geo.addLine(p1,  p2)
l2  = gmsh.model.geo.addLine(p2 , p3)
l3  = gmsh.model.geo.addLine(p3 , p4)
l4  = gmsh.model.geo.addLine(p4 , p5)
l5  = gmsh.model.geo.addLine(p5 , p6)
l6  = gmsh.model.geo.addLine(p6 , p1)
l7  = gmsh.model.geo.addLine(p2 , p5)

# #Surfaces
cl1 = gmsh.model.geo.addCurveLoop([l1, l7, l5, l6]);
s1 = gmsh.model.geo.addPlaneSurface([cl1])
cl2 = gmsh.model.geo.addCurveLoop([l2, l3, l4, -l7]);
s2  = gmsh.model.geo.addPlaneSurface([cl2])

# # # The `setTransfiniteCurve()' meshing constraints explicitly specifies the
# # # location of the nodes on the curve. For example, the following command forces
# # # 10 uniformly placed nodes on curve 2 (including the nodes on the two end
# # # points):
gmsh.model.geo.mesh.setTransfiniteCurve(l1, 20)
gmsh.model.geo.mesh.setTransfiniteCurve(l7, 40)
gmsh.model.geo.mesh.setTransfiniteCurve(l5, 20)
gmsh.model.geo.mesh.setTransfiniteCurve(l6, 40)

gmsh.model.geo.mesh.setTransfiniteSurface(s1)
gmsh.model.geo.mesh.setRecombine(2, s1)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4, l5, l6], 11 , name="outer")
gmsh.model.addPhysicalGroup(2, [s1, s2],13 , name="fluid")

# # Save it to disk
gmsh.model.mesh.generate(2)
gmsh.write("advection.msh")
# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
# This should be called when you are done using the Gmsh Python API:
gmsh.finalize()
