import gmsh
import sys
import math

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize(sys.argv)

# Next add a new model named "cavity" 
gmsh.model.add("mixed")
lc = 5.0

#Points
p1  = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, lc);
p2  = gmsh.model.geo.addPoint(-1, -1, 0,lc);
p3  = gmsh.model.geo.addPoint(1, -1, 0,lc);
p4  = gmsh.model.geo.addPoint(1, 1, 0, lc);
p5  = gmsh.model.geo.addPoint(-1, 1, 0,lc);
p6  = gmsh.model.geo.addPoint(-5*math.sqrt(2)/2,-5*math.sqrt(2)/2, 0,lc);
p7  = gmsh.model.geo.addPoint( 5*math.sqrt(2)/2,-5*math.sqrt(2)/2, 0,lc);
p8  = gmsh.model.geo.addPoint( 5*math.sqrt(2)/2, 5*math.sqrt(2)/2, 0,lc);
p9  = gmsh.model.geo.addPoint(-5*math.sqrt(2)/2, 5*math.sqrt(2)/2, 0,lc);
p10 = gmsh.model.geo.addPoint(-10, -10, 0, lc);
p11 = gmsh.model.geo.addPoint(10, -10, 0, lc);
p12 = gmsh.model.geo.addPoint(10, 10, 0, lc);
p13 = gmsh.model.geo.addPoint(-10, 10, 0, lc);


#As a general rule, elementary entity tags in Gmsh have to be unique per geometrical dimension.
#As a general rule, elementary entity tags in Gmsh have to be unique per geometrical dimension.
l1  = gmsh.model.geo.addLine(p2, p3)
l2  = gmsh.model.geo.addLine(p3, p4)
l3  = gmsh.model.geo.addLine(p4, p5)
l4  = gmsh.model.geo.addLine(p5, p2)
l5  = gmsh.model.geo.addLine(p6, p2)
l6  = gmsh.model.geo.addLine(p7 , p3)
l7  = gmsh.model.geo.addLine(p8 , p4)
l8  = gmsh.model.geo.addLine(p9 , p5)
l9  = gmsh.model.geo.addLine(p10, p11)
l10 = gmsh.model.geo.addLine(p11, p12)
l11 = gmsh.model.geo.addLine(p12, p13)
l12 = gmsh.model.geo.addLine(p13, p10)

l13  = gmsh.model.geo.addCircleArc(p8, p1, p9)
l14  = gmsh.model.geo.addCircleArc(p9, p1, p6)
l15  = gmsh.model.geo.addCircleArc(p6, p1, p7)
l16  = gmsh.model.geo.addCircleArc(p7, p1, p8)


#Surfaces
cl1 = gmsh.model.geo.addCurveLoop([l11, l12, l9, l10]);
cl2 = gmsh.model.geo.addCurveLoop([l13, l14, l15, l16]);
s1 = gmsh.model.geo.addPlaneSurface([cl1,cl2])

cl3 = gmsh.model.geo.addCurveLoop([l7, l3, -l8, -l13]);
s2  = gmsh.model.geo.addPlaneSurface([-cl3])

cl4 = gmsh.model.geo.addCurveLoop([l14, l5, -l4, -l8]);
s3  = gmsh.model.geo.addPlaneSurface([cl4])

cl5 = gmsh.model.geo.addCurveLoop([l1, -l6, -l15, l5]);
s4  = gmsh.model.geo.addPlaneSurface([-cl5])

cl6 = gmsh.model.geo.addCurveLoop([l6, l2, -l7, -l16]);
s5  = gmsh.model.geo.addPlaneSurface([-cl6])

# The `setTransfiniteCurve()' meshing constraints explicitly specifies the
# location of the nodes on the curve. For example, the following command forces
# 20 uniformly placed nodes on curve 2 (including the nodes on the two end
# points):
gmsh.model.geo.mesh.setTransfiniteCurve(l5, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l6, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l7, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l8, 5)

gmsh.model.geo.mesh.setTransfiniteCurve(l1, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l2, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l3, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l4, 5)

gmsh.model.geo.mesh.setTransfiniteCurve(l13, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l14, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l15, 5)
gmsh.model.geo.mesh.setTransfiniteCurve(l16, 5)

gmsh.model.geo.mesh.setTransfiniteSurface(s2)
gmsh.model.geo.mesh.setTransfiniteSurface(s3)
gmsh.model.geo.mesh.setTransfiniteSurface(s4)
gmsh.model.geo.mesh.setTransfiniteSurface(s5)

gmsh.model.geo.mesh.setRecombine(2, s2)
gmsh.model.geo.mesh.setRecombine(2, s3)
gmsh.model.geo.mesh.setRecombine(2, s4)
gmsh.model.geo.mesh.setRecombine(2, s5)

gmsh.model.addPhysicalGroup(1, [l1,l2,l3,l4], 1 , name="drichlet")
gmsh.model.addPhysicalGroup(1, [l9, l10, l11, l12], 2 , name="neumann")
gmsh.model.addPhysicalGroup(2, [s1,s2,s3,s4,s5], 9 , name="fluid")

gmsh.model.geo.synchronize()

# Save it to disk
gmsh.model.mesh.generate(2)
gmsh.write("mixed.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# This should be called when you are done using the Gmsh Python API:
gmsh.finalize()