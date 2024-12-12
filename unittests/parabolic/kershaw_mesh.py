import gmsh
import sys
import math

# Before using any functions in the Python API, Gmsh must be initialized:
gmsh.initialize(sys.argv)
alpha = 0.40



x0 = 0.0
y0 = 0.0
x1 = 1.0
y1 = 1.0



beta  = 0.25
m0 = x0 + alpha*(x1-x0)
m1 = x1 - alpha*(x1-x0)
k0 = y0 + beta*(y1-y0)
k1 = k0 + beta*(y1-y0)
k2 = k1 + beta*(y1-y0)
k3 = k2 + beta*(y1-y0)

# Next add a new model named "cavity" 
gmsh.model.add("mixed")
lc = 0.25
nl = 10

#Points
# gmsh.model.setNumber(alpha, 0.25);
p1  = gmsh.model.geo.addPoint(x0,      y0,            0.0, lc);
p2  = gmsh.model.geo.addPoint(m0,      y0,            0.0, lc);
p3  = gmsh.model.geo.addPoint(x1,      y0,            0.0, lc);

p4  = gmsh.model.geo.addPoint(x1,      k0,            0.0, lc);
p5  = gmsh.model.geo.addPoint(m1,      k0,            0.0, lc);
p6  = gmsh.model.geo.addPoint(x0,      k0,            0.0, lc);
# Second layer
p7  = gmsh.model.geo.addPoint(x1,      k1,            0.0, lc);
p8  = gmsh.model.geo.addPoint(m0,      k1,            0.0, lc);
p9  = gmsh.model.geo.addPoint(x0,      k1,            0.0, lc);

# Second layer
p10  = gmsh.model.geo.addPoint(x1,      k2,            0.0, lc);
p11  = gmsh.model.geo.addPoint(m1,      k2,            0.0, lc);
p12  = gmsh.model.geo.addPoint(x0,      k2,            0.0, lc);

# Second layer
p13  = gmsh.model.geo.addPoint(x1,      k3,            0.0, lc);
p14  = gmsh.model.geo.addPoint(m0,      k3,            0.0, lc);
p15  = gmsh.model.geo.addPoint(x0,      k3,            0.0, lc);

# Straight lines to create transfinite part
l1  = gmsh.model.geo.addLine(p1,  p2)
l2  = gmsh.model.geo.addLine(p2 , p3)
l3  = gmsh.model.geo.addLine(p3 , p4)
l4  = gmsh.model.geo.addLine(p4 , p5)
l5  = gmsh.model.geo.addLine(p5 , p6)
l6  = gmsh.model.geo.addLine(p6 , p1)
l7  = gmsh.model.geo.addLine(p5 , p2)
# Second layer
l8   = gmsh.model.geo.addLine(p4 , p7)
l9   = gmsh.model.geo.addLine(p7 , p8)
l10  = gmsh.model.geo.addLine(p8 , p9)
l11  = gmsh.model.geo.addLine(p9 , p6)
l12  = gmsh.model.geo.addLine(p8 , p5)

# Second layer
l13   = gmsh.model.geo.addLine(p7 , p10)
l14   = gmsh.model.geo.addLine(p10, p11)
l15   = gmsh.model.geo.addLine(p11 , p12)
l16   = gmsh.model.geo.addLine(p12 , p9)
l17   = gmsh.model.geo.addLine(p11 , p8)

# Second layer
l18   = gmsh.model.geo.addLine(p10 , p13)
l19   = gmsh.model.geo.addLine(p13,  p14)
l20   = gmsh.model.geo.addLine(p14 , p15)
l21   = gmsh.model.geo.addLine(p15 , p12)
l22   = gmsh.model.geo.addLine(p14 , p11)





#Surfaces
cl1 = gmsh.model.geo.addCurveLoop([l1,-l7, l5, l6]);
cl2 = gmsh.model.geo.addCurveLoop([l2, l3, l4, l7]);
s1 = gmsh.model.geo.addPlaneSurface([cl1])
s2 = gmsh.model.geo.addPlaneSurface([cl2])

cl3 = gmsh.model.geo.addCurveLoop([-l5,-l12, l10, l11]);
cl4 = gmsh.model.geo.addCurveLoop([-l4, l8, l9, l12]);
s3 = gmsh.model.geo.addPlaneSurface([cl3])
s4 = gmsh.model.geo.addPlaneSurface([cl4])

cl5 = gmsh.model.geo.addCurveLoop([-l10,-l17, l15, l16]);
cl6 = gmsh.model.geo.addCurveLoop([-l9, l13, l14, l17]);
s5 = gmsh.model.geo.addPlaneSurface([cl5])
s6 = gmsh.model.geo.addPlaneSurface([cl6])

cl7 = gmsh.model.geo.addCurveLoop([-l15,-l22, l20, l21]);
cl8 = gmsh.model.geo.addCurveLoop([-l14, l18, l19, l22]);
s7 = gmsh.model.geo.addPlaneSurface([cl7])
s8 = gmsh.model.geo.addPlaneSurface([cl8])




gmsh.model.geo.mesh.setTransfiniteCurve(l1,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l2,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l3,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l4,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l5,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l6,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l7,  nl)

gmsh.model.geo.mesh.setTransfiniteCurve(l8,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l9,  nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l10, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l11, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l12, nl)

gmsh.model.geo.mesh.setTransfiniteCurve(l13, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l14, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l15, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l16, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l17, nl)

gmsh.model.geo.mesh.setTransfiniteCurve(l18, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l19, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l20, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l21, nl)
gmsh.model.geo.mesh.setTransfiniteCurve(l22, nl)


gmsh.model.geo.mesh.setTransfiniteSurface(s1)
gmsh.model.geo.mesh.setTransfiniteSurface(s2)
gmsh.model.geo.mesh.setTransfiniteSurface(s3)
gmsh.model.geo.mesh.setTransfiniteSurface(s4)
gmsh.model.geo.mesh.setTransfiniteSurface(s5)
gmsh.model.geo.mesh.setTransfiniteSurface(s6)
gmsh.model.geo.mesh.setTransfiniteSurface(s7)
gmsh.model.geo.mesh.setTransfiniteSurface(s8)

gmsh.model.geo.mesh.setRecombine(2, s1)
gmsh.model.geo.mesh.setRecombine(2, s2)
gmsh.model.geo.mesh.setRecombine(2, s3)
gmsh.model.geo.mesh.setRecombine(2, s4)
gmsh.model.geo.mesh.setRecombine(2, s5)
gmsh.model.geo.mesh.setRecombine(2, s6)
gmsh.model.geo.mesh.setRecombine(2, s7)
gmsh.model.geo.mesh.setRecombine(2, s8)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l8, l13, l18, l19, l20], 11 , name="outer")
gmsh.model.addPhysicalGroup(1, [l6, l11, l16, l21], 12 , name="inner")
gmsh.model.addPhysicalGroup(2, [s1, s2, s3, s4, s5,s6,s7,s8],13 , name="fluid")

# Save it to disk
gmsh.model.mesh.generate(2)
gmsh.write("parabolic.msh")
# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()
# This should be called when you are done using the Gmsh Python API:
gmsh.finalize()