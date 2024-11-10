cl__1 = 0.4;
Point(1) = {0.5, 0.0, 0, cl__1};
Point(2) = {0.7, 0.3, 0, cl__1};
Point(3) = {0.3, 0.7, 0, cl__1};
Point(4) = {0.0, 0.6, 0, cl__1};
Point(5) = {-1.0, 0.25, 0, cl__1};
Point(6) = {-1.5, 0, 0, cl__1};
Point(7) = {-1.25, -0.5, 0, cl__1};
Point(8) = {-0.25, -1.0, 0, cl__1};
Point(9) = {0.25, -0.2, 0, cl__1};

//
BSpline(1) = {1, 2, 3, 4, 5};
BSpline(2) = {5, 6, 7, 8, 9, 1};
// // //

Curve Loop(1) = {1, 2};
Plane Surface(1) = {1};

Physical Surface("Domain", 9) = {1};

Physical Line("Dirichlet", 1) = {1};
Physical Line("Neumann", 2) = {2};

Mesh 2;
Mesh.Format msh2; 
Save "mymesh.msh";
