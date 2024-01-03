cl__1 = 0.075;
Point(1) = {0, 0, 0      , cl__1};
Point(2) = {0.6, 0.0, 0.0, 0.5*cl__1};
Point(3) = {0.6, 0.2, 0.0, 0.5*cl__1};
Point(4) = {3.0, 0.2, 0.0, cl__1};
Point(5) = {3.0, 1.0, 0.0, cl__1};
Point(6) = {0.0, 1.0, 0.0, cl__1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line Loop(6) = {1, 2, 3, 4, 5, 6};
Plane Surface(6) = {6};

Physical Surface("Domain", 9) = {6};
Physical Line("Wall", 1) = {1, 2, 3, 5};
Physical Line("Inlet", 2) = {6};
Physical Line("Outlet", 3) = {4};

// Mesh 2;
// RecombineMesh; 

