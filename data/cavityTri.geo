cl__1 = 0.1;
Point(1) = {-1, -1, 0, cl__1};
Point(2) = {1, -1, 0, cl__1};
Point(3) = {1, 1, 0, cl__1};
Point(4) = {-1, 1, 0, cl__1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(6) = {4, 1, 2, 3};
Plane Surface(6) = {6};

Physical Surface("Domain", 9) = {6};
Physical Line("Wall", 1) = {1, 2, 3, 4};
Mesh 2;


