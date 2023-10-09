cl__1 = 0.05;
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
Transfinite Line {1, 3} = 50 Using Progression 1;
Transfinite Line {2, 4} = 50 Using Progression 1;
Transfinite Surface {6};
Physical Surface("Domain", 9) = {6};

Physical Line("Inflow", 1) = {1,3};
Physical Line("Inflow2", 2) = {4};
Physical Line("Outflow", 3) = {2};

