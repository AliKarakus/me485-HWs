// Points
Point(1) = {0, 0, 0, 1.0};
Point(2) = {-1, -1, 0, 1.0};
Point(3) = {1, -1, 0, 1.0};
Point(4) = {1, 1, 0, 1.0};
Point(5) = {-1, 1, 0, 1.0};
Point(6) = {-5*Sqrt(2)/2, -5*Sqrt(2)/2, 0, 1.0};
Point(7) = {5*Sqrt(2)/2, -5*Sqrt(2)/2, 0, 1.0};
Point(8) = {5*Sqrt(2)/2, 5*Sqrt(2)/2, 0, 1.0};
Point(9) = {-5*Sqrt(2)/2, 5*Sqrt(2)/2, 0, 1.0};
Point(10) = {-10, -10, 0, 2.0};
Point(11) = {10, -10, 0, 2.0};
Point(12) = {10, 10, 0, 2.0};
Point(13) = {-10, 10, 0, 2.0};

// Lines
Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 5};
Line(4) = {5, 2};
Line(5) = {6, 2};
Line(6) = {7, 3};
Line(7) = {8, 4};
Line(8) = {9, 5};
Line(9) = {10, 11};
Line(10) = {11, 12};
Line(11) = {12, 13};
Line(12) = {13, 10};
Circle(13) = {8, 1, 9};
Circle(14) = {9, 1, 6};
Circle(15) = {6, 1, 7};
Circle(16) = {7, 1, 8};

// Surfaces
Line Loop(1) = {11, 12, 9, 10};
Line Loop(2) = {13, 14, 15, 16};
Plane Surface(1) = {1, 2};
Line Loop(3) = {7, 3, -8, -13};
Plane Surface(2) = {-3};
Line Loop(4) = {14, 5, -4, -8};
Plane Surface(3) = {4};
Line Loop(5) = {1, -6, -15, 5};
Plane Surface(4) = {-5};
Line Loop(6) = {6, 2, -7, -16};
Plane Surface(5) = {-6};

// Meshing
Transfinite Line {8, 7, 6, 5} = 5 Using Progression 1;
Transfinite Line {14, 13, 16, 15, 1, 4, 3, 2} = 5 Using Progression 1;
Transfinite Surface {3};
Transfinite Surface {2};
Transfinite Surface {5};
Transfinite Surface {4};
Recombine Surface {2, 3, 4, 5};

Physical Line("Wall", 1) = {1, 2, 3, 4};
Physical Line("Inflow", 2) = {9, 10, 11, 12};
Physical Surface("Domain", 9) = {1, 2, 3 , 4, 5 };

Mesh 2;
Save "test.msh2";


