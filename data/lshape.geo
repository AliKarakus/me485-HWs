
// Gmsh project created on Sun Feb 17 11:15:46 2013
h=11; lc=1/h;
Point(1) = {0, 0, 0, lc};
Point(2) = {0.5, 0, 0, lc};
Point(3) = {0.5, 0.3, 0, lc};
Point(4) = {0.5, 0.5, 0, lc};
Point(5) = {0.7, 0.5, 0, lc};
Point(6) = {1, 0.5, 0, lc};
Point(7) = {1, 1, 0, lc};
Point(8) = {0, 1, 0, lc};
Point(9) = {0.5, .7, 0, lc};
Point(10)= {0.3, 0.5, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 1};
Circle(9) = {5, 4, 9};
Circle(10) = {9, 4, 10};
Circle(11) = {10, 4, 3};
Line Loop(12) = {1, 2, -11, -10, -9, 5, 6, 7, 8};
Plane Surface(13) = {12};
Line Loop(14) = {3, 4, 9, 10, 11};
Plane Surface(15) = {14};
//Transfinite Surface{15}={3, 4, 5};

n1 = 14;
Transfinite Line{1,6} = n1 ;

n5 = 11 ;
Transfinite Line{9,10,11} = n5 ;
Transfinite Line{3,4} = Floor[n5/2]+1 ;
Transfinite Surface{15}={3, 5, 9, 10};

n3 = 15;
n2 = 11;
Transfinite Line{2,5} = n2 ;
Transfinite Line{7,8} = Floor[((n5-1)*3 + 2*(n2-1))/2]+1 ;
Transfinite Surface{13}={1,2,6,7};

Recombine Surface '*' ;
Mesh.Smoothing = 10;