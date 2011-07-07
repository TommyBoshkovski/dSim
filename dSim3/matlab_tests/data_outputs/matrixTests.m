clear all
close all
clc

A = [[ 0.5 0.5 0 1 0 0 ]
[ 0.5 0.5 0 1 0 0 ]
[ 0 0.5 0.5 0 0 1 ]
[ 0 0.5 0.5 0 0 1 ]
[ 0.5 0 0.5 0 1 0 ]
[ 0.5 0 0.5 0 1 0 ]
[ 0.5 0.5 0 -1 0 0 ]
[ 0.5 0.5 0 -1 0 0 ]
[ 0 0.5 0.5 0 0 -1 ]
[ 0 0.5 0.5 0 0 -1 ]
[ 0.5 0 0.5 0 -1 0 ]
[ 0.5 0 0.5 0 -1 0 ]];


S = [[ 1.41421 0 0 0 0 0 ]
[ 0 1.41421 0 0 0 0 ]
[ 0 0 1.22474 0 0 0 ]
[ 0 0 0 0 0 0 ]
[ 0 0 0 0 0 0 ]
[ 0 0 0 0 0 0 ]];

V = [[ 0 0 -0 -1 -0 -0 ]
[ -0.5 0.541667 -0 -0 -0 1 ]
[ 0 0 -0 -0 -1 -0 ]
[ -1 -0.180556 -0 -0 -0 -0.333333 ]
[ 0 0 -1 -0 -0 -0 ]
[ 0 1 -0 -0 -0 -0.351351 ]];

U = [[ -0.707107 0 0 0.707107 -0 0 ]
[ -0.883883 0.251354 0 -0.883883 0.604908 0 ]
[ -0.176777 0.735035 0 -0.176777 -0.639895 0 ]
[ -0.176777 0.942502 0 -0.176777 0.981786 0 ]
[ 0 0 -0.816497 0 -0 -0.816497 ]
[ 0 0 -0.816497 0 -0 0.632993 ]
[ 0.53033 0.187518 0 0.53033 0.0696672 0 ]
[ 0.53033 0.187518 0 0.53033 0.0696672 0 ]
[ -0.176777 -0.471711 0 -0.176777 -0.432428 0 ]
[ -0.176777 -0.471711 0 -0.176777 -0.432428 0 ]
[ 0 0 0.816497 0 -0 0.367007 ]
[ 0 0 0.816497 0 -0 0.367007 ]];


a = [[ 10 5 0 ]
[ 0 20 0 ]
[ 0 0 50 ]];

b = [[ 3 -2 21 ]
[ 7 0 0 ]
[ 0 9 18 ]];


E = [ -0.614945 -0.762072 0.202707 ;
       0.721926 -0.440631 0.533542 ;
      -0.317278 0.474438 0.821123 ];


