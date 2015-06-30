#ifndef CONFIG_H
#define CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

//typedef int bool;

//#define true 1
//#define false 0

typedef double fType;	// double precision
//typedef float fType;	// single precision

typedef struct atms {
	int Nnodes;	// number of nodes
	int Nvar;	// number of variables per node
	int Nnbr;	// number of neighbors per node

	fType* x;	// x-coordinates
	fType* y;	// y-coordinates
	fType* z;	// z-coordinates

	fType* f; 	// Coriolis force

	fType g; 	// gravitational constant (m/s^2)
	fType a;	// mean raduys of the earth (constant in meters)
	fType gh0;	// Initial condition for the geopotential field

	fType* ghm;	// the profile of the mountain

	// variables for projecting an arbitrary
	// Cartesian vector on the surface
	fType* p_u;
	fType* p_v;
	fType* p_w; 
} atm_struct;

typedef struct DPs { 
	int* idx;	// Nnodes x Nnbr
	fType* DPx;	// Nnodes x Nnbr
	fType* DPy;	// Nnodes x Nnbr
	fType* DPz;	// Nnodes x Nnbr
	fType* L;	// Nnodes x Nnbr
} DP_struct;

#endif
