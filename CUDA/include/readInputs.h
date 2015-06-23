#ifndef READ_INPUT_H
#define READ_INPUT_H

#include <config.h>

// read atm variables from matlab
atm_struct* read_atm();

// read DPx, DPy, DPz and L matrices
DP_struct* read_DPs(int Nnodes, int Nnbr, fType gamma, fType a);

// read H matrix
fType* read_H(int Nnodes);

// read gradghm matrix
fType* read_gradghm(int Nnodes);

#endif
