#ifndef EVAL_CART_H
#define EVAL_CART_H

#include<config.h>
#include<math.h>
/*
__global__ void sumReductionInWarp(__shared__ double array[32], int tid);

__global__ void evalCartRhs_fd( const double* H,
                                
                                const int* idx,    // DP_struct
                                const double* DPx,
                                const double* DPy,
                                const double* DPz,
                                const double* L,
                                
                                const double* x,   // atm_struct
                                const double* y,
                                const double* z,
                                const double* f,
                                
                                const double* ghm, // ghm
        
                                const double* p_u,
                                const double* p_v,
                                const double* p_w,
                                
                                const double* gradghm,

                                const double g, const double a, const double gh0, // constants
                                const int Nnodes, const int Nvar, const int Nnbr,
                                
                                double* F);
*/

void evoke_evalCartRhs_fd(const fType* H_d,
                          const atm_struct* atm_d,
                          const DP_struct* DP_d,
                          const fType* gradghm_d,
                          fType* F_d);

#endif
