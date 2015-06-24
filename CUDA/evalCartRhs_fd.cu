#include <stdio.h>
#include <cuda_runtime.h>
#include <evalCartRhs_fd.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#define BLOCK_SIZE 256

// sum all 32 elements in array
// the sum is stored in array[0]
__device__ void sumReductionInWarp(double array[32], int tid){
	for (int s = 1; s < 32; s *= 2){
		if (tid % (2*s) == 0)
			array[tid] += array[tid+s];
	}
}

__global__ void evalCartRhs_fd(	const double* H,
				
				const int* idx,	   // DP_struct
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

				double* F){	   // output
	/* 
	 * Mapping: 1 stencil per warp
	 */		

	// Declare a shared memory space for all shared arrays 
	// | Tx[4*nWarps] | Ty[4*nWarps] | Tz[4*nWarps] | HV[4*nWarps] | 
	// | Tx_i[32*nWarps] | Ty_i[32*nWarps] | Tz_i[32*nWarps] | HV[32*nWarps] | 
	// | H[4*nWarps] |

	extern __shared__ double sharedMem[];
	
	/**** Variables whose values are shared by threads in a warp ****/
	int lid = threadIdx.x;	// thread id in a block - local id (lid)
	int tid = lid%32;	// thread's id inside a warp

	int warp_id = lid/32;	// warp's id
	
	int nWarpsPerBlock = blockDim.x/32;

	// Compute node id (i)
	int i = blockIdx.x * nWarpsPerBlock + warp_id;	// blockID * # nodes/block + warp_id

	// boundary checking
	if (i >= Nnodes) return;

	/***************************************************************/
	
	/**** Shared data in __shared__ memory space ****/
	// Each warp has a shared chunk of 4 elements in each array

	double* Tx = (double*)(sharedMem + warp_id*4);			// Tx[4]
	double* Ty = (double*)(sharedMem + 1*4*nWarpsPerBlock + warp_id*4);// Ty[4]
	double* Tz = (double*)(sharedMem + 2*4*nWarpsPerBlock + warp_id*4);// Tz[4]
	double* HV = (double*)(sharedMem + 3*4*nWarpsPerBlock + warp_id*4);// HV[4]

	//************************************************/

	/*
	 * Each neighbor of node i is processed by a single thread
	 * in a warp except the 32nd thread in a warp
  	 */

	// get neighbor's id of node i
	// TODO try to ignore the 32nd thread in a warp to get idx value
	int nbr_id = idx[i*(Nnbr+1)+tid];
	
	// each warp has a shared chunk of 32 elements in each array
	double* Tx_i = (double*)(sharedMem + 4*4*nWarpsPerBlock + warp_id*32);			 // Tx_i[32]
	double* Ty_i = (double*)(sharedMem + 4*4*nWarpsPerBlock + 1*32*nWarpsPerBlock + warp_id*32);// Ty_i[32]
	double* Tz_i = (double*)(sharedMem + 4*4*nWarpsPerBlock + 2*32*nWarpsPerBlock + warp_id*32);// Tz_i[32]
	double* HV_i = (double*)(sharedMem + 4*4*nWarpsPerBlock + 3*32*nWarpsPerBlock + warp_id*32);// HV_i[32]
	
	for (int k = 0; k < 4; k++){
		Tx_i[tid] = DPx[i*(Nnbr+1)+tid] * H[nbr_id*4+k];
		Ty_i[tid] = DPy[i*(Nnbr+1)+tid] * H[nbr_id*4+k];
		Tz_i[tid] = DPz[i*(Nnbr+1)+tid] * H[nbr_id*4+k];
		HV_i[tid] =   L[i*(Nnbr+1)+tid] * H[nbr_id*4+k];

		// Do sum reduction on Tx_i[], Ty_i[], Tz_i[] and HV_i[]
		sumReductionInWarp(Tx_i, tid);
		sumReductionInWarp(Ty_i, tid);
		sumReductionInWarp(Tz_i, tid);
		sumReductionInWarp(HV_i, tid);

		// One thread in a warp update the sums in Tx[k], Ty[k], Tz[k], HV[k]
		if (tid == 0){
			Tx[k] = Tx_i[0];
			Ty[k] = Ty_i[0];
			Tz[k] = Tz_i[0];
			HV[k] = HV_i[0];
		}
	}

	// Get pointer to shared H array
	double* s_H = (double*)(sharedMem + 4*4*nWarpsPerBlock + 4*32*nWarpsPerBlock + warp_id*4);
	
	// 4 threads in a warp get load data from global to shared memory
	if (tid < 4) s_H[tid] = H[i*Nvar+tid];

	/////// FINISH SPARSE MATRIX MULTIPLICATION ////////	
	// TODO optimize this serial part
	// compute p,q,s
	if (tid == 0){
		double p = - ( 	s_H[0] * Tx[0] + s_H[1] * Ty[0] + s_H[2] * Tz[0] 
			      + f[i] * (y[i] * s_H[2] - z[i] * s_H[1]) + Tx[3]);

		double q = - (	s_H[0] * Tx[1] + s_H[1] * Ty[1] + s_H[2] * Tz[1]
			      + f[i] * (z[i] * s_H[0] - x[i] * s_H[2]) + Ty[3]);

		double s = - (	s_H[0] * Tx[2] + s_H[1] * Ty[2] + s_H[2] * Tz[2]
			      + f[i] * (x[i] * s_H[1] - y[i] * s_H[0]) + Tz[3]);

		F[i*4+0] = p_u[i*3+0] * p + p_u[i*3+1] * q + p_u[i*3+2] * s + HV[0];
		F[i*4+1] = p_v[i*3+0] * p + p_v[i*3+1] * q + p_v[i*3+2] * s + HV[1];
		F[i*4+2] = p_w[i*3+0] * p + p_w[i*3+1] * q + p_w[i*3+2] * s + HV[2];
		F[i*4+3] = - (	s_H[0] * (Tx[3] - gradghm[i*3+0])
			      + s_H[1] * (Ty[3] - gradghm[i*3+1])
			      + s_H[2] * (Tz[3] - gradghm[i*3+2])
			      +(s_H[3] + gh0 - ghm[i]) * (Tx[0] + Ty[1] + Tz[2]))
			      + HV[3];			
	}
}

void evoke_evalCartRhs_fd(const fType* H_d,
			  const atm_struct* atm_d,
			  const DP_struct* DP_d,
			  const fType* gradghm_d,
			  fType* F_d){

	// Extract constants from atm_struct
	const fType* x = atm_d->x;
        const fType* y = atm_d->y;
        const fType* z = atm_d->z;
        const fType* f = atm_d->f;

        const fType g = atm_d->g;
        const fType a = atm_d->a;
        const fType gh0 = atm_d->gh0;

        const fType* ghm = atm_d->ghm;

        const int Nnodes = atm_d->Nnodes;
        const int Nvar = atm_d->Nvar;
        const int Nnbr = atm_d->Nnbr;

        const fType* p_u = atm_d->p_u;
        const fType* p_v = atm_d->p_v;
        const fType* p_w = atm_d->p_w;

        // extract out constants from the DP structure
        const int* idx = DP_d->idx;
        const fType* DPx = DP_d->DPx;
        const fType* DPy = DP_d->DPy;
        const fType* DPz = DP_d->DPz;
        const fType* L = DP_d->L;

	int nWarpsPerBlock = BLOCK_SIZE/32;	
	size_t sharedMemSize = (4*4*nWarpsPerBlock + 4*32*nWarpsPerBlock + 4*nWarpsPerBlock) * sizeof(double);

	int nBlocksPerGrid = (Nnodes*32 + BLOCK_SIZE - 1)/BLOCK_SIZE;
/*
printf("Shared mem size = %ld\n", sharedMemSize);
printf("nWarps = %d\n", nWarpsPerBlock);
printf("nBlocks = %d\n", nBlocksPerGrid);
*/

cudaProfilerStart();

	// Launch kernel
	evalCartRhs_fd<<<nBlocksPerGrid, BLOCK_SIZE, sharedMemSize>>>(H_d, idx, DPx, DPy, DPz, L,
									 x, y, z, f, ghm,
									 p_u, p_v, p_w,
									 gradghm_d,
									 g, a, gh0, Nnodes, Nvar, Nnbr,
									 F_d);
	
	// wait for kernel to complete
	cudaDeviceSynchronize();
cudaProfilerStop();

}
