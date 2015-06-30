#include <stdio.h>
#include <cuda_runtime.h>
#include <evalCartRhs_fd.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <timer.h>

#define BLOCK_SIZE 256
#define nWarpsPerBlock (BLOCK_SIZE/32)	// total number of warps per block

// Compute offsets of intermediate vectors
#define Tx_offset 0
#define Ty_offset (Tx_offset + 4*nWarpsPerBlock)
#define Tz_offset (Ty_offset + 4*nWarpsPerBlock)
#define HV_offset (Tz_offset + 4*nWarpsPerBlock)

#define Tx_i_offset (HV_offset + 4*nWarpsPerBlock)
#define Ty_i_offset (Tx_i_offset + 32*nWarpsPerBlock)
#define Tz_i_offset (Ty_i_offset + 32*nWarpsPerBlock)
#define HV_i_offset (Tz_i_offset + 32*nWarpsPerBlock)

// Compute offsets of data vectors
//#define DPx_offset (HV_i_offset + 32*nWarpsPerBlock)
//#define DPy_offset (DPx_offset + 32*nWarpsPerBlock) 
//#define DPz_offset (DPy_offset + 32*nWarpsPerBlock)
//#define L_offset   (DPz_offset + 32*nWarpsPerBlock)
#define nbr_H_offset (HV_i_offset + 32*nWarpsPerBlock) // (L_offset + 32*nWarpsPerBlock)
//#define center_H_offset (nbr_H_offset + 4*32*nWarpsPerBlock) 

//#define p_u_offset (center_H_offset + 4*nWarpsPerBlock)
//#define p_v_offset (p_u_offset + 3*nWarpsPerBlock)
//#define p_w_offset (p_v_offset + 3*nWarpsPerBlock)
//#define gradghm_offset (p_w_offset + 3*nWarpsPerBlock)

// sum all 32 elements in array
// the sum is stored in array[0]
__device__ void sumReductionInWarp(fType array[32], int tid){
	for (int s = 1; s < 32; s *= 2){
		if (tid % (2*s) == 0)
			array[tid] += array[tid+s];
	}
}

__global__ void evalCartRhs_fd(	const fType* H,
				const int* idx,	const fType* DPx, const fType* DPy, const fType* DPz, const fType* L,   
				const fType* x, const fType* y, const fType* z, const fType* f, const fType* ghm,
				const fType* p_u, const fType* p_v, const fType* p_w, const fType* gradghm,
				
				const fType g, const fType a, const fType gh0, // constants
				const int Nnodes, const int Nvar, const int Nnbr,
				
				fType* F	   // output
){
	// ================ Thread layout ====================
	//	Each stencil is mapped to a warp
	// ===================================================

	// ================== Compute index and offset ====================
	int tid = threadIdx.x % 32;		// thread id within a warp
	int warpID = threadIdx.x / 32; // warp id within a block
	// ================================================================

	// Declare the shared memory space for this block
	extern __shared__ fType sharedMem[];
	
	// Compute node id (i)
	int node_id = blockIdx.x * nWarpsPerBlock + warpID;	// blockID * # nodes/block + warpID

	// boundary checking
	if (node_id >= Nnodes) return;

//	// Preload data into shared memory to utilize coalesced accesses
//	sharedMem[DPx_offset+warpID*32+tid] = DPx[node_id*(Nnbr+1)+tid];
//	sharedMem[DPy_offset+warpID*32+tid] = DPy[node_id*(Nnbr+1)+tid];
//	sharedMem[DPz_offset+warpID*32+tid] = DPz[node_id*(Nnbr+1)+tid];
//	sharedMem[  L_offset+warpID*32+tid] = L  [node_id*(Nnbr+1)+tid];

	// get neighbor's id of node i
	int nbr_id = idx[node_id*(Nnbr+1)+tid];

	#pragma unroll
	for (int k = 0; k < 4; k++){
		sharedMem[nbr_H_offset+warpID*32*4+tid*4+k] = H[nbr_id*4+k];

		sharedMem[Tx_i_offset+warpID*32+tid] = DPx[node_id*(Nnbr+1)+tid]
											 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];
		sharedMem[Ty_i_offset+warpID*32+tid] = DPy[node_id*(Nnbr+1)+tid] 												 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];
		sharedMem[Tz_i_offset+warpID*32+tid] = DPz[node_id*(Nnbr+1)+tid] 												 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];
		sharedMem[HV_i_offset+warpID*32+tid] = L[node_id*(Nnbr+1)+tid] 												 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];

//		sharedMem[Tx_i_offset+warpID*32+tid] = sharedMem[DPx_offset+warpID*32+tid] 
//											 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];
//		sharedMem[Ty_i_offset+warpID*32+tid] = sharedMem[DPy_offset+warpID*32+tid] 												 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];
//		sharedMem[Tz_i_offset+warpID*32+tid] = sharedMem[DPz_offset+warpID*32+tid] 												 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];
//		sharedMem[HV_i_offset+warpID*32+tid] = sharedMem  [L_offset+warpID*32+tid] 												 * sharedMem[nbr_H_offset+warpID*32*4+tid*4+k];

		// Do sum reduction on Tx_i[], Ty_i[], Tz_i[] and HV_i[]
		sumReductionInWarp(sharedMem+Tx_i_offset+warpID*32, tid);
		sumReductionInWarp(sharedMem+Ty_i_offset+warpID*32, tid);
		sumReductionInWarp(sharedMem+Tz_i_offset+warpID*32, tid);
		sumReductionInWarp(sharedMem+HV_i_offset+warpID*32, tid);

		// One thread in a warp update the sums in Tx[k], Ty[k], Tz[k], HV[k]
		if (tid == 0){
			sharedMem[Tx_offset+warpID*4+k] = sharedMem[Tx_i_offset+warpID*32+0];
			sharedMem[Ty_offset+warpID*4+k] = sharedMem[Ty_i_offset+warpID*32+0];
			sharedMem[Tz_offset+warpID*4+k] = sharedMem[Tz_i_offset+warpID*32+0];
			sharedMem[HV_offset+warpID*4+k] = sharedMem[HV_i_offset+warpID*32+0];
		}
	}

//	// Load H values for the stencil center - need 4 threads only
//	if (tid < 4)
//		sharedMem[center_H_offset+warpID*4+tid] = H[node_id*4+tid];
//			
//	// Load p_u, p_v, p_w and gradghm [3*nComputeWarps] - need only 3 threads
//	if (tid < 3){
//		sharedMem[p_u_offset+warpID*3+tid] = p_u[node_id*3+tid];
//		sharedMem[p_v_offset+warpID*3+tid] = p_v[node_id*3+tid];
//		sharedMem[p_w_offset+warpID*3+tid] = p_w[node_id*3+tid];
//		sharedMem[gradghm_offset+warpID*3+tid] = gradghm[node_id*3+tid];	
//	}	

// ****** Don't do pre-load
	// compute p,q,s
	if (tid == 0){
		fType p = - (H[node_id*4+0] * sharedMem[Tx_offset+warpID*4+0] 
					+ H[node_id*4+1] * sharedMem[Ty_offset+warpID*4+0] 
					+ H[node_id*4+2] * sharedMem[Tz_offset+warpID*4+0] 
			      	+ f[node_id] 	
					* (y[node_id] * H[node_id*4+2] 
					-  z[node_id] * H[node_id*4+1]) 
					+ sharedMem[Tx_offset+warpID*4+3]);

		fType q = - (H[node_id*4+0] * sharedMem[Tx_offset+warpID*4+1] 
					+ H[node_id*4+1] * sharedMem[Ty_offset+warpID*4+1] 
					+ H[node_id*4+2] * sharedMem[Tz_offset+warpID*4+1]
					+ f[node_id] 
					* (z[node_id] * H[node_id*4+0] 
					-  x[node_id] * H[node_id*4+2]) 
					+ sharedMem[Ty_offset+warpID*4+3]);

		fType s = - (H[node_id*4+0] * sharedMem[Tx_offset+warpID*4+2] 
					+ H[node_id*4+1] * sharedMem[Ty_offset+warpID*4+2] 
					+ H[node_id*4+2] * sharedMem[Tz_offset+warpID*4+2]
					+ f[node_id] 
					* (x[node_id] * H[node_id*4+1] 
					-  y[node_id] * H[node_id*4+0]) 
					+ sharedMem[Tz_offset+warpID*4+3]);

		F[node_id*4+0] =  p_u[node_id*3+0] * p 
						+ p_u[node_id*3+1] * q 
						+ p_u[node_id*3+2] * s 
						+ sharedMem[HV_offset+warpID*4+0];

		F[node_id*4+1] =  p_v[node_id*3+0] * p 
						+ p_v[node_id*3+1] * q 
						+ p_v[node_id*3+2] * s 
						+ sharedMem[HV_offset+warpID*4+1];

		F[node_id*4+2] =  p_w[node_id*3+0] * p 
						+ p_w[node_id*3+1] * q 
						+ p_w[node_id*3+2] * s 
						+ sharedMem[HV_offset+warpID*4+2];

		F[node_id*4+3] = - (H[node_id*4+0] * (sharedMem[Tx_offset+warpID*4+3] 
						  - gradghm[node_id*3+0])
						  + H[node_id*4+1] * (sharedMem[Ty_offset+warpID*4+3] 
						  - gradghm[node_id*3+1])
						  + H[node_id*4+2] * (sharedMem[Tz_offset+warpID*4+3] 
						  - gradghm[node_id*3+2])
						  +(H[node_id*4+3] + gh0 - ghm[node_id]) 
						  * (sharedMem[Tx_offset+warpID*4+0] 
						  + sharedMem[Ty_offset+warpID*4+1] 
						  + sharedMem[Tz_offset+warpID*4+2]))
						  + sharedMem[HV_offset+warpID*4+3];			
	}

//// ******* pre-load to shared memory **********
//	// compute p,q,s
//	if (tid == 0){
//		fType p = - (sharedMem[center_H_offset+warpID*4+0] * sharedMem[Tx_offset+warpID*4+0] 
//					+ sharedMem[center_H_offset+warpID*4+1] * sharedMem[Ty_offset+warpID*4+0] 
//					+ sharedMem[center_H_offset+warpID*4+2] * sharedMem[Tz_offset+warpID*4+0] 
//			      	+ f[node_id] 	
//					* (y[node_id] * sharedMem[center_H_offset+warpID*4+2] 
//					-  z[node_id] * sharedMem[center_H_offset+warpID*4+1]) 
//					+ sharedMem[Tx_offset+warpID*4+3]);

//		fType q = - (sharedMem[center_H_offset+warpID*4+0] * sharedMem[Tx_offset+warpID*4+1] 
//					+ sharedMem[center_H_offset+warpID*4+1] * sharedMem[Ty_offset+warpID*4+1] 
//					+ sharedMem[center_H_offset+warpID*4+2] * sharedMem[Tz_offset+warpID*4+1]
//					+ f[node_id] 
//					* (z[node_id] * sharedMem[center_H_offset+warpID*4+0] 
//					-  x[node_id] * sharedMem[center_H_offset+warpID*4+2]) 
//					+ sharedMem[Ty_offset+warpID*4+3]);

//		fType s = - (sharedMem[center_H_offset+warpID*4+0] * sharedMem[Tx_offset+warpID*4+2] 
//					+ sharedMem[center_H_offset+warpID*4+1] * sharedMem[Ty_offset+warpID*4+2] 
//					+ sharedMem[center_H_offset+warpID*4+2] * sharedMem[Tz_offset+warpID*4+2]
//					+ f[node_id] 
//					* (x[node_id] * sharedMem[center_H_offset+warpID*4+1] 
//					-  y[node_id] * sharedMem[center_H_offset+warpID*4+0]) 
//					+ sharedMem[Tz_offset+warpID*4+3]);

//		F[node_id*4+0] =  sharedMem[p_u_offset+warpID*3+0] * p 
//						+ sharedMem[p_u_offset+warpID*3+1] * q 
//						+ sharedMem[p_u_offset+warpID*3+2] * s 
//						+ sharedMem[HV_offset+warpID*4+0];

//		F[node_id*4+1] =  sharedMem[p_v_offset+warpID*3+0] * p 
//						+ sharedMem[p_v_offset+warpID*3+1] * q 
//						+ sharedMem[p_v_offset+warpID*3+2] * s 
//						+ sharedMem[HV_offset+warpID*4+1];

//		F[node_id*4+2] =  sharedMem[p_w_offset+warpID*3+0] * p 
//						+ sharedMem[p_w_offset+warpID*3+1] * q 
//						+ sharedMem[p_w_offset+warpID*3+2] * s 
//						+ sharedMem[HV_offset+warpID*4+2];

//		F[node_id*4+3] = - (sharedMem[center_H_offset+warpID*4+0] * (sharedMem[Tx_offset+warpID*4+3] 
//						  - sharedMem[gradghm_offset+warpID*3+0])
//						  + sharedMem[center_H_offset+warpID*4+1] * (sharedMem[Ty_offset+warpID*4+3] 
//						  - sharedMem[gradghm_offset+warpID*3+1])
//						  + sharedMem[center_H_offset+warpID*4+2] * (sharedMem[Tz_offset+warpID*4+3] 
//						  - sharedMem[gradghm_offset+warpID*3+2])
//						  +(sharedMem[center_H_offset+warpID*4+3] + gh0 - ghm[node_id]) 
//						  * (sharedMem[Tx_offset+warpID*4+0] 
//						  + sharedMem[Ty_offset+warpID*4+1] 
//						  + sharedMem[Tz_offset+warpID*4+2]))
//						  + sharedMem[HV_offset+warpID*4+3];			
//	}
}

void evoke_evalCartRhs_fd(const fType* H_d,
			  const atm_struct* atm_d,
			  const DP_struct* DP_d,
			  const fType* gradghm_d,
			  fType* F_d, long long*  kernelTime){

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

	size_t sharedMemSize = (4*4*nWarpsPerBlock + 4*32*nWarpsPerBlock) * sizeof(fType);
//	sharedMemSize += nWarpsPerBlock * 32 * 4 * sizeof(fType);	// DPx, DPy, DPz, L
	sharedMemSize += nWarpsPerBlock * 32 * 4 * sizeof(fType);	// H[4] * 32 neighbors 
//	sharedMemSize += nWarpsPerBlock * 4 * sizeof(fType);		// H[4] for current stencil point
//	sharedMemSize += nWarpsPerBlock * 4 * 3 * sizeof(fType);	// p_u[3], p_v[3], p_w[3], gradghm[3]	

	int nBlocksPerGrid = (Nnodes*32 + BLOCK_SIZE - 1)/BLOCK_SIZE;

cudaProfilerStart();
long long start = getTime();
	// Launch kernel
	evalCartRhs_fd<<<nBlocksPerGrid, BLOCK_SIZE, sharedMemSize>>>(H_d, idx, DPx, DPy, DPz, L,
									 x, y, z, f, ghm,
									 p_u, p_v, p_w,
									 gradghm_d,
									 g, a, gh0, Nnodes, Nvar, Nnbr,
									 F_d);
	
	// wait for kernel to complete
	cudaDeviceSynchronize();
long long end = getTime();
*kernelTime += (end-start);

cudaProfilerStop();

}
