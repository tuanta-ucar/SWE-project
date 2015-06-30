#include <stdio.h>
#include <cuda_runtime.h>
#include <evalCartRhs_fd.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <timer.h>

#define BLOCK_SIZE 256

#define nWarpsPerBlock (BLOCK_SIZE/32)	// total number of warps per block

#define nMemLoadWarps 2					// number of warps dedicated to global memory load
#define nComputeWarps (nWarpsPerBlock-nMemLoadWarps)	// number of warps dedicated to computation
#define nIters 5					// number of iterations in each block

#define nNodesPerBlock ((nIters-1)*nComputeWarps)	// # nodes to be processed by a block - each compute warp is corresponding to 1 stencil center
#define nBlocksPerGrid (Nnodes/nNodesPerBlock + 1)	// # blocks per grid

#define computePerLoad  (nComputeWarps/nMemLoadWarps)	// ratio btw #compute_warps and #load_warps

// Compute offsets of intermediate vectors
#define Tx_offset 0
#define Ty_offset (Tx_offset + 4*nComputeWarps)
#define Tz_offset (Ty_offset + 4*nComputeWarps)
#define HV_offset (Tz_offset + 4*nComputeWarps)

#define Tx_i_offset  (HV_offset + 4*nComputeWarps)
#define Ty_i_offset  (Tx_i_offset + 32*nComputeWarps)
#define Tz_i_offset  (Ty_i_offset + 32*nComputeWarps)
#define HV_i_offset    (Tz_i_offset + 32*nComputeWarps)

// Compute offsets of data vectors
#define DPx_offset 0
#define DPy_offset (DPx_offset + 32*nComputeWarps) 
#define DPz_offset (DPy_offset + 32*nComputeWarps)
#define L_offset   (DPz_offset + 32*nComputeWarps)
#define nbr_H_offset (L_offset + 32*nComputeWarps)
#define center_H_offset (nbr_H_offset + 4*32*nComputeWarps) 
#define x_offset (center_H_offset + 4*nComputeWarps)
#define y_offset (x_offset + nComputeWarps)
#define z_offset (y_offset + nComputeWarps)
#define f_offset (z_offset + nComputeWarps)
#define ghm_offset (f_offset + nComputeWarps)

#define p_u_offset (ghm_offset + nComputeWarps)
#define p_v_offset (p_u_offset + 3*nComputeWarps)
#define p_w_offset (p_v_offset + 3*nComputeWarps)
#define gradghm_offset (p_w_offset + 3*nComputeWarps)

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
	// ================== Thread Layout =====================
	//	First nMemLoadWarps -> Load data from global mem -> shared mem at step (stp)
	//	Next nComputeWarps  -> Do computation on loaded data in shared mem at step (stp+1)
	//	Number of steps = nInters
	// ======================================================

	// ================== Compute index and offset ====================
	int tid = threadIdx.x % 32;	// thread id within a warp
	// ================================================================

	// Declare the shared memory space for this block
	extern __shared__ fType sharedMem[];

	int step = 0;

	while (step < nIters){
		if (threadIdx.x/32 < nMemLoadWarps){	// load warps 
			if (step != nIters-1){	// no need to load more at the last step
				// nMemLoadWarps load data for nComputeWarps at (step+1)
				
				// each load_warp loads data for (computePerLoad) compute_warps
				for (int k = 0; k < computePerLoad; k++){
					// id of node whose data are going to be loaded
					// node_id = blockID*nNodes/block + step*nNodes/step + loadWarpID*nNodes/load_warp + offset
					int loadWarpID = (threadIdx.x/32) % nMemLoadWarps;	// first nMemLoadWarps dedicated to load data 
					int node_id = blockIdx.x*nNodesPerBlock + step*nComputeWarps + loadWarpID*computePerLoad + k;
					
					// check boundary
					if (node_id >= Nnodes) break;

					// load idx, DPx, DPy, DPz and L
					// offset for 32-element vectors in shared memory (idx, DPx, DPy, DPz and L)
					int offset = (computePerLoad*32)*loadWarpID + k*32;	
					// fType* load_ptr = (fType*)load_space[step & 1];	// step % 2 == step & 1		

					fType* load_ptr;

					if (step % 2 == 0)
						load_ptr = sharedMem + 144*nComputeWarps;
					else 
						load_ptr = sharedMem + 421*nComputeWarps;			

					load_ptr[DPx_offset+offset+tid] = DPx[node_id*(Nnbr+1)+tid]; 
					load_ptr[DPy_offset+offset+tid] = DPy[node_id*(Nnbr+1)+tid]; 
					load_ptr[DPz_offset+offset+tid] = DPz[node_id*(Nnbr+1)+tid]; 
					load_ptr[L_offset+offset+tid] = L[node_id*(Nnbr+1)+tid]; 
					
					// Load H values for all neighbor nodes - offset for local_nbr_H[4*32*nComputeWarps]
					offset = (computePerLoad*32*4)*loadWarpID + k*32*4;
					int nbr_id = idx[node_id*(Nnbr+1)+tid];	// neighbor's id
					for (int var = 0; var < 4; var++)
						load_ptr[nbr_H_offset+offset+tid*4+var] = H[nbr_id*4+var];

					// Load H values for the stencil center - need 4 threads only
					offset = (computePerLoad*4)*loadWarpID + k*4;
					if (tid < 4)
						load_ptr[center_H_offset+offset+tid] = H[node_id*4+tid];
								
					// Load x, y, z, f and ghm - need 1 thread only
					offset = (computePerLoad*1)*loadWarpID + k;
					if (tid == 0){
						load_ptr[x_offset+offset+tid] = x[node_id];
						load_ptr[y_offset+offset+tid] = y[node_id];
						load_ptr[z_offset+offset+tid] = z[node_id];
						load_ptr[f_offset+offset+tid] = f[node_id];
						load_ptr[ghm_offset+offset+tid] = ghm[node_id];
					}

					// Load p_u, p_v, p_w and gradghm [3*nComputeWarps] - need only 3 threads
					offset = (computePerLoad*3)*loadWarpID + k*3;
					if (tid < 3){
						load_ptr[p_u_offset+offset+tid] = p_u[node_id*3+tid];
						load_ptr[p_v_offset+offset+tid] = p_v[node_id*3+tid];
						load_ptr[p_w_offset+offset+tid] = p_w[node_id*3+tid];
						load_ptr[gradghm_offset+offset+tid] = gradghm[node_id*3+tid];	
					}
				}		
			}
		} else {			
			// ********* implementation of compute warps *********
			if (step != 0){	// no need to do computation at step 0
				// nComputeWarps do computation
				// get id of node to be processed. Data have been already loaded at step-1
				int computeWarpID = (threadIdx.x/32 - nMemLoadWarps) % nComputeWarps;	// next nComputeWarps dedicated to do computation
				int node_id = blockIdx.x*nNodesPerBlock + (step-1)*nComputeWarps + computeWarpID;
				
				// check boundary
				if (node_id < Nnodes){ 
					// **** pointers to space used to store intermediate result duing computation phase  ****
					fType* load_ptr;					
					if (step % 2 == 1)
						load_ptr = sharedMem + 144*nComputeWarps;
					else 
						load_ptr = sharedMem + 421*nComputeWarps;

					#pragma unroll
					for (int k = 0; k < 4; k++){
						sharedMem[Tx_i_offset+computeWarpID*32+tid] = load_ptr[DPx_offset+32*computeWarpID+tid] 
															* load_ptr[nbr_H_offset+32*4*computeWarpID+tid*4+k];
						sharedMem[Ty_i_offset+computeWarpID*32+tid] = load_ptr[DPy_offset+32*computeWarpID+tid] 
															* load_ptr[nbr_H_offset+32*4*computeWarpID+tid*4+k];
						sharedMem[Tz_i_offset+computeWarpID*32+tid] = load_ptr[DPz_offset+32*computeWarpID+tid] 
															* load_ptr[nbr_H_offset+32*4*computeWarpID+tid*4+k];
						sharedMem[HV_i_offset+computeWarpID*32+tid] = load_ptr[L_offset+32*computeWarpID+tid] 
															* load_ptr[nbr_H_offset+32*4*computeWarpID+tid*4+k];

						sumReductionInWarp(sharedMem+Tx_i_offset+computeWarpID*32, tid);
						sumReductionInWarp(sharedMem+Ty_i_offset+computeWarpID*32, tid);
						sumReductionInWarp(sharedMem+Tz_i_offset+computeWarpID*32, tid);
						sumReductionInWarp(sharedMem+HV_i_offset+computeWarpID*32, tid);

						if (tid == 0){
							sharedMem[Tx_offset+computeWarpID*4+k] = (sharedMem+Tx_i_offset+computeWarpID*32)[0];
							sharedMem[Ty_offset+computeWarpID*4+k] = (sharedMem+Ty_i_offset+computeWarpID*32)[0];
							sharedMem[Tz_offset+computeWarpID*4+k] = (sharedMem+Tz_i_offset+computeWarpID*32)[0];
							sharedMem[HV_offset+computeWarpID*4+k] = (sharedMem+HV_i_offset+computeWarpID*32)[0];
						}
					}	

					// compute p, q, s
					if (tid == 0){
						fType p =-(load_ptr[center_H_offset+4*computeWarpID+0] * sharedMem[Tx_offset+computeWarpID*4+0] 
							      + load_ptr[center_H_offset+4*computeWarpID+1] * sharedMem[Ty_offset+computeWarpID*4+0] 
							      + load_ptr[center_H_offset+4*computeWarpID+2] * sharedMem[Tz_offset+computeWarpID*4+0] 
							      + load_ptr[f_offset+computeWarpID] 
							      * (load_ptr[y_offset+computeWarpID] * load_ptr[center_H_offset+4*computeWarpID+2] 
							      -  load_ptr[z_offset+computeWarpID] * load_ptr[center_H_offset+4*computeWarpID+1]) 
							      + sharedMem[Tx_offset+computeWarpID*4+3]);

						fType q =-(load_ptr[center_H_offset+4*computeWarpID+0] * sharedMem[Tx_offset+computeWarpID*4+1] 
							      + load_ptr[center_H_offset+4*computeWarpID+1] * sharedMem[Ty_offset+computeWarpID*4+1] 
							      + load_ptr[center_H_offset+4*computeWarpID+2] * sharedMem[Tz_offset+computeWarpID*4+1]
							      + load_ptr[f_offset+computeWarpID] 
							      *(load_ptr[z_offset+computeWarpID] * load_ptr[center_H_offset+4*computeWarpID+0] 
							      - load_ptr[x_offset+computeWarpID] * load_ptr[center_H_offset+4*computeWarpID+2]) 
							      + sharedMem[Ty_offset+computeWarpID*4+3]);

						fType s =-(load_ptr[center_H_offset+4*computeWarpID+0] * sharedMem[Tx_offset+computeWarpID*4+2] 
							      + load_ptr[center_H_offset+4*computeWarpID+1] * sharedMem[Ty_offset+computeWarpID*4+2] 
							      + load_ptr[center_H_offset+4*computeWarpID+2] * sharedMem[Tz_offset+computeWarpID*4+2]
							      + load_ptr[f_offset+computeWarpID] 
							      * (load_ptr[x_offset+computeWarpID] * load_ptr[center_H_offset+4*computeWarpID+1] 
							      -  load_ptr[y_offset+computeWarpID] * load_ptr[center_H_offset+4*computeWarpID+0]) 
							      + sharedMem[Tz_offset+computeWarpID*4+3]);
						
						F[node_id*4+0] =  load_ptr[p_u_offset+3*computeWarpID+0] * p 
							       		+ load_ptr[p_u_offset+3*computeWarpID+1] * q 
							       		+ load_ptr[p_u_offset+3*computeWarpID+2] * s + sharedMem[HV_offset+computeWarpID*4+0];
						
						F[node_id*4+1] =  load_ptr[p_v_offset+3*computeWarpID+0] * p 
							       		+ load_ptr[p_v_offset+3*computeWarpID+1] * q 
							       		+ load_ptr[p_v_offset+3*computeWarpID+2] * s + sharedMem[HV_offset+computeWarpID*4+1];

						F[node_id*4+2] =  load_ptr[p_w_offset+3*computeWarpID+0] * p 
							       		+ load_ptr[p_w_offset+3*computeWarpID+1] * q 
							       		+ load_ptr[p_w_offset+3*computeWarpID+2] * s + sharedMem[HV_offset+computeWarpID*4+2];
						
						F[node_id*4+3] = - (load_ptr[center_H_offset+4*computeWarpID+0] * 
										 (sharedMem[Tx_offset+computeWarpID*4+3] - load_ptr[gradghm_offset+3*computeWarpID+0])
								  		 + load_ptr[center_H_offset+4*computeWarpID+1] * 
										 (sharedMem[Ty_offset+computeWarpID*4+3] - load_ptr[gradghm_offset+3*computeWarpID+1])
								  		 + load_ptr[center_H_offset+4*computeWarpID+2] * 
										 (sharedMem[Tz_offset+computeWarpID*4+3] - load_ptr[gradghm_offset+3*computeWarpID+2])
								  		 +(load_ptr[center_H_offset+4*computeWarpID+3] + gh0 - load_ptr[ghm_offset+computeWarpID]) 
								  		 *(sharedMem[Tx_offset+computeWarpID*4+0] + sharedMem[Ty_offset+computeWarpID*4+1] 
										 + sharedMem[Tz_offset+computeWarpID*4+2]))
								  		 + sharedMem[HV_offset+computeWarpID*4+3];	
					}		
				}
			}
		}

		step++;

		__syncthreads();  // synchronize two groups of warps
	}
}

void evoke_evalCartRhs_fd(const fType* H_d,
			  const atm_struct* atm_d,
			  const DP_struct* DP_d,
			  const fType* gradghm_d,
			  fType* F_d, long long* kernelTime){

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

	// Shared memory space = Space for intermediate results + 2 * Space for input data from global mem
	size_t sharedMemSize = (4*4 + 4*32) * nComputeWarps * sizeof(fType); // shared memory space for intermediate results
	sharedMemSize += 2 * nComputeWarps * 32 * 4 * sizeof(fType);	// DPx, DPy, DPz, L
	sharedMemSize += 2 * nComputeWarps * 32 * 4 * sizeof(fType);	// H[4] * 32 neighbors 
	sharedMemSize += 2 * nComputeWarps * 4 * sizeof(fType);	// H[4] for current stencil point
	sharedMemSize += 2 * nComputeWarps * 5 * sizeof(fType);	// x[1], y[1], z[1], f[1], ghm[1]
	sharedMemSize += 2 * nComputeWarps * 4 * 3 * sizeof(fType);	// p_u[3], p_v[3], p_w[3], gradghm[3]

	/* Shared memory layout
	 *	<<<intermediate_result_space>>> + <<<load_space_1 (p1)>>> + <<<load_space_2 (p2)>>>
	 *		
	 *		intermediate_result_space = Tx[4*nWarps] + Ty[4*nWarps] + Tz[4*nWarps] + HV[4*nWarps] 
	 *					  + Tx_i[32*nWarps] + Ty_i[32*nWarps] + Tz_i[32*nWarps] + HV[32*nWarps] 
	 *
	 *	        load_space_1 = 	DPx[32*nWarps] + DPy[32*nWarps] + DPz[32*nWarps] + L[32*nWarps]
	 *				+ nbr_H[4*32*nWarps] + center_H[4*nWarps] 
	 *				+ x[1*nWarps] + y[1*nWarps] + z[1*nWarps] + f[1*nWarps] + ghm[1*nWarps]
	 *				+ p_u[3*nWarps] + p_v[3*nWarps] + p_w[3*nWarps] + gradghm[3*nWarps] 
	 *
 	 */

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
