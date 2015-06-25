#include <stdio.h>
#include <cuda_runtime.h>
#include <evalCartRhs_fd.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

#define BLOCK_SIZE 256

#define nWarpsPerBlock (BLOCK_SIZE/32)	// total number of warps per block

#define nMemLoadWarps 2					// number of warps dedicated to global memory load
#define nComputeWarps (nWarpsPerBlock-nMemLoadWarps)	// number of warps dedicated to computation
#define nIters 10					// number of iterations in each block

#define nNodesPerBlock ((nIters-1)*nComputeWarps)	// # nodes to be processed by a block - each compute warp is corresponding to 1 stencil center
#define nBlocksPerGrid (Nnodes/nNodesPerBlock + 1)	// # blocks per grid

#define computePerLoad  (nComputeWarps/nMemLoadWarps)	// ratio btw #compute_warps and #load_warps

// sum all 32 elements in array
// the sum is stored in array[0]
__device__ void sumReductionInWarp(double array[32], int tid){
	for (int s = 1; s < 32; s *= 2){
		if (tid % (2*s) == 0)
			array[tid] += array[tid+s];
	}
}

__global__ void evalCartRhs_fd(	const double* H,
				const int* idx,	const double* DPx, const double* DPy, const double* DPz, const double* L,   
				const double* x, const double* y, const double* z, const double* f, const double* ghm,
				const double* p_u, const double* p_v, const double* p_w, const double* gradghm,
				
				const double g, const double a, const double gh0, // constants
				const int Nnodes, const int Nvar, const int Nnbr,
				
				double* F	   // output
){
	// ================== Thread Layout =====================
	//	First nMemLoadWarps -> Load data from global mem -> shared mem at step (stp)
	//	Next nComputeWarps  -> Do computation on loaded data in shared mem at step (stp+1)
	//	Number of steps = nInters
	// ======================================================

	// ================== Compute index and offset ====================
	int tid = threadIdx.x % 32;	// thread id within a warp
	int warp_id = threadIdx.x / 32;	// warp id within a block
	int load_warp_id = warp_id % nMemLoadWarps;	// first nMemLoadWarps dedicated to load data 
	int compute_warp_id = (warp_id - nMemLoadWarps) % nComputeWarps;	// next nComputeWarps dedicated to do computation
	// ================================================================

	// Declare the shared memory space for this block
	extern __shared__ double sharedMem[];

	// ======= Get pointers to different regions in sharedMem[] =========
		// **** intermediate_result_space ****
		double* inter_space = sharedMem;	// beginning of inter_space part

		double* Tx = (double*)(inter_space + compute_warp_id*4);	             // Tx[4]
		double* Ty = (double*)(inter_space + 1*4*nComputeWarps + compute_warp_id*4); // Ty[4]
		double* Tz = (double*)(inter_space + 2*4*nComputeWarps + compute_warp_id*4); // Tz[4]
		double* HV = (double*)(inter_space + 3*4*nComputeWarps + compute_warp_id*4); // HV[4]

		double* Tx_i = (double*)(inter_space + 4*4*nComputeWarps + compute_warp_id*32);			    // Tx_i[32]
		double* Ty_i = (double*)(inter_space + 4*4*nComputeWarps + 1*32*nComputeWarps + compute_warp_id*32);// Ty_i[32]
		double* Tz_i = (double*)(inter_space + 4*4*nComputeWarps + 2*32*nComputeWarps + compute_warp_id*32);// Tz_i[32]
		double* HV_i = (double*)(inter_space + 4*4*nComputeWarps + 3*32*nComputeWarps + compute_warp_id*32);// HV_i[32]
	
		// **** p1_space ****
		double* load_space[2];

		load_space[0] = inter_space + 4*4*nComputeWarps + 4*32*nComputeWarps;
		load_space[1] = load_space[0] + nComputeWarps * (16 + 32*4 + 32*4 + 4 + 5 + 4*3); // first 32*4 bytes represent int numbers
			
	// ==================================================================
	
	int step = 0;
	int load_phase, compute_phase;

	while (step < nIters){
		// load warps and compute warps 
		load_phase = step % 2;		
		compute_phase = (step+1) % 2;
	
		if (warp_id < nMemLoadWarps){	// load warps 
			if (step != nIters-1){	// no need to load more at the last step
				// nMemLoadWarps load data for nComputeWarps at (step+1)
				
				// each load_warp loads data for (computePerLoad) compute_warps
				for (int k = 0; k < computePerLoad; k++){
					// id of node whose data are going to be loaded
					// node_id = blockID*nNodes/block + step*nNodes/step + load_warp_id*nNodes/load_warp + offset
					int node_id = blockIdx.x*nNodesPerBlock + step*nComputeWarps + load_warp_id*computePerLoad + k;
					
					// TODO check boundary
					if (node_id >= Nnodes) break;

					// ================ Load all local pointers =====================
					int* local_idx = (int*)load_space[load_phase];	// head pointer of a compute partition
					double* local_DPx = (double*)&local_idx[32*nComputeWarps];
					double* local_DPy = (double*)&local_DPx[32*nComputeWarps];
					double* local_DPz = (double*)&local_DPy[32*nComputeWarps];
					double* local_L = (double*)&local_DPz[32*nComputeWarps];
					double* local_nbr_H = (double*)&local_L[32*nComputeWarps];
					double* local_center_H = (double*)&local_nbr_H[4*32*nComputeWarps];
					double* local_x = (double*)&local_center_H[4*nComputeWarps];
					double* local_y = (double*)&local_x[1*nComputeWarps];
					double* local_z = (double*)&local_y[1*nComputeWarps];
					double* local_f = (double*)&local_z[1*nComputeWarps];
					double* local_ghm = (double*)&local_f[1*nComputeWarps];
					double* local_p_u = (double*)&local_ghm[1*nComputeWarps];
					double* local_p_v = (double*)&local_p_u[3*nComputeWarps];
					double* local_p_w = (double*)&local_p_v[3*nComputeWarps];
					double* local_gradghm = (double*)&local_p_w[3*nComputeWarps];
					// ==============================================================

					// load idx, DPx, DPy, DPz and L
					// offset for 32-element vectors in shared memory (idx, DPx, DPy, DPz and L)
					int offset = (computePerLoad*32)*load_warp_id + k*32;	

					local_idx[offset+tid] = idx[node_id*(Nnbr+1)+tid];
					int nbr_id = local_idx[offset+tid];	// neighbor's id

					local_DPx[offset+tid] = DPx[node_id*(Nnbr+1)+tid]; 
					local_DPy[offset+tid] = DPy[node_id*(Nnbr+1)+tid]; 
					local_DPz[offset+tid] = DPz[node_id*(Nnbr+1)+tid]; 
					local_L[offset+tid] = L[node_id*(Nnbr+1)+tid]; 
					
					// Load H values for all neighbor nodes - offset for local_nbr_H[4*32*nComputeWarps]
					offset = (computePerLoad*32*4)*load_warp_id + k*32*4;
					for (int var = 0; var < 4; var++)
						local_nbr_H[offset+tid*4+var] = H[nbr_id*4+var];

					// Load H values for the stencil center - need 4 threads only
					offset = (computePerLoad*4)*load_warp_id + k*4;
					if (tid < 4)
						local_center_H[offset+tid] = H[node_id*4+tid];

					// Load x, y, z, f and ghm - need 1 thread only
					offset = (computePerLoad*1)*load_warp_id + k;
					if (tid == 0){
						local_x[offset+tid] = x[node_id];
						local_y[offset+tid] = y[node_id];
						local_z[offset+tid] = z[node_id];
						local_f[offset+tid] = f[node_id];
						local_ghm[offset+tid] = ghm[node_id];
					}

					// Load p_u, p_v, p_w and gradghm [3*nComputeWarps] - need only 3 threads
					offset = (computePerLoad*3)*load_warp_id + k*3;
					if (tid < 3){
						local_p_u[offset+tid] = p_u[node_id*3+tid];
						local_p_v[offset+tid] = p_v[node_id*3+tid];
						local_p_w[offset+tid] = p_w[node_id*3+tid];
						local_gradghm[offset+tid] = gradghm[node_id*3+tid];	
					}
				}		
			}
		} else {			
			// ********* implementation of compute warps *********
			if (step != 0){	// no need to do computation at step 0
				// nComputeWarps do computation
				// get id of node to be processed. Data have been already loaded at step-1
				int node_id = blockIdx.x*nNodesPerBlock + (step-1)*nComputeWarps + compute_warp_id;
				
				// TODO check boundary
				if (node_id >= Nnodes) break;
/*
				// ================ Load all local pointers =====================
				int* local_idx = (int*)load_space[compute_phase];	// head pointer of a compute partition
				double* local_DPx = (double*)&local_idx[32*nComputeWarps];
				double* local_DPy = (double*)&local_DPx[32*nComputeWarps];
				double* local_DPz = (double*)&local_DPy[32*nComputeWarps];
				double* local_L = (double*)&local_DPz[32*nComputeWarps];
				double* local_nbr_H = (double*)&local_L[32*nComputeWarps];
				double* local_center_H = (double*)&local_nbr_H[4*32*nComputeWarps];
				double* local_x = (double*)&local_center_H[4*nComputeWarps];
				double* local_y = (double*)&local_x[1*nComputeWarps];
				double* local_z = (double*)&local_y[1*nComputeWarps];
				double* local_f = (double*)&local_z[1*nComputeWarps];
				double* local_ghm = (double*)&local_f[1*nComputeWarps];
				double* local_p_u = (double*)&local_ghm[1*nComputeWarps];
				double* local_p_v = (double*)&local_p_u[3*nComputeWarps];
				double* local_p_w = (double*)&local_p_v[3*nComputeWarps];
				double* local_gradghm = (double*)&local_p_w[3*nComputeWarps];
				// ==============================================================
		
				int offset = 32*compute_warp_id; // for vector size of 32 

				for (int k = 0; k < 4; k++){
					Tx_i[tid] = local_DPx[offset+tid] * local_nbr_H[offset+tid*4+k];
					Ty_i[tid] = local_DPy[offset+tid] * local_nbr_H[offset+tid*4+k];
					Tz_i[tid] = local_DPz[offset+tid] * local_nbr_H[offset+tid*4+k];
					HV_i[tid] = local_L  [offset+tid] * local_nbr_H[offset+tid*4+k];

					sumReductionInWarp(Tx_i, tid);
					sumReductionInWarp(Ty_i, tid);
					sumReductionInWarp(Tz_i, tid);
					sumReductionInWarp(HV_i, tid);

					if (tid == 0){
						Tx[k] = Tx_i[0];
						Ty[k] = Ty_i[0];
						Tz[k] = Tz_i[0];
						HV[k] = HV_i[0];
					} 
				}	
		
				offset = 4*compute_warp_id; // for vector size of 4

				// compute p, q, s
				if (tid == 0){
					double p = - ( 	local_center_H[offset+0] * Tx[0] 
						      + local_center_H[offset+1] * Ty[0] 
						      + local_center_H[offset+2] * Tz[0] 
						      + local_f[compute_warp_id] 
						      * (local_y[compute_warp_id] * local_center_H[offset+2] 
						      -  local_z[compute_warp_id] * local_center_H[offset+1]) 
						      + Tx[3]);

					double q = - (	local_center_H[offset+0] * Tx[1] 
						      + local_center_H[offset+1] * Ty[1] 
						      + local_center_H[offset+2] * Tz[1]
						      + local_f[compute_warp_id] 
						      * (local_z[compute_warp_id] * local_center_H[offset+0] 
						      -  local_x[compute_warp_id] * local_center_H[offset+2]) 
						      + Ty[3]);

					double s = - (	local_center_H[offset+0] * Tx[2] 
						      + local_center_H[offset+1] * Ty[2] 
						      + local_center_H[offset+2] * Tz[2]
						      + local_f[compute_warp_id] 
						      * (local_x[compute_warp_id] * local_center_H[offset+1] 
						      -  local_y[compute_warp_id] * local_center_H[offset+0]) 
						      + Tz[3]);

					int offset_3 = 3*compute_warp_id; // for vector size of 3

					F[node_id*4+0] = local_p_u[offset_3+0] * p 
						       + local_p_u[offset_3+1] * q 
						       + local_p_u[offset_3+2] * s + HV[0];
					F[node_id*4+1] = local_p_v[offset_3+0] * p 
						       + local_p_v[offset_3+1] * q 
						       + local_p_v[offset_3+2] * s + HV[1];
					F[node_id*4+2] = local_p_w[offset_3+0] * p 
						       + local_p_w[offset_3+1] * q 
						       + local_p_w[offset_3+2] * s + HV[2];
					F[node_id*4+3] = - (local_center_H[offset+0] * (Tx[3] - local_gradghm[offset_3+0])
						      	  + local_center_H[offset+1] * (Ty[3] - local_gradghm[offset_3+1])
						          + local_center_H[offset+2] * (Tz[3] - local_gradghm[offset_3+2])
						      	  +(local_center_H[offset+3] + gh0 - local_ghm[compute_warp_id]) 
							  * (Tx[0] + Ty[1] + Tz[2]))
						          + HV[3];			
				}
*/
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

	// Shared memory space = Space for intermediate results + 2 * Space for input data from global mem
	size_t sharedMemSize = (4*4 + 4*32) * nComputeWarps * sizeof(double); // shared memory space for intermediate results
	sharedMemSize += 2 * nComputeWarps * 32 * sizeof(int);	// idx[]
	sharedMemSize += 2 * nComputeWarps * 32 * 4 * sizeof(double);	// DPx, DPy, DPz, L
	sharedMemSize += 2 * nComputeWarps * 32 * 4 * sizeof(double);	// H[4] * 32 neighbors 
	sharedMemSize += 2 * nComputeWarps * 4 * sizeof(double);	// H[4] for current stencil point
	sharedMemSize += 2 * nComputeWarps * 5 * sizeof(double);	// x[1], y[1], z[1], f[1], ghm[1]
	sharedMemSize += 2 * nComputeWarps * 4 * 3 * sizeof(double);	// p_u[3], p_v[3], p_w[3], gradghm[3]

	/* Shared memory layout
	 *	<<<intermediate_result_space>>> + <<<load_space_1 (p1)>>> + <<<load_space_2 (p2)>>>
	 *		
	 *		intermediate_result_space = Tx[4*nWarps] + Ty[4*nWarps] + Tz[4*nWarps] + HV[4*nWarps] 
	 *					  + Tx_i[32*nWarps] + Ty_i[32*nWarps] + Tz_i[32*nWarps] + HV[32*nWarps] 
	 *
	 *	        load_space_1 = idx[32*nWarps] + DPx[32*nWarps] + DPy[32*nWarps] + DPz[32*nWarps] + L[32*nWarps]
	 *				+ H[4*32*nWarps] + s_H[4*nWarps] 
	 *				+ x[1*nWarps] + y[1*nWarps] + z[1*nWarps] + f[1*nWarps] + ghm[1*nWarps]
	 *				+ p_u[3*nWarps] + p_v[3*nWarps] + p_w[3*nWarps] + gradghm[3*nWarps] 
	 *
 	 */
	
	// Launch kernel
	evalCartRhs_fd<<<nBlocksPerGrid, BLOCK_SIZE, sharedMemSize>>>(H_d, idx, DPx, DPy, DPz, L,
									 x, y, z, f, ghm,
									 p_u, p_v, p_w,
									 gradghm_d,
									 g, a, gh0, Nnodes, Nvar, Nnbr,
									 F_d);
	
	// wait for kernel to complete
	cudaDeviceSynchronize();
}
