#include <CUDA_utils.h>

void setupDevices(){
        // Error code
        cudaError_t err = cudaSuccess;

        // Get device list
        int deviceCount = 0;

        err = cudaGetDeviceCount(&deviceCount);
        checkError(err);

        for (int i = 0; i < deviceCount; i++){
                cudaDeviceProp props;
                err = cudaGetDeviceProperties(&props, i);
                printf("Device name: %s\n", props.name);
        }

        // Set device
        int deviceID = 0;
        err = cudaSetDevice(deviceID);  // choose device 0 by default
        checkError(err);

        printf("Chosen device: %d\n", deviceID);
}

void allocateDevArray(void** devPtr, size_t size){
	cudaError_t err = cudaSuccess;
	err = cudaMalloc(devPtr, size);
	checkError(err);
}

void copyGPUtoCPU(void* dest, const void* source, size_t size){
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(dest, source, size, cudaMemcpyDeviceToHost);
	checkError(err);
}

void copyCPUtoGPU(void* dest, const void* source, size_t size){
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice);
	checkError(err);
}

void initializeDevInputs(const fType* H, const atm_struct* atm, const DP_struct* DP, const fType* gradghm, const fType* F,
                   void** H_d, atm_struct* atm_d, DP_struct* DP_d, void** gradghm_d, void** F_d, void** K_d){
	int Nnodes = atm->Nnodes;
	int Nvar = atm->Nvar;
	int Nnbr = atm->Nnbr;

	// Copy constants from atm -> atm_d
	atm_d->g = atm->g;
	atm_d->a = atm->a;
	atm_d->gh0 = atm->gh0;
	atm_d->Nnodes = atm->Nnodes;
	atm_d->Nvar = atm->Nvar;
	atm_d->Nnbr = atm->Nnbr;
 
	// Copy H to H_d
	allocateDevArray(H_d, Nnodes*Nvar*sizeof(fType));
	copyCPUtoGPU((fType*)(*H_d), H, Nnodes*Nvar*sizeof(fType));

	// Copy atm arrays to atm_d arrays
	atm_d->x = NULL;
	atm_d->y = NULL;
	atm_d->z = NULL;
	atm_d->f = NULL;

	allocateDevArray((void**)&(atm_d->x), Nnodes*sizeof(fType));
	copyCPUtoGPU(atm_d->x, atm->x, Nnodes*sizeof(fType));
	
	allocateDevArray((void**)&(atm_d->y), Nnodes*sizeof(fType));
	copyCPUtoGPU(atm_d->y, atm->y, Nnodes*sizeof(fType));

	allocateDevArray((void**)&(atm_d->z), Nnodes*sizeof(fType));
	copyCPUtoGPU(atm_d->z, atm->z, Nnodes*sizeof(fType));

	allocateDevArray((void**)&(atm_d->f), Nnodes*sizeof(fType));
	copyCPUtoGPU(atm_d->f, atm->f, Nnodes*sizeof(fType));

	atm_d->ghm = NULL;
	atm_d->p_u = NULL;
	atm_d->p_v = NULL;
	atm_d->p_w = NULL;
	
	allocateDevArray((void**)&(atm_d->ghm), Nnodes*sizeof(fType));
	copyCPUtoGPU(atm_d->ghm, atm->ghm, Nnodes*sizeof(fType));

	allocateDevArray((void**)&(atm_d->p_u), Nnodes*3*sizeof(fType));
	copyCPUtoGPU(atm_d->p_u, atm->p_u, Nnodes*3*sizeof(fType));
	
	allocateDevArray((void**)&(atm_d->p_v), Nnodes*3*sizeof(fType));
	copyCPUtoGPU(atm_d->p_v, atm->p_v, Nnodes*3*sizeof(fType));
	
	allocateDevArray((void**)&(atm_d->p_w), Nnodes*3*sizeof(fType));
	copyCPUtoGPU(atm_d->p_w, atm->p_w, Nnodes*3*sizeof(fType));

	// Copy DP arrays to DP_d arrays
	DP_d->idx = NULL;
	DP_d->DPx = NULL;
	DP_d->DPy = NULL;
	DP_d->DPz = NULL;
	DP_d->L = NULL;

	allocateDevArray((void**)&(DP_d->idx), Nnodes*(Nnbr+1)*sizeof(int));
	copyCPUtoGPU(DP_d->idx, DP->idx, Nnodes*(Nnbr+1)*sizeof(int));
	
	allocateDevArray((void**)&(DP_d->DPx), Nnodes*(Nnbr+1)*sizeof(fType));
	copyCPUtoGPU(DP_d->DPx, DP->DPx, Nnodes*(Nnbr+1)*sizeof(fType));

	allocateDevArray((void**)&(DP_d->DPy), Nnodes*(Nnbr+1)*sizeof(fType));
	copyCPUtoGPU(DP_d->DPy, DP->DPy, Nnodes*(Nnbr+1)*sizeof(fType));

	allocateDevArray((void**)&(DP_d->DPz), Nnodes*(Nnbr+1)*sizeof(fType));
	copyCPUtoGPU(DP_d->DPz, DP->DPz, Nnodes*(Nnbr+1)*sizeof(fType));

	allocateDevArray((void**)&(DP_d->L), Nnodes*(Nnbr+1)*sizeof(fType));
	copyCPUtoGPU(DP_d->L, DP->L, Nnodes*(Nnbr+1)*sizeof(fType));

	// copy gradghm to gradghm_d
	allocateDevArray(gradghm_d, Nnodes*3*sizeof(fType));
	copyCPUtoGPU((fType*)(*gradghm_d), gradghm, Nnodes*3*sizeof(fType));

	// Allocate F_d
	allocateDevArray(F_d, Nnodes*Nvar*sizeof(fType));

	// Allocate K_d
	allocateDevArray(K_d, Nnodes*Nvar*sizeof(fType));
}

void freeCudaMem(fType* H_d, atm_struct* atm_d, DP_struct* DP_d, fType* gradghm_d, fType* F_d, fType* K_d){
        // Error code
        cudaError_t err = cudaSuccess;

	err = cudaFree(H_d);
	checkError(err);

	err = cudaFree(atm_d->x);
	checkError(err);
	err = cudaFree(atm_d->y);
	checkError(err);
	err = cudaFree(atm_d->z);
	checkError(err);
	err = cudaFree(atm_d->f);
	checkError(err);
	err = cudaFree(atm_d->ghm);
	checkError(err);
	err = cudaFree(atm_d->p_u);
	checkError(err);
	err = cudaFree(atm_d->p_v);
	checkError(err);
	err = cudaFree(atm_d->p_w);
	checkError(err);

	free(atm_d);

	err = cudaFree(DP_d->idx);
	checkError(err);
	err = cudaFree(DP_d->DPx);
	checkError(err);
	err = cudaFree(DP_d->DPy);
	checkError(err);
	err = cudaFree(DP_d->DPz);
	checkError(err);
	err = cudaFree(DP_d->L);
	checkError(err);
	
	free(DP_d);

	err = cudaFree(gradghm_d);
	checkError(err);
	
	err = cudaFree(F_d);
	checkError(err);

	err = cudaFree(K_d);
	checkError(err);
}

void checkError(cudaError_t err){ 
        if (err != cudaSuccess){
                printf("Error: %s\n",cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }
}
