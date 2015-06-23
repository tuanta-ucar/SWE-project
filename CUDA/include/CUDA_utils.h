#ifndef CUDA_UTILS
#define CUDA_UTILS

#include <cuda_runtime.h>
#include <config.h>
#include <cuda.h>
// print device list and choose device
void setupDevices();

void allocateDevArray(void** devPtr, size_t size);

void copyGPUtoCPU(void* dest, const void* source, size_t size);

void copyCPUtoGPU(void* dest, const void* source, size_t size);

void initializeDevInputs(const fType* H, const atm_struct* atm, const DP_struct* DP, const fType* gradghm, const fType* F,
                   void** H_d, atm_struct* atm_d, DP_struct* DP_d, void** gradghm_d, void** F_d, void** K_d);

void checkError(cudaError_t err);

void freeCudaMem(fType* H_d, atm_struct* atm_d, DP_struct* DP_d, fType* gradghm_d, fType* F_d, fType* K_d);

#endif
