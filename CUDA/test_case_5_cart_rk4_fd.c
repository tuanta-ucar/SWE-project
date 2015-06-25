#include <readInputs.h>
#include <evalCartRhs_fd.h>
#include <reorderNodes.h>
#include <CUDA_utils.h>

void computeK ( const fType* H, const fType* F, 
		const fType dt, const fType coefficient,
		const int Nnodes, const int Nvar,
		const fType d_coefficient, const int d_num,
		fType* K, fType* d);

int main(){
	// *********** TODO CUDA device setup ****************
	setupDevices();
	// ***************************************************

	// constants
	const fType gamma = -6.4e-22;
	const int tend = 15;			// days
	const int nsteps = tend*24*3600;	// seconds
	const fType dt = 90.0;			// time step in second
	
	// ***** read inputs *****
	// read atm variables
	atm_struct* atm = read_atm();	

	int Nnodes = atm->Nnodes;
	int Nnbr = atm->Nnbr;
	int Nvar = atm->Nvar;

	// read H
	fType* H = read_H(Nnodes);	

	// read idx, DPx, DPy, DPz and L
	DP_struct* DP = read_DPs(Nnodes, Nnbr, gamma, atm->a);

	// read gradghm
	fType* gradghm = read_gradghm(Nnodes);

	// reorder node indices. mapping[new_id] = old_id
	int* mapping = reorderNodes(H, DP, atm, gradghm);
	
	fType* F = (fType*) malloc (sizeof(fType) * Nnodes * Nvar);
	fType* K = (fType*) malloc (sizeof(fType) * Nnodes * Nvar);
	fType* d = (fType*) calloc (Nnodes*Nvar, sizeof(fType));

	// ************* TODO device memory space setup *****************
	atm_struct* atm_d = (atm_struct*) malloc (sizeof(atm_struct));
	fType* H_d = NULL;
	DP_struct* DP_d = (DP_struct*) malloc (sizeof(DP_struct));
	fType* gradghm_d = NULL;
	fType* F_d = NULL;
	fType* K_d = NULL;

	// Copy data to GPU
	initializeDevInputs(H,   atm,   DP,   gradghm,   F, 
                      		(void**)&H_d, atm_d, DP_d, (void**)&gradghm_d, (void**)&F_d, (void**)&K_d);
	// **************************************************************


	// ***** main loop *****
	for (int nt = 1; nt <= 1; nt++){   // tend*24*3600
		printf("Step %d\n", nt);

// -----------------------
		memcpy(K, H, sizeof(fType) * Nnodes * Nvar); // K = H
		
		// copy K -> K_d 
		copyCPUtoGPU(K_d, K, sizeof(fType) * Nnodes * Nvar);
		
		// evoke evalCartRhs_fd kernel
		evoke_evalCartRhs_fd(K_d, atm_d, DP_d, gradghm_d, F_d);

		// copy F_d -> F 
		copyGPUtoCPU(F, F_d, sizeof(fType)*Nnodes*Nvar);

		// Compute K
		computeK(H, F, dt, 0.5, Nnodes, Nvar, 1.0, 1, K, d);
// ------------------------
		// copy K -> K_d 
		copyCPUtoGPU(K_d, K, sizeof(fType) * Nnodes * Nvar);

		// evoke evalCartRhs_fd kernel
		evoke_evalCartRhs_fd(K_d, atm_d, DP_d, gradghm_d, F_d);

		// copy F_d -> F 
		copyGPUtoCPU(F, F_d, sizeof(fType)*Nnodes*Nvar);

		for (int i = 0; i < Nnodes*Nvar; i++)
			printf("%f\n", F[i]);
/*
		computeK(H, F, dt, 0.5, Nnodes, Nvar, 2.0, 2, K, d);
// ------------------------
		// copy K -> K_d 
		copyCPUtoGPU(K_d, K, sizeof(fType) * Nnodes * Nvar);

		// evoke evalCartRhs_fd kernel
		evoke_evalCartRhs_fd(K_d, atm_d, DP_d, gradghm_d, F_d);

		// copy F_d -> F 
		copyGPUtoCPU(F, F_d, sizeof(fType)*Nnodes*Nvar);

		computeK(H, F, dt, 1.0, Nnodes, Nvar, 2.0, 3, K, d);
// ------------------------
		// copy K -> K_d 
		copyCPUtoGPU(K_d, K, sizeof(fType) * Nnodes * Nvar);

		// evoke evalCartRhs_fd kernel
		evoke_evalCartRhs_fd(K_d, atm_d, DP_d, gradghm_d, F_d);

		// copy F_d -> F 
		copyGPUtoCPU(F, F_d, sizeof(fType)*Nnodes*Nvar);

		computeK(H, F, dt, 1.0, Nnodes, Nvar, 1.0, 4, K, d);
// ------------------------
		// update H
		for (int i = 0; i < Nnodes * Nvar; i++){
			H[i] += (1.0/6.0) * d[i];
		}
*/
	}
/*
	// ======= DEBUGGING =========
	int count = 0;	
	FILE* file_ptr = fopen("H_debug.bin", "r");

	double* correctH = (double*) malloc (sizeof(double)*atm->Nnodes * atm->Nvar);
	fread(correctH, sizeof(double), atm->Nnodes * atm->Nvar, file_ptr);
	fclose(file_ptr);
	
	for (int i = 0; i < atm->Nnodes; i++){
		for (int j = 0; j < atm->Nvar; j++){
			double abs_err = fabs(H[i*4+j] - correctH[mapping[i]*4+j]);
			//if (abs_err < 1E-10){
				printf("%d %d %.16f %.16f\n", i/4, i%4, H[i*4+j], correctH[mapping[i]*4+j]);
				count++;
			//}
		}
	}

	if (count == 0)	
		printf("No difference that is larger than 1e-10 btw Matlab and C versions\n");
	
	free(correctH);
	// ====== END OF DEBUGGING ======
*/	
	// ***** free variables *****
	free(atm->x);
	free(atm->y);
	free(atm->z);
	free(atm->f);
	free(atm->ghm);
	free(atm->p_u);
	free(atm->p_v);
	free(atm->p_w);
	free(atm);

	free(H);

	free(DP->idx);
	free(DP->DPx);
	free(DP->DPy);
	free(DP->DPz);
	free(DP->L);
	free(DP);

	free(gradghm);	
	
	free(F);
	free(K);

	free(d);
	free(mapping);

	freeCudaMem(H_d, atm_d, DP_d, gradghm_d, F_d, K_d);
	
	return 0;
}

void computeK ( const fType* H, const fType* F, 
		const fType dt, const fType coefficient,
		const int Nnodes, const int Nvar,
		const fType d_coefficient, const int d_num,
		fType* K, fType* d){

	fType di = 0.0;
	
	if (d_num == 1)
		for (int i = 0; i < Nnodes * Nvar; i++){
			di = dt * F[i];			// d = dt*F
			K[i] = H[i] + coefficient*di;	// K = H + coefficient*d
			d[i] = d_coefficient * di;	// initialize d[i]
		}
	else if (d_num != 4)
		for (int i = 0; i < Nnodes * Nvar; i++){
			di = dt * F[i];			// d = dt*F
			K[i] = H[i] + coefficient*di;	// K = H + coefficient*d
			d[i] += d_coefficient * di;
		}
	else 	// don't update K in d4 case
		for (int i = 0; i < Nnodes * Nvar; i++){
			di = dt * F[i];			// d = dt*F
			d[i] += d_coefficient * di;
		}
}
