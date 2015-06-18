#include <readInputs.h>
#include <evalCartRhs_fd.h>
#include <timer.h>
#include <reorderNodes.h>

void computeK ( const fType* H, const fType* F, 
		const fType dt, const fType coefficient,
		const int Nnodes, const int Nvar,
		const fType d_coefficient, const int d_num,
		fType* K, fType* d);

void allocateMemoryToThreads(	fType* H, fType* H_t,
				DP_struct* DP, DP_struct* DP_t,
				atm_struct* atm, atm_struct* atm_t,
				fType* gradghm, fType* gradghm_t,
				fType* F, fType* K,
				int start_id, int end_id);	 
int main(){

for (int nthreads = 16; nthreads <= 16; nthreads++){
	omp_set_num_threads(nthreads);

	int ntimes = 1;
	double perf[ntimes][2];
	
	for (int attempt = 0; attempt < ntimes; attempt++){
		// timing variables
		double tps, tps1;

		// constants
		const fType gamma = -6.4e-22;
		const int tend = 15;			// days
		const int nsteps = tend*24*3600;	// seconds
		const fType dt = 90.0;			// time step in second
		
		// ***** read inputs *****
		// read atm variables
		atm_struct* atm = read_atm();	

		// read H
		fType* H = read_H(atm->Nnodes);	

		// read idx, DPx, DPy, DPz and L
		DP_struct* DP = read_DPs(atm->Nnodes, atm->Nnbr, gamma, atm->a);

		// read gradghm
		fType* gradghm = read_gradghm(atm->Nnodes);

		// reorder node indices. mapping[new_id] = old_id
		int* mapping = reorderNodes(H, DP, atm, gradghm);
		
		// ***** main loop *****
		fType* F = (fType*) _mm_malloc (sizeof(fType) * atm->Nnodes * atm->Nvar, 64);
		fType* K = (fType*) _mm_malloc (sizeof(fType) * atm->Nnodes * atm->Nvar, 64);
		fType* d = (fType*) calloc (atm->Nnodes*atm->Nvar, sizeof(fType));

		// =========== Declare Threaded Arrays ==============
		atm_struct* atm_t = (atm_struct*) malloc(sizeof(atm_struct));
		atm_t->Nnodes = atm->Nnodes;
		atm_t->Nvar = atm->Nvar;
		atm_t->Nnbr = atm->Nnbr;
		atm_t->x = (fType*) malloc (sizeof(fType) * atm->Nnodes);
		atm_t->y = (fType*) malloc (sizeof(fType) * atm->Nnodes);
		atm_t->z = (fType*) malloc (sizeof(fType) * atm->Nnodes);
		atm_t->f = (fType*) malloc (sizeof(fType) * atm->Nnodes);
		atm_t->ghm = (fType*) malloc (sizeof(fType) * atm->Nnodes);
		atm_t->g = atm->g;
		atm_t->a = atm->a;
		atm_t->gh0 = atm->gh0;
		atm_t->p_u = (fType*) malloc (sizeof(fType) * atm->Nnodes * 3);
		atm_t->p_v = (fType*) malloc (sizeof(fType) * atm->Nnodes * 3);
		atm_t->p_w = (fType*) malloc (sizeof(fType) * atm->Nnodes * 3);
	
		int paddedSize = atm->Nnodes * (atm->Nnbr+1);

		DP_struct* DP_t = (DP_struct*) malloc (sizeof(DP_struct));
		DP_t->idx = (int*) malloc(sizeof(int)*paddedSize);
		DP_t->DPx = (fType*) malloc(sizeof(fType)*paddedSize);
		DP_t->DPy = (fType*) malloc(sizeof(fType)*paddedSize);
		DP_t->DPz = (fType*) malloc(sizeof(fType)*paddedSize);
		DP_t->L = (fType*) malloc(sizeof(fType)*paddedSize);

		fType* H_t = (fType*) malloc(sizeof(fType) * atm->Nnodes * 4);
		fType* gradghm_t = (fType*) malloc(sizeof(fType) * atm->Nnodes * 3);
		// ==========================================		

		tps = 0.0;
		tps1 = 0.0;

		// ########### Thread team created ############
		#pragma omp parallel shared(atm,H,DP,gradghm,F,K,d,tps1,tps,\
						atm_t,H_t,DP_t,gradghm_t)
		{
		// thread allocation
		int thread_id = omp_get_thread_num();
		int chunkSize = atm->Nnodes/nthreads;

		int start_id, end_id;

		if (thread_id != nthreads-1){
			start_id = thread_id*chunkSize;
			end_id = (thread_id+1)*chunkSize-1;
		} else {
			start_id = thread_id*chunkSize;
			end_id = atm->Nnodes-1;
		}


		// Allocate memory to threads 
		allocateMemoryToThreads(H, H_t, DP, DP_t, atm, atm_t, gradghm, gradghm_t, F, K, start_id, end_id);
		
		#pragma omp barrier
	
		double tstart = omp_get_wtime();
		
		for (int nt = 1; nt <= 100; nt++){   // tend*24*3600
			#pragma omp single
			{
			memcpy(K, H_t, sizeof(fType) * atm_t->Nnodes * atm_t->Nvar); // K = H
			}

			
			evalCartRhs_fd(K, DP_t, atm_t, gradghm_t, F, start_id, end_id, &tps1);
			
			#pragma omp single
			{
			computeK(H_t, F, dt, 0.5, atm_t->Nnodes, atm_t->Nvar, 1.0, 1, K, d);
			}

			evalCartRhs_fd(K, DP_t, atm_t, gradghm_t, F, start_id, end_id, &tps1);
			
			#pragma omp single
			{			
			computeK(H_t, F, dt, 0.5, atm_t->Nnodes, atm_t->Nvar, 2.0, 2, K, d);
			}

			evalCartRhs_fd(K, DP_t, atm_t, gradghm_t, F, start_id, end_id, &tps1);

			#pragma omp single
			{			
			computeK(H_t, F, dt, 1.0, atm_t->Nnodes, atm_t->Nvar, 2.0, 3, K, d);
			}

			evalCartRhs_fd(K, DP_t, atm_t, gradghm_t, F, start_id, end_id, &tps1);
			
			#pragma omp single
			{			
			computeK(H_t, F, dt, 1.0, atm_t->Nnodes, atm_t->Nvar, 1.0, 4, K, d);

			// update H
			for (int i = 0; i < atm_t->Nnodes * atm_t->Nvar; i++){
				H_t[i] += (1.0/6.0) * d[i];
			}
			}
		}
		
		#pragma omp barrier
		double tstop = omp_get_wtime();
		
		if (omp_get_thread_num() == 0)
			tps = tstop-tstart;
	
		} // end of OMP region

		perf[attempt][0] = tps1/100;
		perf[attempt][1] = tps/100 ;
		
		printf("#attempt = %d Fused loop time (seconds): %lf\n", attempt, perf[attempt][0]);
		printf("#attempt = %d Total time (seconds): %lf\n", attempt, perf[attempt][1]);


		// ======= DEBUGGING =========
			int count = 0;	
			FILE* file_ptr = fopen("H_debug.bin", "r");

			double* correctH = (double*) malloc (sizeof(double)*atm->Nnodes * atm->Nvar);
			fread(correctH, sizeof(double), atm->Nnodes * atm->Nvar, file_ptr);
			fclose(file_ptr);
			
			for (int i = 0; i < atm->Nnodes; i++){
				for (int j = 0; j < atm->Nvar; j++){
					double abs_err = fabs(H_t[i*4+j] - correctH[mapping[i]*4+j]);
					if (abs_err > 1E-10){
						printf("%d %d %.16f %.16f\n", i/4, i%4, H_t[i*4+j], correctH[mapping[i]*4+j]);
						count++;
					}
				}
			}
		
			if (count == 0)	
				printf("No difference that is larger than 1e-10 btw Matlab and C versions\n");
			
			free(correctH);
		// ====== END OF DEBUGGING ======

		
		// ***** free variables *****
		free(atm_t->x);
		free(atm_t->y);
		free(atm_t->z);
		free(atm_t->f);
		free(atm_t->ghm);
		free(atm_t->p_u);
		free(atm_t->p_v);
		free(atm_t->p_w);
		free(atm_t);

		free(H_t);

		free(DP_t->idx);
		free(DP_t->DPx);
		free(DP_t->DPy);
		free(DP_t->DPz);
		free(DP_t->L);
		free(DP_t);

		free(gradghm_t);	
		
		_mm_free(F);
		_mm_free(K);

		free(d);
		free(mapping);
	} // end an attempt

	FILE *prof_file = fopen("profiling_file.txt", "a");
	
	// write profiling time to a file
	fprintf(prof_file,"nthreads = %d\n", nthreads);
	for (int i = 0; i < ntimes; i++)
		fprintf(prof_file, "%lf,", perf[i][0]);
	fprintf(prof_file, "\n");
	for (int i = 0; i < ntimes; i++)
		fprintf(prof_file, "%lf,", perf[i][1]);
	fprintf(prof_file, "\n\n");

	fclose(prof_file);
} // end of one nthread

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

void allocateMemoryToThreads(	fType* H, fType* H_t,
				DP_struct* DP, DP_struct* DP_t,
				atm_struct* atm, atm_struct* atm_t,
				fType* gradghm, fType* gradghm_t,
				fType* F, fType* K,
				int start_id, int end_id){
	for (int i = start_id; i <= end_id; i++){
		atm_t->x[i] = atm->x[i];
		atm_t->y[i] = atm->y[i];
		atm_t->z[i] = atm->z[i];
		atm_t->f[i] = atm->f[i];
		atm_t->ghm[i] = atm->ghm[i];	
	
		for (int j = 0; j < 3; j++){
			atm_t->p_u[i*3+j] = atm->p_u[i*3+j];		
			atm_t->p_v[i*3+j] = atm->p_v[i*3+j];		
			atm_t->p_w[i*3+j] = atm->p_w[i*3+j];		
			gradghm_t[i*3+j] = gradghm[i*3+j];
		}

		for (int inbr = 0; inbr < atm->Nnbr+1; inbr++){
			DP_t->idx[i*(atm->Nnbr+1)+inbr] = DP->idx[i*(atm->Nnbr+1)+inbr];	
			DP_t->DPx[i*(atm->Nnbr+1)+inbr] = DP->DPx[i*(atm->Nnbr+1)+inbr];	
			DP_t->DPy[i*(atm->Nnbr+1)+inbr] = DP->DPy[i*(atm->Nnbr+1)+inbr];	
			DP_t->DPz[i*(atm->Nnbr+1)+inbr] = DP->DPz[i*(atm->Nnbr+1)+inbr];	
			DP_t->L[i*(atm->Nnbr+1)+inbr] = DP->L[i*(atm->Nnbr+1)+inbr];	
		}

		for (int ivar = 0; ivar < 4; ivar++){
			H_t[i*4+ivar] = H[i*4+ivar];
			F[i*4+ivar] = 0;
			K[i*4+ivar] = 0;
		}

	}	

	#pragma omp barrier

	if (omp_get_thread_num() == 0){
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

		_mm_free(H);

		_mm_free(DP->idx);
		_mm_free(DP->DPx);
		_mm_free(DP->DPy);
		_mm_free(DP->DPz);
		_mm_free(DP->L);
		free(DP);

		free(gradghm);	
	}			
}
