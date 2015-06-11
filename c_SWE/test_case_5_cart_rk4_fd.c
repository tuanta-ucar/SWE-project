#include <readInputs.h>
#include <evalCartRhs_fd.h>
#include <timer.h>

void computeK ( const fType* H, const fType* F, 
		const fType dt, const fType coefficient,
		const int Nnodes, const int Nvar,
		const fType d_coefficient, const int d_num,
		fType* K, fType* d);
int main(){
	FILE *prof_file = fopen("profiling_file.txt", "w");

	for (int attempt = 0; attempt < 6; attempt++){
		// timing variables
		long long tstart, tstop, tps;		// for main loop
		long long tstart0, tstop0, tps0;	// for evalCartRhs_fd function
		long long tps1, tps2;			// for 1st and 2nd loop in evalCartRhs_fd

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

		// ***** main loop *****
		fType* F = (fType*) _mm_malloc (sizeof(fType) * atm->Nnodes * atm->Nvar, 64);
		fType* K = (fType*) _mm_malloc (sizeof(fType) * atm->Nnodes * atm->Nvar, 64);

		fType* d = (fType*) calloc (atm->Nnodes*atm->Nvar, sizeof(fType));

		tps0 = 0;
		tps1 = 0;

		tstart = getTime();
		// ########### Thread team created ############
		#pragma omp parallel shared(atm,H,DP,gradghm,F,K,d,tps1)
		{	
		for (int nt = 1; nt <= 100; nt++){   // tend*24*3600
			#pragma omp single
			{
			memcpy(K, H, sizeof(fType) * atm->Nnodes * atm->Nvar); // K = H
			}

			evalCartRhs_fd(K, DP, atm, gradghm, F, &tps1);
			
			#pragma omp single
			{
			computeK(H, F, dt, 0.5, atm->Nnodes, atm->Nvar, 1.0, 1, K, d);
			}

			evalCartRhs_fd(K, DP, atm, gradghm, F, &tps1);
			
			#pragma omp single
			{			
			computeK(H, F, dt, 0.5, atm->Nnodes, atm->Nvar, 2.0, 2, K, d);
			}

			evalCartRhs_fd(K, DP, atm, gradghm, F, &tps1);

			#pragma omp single
			{			
			computeK(H, F, dt, 1.0, atm->Nnodes, atm->Nvar, 2.0, 3, K, d);
			}

			evalCartRhs_fd(K, DP, atm, gradghm, F, &tps1);
			
			#pragma omp single
			{			
			computeK(H, F, dt, 1.0, atm->Nnodes, atm->Nvar, 1.0, 4, K, d);

			// update H
			for (int i = 0; i < atm->Nnodes * atm->Nvar; i++){
				H[i] += (1.0/6.0) * d[i];
			}
			}
		}
		} // end of OMP region
		tstop = getTime();

		tps = (tstop-tstart)/100 ;
		tps0 = tps0/100;
		tps1 = tps1/100;

		printf("#attempt = %d Fused loop time (seconds): %lf\n", attempt, tps1*1e-6);
		printf("#attempt = %d evalCartRhs_fd time (seconds): %lf\n", attempt, tps0*1e-6);
		printf("#attempt = %d Total time (seconds): %lf\n", attempt, tps*1e-6);

		fprintf(prof_file, "%lf\n%lf\n%lf\n\n", tps1*1e-6, tps0*1e-6, tps*1e-6);

		// ======= DEBUGGING =========
			int count = 0;	
			FILE* file_ptr = fopen("H_debug.bin", "r");
			double temp_num = 0.0;
			for (int i = 0; i < atm->Nnodes * atm->Nvar; i++){
				fread(&temp_num, sizeof(double), 1, file_ptr);
				double abs_err = fabs(temp_num - H[i]);
				if (abs_err > 1E-10){
					printf("%d %d %.16f %.16f\n", i/4, i%4, temp_num, H[i]);
					count++;
				}
			}
		
			if (count == 0)	
				printf("No difference that is larger than 1e-10 btw Matlab and C versions\n");

			fclose(file_ptr);
		// ====== END OF DEBUGGING ======

		// ***** write outputs *****
		
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
		
		_mm_free(F);
		_mm_free(K);

		free(d);
	} // end an attempt

	fclose(prof_file);

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
	// ***** verify output H *****
	/*
	FILE* file_ptr = fopen("H_correct_binary.bin", "r");
	
	double temp_num = 0.0;
	double abs_err = 0.0;
	double rel_err = 0.0;

	int rank1_count = 0;	// 0.1% rel_err or less
	int rank2_count = 0;	// 0.1% -> 1%
	int rank3_count = 0;	// 1% -> 10%
	int rank4_count = 0;	// 10% -> 20%
	int rank5_count = 0;	// > 20%

        for (int i = 0; i < atm->Nnodes * atm->Nvar; i++){
                fread(&temp_num, sizeof(double), 1, file_ptr);
		abs_err = fabs(temp_num - H[i]);
		
		if (fabs(temp_num) < 0.000000000001)
			rel_err = 100.0*abs_err/fabs(temp_num);
		else 
			rel_err = 0.0;

		if (rel_err < 0.1)
			rank1_count++;
		else if (rel_err < 1)
			rank2_count++;
		else if (rel_err < 10)
			rank3_count++;
		else if (rel_err < 20)
			rank4_count++;
		else {
			rank5_count++;
			//printf("%d %f %f\n", i, temp_num, H[i]);
		}	
        }

        fclose(file_ptr);

	double total = atm->Nnodes * atm->Nvar;
        printf("Relative error < 0.1%% :      %f\n", 100.0*(double)rank1_count/total);
        printf("Relative error 0.1%% -> 1%% : %f\n", 100.0*(double)rank2_count/total);
        printf("Relative error 1%% -> 10%% :  %f\n", 100.0*(double)rank3_count/total);
        printf("Relative error 10%% -> 20%% : %f\n", 100.0*(double)rank4_count/total);
        printf("Relative error > 20%% :       %f\n", 100.0*(double)rank5_count/total);
	*/
