#include <evalCartRhs_fd.h>
#include <timer.h>

/*
 * Output: F matrix with size Nnodes x 4
 */


void  evalCartRhs_fd( 	const fType* H,
			const DP_struct* DP,
			const atm_struct* atm,
			const fType* gradghm,
			fType* F,
			int* start1, int* start2, int chunkSize,
			double* tps1){	// 1st loop time

	// extract out some constants from the atm structure
	const fType* x = atm->x;
	const fType* y = atm->y;
	const fType* z = atm->z;
	const fType* f = atm->f;
	
	const fType g = atm->g;
	const fType a = atm->a;
	const fType gh0 = atm->gh0;

	const fType* ghm = atm->ghm;

	const int Nnodes = atm->Nnodes;
	const int Nvar = atm->Nvar;
	const int Nnbr = atm->Nnbr;

	const fType* p_u = atm->p_u;
	const fType* p_v = atm->p_v;
	const fType* p_w = atm->p_w;
 
	// extract out constants from the DP structure
	const int* idx = DP->idx;
	const fType* DPx = DP->DPx;
	const fType* DPy = DP->DPy;
	const fType* DPz = DP->DPz;
	const fType* L = DP->L;

	// timing variables
	double tstart, tstop;

	__assume_aligned(idx, 32);
	__assume_aligned(DPx, 64);
	__assume_aligned(DPy, 64);
	__assume_aligned(DPz, 64);
	__assume_aligned(L, 64);
	__assume_aligned(H, 64);
	__assume_aligned(F, 64);

	// This is the computation for the right hand side 
	// of the Cartesia momentum equation
	fType p, q, s;

	fType H_i1, H_i2, H_i3, H_i4;
	fType Tx_i1, Tx_i2, Tx_i3, Tx_i4;
	fType Ty_i1, Ty_i2, Ty_i3, Ty_i4;
	fType Tz_i1, Tz_i2, Tz_i3, Tz_i4;
	fType HV_i1, HV_i2, HV_i3, HV_i4;
	
	int thread_id = omp_get_thread_num();

	#pragma omp barrier	

	int start_id, end_id;	// each threads has its own start and end indices

	tstart = omp_get_wtime();

	while (true){
		// Each thread gets its own start_id from the critical section
		if (thread_id / 8 == 0){
			// thread 0 -> 7
			#pragma omp atomic capture
			{
				start_id = *start1;
				*start1 += chunkSize;
			}

			if (start_id > Nnodes/2) break;
			end_id = start_id + chunkSize - 1;	
			if (end_id > Nnodes/2) end_id = Nnodes/2;
		} else {
			// thread 8 to 15
			#pragma omp atomic capture
			{
				start_id = *start2;
				*start2 += chunkSize;
			}
			
			if (start_id > Nnodes-1) break;
			end_id = start_id + chunkSize - 1;	
			if (end_id > Nnodes-1) end_id = Nnodes-1;
		}

		// ####### Parallel region #######
		for (int i = start_id; i <= end_id; i++){
			Tx_i1 = 0.0;
			Tx_i2 = 0.0; 
			Tx_i3 = 0.0; 
			Tx_i4 = 0.0;  
		
			Ty_i1 = 0.0;  
			Ty_i2 = 0.0;  
			Ty_i3 = 0.0;  
			Ty_i4 = 0.0;  
			
			Tz_i1 = 0.0;  
			Tz_i2 = 0.0;  
			Tz_i3 = 0.0;  
			Tz_i4 = 0.0;  

			HV_i1 = 0.0; 
			HV_i2 = 0.0; 
			HV_i3 = 0.0; 
			HV_i4 = 0.0; 

			for (int inbr = 0; inbr < Nnbr; inbr++){
				int dp_idx = i*(Nnbr+1) + inbr;
				int h_idx = idx[dp_idx] * Nvar;	   // neighbor's index in H
				
				fType dp_x = DPx[dp_idx];
				fType dp_y = DPy[dp_idx];
				fType dp_z = DPz[dp_idx];
				fType l = L[dp_idx];

if (i == 12345)
	printf("n = %d nbr_id %d H_nbr = %lf %lf %lf %lf\n", inbr, h_idx, H[h_idx+0], 
						H[h_idx+1], H[h_idx+2], H[h_idx+3]);

				Tx_i1 += dp_x * H[h_idx+0]; // DPx[i][inbr]*H[nbr_idx][ivar]
				Ty_i1 += dp_y * H[h_idx+0]; // DPy[i][inbr]*H[nbr_idx][ivar]
				Tz_i1 += dp_z * H[h_idx+0]; // DPz[i][inbr]*H[nbr_idx][ivar]
				HV_i1 += l * H[h_idx+0];   // L[i][inbr]*H[nbr_idx][ivar]
				
				Tx_i2 += dp_x * H[h_idx+1]; // DPx[i][inbr]*H[nbr_idx][ivar]
				Ty_i2 += dp_y * H[h_idx+1]; // DPy[i][inbr]*H[nbr_idx][ivar]
				Tz_i2 += dp_z * H[h_idx+1]; // DPz[i][inbr]*H[nbr_idx][ivar]
				HV_i2 += l * H[h_idx+1];   // L[i][inbr]*H[nbr_idx][ivar]
				
				Tx_i3 += dp_x * H[h_idx+2]; // DPx[i][inbr]*H[nbr_idx][ivar]
				Ty_i3 += dp_y * H[h_idx+2]; // DPy[i][inbr]*H[nbr_idx][ivar]
				Tz_i3 += dp_z * H[h_idx+2]; // DPz[i][inbr]*H[nbr_idx][ivar]
				HV_i3 += l * H[h_idx+2];   // L[i][inbr]*H[nbr_idx][ivar]
				
				Tx_i4 += dp_x * H[h_idx+3]; // DPx[i][inbr]*H[nbr_idx][ivar]
				Ty_i4 += dp_y * H[h_idx+3]; // DPy[i][inbr]*H[nbr_idx][ivar]
				Tz_i4 += dp_z * H[h_idx+3]; // DPz[i][inbr]*H[nbr_idx][ivar]
				HV_i4 += l * H[h_idx+3];   // L[i][inbr]*H[nbr_idx][ivar]
			}
//if (i == 12345)
//	printf("%lf\n", Tx_i1);
		

			H_i1 = H[i*Nvar+0];
			H_i2 = H[i*Nvar+1];
			H_i3 = H[i*Nvar+2];
			H_i4 = H[i*Nvar+3];	
		
			// compute p, q, s 
			p = -(	H_i1 * Tx_i1 
				+ H_i2 * Ty_i1 
				+ H_i3 * Tz_i1 
				+ f[i] * (y[i] * H_i3 - z[i] * H_i2) 
				+ Tx_i4);

			q = -(	H_i1 * Tx_i2 
				+ H_i2 * Ty_i2 
				+ H_i3 * Tz_i2 
				+ f[i] * (z[i] * H_i1 - x[i] * H_i3) 
				+ Ty_i4);

			s = -(	H_i1 * Tx_i3 
				+ H_i2 * Ty_i3 
				+ H_i3 * Tz_i3 
				+ f[i] * (x[i] * H_i2 - y[i] * H_i1) 
				+ Tz_i4);

			// Project the momentum equations onto the surface of the sphere
			F[i*4+0] = p_u[i*3+0] * p
				 + p_u[i*3+1] * q
				 + p_u[i*3+2] * s
				 + HV_i1;

			F[i*4+1] = p_v[i*3+0] * p
				 + p_v[i*3+1] * q
				 + p_v[i*3+2] * s
				 + HV_i2;
					
			F[i*4+2] = p_w[i*3+0] * p
				 + p_w[i*3+1] * q
				 + p_w[i*3+2] * s
				 + HV_i3;
			
			// right-hand side for the geopotential (Does not need to be projected, this
			// has already been accounted for in the DPx, DPy, and DPz operations for
			// this equation
			F[i*4+3] = -(	  H_i1 * (Tx_i4 - gradghm[i*3+0]) 
					+ H_i2 * (Ty_i4 - gradghm[i*3+1]) 
					+ H_i3 * (Tz_i4 - gradghm[i*3+2]) 
					+ (H_i4 + gh0 - ghm[i]) * (Tx_i1 + Ty_i2 + Tz_i3)
				    ) 	+ HV_i4;

		} // end of parallelized for-loop
	} // end of 1 iteration
	#pragma omp barrier
	
	tstop = omp_get_wtime();

	if (omp_get_thread_num() == 0){
		*tps1 += (tstop-tstart);
	}
}
