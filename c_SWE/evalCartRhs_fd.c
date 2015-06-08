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
			long long* tps1,	// 1st loop time
			long long* tps2){	// 2nd loop time

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
	long long tstart, tstop;

	// compute the (projected) Cartesian derivarives 
	// applied to the velocity and geopotential
	fType sum1, sum2, sum3, sum4;

	fType* Tx = (fType*) _mm_malloc (sizeof(fType) * Nnodes * Nvar, 64);
	fType* Ty = (fType*) _mm_malloc (sizeof(fType) * Nnodes * Nvar, 64);
	fType* Tz = (fType*) _mm_malloc (sizeof(fType) * Nnodes * Nvar, 64);
	fType* HV = (fType*) _mm_malloc (sizeof(fType) * Nnodes * Nvar, 64);
	
	__assume_aligned(idx, 32);
	__assume_aligned(DPx, 64);
	__assume_aligned(DPy, 64);
	__assume_aligned(DPz, 64);
	__assume_aligned(L, 64);

	tstart = getTime();
	for (int i = 0; i < Nnodes; i++){
		for (int ivar = 0; ivar < Nvar; ivar++){
			int t_idx = i*Nvar+ivar; 

			sum1 = 0.0;
			sum2 = 0.0;
			sum3 = 0.0;
			sum4 = 0.0;
		
			for (int inbr = 0; inbr < Nnbr; inbr++){
				int dp_idx = i*(Nnbr+1) + inbr;
				int h_idx = idx[dp_idx] * Nvar + ivar;// neighbor's index in H

				sum1 += DPx[dp_idx] * H[h_idx]; // DPx[i][inbr]*H[nbr_idx][ivar]
				sum2 += DPy[dp_idx] * H[h_idx]; // DPy[i][inbr]*H[nbr_idx][ivar]
				sum3 += DPz[dp_idx] * H[h_idx]; // DPz[i][inbr]*H[nbr_idx][ivar]
				sum4 += L[dp_idx] * H[h_idx];   // L[i][inbr]*H[nbr_idx][ivar]
			}

			Tx[t_idx] = sum1;
			Ty[t_idx] = sum2;
			Tz[t_idx] = sum3;
			HV[t_idx] = sum4;		
		}
	} 	
	tstop = getTime();
	*tps1 += (tstop-tstart);	

	// This is the computation for the right hand side 
	// of the Cartesia momentum equation
	fType p, q, s;

	fType H_i1, H_i2, H_i3, H_i4;
	fType Tx_i1, Tx_i2, Tx_i3, Tx_i4;
	fType Ty_i1, Ty_i2, Ty_i3, Ty_i4;
	fType Tz_i1, Tz_i2, Tz_i3, Tz_i4;

	__assume_aligned(Tx, 64);
	__assume_aligned(Ty, 64);
	__assume_aligned(Tz, 64);
	__assume_aligned(HV, 64);

	__assume_aligned(H, 64);
	__assume_aligned(F, 64);
	
	tstart = getTime();
	
	for (int i = 0; i < Nnodes; i++){
		H_i1 = H[i*Nvar+0];
		H_i2 = H[i*Nvar+1];
		H_i3 = H[i*Nvar+2];
		H_i4 = H[i*Nvar+3];	
	
		Tx_i1 = Tx[i*Nvar+0]; 
		Tx_i2 = Tx[i*Nvar+1]; 
		Tx_i3 = Tx[i*Nvar+2]; 
		Tx_i4 = Tx[i*Nvar+3]; 
	
		Ty_i1 = Ty[i*Nvar+0]; 
		Ty_i2 = Ty[i*Nvar+1]; 
		Ty_i3 = Ty[i*Nvar+2]; 
		Ty_i4 = Ty[i*Nvar+3]; 
		
		Tz_i1 = Tz[i*Nvar+0]; 
		Tz_i2 = Tz[i*Nvar+1]; 
		Tz_i3 = Tz[i*Nvar+2]; 
		Tz_i4 = Tz[i*Nvar+3];

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
			 + HV[i*4+0];

		F[i*4+1] = p_v[i*3+0] * p
			 + p_v[i*3+1] * q
			 + p_v[i*3+2] * s
			 + HV[i*4+1];
				
		F[i*4+2] = p_w[i*3+0] * p
			 + p_w[i*3+1] * q
			 + p_w[i*3+2] * s
			 + HV[i*4+2];
		
		// right-hand side for the geopotential (Does not need to be projected, this
		// has already been accounted for in the DPx, DPy, and DPz operations for
		// this equation
		F[i*4+3] = -(	  H_i1 * (Tx_i4 - gradghm[i*3+0]) 
				+ H_i2 * (Ty_i4 - gradghm[i*3+1]) 
				+ H_i3 * (Tz_i4 - gradghm[i*3+2]) 
				+ (H_i4 + gh0 - ghm[i]) * (Tx_i1 + Ty_i2 + Tz_i3)
			    ) 	+ HV[i*4+3];
	}	
	tstop = getTime();
	*tps2 += (tstop-tstart);	
	
	// free memory
	_mm_free(Tx);
	_mm_free(Ty);
	_mm_free(Tz);
	_mm_free(HV);
}

// ------------- DEBUGGING ---------------
/*
		if (i == 13651){
			printf("%f %f %f %f\n", H_i1, H_i2, H_i3, H_i4);
			printf("%f %f %f %f\n", Tx_i1, Tx_i2, Tx_i3, Tx_i4);
			printf("%f %f %f %f\n", Ty_i1, Ty_i2, Ty_i3, Ty_i4);
			printf("%f %f %f %f\n", Tz_i1, Tz_i2, Tz_i3, Tz_i4);
			printf("%f %f %f\n", x[i], y[i], z[i]);
			printf("%f\n", f[i]);

			printf("%e\n", p);
			printf("%e\n", q);
			printf("%e\n", s);

			exit(-1);
		}
*/
// ------------- End of debugging ---------------
