#ifndef EVAL_CART_H
#define EVAL_CART_H

#include<config.h>
#include<math.h>

void evalCartRhs_fd(	const fType* H,
			const DP_struct* DP, 
			const atm_struct* atm,
			const fType* gradghm,
			fType* F,
			int start_id, int end_id,
			double* tps1);

#endif
