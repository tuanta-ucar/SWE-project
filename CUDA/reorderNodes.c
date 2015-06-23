#include <reorderNodes.h>

// return mapping: new_id -> old_id
int* reorderNodes(fType* H, DP_struct* DP, atm_struct* atm, fType* gradghm){
	// extract out some constants from the atm structure
        fType* x = atm->x;
        fType* y = atm->y;
        fType* z = atm->z;
        fType* f = atm->f;
        
        fType* ghm = atm->ghm;

        const int Nnodes = atm->Nnodes;
        const int Nvar = atm->Nvar;
        const int Nnbr = atm->Nnbr;

        fType* p_u = atm->p_u;
        fType* p_v = atm->p_v;
        fType* p_w = atm->p_w;
 
        // extract out constants from the DP structure
        int* idx = DP->idx;
        fType* DPx = DP->DPx;
        fType* DPy = DP->DPy;
        fType* DPz = DP->DPz;
        fType* L = DP->L;

	// ******  sort nodes with respect to z coordinate *******

	// mapping maintains the connection btw old_id and new_id
	// mapping[new_id] == old_id
	int* mapping = (int*) malloc (sizeof(int)*Nnodes);

	// Initialize mapping
	for (int i = 0; i < Nnodes; i++)
		mapping[i] = i;

	fType* sorted_z = (fType*) malloc (sizeof(fType)*Nnodes);
	
	memcpy(sorted_z, z, sizeof(fType)*Nnodes);
	mergeSort(mapping, sorted_z, Nnodes);
	memcpy(z, sorted_z, sizeof(fType)*Nnodes);
	
	free(sorted_z);
	// *****  Rearrange x, y, f, ghm  *******
	fType* x_temp = (fType*) malloc(sizeof(fType)*Nnodes);
	fType* y_temp = (fType*) malloc(sizeof(fType)*Nnodes);
	fType* f_temp = (fType*) malloc(sizeof(fType)*Nnodes);
	fType* ghm_temp = (fType*) malloc(sizeof(fType)*Nnodes);
	
	for (int i = 0; i < Nnodes; i++){
		x_temp[i] = x[mapping[i]];
		y_temp[i] = y[mapping[i]];
		f_temp[i] = f[mapping[i]];
		ghm_temp[i] = ghm[mapping[i]];
	}

	memcpy(x, x_temp, sizeof(fType)*Nnodes);
	memcpy(y, y_temp, sizeof(fType)*Nnodes);
	memcpy(f, f_temp, sizeof(fType)*Nnodes);
	memcpy(ghm, ghm_temp, sizeof(fType)*Nnodes);

	free(x_temp);
	free(y_temp);
	free(f_temp);
	free(ghm_temp);	
	// *********************************
	
	// ****** Rearrange p_u, p_v, p_w and gradghm  ***********
	fType* p_u_temp = (fType*) malloc (sizeof(fType)*Nnodes*3);
	fType* p_v_temp = (fType*) malloc (sizeof(fType)*Nnodes*3);
	fType* p_w_temp = (fType*) malloc (sizeof(fType)*Nnodes*3);
	fType* gradghm_temp = (fType*) malloc (sizeof(fType)*Nnodes*3);

	for (int i = 0; i < Nnodes; i++){
		memcpy(p_u_temp+(i*3), p_u+(mapping[i]*3), sizeof(fType)*3);
		memcpy(p_v_temp+(i*3), p_v+(mapping[i]*3), sizeof(fType)*3);
		memcpy(p_w_temp+(i*3), p_w+(mapping[i]*3), sizeof(fType)*3);
		memcpy(gradghm_temp+(i*3), gradghm+(mapping[i]*3), sizeof(fType)*3);
	}

	memcpy(p_u,p_u_temp, sizeof(fType)*Nnodes*3);
	memcpy(p_v,p_v_temp, sizeof(fType)*Nnodes*3);
	memcpy(p_w,p_w_temp, sizeof(fType)*Nnodes*3);
	memcpy(gradghm,gradghm_temp, sizeof(fType)*Nnodes*3);

	free(p_u_temp);
	free(p_v_temp);
	free(p_w_temp);
	free(gradghm_temp);	
	// *********************************************

	// ***** Rearrange idx ******
	int* idx_temp = (int*) malloc (sizeof(int)*Nnodes*(Nnbr+1));

	for (int i = 0; i < Nnodes; i++){
		memcpy(idx_temp+(i*(Nnbr+1)), idx+(mapping[i]*(Nnbr+1)), sizeof(int)*(Nnbr+1));
	}

	int inv_mapping[Nnodes];	// inversed mapping: inv_mapping[old_idx] == new_idx

	for (int i = 0; i < Nnodes; i++)
		inv_mapping[mapping[i]] = i;

	for (int i = 0; i < Nnodes; i++){
		for (int j = 0; j < Nnbr; j++){
			int old_id = idx_temp[i*(Nnbr+1)+j];
			idx_temp[i*(Nnbr+1)+j] = inv_mapping[old_id];
		}			
	}

	memcpy(idx, idx_temp, sizeof(int)*Nnodes*(Nnbr+1));

	free(idx_temp);
	// **************************	
	
	// ***** rearrange DPx, DPy, DPz and L ****
	fType* DPx_temp = (fType*) malloc (sizeof(fType)*Nnodes*(Nnbr+1));
	fType* DPy_temp = (fType*) malloc (sizeof(fType)*Nnodes*(Nnbr+1));
	fType* DPz_temp = (fType*) malloc (sizeof(fType)*Nnodes*(Nnbr+1));
	fType* L_temp = (fType*) malloc (sizeof(fType)*Nnodes*(Nnbr+1));

	for (int i = 0; i < Nnodes; i++){
		memcpy(DPx_temp+(i*(Nnbr+1)), DPx+(mapping[i]*(Nnbr+1)), sizeof(fType)*(Nnbr+1));
		memcpy(DPy_temp+(i*(Nnbr+1)), DPy+(mapping[i]*(Nnbr+1)), sizeof(fType)*(Nnbr+1));
		memcpy(DPz_temp+(i*(Nnbr+1)), DPz+(mapping[i]*(Nnbr+1)), sizeof(fType)*(Nnbr+1));
		memcpy(L_temp+(i*(Nnbr+1)), L+(mapping[i]*(Nnbr+1)), sizeof(fType)*(Nnbr+1));
	}

	memcpy(DPx, DPx_temp, sizeof(fType)*Nnodes*(Nnbr+1));
	memcpy(DPy, DPy_temp, sizeof(fType)*Nnodes*(Nnbr+1));
	memcpy(DPz, DPz_temp, sizeof(fType)*Nnodes*(Nnbr+1));
	memcpy(L, L_temp, sizeof(fType)*Nnodes*(Nnbr+1));
	
	free(DPx_temp);
	free(DPy_temp);
	free(DPz_temp);
	free(L_temp);
	// ****************************************

	// ******* Rearrange H ***********
	fType* H_temp = (fType*) malloc (sizeof(fType)*Nnodes*4);

	for (int i = 0; i < Nnodes; i++){
		memcpy(H_temp+(i*4), H+(mapping[i]*4), sizeof(fType)*4);
	}
	
	memcpy(H, H_temp, sizeof(fType)*Nnodes*4);

	free(H_temp);
	// ******************************

	return mapping;	
}

// function to merge the two sub arrays arr[l..m] and arr[m+1..r] of arr[l..r]
void merge(int mapping[], fType arr[], int l, int m, int r){
	int i, j, k;
	int n1 = m-l+1;	// size of the first sub-array
	int n2 = r-m;	// size of the second sub-array

	// create temp arrays
	fType L[n1], R[n2];
	int L_m[n1], R_m[n2];
	
	// Initialize L and R
	for (i = 0; i < n1; i++){
		L[i] = arr[l+i];
		L_m[i] = mapping[l+i];
	}

	for (j = 0; j < n2; j++){
		R[j] = arr[m+1+j];
		R_m[j] = mapping[m+1+j];
	}

	// merge temp arrays back into arr[l..r]
	i = 0; j = 0; k = l;

	while (i < n1 && j < n2){
		if (L[i] <= R[j]){
			arr[k] = L[i];
			mapping[k] = L_m[i];
			i++;
		} else {
			arr[k] = R[j];
			mapping[k] = R_m[j];
			j++;
		}
		k++;
	}

	// copy remaining elements in L and R if any
	while (i < n1){
		arr[k] = L[i];
		mapping[k] = L_m[i];
		i++;
		k++;
	}

	while (j < n2){
		arr[k] = R[j];
		mapping[k] = R_m[j];
		j++;
		k++;
	}
}

void mergeSort(int mapping[], fType arr[], int n){
	int curr_size;	// size of subarrays to be merged
	int left_start;	// starting index of left subarray to be merged

	for (curr_size=1; curr_size <= n-1; curr_size = 2*curr_size){
		for (left_start=0; left_start<n-1; left_start += 2*curr_size){
			int mid = left_start + curr_size - 1;
			int right_end;
	
			if ((left_start + curr_size - 1) < (n-1)) 
				mid = left_start + curr_size - 1;
			else 
				mid = n-1;

			if ((left_start + 2*curr_size - 1) < (n-1))
				right_end = left_start + 2*curr_size - 1;
			else 
				right_end = n-1;

			// Merge subarrays arr[left_start..mid] and arr[mid+1..right_end] 
			merge(mapping, arr, left_start, mid, right_end);
		}
	}
} 
