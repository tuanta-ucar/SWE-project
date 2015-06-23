#include <readInputs.h>

atm_struct* read_atm(){
	// allocate an atm_struct
	atm_struct* atm = (atm_struct*)malloc(sizeof(atm_struct) * 1);

	// open atm_binary.bin file
	FILE* file_ptr = fopen("./inputBins/atm_binary.bin", "r");

	// read Nnodes
	fread(&(atm->Nnodes), sizeof(int), 1, file_ptr);

	// read Nvar
	fread(&(atm->Nvar), sizeof(int), 1, file_ptr);

	// read Nnbr
	fread(&(atm->Nnbr), sizeof(int), 1, file_ptr);

	double number = 0;

	// read x[Nnodes] 
	atm->x = (fType*)malloc(sizeof(fType) * atm->Nnodes);
	for (int i = 0; i < atm->Nnodes; i++){
		fread(&number, sizeof(double), 1, file_ptr);
		(atm->x)[i] = (fType)number;
	}
	
	// read y[Nnodes]
	atm->y = (fType*)malloc(sizeof(fType) * atm->Nnodes);
	for (int i = 0; i < atm->Nnodes; i++){
		fread(&number, sizeof(double), 1, file_ptr);
		(atm->y)[i] = (fType)number;
	}
 
	// read z[Nnodes]
	atm->z = (fType*)malloc(sizeof(fType) * atm->Nnodes);
	for (int i = 0; i < atm->Nnodes; i++){
		fread(&number, sizeof(double), 1, file_ptr);
		(atm->z)[i] = (fType)number;
	}

	// read f[Nnodes]
	atm->f = (fType*)malloc(sizeof(fType) * atm->Nnodes);
	for (int i = 0; i < atm->Nnodes; i++){
		fread(&number, sizeof(double), 1, file_ptr);
		(atm->f)[i] = (fType)number;
	}
	
	// read g
	fread(&number, sizeof(double), 1, file_ptr);
	atm->g = (fType)number;

	// read a
	fread(&number, sizeof(double), 1, file_ptr);
	atm->a = (fType)number;

	// read gh0
	fread(&number, sizeof(double), 1, file_ptr);
	atm->gh0 = (fType)number;

	// read ghm[Nnodes]
	atm->ghm = (fType*)malloc(sizeof(fType) * atm->Nnodes);
	for (int i = 0; i < atm->Nnodes; i++){
		fread(&number, sizeof(double), 1, file_ptr);
		(atm->ghm)[i] = (fType)number;
	}

	// close file
	fclose(file_ptr);

	// read p_u, p_v and p_w: size Nnodes x 3
	FILE* file_ptr1 = fopen("./inputBins/atm_p_binary.bin", "r");

	atm->p_u = (fType*) malloc (sizeof(fType) * atm->Nnodes * 3);
	atm->p_v = (fType*) malloc (sizeof(fType) * atm->Nnodes * 3);
	atm->p_w = (fType*) malloc (sizeof(fType) * atm->Nnodes * 3);

	for (int i = 0; i < atm->Nnodes * 3; i++){
		fread(&number, sizeof(double), 1, file_ptr1);
		(atm->p_u)[i] = (fType)number;
	}

	for (int i = 0; i < atm->Nnodes * 3; i++){
		fread(&number, sizeof(double), 1, file_ptr1);
		(atm->p_v)[i] = (fType)number;
	}

	for (int i = 0; i < atm->Nnodes * 3; i++){
		fread(&number, sizeof(double), 1, file_ptr1);
		(atm->p_w)[i] = (fType)number;
	}

	fclose(file_ptr1);

	return atm;
}

DP_struct* read_DPs(int Nnodes, int Nnbr, fType gamma, fType a){
	// open idx_transposed_binary.bin file
	FILE* file_ptr = fopen("./inputBins/idx_transposed_binary.bin", "r");

	// allocate DP_struct
	DP_struct* DP = (DP_struct*) malloc(sizeof(DP_struct)*1);

	int paddedSize = Nnodes * (Nnbr+1);

	// read idx: Nnodes x Nnbr. No need to convert to row-major indexing
	// Pad 1 more column to idx -> actual size = Nnodes x (Nnbr+1)
	DP->idx = (int*) malloc (sizeof(int) * paddedSize);

	for (int i = 0; i < Nnodes; i++)
		fread(DP->idx + i*(Nnbr+1), sizeof(int) * Nnbr, 1, file_ptr);

	fclose(file_ptr);

	// substract 1 from each index (Matlab starts with index 0)
	for (int i = 0; i < paddedSize; i++)
		DP->idx[i]--;

	// --------------- DP matrices -----------------
	// read weightsDx: Nnodes x Nnbr
	FILE* file_ptr1 = fopen("./inputBins/DP_binary.bin", "r");
 
	DP->DPx = (fType*) malloc (sizeof(fType) * paddedSize);

	double number = 0.0;
	int row_id = 0;
	int col_id = 0;
	int id = 0;

	for (int i = 0; i < Nnodes * Nnbr; i++){
		row_id = i/Nnbr;
		col_id = i%Nnbr;
		id = row_id*(Nnbr+1) + col_id;

		fread(&number, sizeof(double), 1, file_ptr1);
		DP->DPx[id] = (fType) number/a;
	}

	// read weightsDy: Nnodes x Nnbr 
	DP->DPy = (fType*) malloc (sizeof(fType) * paddedSize);

	for (int i = 0; i < Nnodes * Nnbr; i++){
		row_id = i/Nnbr;
		col_id = i%Nnbr;
		id = row_id*(Nnbr+1) + col_id;
		
		fread(&number, sizeof(double), 1, file_ptr1);
		DP->DPy[id] = (fType) number/a;
	}

	// read weightsDz: Nnodes x Nnbr
	DP->DPz = (fType*) malloc (sizeof(fType) * paddedSize);

	for (int i = 0; i < Nnodes * Nnbr; i++){
		row_id = i/Nnbr;
		col_id = i%Nnbr;
		id = row_id*(Nnbr+1) + col_id;
		
		fread(&number, sizeof(double), 1, file_ptr1);
		DP->DPz[id] = (fType) number/a;
	}

	// read weightsL: Nnodes x Nnbr. Pad extra space.
	DP->L = (fType*) malloc (sizeof(fType) * paddedSize);

	for (int i = 0; i < Nnodes * Nnbr; i++){
		row_id = i/Nnbr;
		col_id = i%Nnbr;
		id = row_id*(Nnbr+1) + col_id;
		
 		fread(&number, sizeof(double), 1, file_ptr1);
		DP->L[id] = (fType) number * gamma; // mulitply with gamma
	}

	fclose(file_ptr1);

	// Intialize all padded memory locations to 0
	for (int i = 0; i < Nnodes; i++){
		DP->idx[i*32+31] = 0;
		DP->DPx[i*32+31] = 0.0;
		DP->DPy[i*32+31] = 0.0;
		DP->DPz[i*32+31] = 0.0;
		DP->L[i*32+31] = 0.0;
	}

	return DP;
}

fType* read_H(int Nnodes){
	// H matrix size: Nnodes x 4
	fType* H = (fType*) malloc(sizeof(fType)* Nnodes * 4);

	// read the transpose of H: no need to convert to row-major style
	FILE* file_ptr = fopen("./inputBins/H_transposed_binary.bin", "r");

	double number = 0.0;

	for (int i = 0; i < Nnodes * 4; i++){
		fread(&number, sizeof(double), 1, file_ptr);
		H[i] = (fType)number;
	}
	
	fclose(file_ptr);

	return H;
}

fType* read_gradghm(int Nnodes){
	// gradghm matrix size: Nnodes x 3
	fType* gradghm = (fType*) malloc(sizeof(fType)* Nnodes * 3);

	// open gradghm_binary.bin file
	FILE* file_ptr = fopen("./inputBins/gradghm_transposed_binary.bin", "r");	

 	// read the transpose of gradgm: No need to convert to row-major style
	double number = 0.0;
        for (int i = 0; i < Nnodes * 3; i++){
                fread(&number, sizeof(double), 1, file_ptr);
                gradghm[i] = (fType)number;
        }
        
        fclose(file_ptr);

	return gradghm;
}

/*
 	// read the transpose of H
        FILE* file_ptr1 = fopen("./inputBins/H_transposed_binary.bin", "r");

        fType* H_trans = (fType*) malloc(sizeof(fType) * Nnodes * 4);
        
        for (int i = 0; i < Nnodes * 4; i++){
                fread(&number, sizeof(double), 1, file_ptr1);
                H_trans[i] = (fType)number;
        }
        
        fclose(file_ptr1);

        for (int i = 0; i < Nnodes * 4; i++){
                int row_id = i/4 + 1;
                int col_id = i%4 + 1;
                if (H_trans[i] != H[row_id*5+col_id]){
                        printf("Mismatch\n");
                        return H;
                }
        }

        printf("Correct\n");
*/
