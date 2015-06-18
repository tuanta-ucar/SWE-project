#ifndef REORDER_NODES
#define REORDER_NODES

#include <config.h>

int* reorderNodes(fType* H, DP_struct* DP, atm_struct* atm, fType* gradgm);
void merge(int mapping[], fType arr[], int l, int m, int r);
void mergeSort(int mapping[], fType arr[], int n);

#endif 
