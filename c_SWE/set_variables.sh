#!/bin/sh

export KMP_AFFINITY=verbose,granularity=thread,compact,1,0
#export OMP_PROC_BIND=spread
export OMP_NUM_THREADS=8

