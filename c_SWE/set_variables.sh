#!/bin/sh

#export KMP_AFFINITY=verbose,granularity=thread,compact,1,0 # 1 thread per core first
export KMP_AFFINITY=verbose,granularity=thread,compact      # 2 threads per core first

#export KMP_AFFINITY=verbose,granularity=thread,scatter

#export OMP_NUM_THREADS=1
#./swe
