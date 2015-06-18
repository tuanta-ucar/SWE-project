#NVIDIA_OPENCL := /home/tuanta/NVIDIA_GPU_SDK/OpenCL/common
#INTEL_OPENCL := /opt/intel/opencl-1.2-5.0.0.57/lib64

# compiler option
#CC = gcc
CC = icc

OPT = -O3 

# Compile-time flags
ifeq ($(CC),gcc)
	CFLAGS = $(OPT) -std=c99 -Wno-unused-result
else
	CFLAGS = $(OPT) -xHost -std=c99 -qopt-report=5 -fno-alias -openmp
endif

INC = ./include

SRCS = reorderNodes.c readInputs.c timer.c evalCartRhs_fd.c test_case_5_cart_rk4_fd.c

LIB = 

EXE = swe

all: test_case_5_cart_rk4_fd.c 
	$(CC) -g $(CFLAGS) -I$(INC) $(SRCS) $(LIB) -o $(EXE)
clean: 
	rm ./swe
