#NVIDIA_OPENCL := /home/tuanta/NVIDIA_GPU_SDK/OpenCL/common
#INTEL_OPENCL := /opt/intel/opencl-1.2-5.0.0.57/lib64

# compiler option
#CC = gfortran
CC = ifort

OPT = -O3

# Compile-time flags
ifeq ($(CC),gfortran)
	CFLAGS = $(OPT) 
else
	CFLAGS = $(OPT) -qopt-report=5 -assume byterecl -xHost -align array64byte
endif

#INC = 

SRCS = swerbf-test.f90

#LIB = 

EXE = swe

all: $(SRCS) 
	$(CC) $(CFLAGS) $(SRCS) -o $(EXE)

clean: 
	rm ./swe
