# compiler option
CUDA_PATH = /usr/local/cuda-6.5

INCLUDES := -I$(CUDA_PATH)/include -I./include
LIBS := -L$(CUDA_PATH)/lib64/

SRCS =  readInputs.c reorderNodes.c test_case_5_cart_rk4_fd.c
CUDA_SRCS =  evalCartRhs_fd.cu CUDA_utils.cu

all:
	nvcc -G -arch=compute_52 $(INCLUDES) $(CUDA_SRCS) -c 
	nvcc -g --x 'c++' $(INCLUDES) $(SRCS) -c 
	nvcc *.o -o swe

clean: 
	rm ./*.o
	rm ./swe
