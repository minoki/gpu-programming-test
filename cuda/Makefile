nvcc= nvcc

CUDA_PATH= /usr/local/cuda

all: add

add: add.cu
	$(nvcc) -o $@ $<

add_nvrtc: add_nvrtc.c
	$(CC) -o $@ -I$(CUDA_PATH)/include $< -L$(CUDA_PATH)/lib64 -lcuda -lcudart -lnvrtc

.PHONY: all
