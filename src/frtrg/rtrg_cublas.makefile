#!/bin/make -f
PYTHON_VERSION_STR="310-x86_64-linux-gnu"

all: libcuda_helpers.a rtrg_cublas.cpython-$(PYTHON_VERSION_STR).so

libcuda_helpers.a: cuda_helpers.cu rtrg_cublas.h
	rm -f libcuda_helpers.a
	CC=icc nvcc -rdc=true --compiler-options '-fPIC' -c -o temp.o cuda_helpers.cu
	LDSHARED=icc nvcc -dlink --compiler-options '-fPIC' -o cuda_helpers.o temp.o -lcublas -lcudart
	ar crs libcuda_helpers.a cuda_helpers.o temp.o

rtrg_cublas.cpython-$(PYTHON_VERSION_STR).so: libcuda_helpers.a rtrg_cublas.c rtrg_cublas.h
	python setup_cublas.py build_ext --inplace
	echo "Check the last command and repeat it with a different linker if necessary"

clean:
	rm -f libcuda_helpers.a temp.o cuda_helpers.o
