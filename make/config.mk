# Directories
PREFIX = .
BIN_DIR = $(PREFIX)/bin
LIB_DIR = $(PREFIX)/lib
FORTRAN_DIR = /usr/local/gfortran/lib

# Compilers for the executable
CC_EXE  = mpiicc
CXX_EXE = mpiicpc
FC_EXE  = mpiifort

# Compilers for the library
CC_LIB  = icc
CXX_LIB = icpc
FC_LIB  = ifort

# Compile and link flags
CPPFLAGS += -O3 
FFLAGS += -O3 
CXXFLAGS += -DMLIP_MPI
LDFLAGS += -L/home/jinghuang/intel/oneapi/mkl/2022.0.2/lib/intel64 -lmkl_rt -lifcore
CPPFLAGS += -I/home/jinghuang/intel/oneapi/mkl/2022.0.2/include
CXXFLAGS += -DMLIP_INTEL_MKL 
CXXFLAGS += -O3 -xHost -ipo -no-prec-div -fp-model fast=2 -qoverride-limits -qopt-zmm-usage=high  -qopt-mem-layout-trans=3  -qopt-streaming-stores=auto -fimf-use-svml=true -std=c++11 -qopt-report=5


# Extra variables
