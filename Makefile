#
# Makefile
# 
# This script defines rules and variables that could change in the course of
# development.
# In particular:
#   * defines srource and object file's variables
#   * specifies phony and files targets for binaries and libraries
#   * define some other rules
# Note: 
# This script assumes having include files in the source directories.
#

# Targets to be generated:
#   <bin_dir>/mlp
#   <lib_dir>/lib_mlip_interface.a
#   <lib_dir>/lib_mlip_cblas.a

MK_THIS := $(lastword $(MAKEFILE_LIST))

# configuration makefile
-include $(CURDIR)/make/config.mk

-include $(CURDIR)/make/auto_vars.mk

ifneq (0, $(words $(filter mlp libinterface,$(MAKECMDGOALS))))
    # source file's directories
    SRC_DIR = $(CURDIR)/src
    SRC_DEV_DIR = $(CURDIR)/dev_src
    # source files
	SRC_COMMON := $(wildcard $(SRC_DIR)/common/*.cpp) 
    SRC_COMMON += $(wildcard $(SRC_DIR)/drivers/*.cpp)
    SRC_COMMON += $(wildcard $(SRC_DIR)/*.cpp) 
    ifneq ($(USE_MLIP_PUBLIC), 1)
      SRC_EXTRA := $(wildcard $(SRC_DEV_DIR)/*.cpp) 
      SRC_EXTRA += $(wildcard $(SRC_DEV_DIR)/*.f90)
    endif

    ifneq (0, $(words $(filter mlp,$(MAKECMDGOALS))))
        CXX = $(CXX_EXE)
        FC = $(FC_EXE)
        # mlp target source files
        ifeq ($(USE_MLIP_PUBLIC), 1)
          SRC_COMMON += $(wildcard $(SRC_DIR)/mlp/*.cpp) 
          SRC_COMMON += $(SRC_DIR)/mlp/self_test.cpp
        else
          SRC_COMMON += $(wildcard $(SRC_DIR)/mlp/*.cpp) 
          SRC_COMMON := $(filter-out $(SRC_DIR)/mlp/mlp.cpp, $(SRC_COMMON))
          SRC_EXTRA += $(wildcard $(SRC_DEV_DIR)/mlp/*.cpp) 
          SRC_EXTRA += $(wildcard $(SRC_DEV_DIR)/utils/mtpr_train.cpp)
        endif

        #EXE_TARGET = mlp$(PROGRAM_NAME_SUFFIX)
	EXE_TARGET = mlp-sus2
        TARGET_EXE = mlp

       # obj directory name suffix 
        ifeq (1, $(USE_MPI)) 
          OBJ_SUFFIX = /mpi
        else
          OBJ_SUFFIX = /ser
        endif
#
    else ifneq (0, $(words $(filter libinterface,$(MAKECMDGOALS))))
        CXX = $(CXX_LIB)
        FC = $(FC_LIB)
          SRC_COMMON += $(SRC_DIR)/external/interface.cpp
        ifneq ($(USE_MLIP_PUBLIC), 1)
          SRC_EXTRA  += $(wildcard $(SRC_DEV_DIR)/utils/mtpr_train.cpp)
        endif
        ifneq (0, $(words $(filter -DMLIP_MPI,$(CXXFLAGS))))
          CXXFLAGS := $(filter-out -DMLIP_MPI, $(CXXFLAGS))
        endif

        CXXFLAGS += -fPIC
        CPPFLAGS += -fPIC
        FFLAGS += -fPIC
        LIB_TARGET = lib_mlip_interface.a
        TARGET_LIB = libinterface

       # obj directory name suffix 
        OBJ_SUFFIX = /ser

        ifneq (1, $(words $(filter -DMLIP_INTEL_MKL, $(CXXFLAGS))))
          SRC_BLAS := $(wildcard ./blas/*.f) 
          SRC_CBLAS := $(wildcard ./cblas/*.c) 
          SRC_CBLAS += $(wildcard ./cblas/*.f) 

          SRC_FILES += $(SRC_BLAS)
          SRC_FILES += $(SRC_CBLAS)

          OBJ_FILES += $(SRC_BLAS:./blas/%=%.o) 
          OBJ_FILES += $(SRC_CBLAS:./cblas/%=%.o) 

          TARGET_PRERQ =

        endif

   endif
   # use C++11 standart for c++ files
   CXXFLAGS += -std=c++11
   # this suppresses the error when .cfg files are read with "Stress" instead of "PlusStress"
   # this luxury is now removed - you were warned
   # CXXFLAGS += -DMLIP_NO_WRONG_STRESS_ERROR
   ifneq ($(USE_MLIP_PUBLIC), 1)
       CXXFLAGS += -DMLIP_DEV
   endif

   SRC_FILES += $(SRC_COMMON)
   SRC_FILES += $(SRC_EXTRA)

   OBJ_FILES += $(SRC_COMMON:$(SRC_DIR)/%=%.o) 
   OBJ_FILES += $(SRC_EXTRA:$(SRC_DEV_DIR)/%=%.o) 

else ifneq (0, $(words $(filter cblas,$(MAKECMDGOALS))))

   CC = $(CC_LIB)
   FC = $(FC_LIB)

   # source files
   SRC_BLAS := $(wildcard ./blas/*.f) 
   SRC_CBLAS := $(wildcard ./cblas/*.c) 
   SRC_CBLAS += $(wildcard ./cblas/*.f) 

   SRC_FILES := $(SRC_BLAS)
   SRC_FILES += $(SRC_CBLAS)

   OBJ_FILES := $(SRC_BLAS:./blas/%=%.o) 
   OBJ_FILES += $(SRC_CBLAS:./cblas/%=%.o) 

   CPPFLAGS += -fPIC
   FFLAGS += -fPIC

   LIB_TARGET = lib_mlip_cblas.a
   TARGET_LIB = cblas

   # obj directory name suffix 
   OBJ_SUFFIX = /cblas

   TARGET_PRERQ =

endif

NODEPS += distclean help

.PHONY: all
all:

# include base makefile 
# variables is to be exported to the  base makefile
#   SRC_FILES
#   OBJ_FILES
#   EXE_TARGET
#   TARGET_EXE
#   LIB_TARGET
#   TARGET_LIB
#   OBJ_SUFFIX
#
-include $(CURDIR)/make/Makefile

ifneq (0, $(words $(filter mlp libinterface,$(MAKECMDGOALS))))
# target to build inner blas
$(LIB_DIR)/lib_mlip_cblas.a:
	@ $(MAKE) --no-print-directory -f $(MK_THIS) cblas

endif

.PHONY: distclean
distclean: clean
	@ $(RM) -f make/config.mk

.PHONY: test
test:
	@ $(MAKE) --no-print-directory -C $(CURDIR)/test

.PHONY: clean-test
clean-test:
	@ $(MAKE) --no-print-directory -C $(CURDIR)/test clean

.PHONY: help
help:
	@echo ""
	@echo "make mlp            - builds mlp program"
	@echo "make libinterface   - builds interface library for external codes"
	@echo "make clean          - removes intermediate objects"
	@echo "make test           - run integration test"
	@echo "make help           - this info"
	@echo ""
