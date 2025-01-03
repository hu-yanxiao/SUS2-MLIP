# Here the automatic variables are set for all Makefiles

ifneq (0, $(words $(filter -DMLIP_MPI,$(CXXFLAGS))))
    USE_MPI = 1
endif

