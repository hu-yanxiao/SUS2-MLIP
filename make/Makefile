#
# Makefile
#
# This makefile is a base makefile script that create object files,
# library and executables files. Imported variables and targets shoud be
# defined in external script.
# 
# 
#  Imported variables:
#   SRC_FILES   - define the source files
#   OBJ_FILES   - define the object files
#   OBJ_SUFFIX  - define the suffix of the name of obj directory
#
# Imported phony targets:
#   TARGET_EXE 	- the target of executable
#   TARGET_LIB  - the target of library 
#
# Imported File name targets:
#   EXE_TARGET  - executable file target
#   LIB_TARGET  - library file target
#

# ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST)) ) )
	
OBJ_DIR := $(CURDIR)/obj$(OBJ_SUFFIX)

OBJ_FILES := $(addprefix $(OBJ_DIR)/,$(OBJ_FILES) )
DEP_FILES := $(OBJ_FILES:.o=.d)

EXE_TARGET := $(addprefix $(BIN_DIR)/,$(EXE_TARGET) )
LIB_TARGET := $(addprefix $(LIB_DIR)/,$(LIB_TARGET) )

OBJ_SUBDIRS := $(sort $(dir $(OBJ_FILES)) )

VPATH += $(sort $(dir $(SRC_FILES)) )

# vpath %.o $(OBJ_SUBDIRS)

.SUFFIXES:
.SUFFIXES: .cpp .c .h .f .f90 .o .d

.PHONY: clean 
.PHONY: $(TARGET_LIB) $(TARGET_EXE) $(OBJ_SUBDIRS) $(BIN_DIR) $(LIB_DIR)
.PRECIOUS: $(OBJ_DIR)/%.cpp.d $(OBJ_DIR)/%.c.d

NODEPS += clean

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
CPPFLAGS += -MMD 
endif

$(TARGET_EXE): $(OBJ_SUBDIRS) $(BIN_DIR) $(TARGET_PRERQ) $(EXE_TARGET) 
$(TARGET_LIB): $(OBJ_SUBDIRS) $(LIB_DIR) $(TARGET_PRERQ) $(LIB_TARGET)

clean:
	@$(RM) -rf $(CURDIR)/obj $(LIB_DIR) $(BIN_DIR)

$(EXE_TARGET): $(OBJ_FILES) 
	$(CXX) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(LIB_TARGET): $(OBJ_FILES) #$(LIB_BLAS)
	$(AR) -rs $@ $^

$(OBJ_SUBDIRS) $(BIN_DIR) $(LIB_DIR):
	@mkdir -p $@

$(OBJ_DIR)/%.f.o: %.f $(PREREQ)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJ_DIR)/%.f90.o: %.f90 $(PREREQ)
	$(FC) $(FFLAGS) -c $< -o $@

$(OBJ_DIR)/%.c.o: %.c $(PREREQ)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cpp.o: %.cpp $(PREREQ) 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
-include $(DEP_FILES)
endif

