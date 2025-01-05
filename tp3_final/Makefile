################################################################################
# Simplified Makefile for fluid simulation
################################################################################

SHELL = /bin/sh
BIN_NAME = fluid_simulation

CXX = nvcc
LD  = nvcc

CXXFLAGS   = -O2 -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets --restrict --use_fast_math

SRC_DIR = src
SRC = $(SRC_DIR)/main.cu $(SRC_DIR)/fluid_solver.cu $(SRC_DIR)/EventManager.cpp
SBATCH_SCRIPT = $(SRC_DIR)/runcuda.sh

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

all: $(BIN_NAME)

$(BIN_NAME): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

run: $(BIN_NAME)
	cd $(SRC_DIR) && sbatch runcuda.sh

clean:
	rm -f $(BIN_NAME)
