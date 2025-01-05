################################################################################
# Simplified Makefile for fluid simulation
################################################################################

SHELL = /bin/sh
BIN_NAME = fluid_simulation

CXX = nvcc
LD  = nvcc

CXXFLAGS   = -O2 -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets --restrict --use_fast_math

SRC = main.cu fluid_solver.cu EventManager.cpp

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

all: $(BIN_NAME)

$(BIN_NAME): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f $(BIN_NAME)
