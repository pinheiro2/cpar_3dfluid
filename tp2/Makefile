# Compiler and flags
CPP = g++ -Wall -march=native -funroll-loops -Ofast -ftree-vectorize
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

# Phony targets
.PHONY: all fluid_sim_seq phase2 runseq runpar clean

# Build all targets
all: fluid_sim_seq phase2

# Build sequential version
fluid_sim_seq:
	$(CPP) $(SRCS) -o fluid_sim_seq

# Build parallel version
phase2:
	$(CPP) -fopenmp $(SRCS) -o fluid_sim

# Run sequential version
runseq:
	./fluid_sim_seq

# Run parallel version with 16 threads
runpar:
	OMP_NUM_THREADS=16 ./fluid_sim

# Clean up generated files
clean:
	@echo Cleaning up...
	@rm -f fluid_sim_seq fluid_sim
	@echo Done.
