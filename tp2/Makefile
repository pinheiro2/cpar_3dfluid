CPP = g++ -Wall -march=native -funroll-loops -Ofast -ftree-vectorize
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	fluid_sim_seq phase2

fluid_sim_seq: $(CPP) $(SRCS) -o fluid_sim_seq

phase2: $(CPP) -fopenmp $(SRCS) -o fluid_sim

runseq:
	./fluid_sim_seq

runpar:
	export OMP_NUM_THREADS=16 ./fluid_sim
clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.
