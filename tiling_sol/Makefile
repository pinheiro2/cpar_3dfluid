# Define the compiler
CPP = g++ -Wall
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all: \
	fluid_sim_Ofast_march_native \
	fluid_sim_Ofast_mavx \
	fluid_sim_Ofast_mavx2 \
	fluid_sim_Ofast_msse2 \
	fluid_sim_Ofast_msse3 \
	fluid_sim_Ofast_msse \
	fluid_sim_Ofast_mavx512 

# Compile commands for each combination
fluid_sim_Ofast_march_native:
	$(CPP) $(SRCS) -Ofast -march=native -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_march_native

fluid_sim_Ofast_mavx:
	$(CPP) $(SRCS) -Ofast -mavx -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_mavx

fluid_sim_Ofast_mavx2:
	$(CPP) $(SRCS) -Ofast -mavx2 -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_mavx2

fluid_sim_Ofast_msse2:
	$(CPP) $(SRCS) -Ofast -msse2 -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_msse2

fluid_sim_Ofast_msse3:
	$(CPP) $(SRCS) -Ofast -msse3 -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_msse3

fluid_sim_Ofast_msse:
	$(CPP) $(SRCS) -Ofast -msse -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_msse

fluid_sim_Ofast_mavx512:
	$(CPP) $(SRCS) -Ofast -mavx512 -funroll-loops -ftree-vectorize -o fluid_sim_Ofast_mavx512

.PHONY: clean
clean:
	@echo Cleaning up...
	@rm -f fluid_sim_*  # Remove all generated executables
	@echo Done.
