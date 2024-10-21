CPP = g++ -Wall 
SRCS = main.cpp fluid_solver.cpp EventManager.cpp

all:
	$(CPP) $(SRCS) -march=native -funroll-loops -Ofast -ftree-vectorize -o fluid_sim
	
clean:
	@echo Cleaning up...
	@rm fluid
	@echo Done.
