#!/bin/bash
#SBATCH --output=fluid_simulation.out  # Output file in the working directory
#SBATCH --partition=day               # Queue/partition name
#SBATCH --constraint=k20              # Specific GPU constraint
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --time=5:00                   # Max runtime (hh:mm:ss)

# Run the CUDA executable
perf stat -e instructions,cycles ../fluid_simulation

# Move the output file to the main folder
mv fluid_simulation.out ../fluid_simulation.out
