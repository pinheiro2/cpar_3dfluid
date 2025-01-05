#!/bin/bash
#SBATCH --output=fluid_simulation.out  # Nome do arquivo de saída
#SBATCH --partition day
#SBATCH --constraint=k20
#SBATCH --ntasks=1
#SBATCH --time=5:00

# Carregar módulos necessários (ajuste para o seu sistema)
module load gcc/7.2.0
module load cuda/11.3.1         

# Executar o programa CUDA
perf stat -e instructions,cycles ./fluid_simulation
