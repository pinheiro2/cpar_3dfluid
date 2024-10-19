#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=1
#SBATCH --partition=cpar

perf record ./fluid_sim
perf report --stdio>result
