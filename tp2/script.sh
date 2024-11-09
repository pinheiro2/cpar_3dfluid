#!/bin/bash
#SBATCH --time=1:00
#SBATCH --ntasks=1
#SBATCH --partition=cpar
#SBATCH --cpus-per-task=16

perf record ./fluid_sim
perf report --stdio>result
