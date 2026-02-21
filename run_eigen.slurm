#!/bin/bash
#SBATCH --job-name=eigen_vals
#SBATCH --output=eigen_%j.out
#SBATCH --time=0:15:00
#SBATCH --ntasks=1
#SBATCH --partition=cpu-preempt

for N in 10 20 40 80 160 320 640
do
    python src/eigen.py --N $N --potential well --n_eigs 5
done
