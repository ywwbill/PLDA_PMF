#!/bin/tcsh
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

module unload intel
module load openmpi/gnu

mpirun -n 1 ./PLDATrain data/train_corpus.txt model/train_theta.txt model/phi.txt 10 7167 500 0.1 0.1
