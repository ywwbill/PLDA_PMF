#!/bin/tcsh
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

module unload intel
module load openmpi/gnu

mpirun -n 8 ./PLDATrain train_corpus.txt train_par_theta.txt par_phi.txt 10 7167 100 0.1 0.1
