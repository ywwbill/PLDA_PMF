#!/bin/tcsh
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

module unload intel
module load openmpi/gnu

mpirun -n 1 ./PLDATest test_corpus.txt test_seq_theta.txt seq_phi.txt 10 7167 100 0.1 0.1
