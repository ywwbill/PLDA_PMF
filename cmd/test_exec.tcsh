#!/bin/tcsh
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

module unload intel
module load openmpi/gnu

mpirun -n 1 ./PLDATest data/test_corpus.txt model/test_theta.txt model/phi.txt 10 7167 500 0.1 0.1
