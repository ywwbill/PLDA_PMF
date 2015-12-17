#!/bin/tcsh
#SBATCH --nodes=$1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$2
#SBATCH -t 72:00:00

setenv OMP_NUM_THREADS $2

module load openmpi

mpic++ -fopenmp -o PLDATestHybrid PLDATestHybrid.cpp

module unload intel
module load openmpi/gnu

mpirun -n $1 ./PLDATestHybrid data/test_corpus.txt model/test_theta.txt data/LDA_model.txt 10 7167 500 0.1 0.1
