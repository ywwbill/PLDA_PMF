#!/bin/tcsh
#SBATCH --nodes=$1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$2
#SBATCH -t 72:00:00

setenv OMP_NUM_THREADS $2

module load openmpi

mpic++ -fopenmp -o PLDATrainHybrid PLDATrainHybrid.cpp

module unload intel
module load openmpi/gnu

mpirun -n $1 ./PLDATrainHybrid data/train_corpus.txt model/train_theta.txt model/phi.txt 10 7167 500 0.1 0.1
