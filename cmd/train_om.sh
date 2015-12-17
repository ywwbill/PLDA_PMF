#!/bin/bash
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

g++ -fopenmp -o PLDATrainOpenMP PLDATrainOpenMP.cpp

OMP_NUM_THREADS=$1
export OMP_NUM_THREADS

./PLDATrainOpenMP data/train_corpus.txt model/train_theta.txt model/phi.txt 10 7167 500 0.1 0.1