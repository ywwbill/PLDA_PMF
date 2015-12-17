#!/bin/bash
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

g++ -fopenmp -o PLDATestOpenMP PLDATestOpenMP.cpp

OMP_NUM_THREADS=$1
export OMP_NUM_THREADS

./PLDATestOpenMP data/test_corpus.txt model/test_theta.txt model/phi.txt 10 7167 500 0.1 0.1