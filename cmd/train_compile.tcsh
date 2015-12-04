#!/bin/tcsh
#SBATCH --ntasks=40
#SBATCH -t 00:01:00

module load openmpi
mpic++ -o PLDATrain PLDATrain.cpp

