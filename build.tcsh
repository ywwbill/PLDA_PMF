module load openmpi

mpic++ -o pmf src/pmf.cpp src/pmf.h src/mf.h src/mf_common.cpp -std=c++0x 

