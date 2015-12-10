module load openmpi

mpic++ -o pmf src/pmf.cpp src/pmf.h src/mf.h src/mf_common.cpp src/pmf_block.cpp src/pmf_block.h -std=c++0x 

