BATCH --ntasks=40
#SBATCH -t 00:01:00

module unload intel
module load openmpi/gnu

time mpirun -n 17 pmf /homes/cmsc-xi5/PLDA_PMF/data/train_corpus.txt /homes/cmsc-xi5/PLDA_PMF/data/test_corpus.txt

