#include <iostream>
#include "mpi.h"
using namespace std;

int main(int argc, char** argv)
{
	int no;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &no);

	//printf("Hello from thread %d\n", no);
	cout << "Hello from thread " << no << endl;

	MPI_Finalize();

	return 0;
}


