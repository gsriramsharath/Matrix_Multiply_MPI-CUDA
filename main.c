/*
 * main.c
 *
 *  Created on: May 25, 2014
 *      Author: Maxime Hugues <maxime.hugues inria.fr>
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "perf.h"
#include "matblock.h"


int main(int argc, char ** argv)
{
	// Matrix Dimension
	int Nb_block =0;
	int block_size =0;
	double * Alocal, *Blocal, *Clocal;

	// MPI
	int rank, communicator_size;	
	int tag=1;

	// Cartesian index
	int I,J;
	MPI_Comm row,column;

	// Performance timers
	double start=0,stop;


	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&communicator_size);


	if(argc < 3)
	{
		printf("Usage MatProd Nb_block Block_Size\n");
		exit(1);
	}

	Nb_block=(int)strtol(argv[1],(char**)NULL,10);
	block_size=(int)strtol(argv[2],(char**)NULL,10);

	if(communicator_size < Nb_block*Nb_block)
	{
		printf("Resources are insufficient\n");
		exit(2);
	}

	if(rank==0)
	{
		printf("Matrix Block Product Parameters\n");
		printf("\t Nodes:%i Blocks:%i Block Size:%i\n",communicator_size,Nb_block,block_size);
	}

	alloc_MatBlock(&Alocal, block_size);
	alloc_MatBlock(&Blocal, block_size);
	alloc_MatBlock(&Clocal, block_size);


	if(rank==0) start=MPI_Wtime();

	compute_dgemm(Alocal, Blocal, Clocal, Nb_block, block_size, rank);

#ifdef DEBUG
	printf("Rank:%i Finished\n",rank);
#endif

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0)
	{
		stop=MPI_Wtime();
		perf_matbloc(start,stop,Nb_block,block_size);

	}

	free_MatBlock(Alocal, block_size);
	free_MatBlock(Blocal, block_size);
	free_MatBlock(Clocal, block_size);


	MPI_Finalize();

	return 0;
}

