/*
 * main.c
 *
 *  Created on: Jan 25, 2010
 *      Author: mhugues
 *
 *    Last modified on April 2nd 2010
 */


#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include "perf.h"
#include "worker.h"


int main(int argc, char ** argv)
{

	int rank, cluster_size;
	int i,j,k;
	// Cartesian index
	int I,J;
MPI_Comm row,column;
	int Nb_block =0;
	int block_size =0;

	int tag=1;
	int provided;
	double start=0,stop;

	//Init
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD , &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);


	if(argc < 3)
	{
		printf("Usage MatProd Nb_block Block_Size\n");
		exit(1);
	}

	Nb_block=(int)strtol(argv[1],(char**)NULL,10);
	block_size=(int)strtol(argv[2],(char**)NULL,10);

	if(cluster_size < Nb_block*Nb_block)
	{
		printf("Resources are insufficient\n");
		exit(2);
	}

	if(rank==0)
	{
		printf("Matrix Block Product Parameters\n");
		printf("\t Nodes:%i Blocks:%i Block Size:%i\n",cluster_size,Nb_block,block_size);
	}

	I=rank/Nb_block;
	J=rank%Nb_block;

	MPI_Comm_split(MPI_COMM_WORLD,I,J,&row);
	MPI_Comm_split(MPI_COMM_WORLD,J,I,&column);



	if(rank==0) start=MPI_Wtime();

	worker_exec(rank, row, column,  I,J, block_size, Nb_block);

#ifdef DEBUG
	printf("Rank:%i Finished\n",rank);
#endif

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank==0)
	{
		stop=MPI_Wtime();
		perf_matbloc(start,stop,Nb_block,block_size);

	}


	MPI_Finalize();

	return 0;
}

