/*
 * worker.c
 *
 *  Created on: May 25, 2014
 *      Author: Maxime Hugues <maxime.hugues inria.fr>
 */

#include "worker.h"
#include "matblock.h"

void worker_exec(int rank,  MPI_Comm row, MPI_Comm column,int I, int J, int block_size, int Nb_block)
{

	double * Alocal,*Blocal,*Clocal;
	double * Abuf, *Bbuf;
	int K=0;
        int i;

#ifdef DEBUG
	printf("Init Rank:%i Cart:%i,%i\n",rank,I,J);
#endif

	create_MatBlock(&Alocal, block_size);
	create_MatBlock(&Blocal, block_size);
	create_MatBlock(&Clocal, block_size);

	create_MatBlock(&Abuf, block_size);
	create_MatBlock(&Bbuf, block_size);

	for(i=0;i<block_size*block_size;i++)
	{
		Alocal[i]=1.0;
		Blocal[i]=2.0;
	}

#ifdef DEBUG
	printf("Start Computations Rank:%i \n",rank);
#endif

	while( K < Nb_block)
	{


		if(J==K)
		{
			memcpy(Abuf,Alocal,block_size*block_size*sizeof(double));
		}


		MPI_Bcast(Abuf,block_size*block_size,MPI_DOUBLE,K,row);

		if(I==K)
		{

			memcpy(Bbuf,Blocal,block_size*block_size*sizeof(double));
		}

		MPI_Bcast(Bbuf,block_size*block_size,MPI_DOUBLE,K,column);

		block_MatrixProd_GPU(Abuf,Bbuf,Clocal,block_size,rank);
		K++;
	}

	free_MatBlock(Abuf, block_size);
	free_MatBlock(Bbuf, block_size);

	free_MatBlock(Alocal, block_size);
	free_MatBlock(Blocal, block_size);
	free_MatBlock(Clocal, block_size);

}
