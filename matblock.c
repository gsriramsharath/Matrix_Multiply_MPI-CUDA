#include "matblock.h"

void compute_dgemm(int rank,  MPI_Comm row, MPI_Comm column,int I, int J, int block_size, int Nb_block)
{

	double * Alocal,*Blocal,*Clocal;
	double * Abuf, *Bbuf;
	int K=0;
    int i;

#ifdef DEBUG
	printf("Init Rank:%i Cart:%i,%i\n",rank,I,J);
#endif

	alloc_MatBlock(&Alocal, block_size);
	alloc_MatBlock(&Blocal, block_size);
	alloc_MatBlock(&Clocal, block_size);

	alloc_MatBlock(&Abuf, block_size);
	alloc_MatBlock(&Bbuf, block_size);

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




void block_MatrixProd(double * A, double * B, double * C, int block_size)
{


//	cblas_dgemm(CblasRowMajor, CblasNoTrans,  CblasTrans, block_size, block_size, block_size, 1.0, A, block_size,B, block_size, 1.0, C, block_size);

	return;

}

int block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank)
{
	double *d_A, *d_B, *d_C;
	double alpha=1.0, beta=0.0;
	int full_size=block_size*block_size;

	cudaError_t cudaStat;
	cublasStatus_t stat;
	cublasHandle_t handle;

	cudaSetDevice(rank%2);

	cudaStat = cudaMalloc((void**)&d_A, full_size*sizeof(double));
	if (cudaStat != cudaSuccess)
	{
		printf("Memory alloction of A on device failed\n");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&d_B, full_size*sizeof(double));
	if (cudaStat != cudaSuccess)
	{
		printf("Memory alloction of B on device failed\n");
		return EXIT_FAILURE;
	}
	cudaStat = cudaMalloc((void**)&d_C, full_size*sizeof(double));
	if (cudaStat != cudaSuccess)
	{
		printf("Memory alloction of C on device failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("CUBLAS Initialization failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasSetVector(full_size, sizeof(double), A, 1, d_A, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("Data transfer of A to device failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasSetVector(full_size, sizeof(double), B, 1, d_B, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("Data transfer of B to device failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasSetVector(full_size, sizeof(double), C, 1, d_C, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("Data transfer of C to device failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, block_size, block_size, block_size, 
					   &alpha, d_A, block_size, d_B, block_size, &beta, d_C, block_size);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("DGEMM Computation failed\n");
		return EXIT_FAILURE;
	}

	stat = cublasGetVector(full_size, sizeof(double), d_C, 1, C, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("Data transfer of C from device failed\n");
		return EXIT_FAILURE;
	}

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cublasDestroy(handle);

	return EXIT_SUCCESS;

}

void alloc_MatBlock(double **A, const int block_size)
{
	int i=0;
	*A=(double*)malloc(block_size*block_size*sizeof(double));

	for(i=0;i<block_size*block_size;i++)
		(*A)[i]=0.0;

}

void free_MatBlock(double *A, const int block_size)
{
	free(A);
}


