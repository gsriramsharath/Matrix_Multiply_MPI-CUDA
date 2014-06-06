#include "matblock.h"

int compute_dgemm(double *A, double *B, double *C, int Nb_block, int block_size, int rank)
{

	double * Abuf, *Bbuf;
	double alpha=1.0, beta=0.0;
	int K=0;
    int i, err;

	double *d_A, *d_B, *d_C;
	cublasHandle_t context;

    // Cartesian Index
    MPI_Comm row, column;
    int I, J;

	I=rank/Nb_block;
	J=rank%Nb_block;

	MPI_Comm_split(MPI_COMM_WORLD,I,J,&row);
	MPI_Comm_split(MPI_COMM_WORLD,J,I,&column);

	setDevice(rank);

#ifdef DEBUG
	printf("Init Rank:%i Cart:%i,%i\n",rank,I,J);
#endif

	alloc_MatBlock(&Abuf, block_size);
	alloc_MatBlock(&Bbuf, block_size);

	create_context(&context);

	err = alloc_MatBlock_device(&d_A, block_size*block_size);
	if (err != EXIT_SUCCESS)
	{
		printf("Memory allocation of A on device failed\n");
		return EXIT_FAILURE;
	}
	err = alloc_MatBlock_device(&d_B, block_size*block_size);
	if (err != EXIT_SUCCESS)
	{
		printf("Memory allocation of B on device failed\n");
		return EXIT_FAILURE;
	}
	err = alloc_MatBlock_device(&d_C, block_size*block_size);
	if (err != EXIT_SUCCESS)
	{
		printf("Memory allocation of C on device failed\n");
		return EXIT_FAILURE;
	}


	for(i=0;i<block_size*block_size;i++)
	{
		A[i]=1.0;
		B[i]=2.0;
	}

	err = copy_data_toDevice(C, d_C, block_size*block_size);
	if (err != EXIT_SUCCESS)
	{
		printf("Data transfer of C to device failed\n");
		return EXIT_FAILURE;
	}

#ifdef DEBUG
	printf("Start Computations Rank:%i \n",rank);
#endif

	while( K < Nb_block)
	{


		if(J==K) memcpy(Abuf,A,block_size*block_size*sizeof(double));

		MPI_Bcast(Abuf,block_size*block_size,MPI_DOUBLE,K,row);

		if(I==K) memcpy(Bbuf,B,block_size*block_size*sizeof(double));

		MPI_Bcast(Bbuf,block_size*block_size,MPI_DOUBLE,K,column);

		err = copy_data_toDevice(Abuf, d_A, block_size*block_size);
		if (err != EXIT_SUCCESS)
		{
			printf("Data transfer of A to device failed\n");
			return EXIT_FAILURE;
		}

		err = copy_data_toDevice(Bbuf, d_B, block_size*block_size);
		if (err != EXIT_SUCCESS)
		{
			printf("Data transfer of B to device failed\n");
			return EXIT_FAILURE;
		}

		err = block_MatrixProd_GPU(&context,alpha,d_A,d_B,beta,d_C,block_size);
		if (err != EXIT_SUCCESS)
		{
			printf("Computation on device failed\n");
			return EXIT_FAILURE;
		}

		K++;
	}

	err = copy_data_fromDevice(C, d_C, block_size*block_size);
	if (err != EXIT_SUCCESS)
	{
		printf("Data transfer of C from device failed\n");
		return EXIT_FAILURE;
	}

	destroy_context(&context);

	free_MatBlock_device(d_A);
	free_MatBlock_device(d_B);
	free_MatBlock_device(d_C);

	free_MatBlock(Abuf);
	free_MatBlock(Bbuf);

	return EXIT_SUCCESS;

}


void block_MatrixProd(double * A, double * B, double * C, int block_size)
{


//	cblas_dgemm(CblasRowMajor, CblasNoTrans,  CblasTrans, block_size, block_size, block_size, 1.0, A, block_size,B, block_size, 1.0, C, block_size);

	return;

}

int block_MatrixProd_GPU(cublasHandle_t *context, double alpha, double * A, double * B,
						 double beta, double * C, int block_size)
{

	cublasStatus_t stat;

	stat = cublasDgemm(*context, CUBLAS_OP_N, CUBLAS_OP_N, block_size, block_size, block_size, 
					   &alpha, A, block_size, B, block_size, &beta, C, block_size);
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		printf("DGEMM Computation failed\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;

}

void alloc_MatBlock(double **A, const int block_size)
{
	int i=0;
	*A=(double*)malloc(block_size*block_size*sizeof(double));

	for(i=0;i<block_size*block_size;i++)
		(*A)[i]=0.0;

}

void free_MatBlock(double *A)
{
	free(A);
}

int alloc_MatBlock_device(double ** d_A, const int size)
{
	cudaError_t cudaStat;

	cudaStat = cudaMalloc((void**)d_A, size*sizeof(double));
	if (cudaStat != cudaSuccess)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;

}

int copy_data_toDevice(double * A, double * d_A, const int size)
{
	cublasStatus_t stat;
	stat = cublasSetVector(size, sizeof(double), A, 1, d_A, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}

int copy_data_fromDevice(double * A, double * d_A, const int size)
{
	cublasStatus_t stat;
	stat = cublasGetVector(size, sizeof(double), d_A, 1, A, 1);
	if (stat != CUBLAS_STATUS_SUCCESS)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}

void free_MatBlock_device(double * d_A)
{
	cudaFree(d_A);
}

int create_context(void *context)
{
	cublasStatus_t stat;
	stat = cublasCreate((cublasHandle_t *)context);
	if (stat != CUBLAS_STATUS_SUCCESS)
		return EXIT_FAILURE;

	return EXIT_SUCCESS;
}

void destroy_context(void *context)
{
	cublasHandle_t *tmp = context;
	cublasDestroy(*tmp);
}

void setDevice(int id)
{
	cudaSetDevice(id%GPUPERNODE);
}




