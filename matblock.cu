#include "matblock.h"
#include <cuda_runtime.h>
#include "cublas.h"

void MatrixProd(double * A, double * B, double * C, int block_size)
{
	int i,j,k;
	double temp;


	for(i=0; i < block_size; i++)
		for(j=0; j< block_size; j++)
		{
			temp=0.0;

			for(k=0; k< block_size; k++)
			{
				temp+=A[i*block_size+k] * B[k*block_size+j];

			}
			C[i*block_size+j]=temp;

		}

	return;
}

void block_MatrixProd(double * A, double * B, double * C, int block_size)
{
	MatrixProd(A,B,C,block_size);

//	cblas_dgemm(CblasRowMajor, CblasNoTrans,  CblasTrans, block_size, block_size, block_size, 1.0, A, block_size,B, block_size, 1.0, C, block_size);

	return;

}

int block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank)
{
	double *d_A, *d_B, *d_C;
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

	stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, block_size, block_size, block_size, 1.0f, d_A, block_size, 
				d_B, block_size, 0.0f, d_C, block_size);
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

void create_MatBlock(double **A, const int block_size)
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


