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

void block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank)
{
	double *d_A, *d_B, *d_C;
	int full_size=block_size*block_size;
	cublasHandle_t handle;

	cudaSetDevice(rank%2);

	//cublasCreate(&handle);

	cublasAlloc(full_size, sizeof(double), (void**)&d_A);
	cublasAlloc(full_size, sizeof(double), (void**)&d_B);
	cublasAlloc(full_size, sizeof(double), (void**)&d_C);

	cublasSetVector(full_size, sizeof(double), A, 1, d_A, 1);
	cublasSetVector(full_size, sizeof(double), B, 1, d_B, 1);
	cublasSetVector(full_size, sizeof(double), C, 1, d_C, 1);

	float t0, error_norm=0, ref_norm=0;
	cudaThreadSynchronize();

	cublasDgemm('n', 'n', block_size, block_size, block_size, 1.0f, d_A, block_size, d_B, block_size, 0.0f, d_C, block_size);

	cudaThreadSynchronize();

	cublasGetVector(full_size, sizeof(double), d_C, 1, C, 1);

	cublasFree(d_A);
	cublasFree(d_B);
	cublasFree(d_C);


	return;

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
	int i;
	free(A);

}


