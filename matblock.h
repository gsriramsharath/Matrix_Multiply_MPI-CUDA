/*
 * matblock.h
 *
 *  Created on: May 25, 2014
 *      Author: Maxime Hugues <maxime.hugues inria.fr>
 */

#ifndef MATBLOCK_H_
#define MATBLOCK_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define GPUPERNODE 2

//#include <cblas.h>

int compute_dgemm(double *A, double *B, double *C, int Nb_block, int block_size, int rank);

// Block Matrix Product
void block_MatrixProd(double * A, double * B, double * C, int block_size);
int block_MatrixProd_GPU(cublasHandle_t *context, double alpha, double * A, double * B, 
						 double beta, double * C, int block_size);

void alloc_MatBlock(double **A, const int block_size);
void free_MatBlock(double *A);

// GPU functions
void setDevice(int id);
int create_context(void *context);
void destroy_context(void *context);

int alloc_MatBlock_device(double ** d_A, const int size);
void free_MatBlock_device(double * d_A);

int copy_data_toDevice(double * A, double * d_A, const int size);
int copy_data_fromDevice(double * A, double * d_A, const int size);

#endif /* MATBLOCK_H_ */