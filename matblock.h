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

//#include <cblas.h>

int compute_dgemm(double *A, double *B, double *C, int Nb_block, int block_size, int rank);

// Block Matrix Product
void block_MatrixProd(double * A, double * B, double * C, int block_size);
int block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank);

void alloc_MatBlock(double **A, const int block_size);
void free_MatBlock(double *A, const int block_size);



#endif /* MATBLOCK_H_ */
