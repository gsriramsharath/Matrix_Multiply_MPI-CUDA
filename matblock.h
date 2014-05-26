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

void compute_dgemm(int rank,  MPI_Comm row, MPI_Comm column,int I, int J, int block_size, int Nb_block);

// Block Matrix Product
void block_MatrixProd(double * A, double * B, double * C, int block_size);
int block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank);

void create_MatBlock(double **A, const int block_size);
void free_MatBlock(double *A, const int block_size);



#endif /* MATBLOCK_H_ */
