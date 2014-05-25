/*
 * matblock.h
 *
 *  Created on: Feb 12, 2010
 *      Author: mhugues
 */

#ifndef MATBLOCK_H_
#define MATBLOCK_H_



#include <stdlib.h>
//#include <cblas.h>


// Block Matrix Product
void block_MatrixProd(double * A, double * B, double * C, int block_size);
void block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank);

// sub-Block Matrix Product for Cache & Register blocking
void Cacheblocking_MatrixProd(double ** A, double ** B, double ** C, int block_size, int i, int j, int k);
void MatrixProd(double * A, double * B, double * C, int block_size);

void create_MatBlock(double **A, const int block_size);
void free_MatBlock(double *A, const int block_size);



#endif /* MATBLOCK_H_ */
