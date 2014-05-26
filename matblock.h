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
//#include <cblas.h>


// Block Matrix Product
void block_MatrixProd(double * A, double * B, double * C, int block_size);
int block_MatrixProd_GPU(double * A, double * B, double * C, int block_size, int rank);

// sub-Block Matrix Product for Cache & Register blocking
void Cacheblocking_MatrixProd(double ** A, double ** B, double ** C, int block_size, int i, int j, int k);
void MatrixProd(double * A, double * B, double * C, int block_size);

void create_MatBlock(double **A, const int block_size);
void free_MatBlock(double *A, const int block_size);



#endif /* MATBLOCK_H_ */
