#ifndef PERF_H
#define PERF_H

#include <stdio.h>
#include <sys/time.h>
#include <math.h>

//Print performance in Gflops of the SparseMatrixVector
//elapsedTime in ms for one iteration
void perf_spmv(struct timeval start, struct timeval stop, int Nb_block, int block_size);
void perf_matbloc(double start, double stop, int Nb_block, int block_size);


#endif
