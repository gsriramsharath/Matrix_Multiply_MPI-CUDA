/*
 * worker.h
 *
 *  Created on: Feb 12, 2010
 *      Author: mhugues
 */

#ifndef WORKER_H_
#define WORKER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>



#include <unistd.h>

#include "matblock.h"



//MPI_Comm row,column;

void worker_exec(int rank, MPI_Comm row, MPI_Comm column, int I, int J, int block_size, int Nb_block);

#endif /* WORKER_H_ */
