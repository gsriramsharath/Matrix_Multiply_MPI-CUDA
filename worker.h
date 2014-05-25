/*
 * worker.h
 *
 *  Created on: May 25, 2014
 *      Author: Maxime Hugues <maxime.hugues inria.fr>
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

void worker_exec(int rank, MPI_Comm row, MPI_Comm column, int I, int J, int block_size, int Nb_block);

#endif /* WORKER_H_ */
