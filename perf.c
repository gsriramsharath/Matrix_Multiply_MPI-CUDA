#include "perf.h"

//Print performance in Gflops
//elapsedTime in ms for one iteration
void perf_spmv(struct timeval start, struct timeval stop, int Nb_block, int block_size)
{
	double Time=0.0;

	Time=(double)((stop.tv_sec-start.tv_sec)+(stop.tv_usec-start.tv_usec)/1000000.0);


	//Time for one computation
	Time=2.0*powf((float)(Nb_block*block_size),(float)3)/Time/1e9;

	printf("\tPerformance [%.3f GFLOPS]\n\n",Time);

}

void perf_matbloc(double start, double stop, int Nb_block, int block_size)
{
	double Time=0.0;
	FILE *fd;
	Time=stop-start;

	//Time for one computation
	Time=2.0*pow((double)(Nb_block*block_size),(double)3)/Time/1e9;

	printf("Performance [%.3f GFLOPS]\n\n",Time);


    fd=fopen("MatProdIO_BLAS_results","a");
    fseek(fd,0L,SEEK_END);
    fprintf(fd,"%i;%.3f;\n",block_size, Time);
    fclose(fd);

}
