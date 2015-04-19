/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;
//a overflows the stack as array. Malloc a instead. 
double** a; 

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
  {
//allocate a in here. Allocations do not seem to carry on to the inside of omp parallelizations.
a = (double **)malloc(N*sizeof(double*));
if (a == NULL) {
  printf("First allocation failed.\n");
}
for (i=0;i<N;i++)
{
  a[i] = (double*)malloc(N*sizeof(double));
  if (a[i] == NULL) {
    printf("%d th allocation failed.\n",i);
  }
}
  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  //added OMP barrier for debugging
  #pragma omp barrier

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      a[i][j] = tid + i + j;
    }
  }
  /* For confirmation */
  //changed printf flag to lf for double
  printf("Thread %d done. Last element= %lf\n",tid,a[N-1][N-1]);

  //free a after everything is done.
  for (i=0;i<N;i++) {
    free(a[i]);
  }
  free(a);
  }  /* All threads join master thread and disband */
  return 0;
}
