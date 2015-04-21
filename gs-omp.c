#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

void gauss_seidel(double* u_new, double* u, double h2,unsigned int dim)
{
  //red-black grid parallelized algo
  unsigned int i;
  u_new[0] = 0.5*(h2+u[1]);
  //first update the red (even) entries
  #pragma omp for
  for (i = 2;i<dim-1;i+=2)
  {
    u_new[i] = 0.5*(h2+u[i-1]+u[i+1]);
  }
  if (dim % 2 == 1) u_new[dim-1] = 0.5*(h2+u[dim-2]);
  //then update the black (odd) entries with new reds
  #pragma omp for
  for (i = 1;i<dim-1;i+=2)
  {
    u_new[i] = 0.5*(h2+u_new[i-1]+u_new[i+1]);
  }
  if (dim % 2 == 0) u_new[dim-1] = 0.5*(h2+u_new[dim-2]);
}

int main(int argc,char* argv[])
{

  /* compute time elapsed. The function is taken from the following URL:
     http://stackoverflow.com/questions/5248915/execution-time-of-c-program
  */
  struct timeval t1,t2;
  //dimension of matrix and vector, and number of iterations. Matrix is square matrix
  unsigned int dim,iter;

  //verbose flag
  char verbose = 0;
  //initialize values
  if (argc == 1) {
    printf("No dim or iteration specified. Defaulting values dim = 100, iter = 10.\n");
    dim = 100;
    iter = 10;
  } else if (argc == 2) {
    dim  = atoi(argv[1]);
    printf("No iteration specified. Defaulting value iter = 10.\n");
    iter = 10;
  } else if (argc == 3) {
    dim = atoi(argv[1]);
    iter = atoi(argv[2]);
  } else {
    dim = atoi(argv[2]);
    iter = atoi(argv[3]);
    if (strcmp(argv[1],"-v") == 0) verbose = 1;
  }
  if (dim < 0) {
    printf("Dim input is negative. Defaulting value dim = 100.\n");
    dim = 100;
  }
  if (iter <= 0) {
    printf("Iter is nonpositive. Defaulting value iter = 10.\n");
    iter = 10;
  }

  double h = 1.0/(dim+1);

    //start time
    gettimeofday(&t1,NULL);

    double *u = malloc(sizeof(double)*dim);
    //function f is 1, u^0 starts as 0 vec
    double f = h*h;
    int i;
    #pragma omp for
    for (i = 0;i<dim;i++) u[i] = 0.0;

    //solve u. Use argv[2] to pick numerical algorithm.
    /* in both cases, Au^0 = 0, so res_0 is just the norm of f
       For f = 1, the norm^2 would be dim, so norm(f,dim) = sqrt(dim)
    */
      //Gauss-Seidel
      for (i=0;i<iter;i++)
      {
        double *un = malloc(sizeof(double)*dim);
        #pragma omp parallel shared(un)
        gauss_seidel(un,u,f,dim);
        unsigned int k;
        #pragma omp for
        for (k = 0;k<dim;k++) u[k] = un[k];

        free(un);
      }
    free(u);
        //print out vectors to check.
        if (verbose) {
        int k;
        #pragma omp for
        for (k=0;k<dim;k++) printf("%lf ",u[k]);
        printf("\n");
        }

    //end time
    gettimeofday(&t2,NULL);

    printf("Time elapsed: %lf sec.\n", (double)(t2.tv_usec-t1.tv_usec)/1000000 + (double)(t2.tv_sec - t1.tv_sec));
    printf("Total iterations: %d\n",iter); 

  return 0;
}
