#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

void jacobi(double* u_new, double* u, double h2,unsigned int dim)
{
  unsigned int i;
  u_new[0] = 0.5*(h2+u[1]);
  u_new[dim-1] = 0.5*(h2+u[dim-2]);
  #pragma omp for
  for (i = 1;i<dim-1;i++)
  {
    u_new[i] = 0.5*(h2+u[i-1]+u[i+1]);
  }
}

int main(int argc,char* argv[])
{

  /* compute time elapsed. The function is taken from the following URL:
     http://stackoverflow.com/questions/5248915/execution-time-of-c-program
  */
  struct timeval t1,t2;
  //dimension of matrix and vector, and number of iterations. Matrix is square matrix
  unsigned int dim,iter;
  if (argc == 0) {
    printf("No dim or iteration specified. Defaulting values dim = 100, iter = 10.\n");
    dim = 100;
    iter = 10;
  } else if (argc == 1) {
    printf("No iteration specified. Defaulting value iter = 10.\n");
    iter = 10;
    dim = atoi(argv[1]);
  } else {
    dim = atoi(argv[1]);
    iter = atoi(argv[2]);
  }
  if (dim < 0) {
    printf("Dim input is negative. Defaulting value dim = 100.\n");
    dim = 100;
  }
  if (iter <= 0) {
    printf("Iter is nonpositive. Defaulitng value iter = 10.\n");
    iter = 10;
  }

  double h = 1.0/(dim+1);

    //start time
    gettimeofday(&t1,NULL);

    double *u = malloc(sizeof(double)*dim);
    double f = h*h;
    //function f is 1, u^0 starts as 0 vec
    int i;
    #pragma omp for
    for (i = 0;i<dim;i++)
    {
      u[i] = 0.0;
    }

      //Jacobi

      /* Since we scale everything by h^2, we scale the residue accordingly
       so that it reflects Au = f rather than h^2*Au = h^2f
      */
      for (i = 0;i<iter;i++)
      { 
        double *un = malloc(sizeof(double)*dim);
        #pragma omp parallel shared(un)
        jacobi(un,u,f,dim);
        int k;
        #pragma omp for
        for (k = 0;k<dim;k++)
        {
          u[k] = un[k];
        }

        #pragma omp for
        for (k=0;k<dim;k++){
          printf("%lf ",u[k]);
        }
        printf("\n");
        free(un);
      }
    free(u);

    //end time
    gettimeofday(&t2,NULL);

    printf("Total iterations: %d\n",iter); 
    printf("Time elapsed: %lf sec.\n", (double)(t2.tv_usec-t1.tv_usec)/1000000 + (double)(t2.tv_sec - t1.tv_sec));

  return 0;
}
