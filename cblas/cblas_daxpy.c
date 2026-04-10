#include "cblas.h"
#include "cblas_f77.h"
void cblas_daxpy( const int N, const double alpha, const double *X,
                       const int incX, double *Y, const int incY)
{
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
   F77_daxpy( &F77_N, &alpha, X, &F77_incX, Y, &F77_incY);
} 
