#include "cblas.h"
#include "cblas_f77.h"
CBLAS_INDEX cblas_idamax( const int N, const double *X, const int incX)
{
   int iamax;
   #define F77_N N
   #define F77_incX incX
   F77_idamax_sub( &F77_N, X, &F77_incX, &iamax);
   return iamax ? iamax-1 : 0;
}
