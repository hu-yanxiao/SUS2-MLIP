#include "cblas.h"
#include "cblas_f77.h"
void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double  *X, const int incX,
                const double  *Y, const int incY, double  *A, const int lda)
{
   #define F77_M M
   #define F77_N N
   #define F77_incX incX
   #define F77_incY incY
   #define F77_lda lda

   if (order == CblasColMajor)
   {
      F77_dger( &F77_M, &F77_N, &alpha, X, &F77_incX, Y, &F77_incY, A, 
                      &F77_lda);
   }
   else if (order == CblasRowMajor)
   {
      F77_dger( &F77_N, &F77_M ,&alpha, Y, &F77_incY, X, &F77_incX, A, 
                      &F77_lda);
   }
   return;
}
