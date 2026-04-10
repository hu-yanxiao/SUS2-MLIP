#include "cblas.h"
#include "cblas_f77.h"
void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double  *A, const int lda,
                 const double  *X, const int incX, const double beta,
                 double  *Y, const int incY)
{
   char TA;
   #define F77_TA &TA   
   #define F77_M M
   #define F77_N N
   #define F77_lda lda
   #define F77_incX incX
   #define F77_incY incY

   if (order == CblasColMajor)
   {
      if (TransA == CblasNoTrans) TA = 'N';
      else if (TransA == CblasTrans) TA = 'T';
      else if (TransA == CblasConjTrans) TA = 'C';

      F77_dgemv(F77_TA, &F77_M, &F77_N, &alpha, A, &F77_lda, X, &F77_incX, 
                &beta, Y, &F77_incY);
   }
   else if (order == CblasRowMajor)
   {
      if (TransA == CblasNoTrans) TA = 'T';
      else if (TransA == CblasTrans) TA = 'N';
      else if (TransA == CblasConjTrans) TA = 'N';

      F77_dgemv(F77_TA, &F77_N, &F77_M, &alpha, A, &F77_lda, X,
                &F77_incX, &beta, Y, &F77_incY);
   }
   return;
}
