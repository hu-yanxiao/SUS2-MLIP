#ifndef CBLAS_F77_H
#define CBLAS_F77_H

#include <stdlib.h>
#include <string.h>

   #define FCHAR char *

   #define FINT const int *
   #define FINT2 int *

/*
 * Level 1 BLAS
 */
   #define F77_dswap      dswap_
   #define F77_daxpy      daxpy_
   #define F77_idamax_sub idamaxsub_

/*
 * Level 2 BLAS
 */
   #define F77_dger       dger_
   #define F77_dgemv      dgemv_

/*
 * Level 3 BLAS
 */
   #define F77_dgemm      dgemm_

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Level 1 Fortran Prototypes
 */
   void F77_dswap( FINT, double *, FINT, double *, FINT);
   void F77_daxpy( FINT, const double *, const double *, FINT, double *, FINT);
   void F77_idamax_sub( FINT, const double * , FINT, FINT2);

/*
 * Level 2 Fortran Prototypes
 */
   void F77_dgemv(FCHAR, FINT, FINT, const double *, const double *, FINT, const double *, FINT, const double *, double *, FINT);
   void F77_dger( FINT, FINT, const double *, const double *, FINT, const double *, FINT, double *, FINT);

/*
 * Level 3 Fortran Prototypes
 */
   void F77_dgemm(FCHAR, FCHAR, FINT, FINT, FINT, const double *, const double *, FINT, const double *, FINT, const double *, double *, FINT);

#endif // CBLAS_F77_H
