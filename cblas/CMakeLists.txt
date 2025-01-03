cmake_minimum_required(VERSION 3.10)

project(CBLAS LANGUAGES C Fortran)


add_library(mlip_cblas STATIC
  cblas_daxpy.c
  cblas_dgemm.c
  cblas_dgemv.c
  cblas_dger.c
  cblas_dswap.c
  cblas_idamax.c
  daxpy.f
  dgemm.f
  dgemv.f
  dger.f
  dswap.f
  idamax.f
  idamaxsub.f
  lsame.f
  xerbla.f
  )

set_target_properties(mlip_cblas
  PROPERTIES
  OUTPUT_NAME "cblas"
  POSITION_INDEPENDENT_CODE ON
  )
target_include_directories(mlip_cblas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS mlip_cblas)