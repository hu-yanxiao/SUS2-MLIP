set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/eam.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matvec_oper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mtpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mtpr_trainer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/non_linear_regression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sw_basis.cpp
  )

target_sources(mlp PRIVATE ${SOURCES})
target_sources(libinterface PRIVATE ${SOURCES})

add_subdirectory(mlp)
