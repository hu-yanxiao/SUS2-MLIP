set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/active_learning.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_mlip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_potentials.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/basic_trainer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/configuration.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/error_monitor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lammps_potential.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/linear_regression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lotf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/maxvol.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlip_wrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mtp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/neighborhoods.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pair_potentials.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/radial_basis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/vasp_potential.cpp
  )

target_sources(mlp PRIVATE ${SOURCES})
target_sources(libinterface PRIVATE ${SOURCES})

add_subdirectory(mlp)
add_subdirectory(common)
add_subdirectory(drivers)
add_subdirectory(external)
