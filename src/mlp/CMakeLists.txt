set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/calc_errors.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp_commands.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/self_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/train.cpp
  )

target_sources(mlp PRIVATE ${SOURCES})
