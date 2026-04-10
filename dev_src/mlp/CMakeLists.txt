set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/dev_mlp_commands.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/dev_self_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mtpr_train.cpp
  )

target_sources(mlp PRIVATE ${SOURCES})
