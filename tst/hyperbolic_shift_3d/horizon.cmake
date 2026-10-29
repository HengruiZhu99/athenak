# Existing FastFlow with optional patch-aware physical-geometry sampling.
add_library(hyperboloidal_fastflow STATIC
  ${CMAKE_SOURCE_DIR}/src/z4c/fastflow.cpp
  ${CMAKE_SOURCE_DIR}/src/geodesic-grid/gauss_legendre.cpp
  ${CMAKE_SOURCE_DIR}/src/parameter_input.cpp
  ${CMAKE_SOURCE_DIR}/src/outputs/io_wrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/globals.cpp)
target_include_directories(hyperboloidal_fastflow PUBLIC
  ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_fastflow PUBLIC Kokkos::kokkos Threads::Threads)
if(ENABLE_MPI)
  target_link_libraries(hyperboloidal_fastflow PUBLIC MPI::MPI_CXX)
endif()
add_executable(hyperbolic_shift_3d_horizon test_horizon.cpp)
target_link_libraries(hyperbolic_shift_3d_horizon PRIVATE hyperboloidal_fastflow)
target_compile_options(hyperbolic_shift_3d_horizon PRIVATE -Wall -Wextra -ffp-contract=off)
add_test(NAME hyperbolic_shift_3d_horizon COMMAND hyperbolic_shift_3d_horizon)
add_executable(hyperbolic_shift_3d_horizon_grid horizon_grid.cpp)
target_link_libraries(hyperbolic_shift_3d_horizon_grid PRIVATE hyperboloidal_fastflow)
target_compile_options(hyperbolic_shift_3d_horizon_grid PRIVATE -Wall -Wextra -ffp-contract=off)
if(ENABLE_MPI)
  add_test(NAME hyperbolic_shift_3d_horizon_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:hyperbolic_shift_3d_horizon> ${MPIEXEC_POSTFLAGS})
  # Both tests produce their own horizon summary files in the test directory.
  set_tests_properties(hyperbolic_shift_3d_horizon hyperbolic_shift_3d_horizon_mpi
    PROPERTIES RUN_SERIAL TRUE)
endif()
