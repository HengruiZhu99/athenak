add_executable(hyperboloidal_topology_tests topology_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/meshblock_tree.cpp
    ${CMAKE_SOURCE_DIR}/src/mesh/chart_atlas.cpp)
target_include_directories(hyperboloidal_topology_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_topology_tests PRIVATE Kokkos::kokkos)
if (Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_topology_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_topology COMMAND hyperboloidal_topology_tests)

add_executable(hyperboloidal_location_tests location_tests.cpp)
target_include_directories(hyperboloidal_location_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
if (Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_location_tests PRIVATE MPI::MPI_CXX)
  add_test(NAME hyperboloidal_locations_mpi COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:hyperboloidal_location_tests> ${MPIEXEC_POSTFLAGS})
endif()
add_test(NAME hyperboloidal_locations COMMAND hyperboloidal_location_tests)

add_executable(hyperboloidal_rk4_tests physics_rk4.cpp)
target_include_directories(hyperboloidal_rk4_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_rk4_tests PRIVATE Kokkos::kokkos)
if (Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_rk4_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_rk4 COMMAND hyperboloidal_rk4_tests)

add_executable(hyperboloidal_exchange_tests exchange_tests.cpp
    ${CMAKE_SOURCE_DIR}/src/bvals/mapped_exchange.cpp)
target_include_directories(hyperboloidal_exchange_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_exchange_tests PRIVATE Kokkos::kokkos)
if (Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_exchange_tests PRIVATE MPI::MPI_CXX)
  foreach(ranks 2 3)
    add_test(NAME hyperboloidal_exchange_${ranks} COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:hyperboloidal_exchange_tests> ${MPIEXEC_POSTFLAGS})
  endforeach()
endif()
add_test(NAME hyperboloidal_exchange COMMAND hyperboloidal_exchange_tests)

add_executable(hyperboloidal_task_graph_tests task_graph.cpp)
target_include_directories(hyperboloidal_task_graph_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_task_graph_tests PRIVATE Kokkos::kokkos)
if (Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_task_graph_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_task_graph COMMAND hyperboloidal_task_graph_tests)

add_executable(hyperboloidal_residual_weight_tests residual_weight.cpp)
target_include_directories(hyperboloidal_residual_weight_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_residual_weight_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_residual_weight_tests PRIVATE MPI::MPI_CXX)
endif()
target_compile_options(hyperboloidal_residual_weight_tests PRIVATE -UNDEBUG)
add_test(NAME hyperboloidal_residual_weight COMMAND hyperboloidal_residual_weight_tests)

add_executable(hyperboloidal_binary_coordinate_tests binary_coordinates.cpp)
target_include_directories(hyperboloidal_binary_coordinate_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(hyperboloidal_binary_coordinate_tests PRIVATE -UNDEBUG)
add_test(NAME hyperboloidal_binary_coordinates COMMAND hyperboloidal_binary_coordinate_tests)

add_executable(hyperboloidal_scri_reconstruction_tests scri_reconstruction.cpp)
target_include_directories(hyperboloidal_scri_reconstruction_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_scri_reconstruction_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_scri_reconstruction_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_scri_reconstruction COMMAND hyperboloidal_scri_reconstruction_tests)

add_executable(hyperboloidal_scri_reference_tests scri_reference.cpp)
target_include_directories(hyperboloidal_scri_reference_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_scri_reference_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_scri_reference_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_scri_reference COMMAND hyperboloidal_scri_reference_tests)

add_executable(hyperboloidal_scri_symbol scri_symbol.cpp)
target_include_directories(hyperboloidal_scri_symbol PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_scri_symbol PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_scri_symbol PRIVATE MPI::MPI_CXX)
endif()

add_executable(hyperboloidal_scri_fourier scri_fourier.cpp)
target_include_directories(hyperboloidal_scri_fourier PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_scri_fourier PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_scri_fourier PRIVATE MPI::MPI_CXX)
endif()

add_executable(hyperboloidal_scri_quadrature_tests scri_quadrature.cpp)
target_include_directories(hyperboloidal_scri_quadrature_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_scri_quadrature_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_scri_quadrature_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_scri_quadrature COMMAND hyperboloidal_scri_quadrature_tests)

add_executable(hyperboloidal_scri_rate_tests scri_rate_diagnostic.cpp)
target_include_directories(hyperboloidal_scri_rate_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperboloidal_scri_rate_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(hyperboloidal_scri_rate_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME hyperboloidal_scri_rate COMMAND hyperboloidal_scri_rate_tests)

# Inventory programs report measured values without imposing a science tolerance.
foreach(inventory stationarity characteristics gauge_admission)
  add_executable(hyperboloidal_nonradiative_${inventory} nonradiative_${inventory}.cpp)
  target_include_directories(hyperboloidal_nonradiative_${inventory} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperboloidal_nonradiative_${inventory} PRIVATE Kokkos::kokkos)
  if(Athena_ENABLE_MPI)
    target_link_libraries(hyperboloidal_nonradiative_${inventory} PRIVATE MPI::MPI_CXX)
  endif()
endforeach()
