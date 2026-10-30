find_package(Threads REQUIRED)
add_library(hyperbolic_shift_3d_native SHARED native_api.cpp ../elliptic_shift_1d/spherical_api.cpp)
target_include_directories(hyperbolic_shift_3d_native PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(hyperbolic_shift_3d_native PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_native PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(hyperbolic_shift_3d_mapped test_mapped.cpp)
target_include_directories(hyperbolic_shift_3d_mapped PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_mapped PRIVATE Kokkos::kokkos Threads::Threads)
add_test(NAME hyperbolic_shift_3d_mapped COMMAND hyperbolic_shift_3d_mapped)
add_executable(hyperbolic_shift_3d_coupled coupled.cpp)
target_include_directories(hyperbolic_shift_3d_coupled PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_coupled PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_coupled PRIVATE -Wall -Wextra -ffp-contract=off)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hyperbolic_shift_3d_point_and_symbol
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/audit.py
  --library $<TARGET_FILE:hyperbolic_shift_3d_native>
  --output ${CMAKE_BINARY_DIR}/Testing/hyperbolic_shift_3d)
add_executable(hyperbolic_shift_3d_tensor_scri test_tensor_scri.cpp)
target_include_directories(hyperbolic_shift_3d_tensor_scri PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_tensor_scri PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_tensor_scri PRIVATE -Wall -Wextra -ffp-contract=off)
add_test(NAME hyperbolic_shift_3d_tensor_scri COMMAND hyperbolic_shift_3d_tensor_scri)
add_executable(hyperbolic_shift_3d_reanalyze reanalyze.cpp)
target_include_directories(hyperbolic_shift_3d_reanalyze PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_reanalyze PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_reanalyze PRIVATE -Wall -Wextra -ffp-contract=off)
include(horizon.cmake)
foreach(test composite_device composite_exchange scri_device scri_angular algebraic_projection angular_dissipation scri_dissipation_closure sampled_normal_device scri_radial_difference scri_radial_device)
  add_executable(hyperbolic_shift_3d_${test} test_${test}.cpp)
  target_include_directories(hyperbolic_shift_3d_${test} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperbolic_shift_3d_${test} PRIVATE Kokkos::kokkos Threads::Threads)
  target_compile_options(hyperbolic_shift_3d_${test} PRIVATE -Wall -Wextra -ffp-contract=off)
  add_test(NAME hyperbolic_shift_3d_${test} COMMAND hyperbolic_shift_3d_${test})
endforeach()
add_executable(hyperbolic_shift_3d_donor_seven_device test_sampled_normal_device.cpp)
target_include_directories(hyperbolic_shift_3d_donor_seven_device PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_donor_seven_device PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_donor_seven_device PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_donor_seven_device PRIVATE HYPERBOLIC_TEST_DONOR_SEVEN=1)
add_test(NAME hyperbolic_shift_3d_donor_seven_device COMMAND hyperbolic_shift_3d_donor_seven_device)
add_executable(hyperbolic_shift_3d_inner_transition_device test_sampled_normal_device.cpp)
target_include_directories(hyperbolic_shift_3d_inner_transition_device PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_inner_transition_device PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_inner_transition_device PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_inner_transition_device PRIVATE HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1)
add_test(NAME hyperbolic_shift_3d_inner_transition_device COMMAND hyperbolic_shift_3d_inner_transition_device)
add_executable(hyperbolic_shift_3d_core_coordinates core_coordinates.cpp)
target_include_directories(hyperbolic_shift_3d_core_coordinates PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_core_coordinates PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_core_coordinates PRIVATE -Wall -Wextra -ffp-contract=off)
# Component qualification on static Cartesian leaves, including a coarse/fine
# seam. This does not exercise regridding, field transfer or evolution restart.
add_executable(hyperbolic_shift_3d_refined_core_device test_sampled_normal_device.cpp)
target_include_directories(hyperbolic_shift_3d_refined_core_device PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_refined_core_device PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_refined_core_device PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_refined_core_device PRIVATE HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1 HYPERBOLIC_TEST_REFINED_CORE=1)
add_test(NAME hyperbolic_shift_3d_refined_core_device COMMAND hyperbolic_shift_3d_refined_core_device)
set_tests_properties(hyperbolic_shift_3d_refined_core_device PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
add_executable(hyperbolic_shift_3d_core2_device test_sampled_normal_device.cpp)
target_include_directories(hyperbolic_shift_3d_core2_device PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_core2_device PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_core2_device PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_core2_device PRIVATE HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1 HYPERBOLIC_TEST_CORE2=1)
add_test(NAME hyperbolic_shift_3d_core2_device COMMAND hyperbolic_shift_3d_core2_device)
set_tests_properties(hyperbolic_shift_3d_core2_device PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
add_executable(hyperbolic_shift_3d_scri_sampled_normal scri_sampled_normal.cpp)
target_include_directories(hyperbolic_shift_3d_scri_sampled_normal PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_scri_sampled_normal PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_scri_sampled_normal PRIVATE -Wall -Wextra -ffp-contract=off)
add_test(NAME hyperbolic_shift_3d_scri_sampled_normal COMMAND hyperbolic_shift_3d_scri_sampled_normal)
set_tests_properties(hyperbolic_shift_3d_scri_sampled_normal PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
# The bounded manufactured-resolution series is driven explicitly by Python.
add_executable(hyperbolic_shift_3d_scri_sampled_accuracy scri_sampled_accuracy.cpp)
target_include_directories(hyperbolic_shift_3d_scri_sampled_accuracy PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_scri_sampled_accuracy PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_scri_sampled_accuracy PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(hyperbolic_shift_3d_scri_radial_duration scri_radial_duration.cpp)
target_include_directories(hyperbolic_shift_3d_scri_radial_duration PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_scri_radial_duration PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_scri_radial_duration PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(hyperbolic_shift_3d_scri_device_inline test_scri_device.cpp)
target_include_directories(hyperbolic_shift_3d_scri_device_inline PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_scri_device_inline PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_scri_device_inline PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_scri_device_inline PRIVATE HYPERBOLIC_SCRI_INLINE_SAMPLING=1)
add_test(NAME hyperbolic_shift_3d_scri_device_inline COMMAND hyperbolic_shift_3d_scri_device_inline)
if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_communication benchmark_communication.cpp)
  target_include_directories(hyperbolic_shift_3d_mpi_communication PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperbolic_shift_3d_mpi_communication PRIVATE
    Kokkos::kokkos Threads::Threads MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_communication PRIVATE
    -Wall -Wextra -ffp-contract=off)
  add_test(NAME hyperbolic_shift_3d_communication_cli
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_communication.py
      --binary $<TARGET_FILE:hyperbolic_shift_3d_mpi_communication> --cli-tests)
  add_test(NAME hyperbolic_shift_3d_communication_mpi
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_communication.py
      --binary $<TARGET_FILE:hyperbolic_shift_3d_mpi_communication>
      --mpiexec ${MPIEXEC_EXECUTABLE} --ranks 2)
  set_tests_properties(hyperbolic_shift_3d_communication_cli hyperbolic_shift_3d_communication_mpi
    PROPERTIES RUN_SERIAL TRUE TIMEOUT 240)
  add_executable(hyperbolic_shift_3d_mpi coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi PRIVATE -Wall -Wextra -ffp-contract=off)
  add_executable(hyperbolic_shift_3d_mpi_projected coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_projected PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_projected PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_projected PRIVATE HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1)
  add_executable(hyperbolic_shift_3d_mpi_projected_angular coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_projected_angular PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_projected_angular PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_projected_angular PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1)
  add_executable(hyperbolic_shift_3d_mpi_dissipative coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_dissipative PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_dissipative PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_dissipative PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1 HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1)
  add_executable(hyperbolic_shift_3d_mpi_sampled_normal coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_sampled_normal PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_sampled_normal PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_sampled_normal PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1)
  add_executable(hyperbolic_shift_3d_mpi_donor_seven coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_donor_seven PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_donor_seven PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_donor_seven PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1 HYPERBOLIC_SCRI_DONOR_SEVEN=1)
  add_executable(hyperbolic_shift_3d_mpi_inner_transition coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_inner_transition PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_inner_transition PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_inner_transition PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1 HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1)
  add_test(NAME hyperbolic_shift_3d_sampled_normal_mpi
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_sampled_normal_mpi.py
      --binary $<TARGET_FILE:hyperbolic_shift_3d_mpi_sampled_normal>
      --control $<TARGET_FILE:hyperbolic_shift_3d_mpi_dissipative>
      --mpiexec ${MPIEXEC_EXECUTABLE})
  add_executable(hyperbolic_shift_3d_mpi_core2 coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_core2 PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_core2 PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_core2 PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1 HYPERBOLIC_CORE_SUBDIVISIONS=2)
  add_test(NAME hyperbolic_shift_3d_core2_mpi
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_sampled_normal_mpi.py
      --version 8 --binary $<TARGET_FILE:hyperbolic_shift_3d_mpi_core2>
      --control $<TARGET_FILE:hyperbolic_shift_3d_mpi_inner_transition> --mpiexec ${MPIEXEC_EXECUTABLE})
  set_tests_properties(hyperbolic_shift_3d_core2_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
  set_tests_properties(hyperbolic_shift_3d_sampled_normal_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
  add_test(NAME hyperbolic_shift_3d_dissipative_mpi
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_dissipative_mpi.py
      --binary $<TARGET_FILE:hyperbolic_shift_3d_mpi_dissipative>
      --control $<TARGET_FILE:hyperbolic_shift_3d_mpi_projected_angular>
      --mpiexec ${MPIEXEC_EXECUTABLE})
  set_tests_properties(hyperbolic_shift_3d_dissipative_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
endif()

add_executable(hyperbolic_shift_3d_evolved_overlap test_evolved_overlap.cpp)
target_include_directories(hyperbolic_shift_3d_evolved_overlap PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_evolved_overlap PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_evolved_overlap PRIVATE -Wall -Wextra -ffp-contract=off)

add_executable(hyperbolic_shift_3d_overlap_device test_sampled_normal_device.cpp)
target_include_directories(hyperbolic_shift_3d_overlap_device PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_overlap_device PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_overlap_device PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_overlap_device PRIVATE HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1 HYPERBOLIC_TEST_OVERLAP=3)
if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_overlap coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_overlap PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_overlap PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_overlap PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1 HYPERBOLIC_EVOLVED_OVERLAP=3)
endif()

add_test(NAME hyperbolic_shift_3d_evolved_overlap COMMAND hyperbolic_shift_3d_evolved_overlap 12 3)
add_test(NAME hyperbolic_shift_3d_overlap_device COMMAND hyperbolic_shift_3d_overlap_device)
set_tests_properties(hyperbolic_shift_3d_evolved_overlap hyperbolic_shift_3d_overlap_device PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)

foreach(kind IN ITEMS geometry device)
  if(kind STREQUAL "geometry")
    set(spherical_source test_evolved_overlap.cpp)
  else()
    set(spherical_source test_sampled_normal_device.cpp)
  endif()
  add_executable(hyperbolic_shift_3d_spherical_overlap_${kind} ${spherical_source})
  target_include_directories(hyperbolic_shift_3d_spherical_overlap_${kind} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperbolic_shift_3d_spherical_overlap_${kind} PRIVATE Kokkos::kokkos Threads::Threads)
  target_compile_options(hyperbolic_shift_3d_spherical_overlap_${kind} PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_spherical_overlap_${kind} PRIVATE HYPERBOLIC_TEST_SPHERICAL_OVERLAP=1 HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1 HYPERBOLIC_TEST_OVERLAP=3)
  add_test(NAME hyperbolic_shift_3d_spherical_overlap_${kind} COMMAND hyperbolic_shift_3d_spherical_overlap_${kind})
  set_tests_properties(hyperbolic_shift_3d_spherical_overlap_${kind} PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
endforeach()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_spherical_overlap coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_spherical_overlap PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_spherical_overlap PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_spherical_overlap PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1)
endif()

foreach(kind IN ITEMS geometry device)
  if(kind STREQUAL "geometry")
    set(spherical_source test_evolved_overlap.cpp)
  else()
    set(spherical_source test_sampled_normal_device.cpp)
  endif()
  add_executable(hyperbolic_shift_3d_continuous_positive_${kind} ${spherical_source})
  target_include_directories(hyperbolic_shift_3d_continuous_positive_${kind} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperbolic_shift_3d_continuous_positive_${kind} PRIVATE Kokkos::kokkos Threads::Threads)
  target_compile_options(hyperbolic_shift_3d_continuous_positive_${kind} PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_continuous_positive_${kind} PRIVATE HYPERBOLIC_TEST_CONTINUOUS_POSITIVE=1 HYPERBOLIC_TEST_SPHERICAL_OVERLAP=1 HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1 HYPERBOLIC_TEST_OVERLAP=3)
  add_test(NAME hyperbolic_shift_3d_continuous_positive_${kind} COMMAND hyperbolic_shift_3d_continuous_positive_${kind})
  set_tests_properties(hyperbolic_shift_3d_continuous_positive_${kind} PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)
endforeach()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_continuous_positive coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_continuous_positive PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_continuous_positive PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_continuous_positive PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1 HYPERBOLIC_CONTINUOUS_POSITIVE=1)
endif()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_continuous_positive_fast coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_continuous_positive_fast PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_continuous_positive_fast PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_continuous_positive_fast PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1 HYPERBOLIC_CONTINUOUS_POSITIVE=1 HYPERBOLIC_BOUNDARY_REUSE=1)
endif()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_continuous_positive_sparse coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_continuous_positive_sparse PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_continuous_positive_sparse PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_continuous_positive_sparse PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1 HYPERBOLIC_CONTINUOUS_POSITIVE=1 HYPERBOLIC_BOUNDARY_REUSE=1 HYPERBOLIC_SPARSE_STAGE=1)
endif()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_sparse_exchange test_sparse_stage_exchange.cpp)
  target_include_directories(hyperbolic_shift_3d_sparse_exchange PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperbolic_shift_3d_sparse_exchange PRIVATE Kokkos::kokkos MPI::MPI_CXX Threads::Threads)
  target_compile_options(hyperbolic_shift_3d_sparse_exchange PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_continuous_positive_balanced coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_continuous_positive_balanced PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_continuous_positive_balanced PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_continuous_positive_balanced PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1 HYPERBOLIC_CONTINUOUS_POSITIVE=1 HYPERBOLIC_BOUNDARY_REUSE=1 HYPERBOLIC_SPARSE_STAGE=1 HYPERBOLIC_MEASURED_PARTITION=1)
endif()

add_executable(hyperbolic_shift_3d_measured_partition test_measured_partition.cpp)
target_include_directories(hyperbolic_shift_3d_measured_partition PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(hyperbolic_shift_3d_measured_partition PRIVATE -Wall -Wextra)


if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_continuous_positive_radial_batch coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_continuous_positive_radial_batch PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_continuous_positive_radial_batch PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_continuous_positive_radial_batch PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1 HYPERBOLIC_CONTINUOUS_POSITIVE=1 HYPERBOLIC_BOUNDARY_REUSE=1 HYPERBOLIC_SPARSE_STAGE=1 HYPERBOLIC_RADIAL_FIELD_BATCH=1)
endif()

add_executable(hyperbolic_shift_3d_radial_field_batch test_radial_field_batch.cpp)
target_include_directories(hyperbolic_shift_3d_radial_field_batch PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_radial_field_batch PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_radial_field_batch PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_radial_field_batch PRIVATE HYPERBOLIC_RADIAL_FIELD_BATCH=1)
add_test(NAME hyperbolic_shift_3d_radial_field_batch COMMAND hyperbolic_shift_3d_radial_field_batch)
set_tests_properties(hyperbolic_shift_3d_radial_field_batch PROPERTIES TIMEOUT 420)

add_executable(hyperbolic_shift_3d_radial_batch_device test_sampled_normal_device.cpp)
target_include_directories(hyperbolic_shift_3d_radial_batch_device PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_radial_batch_device PRIVATE Kokkos::kokkos Threads::Threads)
target_compile_options(hyperbolic_shift_3d_radial_batch_device PRIVATE -Wall -Wextra -ffp-contract=off)
target_compile_definitions(hyperbolic_shift_3d_radial_batch_device PRIVATE HYPERBOLIC_RADIAL_FIELD_BATCH=1 HYPERBOLIC_TEST_CONTINUOUS_POSITIVE=1 HYPERBOLIC_TEST_SPHERICAL_OVERLAP=1 HYPERBOLIC_TEST_DONOR_SEVEN=1 HYPERBOLIC_TEST_INNER_TRANSITION=1 HYPERBOLIC_TEST_OVERLAP=3)
add_test(NAME hyperbolic_shift_3d_radial_batch_device COMMAND hyperbolic_shift_3d_radial_batch_device)
set_tests_properties(hyperbolic_shift_3d_radial_batch_device PROPERTIES RUN_SERIAL TRUE TIMEOUT 420)

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mesh_plan_bundle test_mesh_plan_bundle.cpp)
  target_include_directories(hyperbolic_shift_3d_mesh_plan_bundle PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(hyperbolic_shift_3d_mesh_plan_bundle PRIVATE Kokkos::kokkos MPI::MPI_CXX Threads::Threads)
  target_compile_options(hyperbolic_shift_3d_mesh_plan_bundle PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mesh_plan_bundle PRIVATE HYPERBOLIC_RADIAL_FIELD_BATCH=1)
  add_test(NAME hyperbolic_shift_3d_mesh_plan_bundle_mpi COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:hyperbolic_shift_3d_mesh_plan_bundle> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(hyperbolic_shift_3d_mesh_plan_bundle_mpi PROPERTIES RUN_SERIAL TRUE TIMEOUT 420 PROCESSORS 2)
endif()

if(ENABLE_MPI)
  add_executable(hyperbolic_shift_3d_mpi_continuous_positive_planned coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_continuous_positive_planned PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_continuous_positive_planned PRIVATE -Wall -Wextra -ffp-contract=off)
  target_compile_definitions(hyperbolic_shift_3d_mpi_continuous_positive_planned PRIVATE
    HYPERBOLIC_ALGEBRAIC_FINAL_PROJECTION=1 HYPERBOLIC_SCRI_ANGULAR_EXCHANGE=1
    HYPERBOLIC_SCRI_ANGULAR_DISSIPATION=1 HYPERBOLIC_SCRI_SAMPLED_NORMAL=1
    HYPERBOLIC_SCRI_DONOR_SEVEN=1 HYPERBOLIC_INNER_TRANSITION_LOGICAL=1
    HYPERBOLIC_CORE_SUBDIVISIONS=2 HYPERBOLIC_EVOLVED_OVERLAP=3 HYPERBOLIC_SPHERICAL_OVERLAP=1
    HYPERBOLIC_CONTINUOUS_POSITIVE=1 HYPERBOLIC_BOUNDARY_REUSE=1 HYPERBOLIC_SPARSE_STAGE=1
    HYPERBOLIC_RADIAL_FIELD_BATCH=1 HYPERBOLIC_COORDINATED_PLANS=1)
  add_executable(hyperbolic_shift_3d_repartition_transaction test_repartition_transaction.cpp)
  target_link_libraries(hyperbolic_shift_3d_repartition_transaction PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_repartition_transaction PRIVATE -Wall -Wextra -ffp-contract=off)
  get_target_property(hyperbolic_planned_definitions hyperbolic_shift_3d_mpi_continuous_positive_planned COMPILE_DEFINITIONS)
  target_compile_definitions(hyperbolic_shift_3d_repartition_transaction PRIVATE ${hyperbolic_planned_definitions})
  add_executable(hyperbolic_shift_3d_mpi_radial_batch_timestep_experiment coupled_mpi.cpp)
  target_link_libraries(hyperbolic_shift_3d_mpi_radial_batch_timestep_experiment PRIVATE hyperboloidal_fastflow MPI::MPI_CXX)
  target_compile_options(hyperbolic_shift_3d_mpi_radial_batch_timestep_experiment PRIVATE -Wall -Wextra -ffp-contract=off)
  get_target_property(hyperbolic_radial_batch_definitions hyperbolic_shift_3d_mpi_continuous_positive_radial_batch COMPILE_DEFINITIONS)
  target_compile_definitions(hyperbolic_shift_3d_mpi_radial_batch_timestep_experiment PRIVATE ${hyperbolic_radial_batch_definitions} HYPERBOLIC_EXPERIMENTAL_TIMESTEP=1)
endif()
add_executable(hyperbolic_shift_3d_timestep_penalty test_timestep_penalty.cpp)
target_link_libraries(hyperbolic_shift_3d_timestep_penalty PRIVATE Kokkos::kokkos)
target_include_directories(hyperbolic_shift_3d_timestep_penalty PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_compile_definitions(hyperbolic_shift_3d_timestep_penalty PRIVATE HYPERBOLIC_EXPERIMENTAL_TIMESTEP=1)
add_test(NAME hyperbolic_shift_3d_timestep_penalty COMMAND hyperbolic_shift_3d_timestep_penalty)
