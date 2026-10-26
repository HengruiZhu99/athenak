cmake_minimum_required(VERSION 3.16)
project(elliptic_shift_1d LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(HYP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_subdirectory("${HYP_ROOT}/kokkos" kokkos)
add_library(spherical_z4c SHARED spherical_api.cpp)
target_include_directories(spherical_z4c PRIVATE "${HYP_ROOT}/src")
target_link_libraries(spherical_z4c PRIVATE Kokkos::kokkos)
target_compile_options(spherical_z4c PRIVATE -Wall -Wextra)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME nonlinear_elliptic_shift
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_elliptic.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME hyperboloidal_wormhole_initial_data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_wormhole_data.py --library $<TARGET_FILE:spherical_z4c>)
add_executable(scri_chart_probe scri_chart_probe.cpp)
target_include_directories(scri_chart_probe PRIVATE "${HYP_ROOT}/src")
target_link_libraries(scri_chart_probe PRIVATE Kokkos::kokkos)
add_test(NAME spherical_g1_tensor_agreement
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_g1_rhs.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME scalar_nonlinear_geometry_and_analytic_wave
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_rhs.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME analytic_regular_center_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_origin_limits.py)
