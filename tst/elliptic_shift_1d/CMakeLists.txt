# Minimal build harness for the selectively imported, unchanged spherical donor.
cmake_minimum_required(VERSION 3.16)
project(hyperbolic_shift_donor LANGUAGES CXX)
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
foreach(pair IN ITEMS "boundary;audit" "moving;test_gamma_driver"
                      "gamma_inner;test_gamma_inner_stencil"
                      "inner_lapse;test_constraint_damping")
  list(GET pair 0 label)
  list(GET pair 1 script)
  add_test(NAME hyperbolic_shift_${label}
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../hyperbolic_shift_1d/${script}.py
      --library $<TARGET_FILE:spherical_z4c>
      --output ${CMAKE_CURRENT_BINARY_DIR}/${label}-audit.json)
endforeach()
