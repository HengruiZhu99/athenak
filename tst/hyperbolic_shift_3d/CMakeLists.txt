add_library(hyperbolic_shift_3d_native SHARED native_api.cpp ../elliptic_shift_1d/spherical_api.cpp)
target_include_directories(hyperbolic_shift_3d_native PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(hyperbolic_shift_3d_native PRIVATE Kokkos::kokkos)
target_compile_options(hyperbolic_shift_3d_native PRIVATE -Wall -Wextra -ffp-contract=off)
add_executable(hyperbolic_shift_3d_mapped test_mapped.cpp)
target_include_directories(hyperbolic_shift_3d_mapped PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(hyperbolic_shift_3d_mapped PRIVATE Kokkos::kokkos)
add_test(NAME hyperbolic_shift_3d_mapped COMMAND hyperbolic_shift_3d_mapped)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hyperbolic_shift_3d_point_and_symbol
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/audit.py
  --library $<TARGET_FILE:hyperbolic_shift_3d_native>
  --output ${CMAKE_BINARY_DIR}/Testing/hyperbolic_shift_3d)
