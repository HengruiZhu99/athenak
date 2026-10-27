# Master CMakeLists.txt file for Athena++ (Kokkos version)
#
#------ start with setting various compiler, environment, and build defaults -------------

cmake_minimum_required(VERSION 3.10)
project(athena VERSION 0.1 LANGUAGES C CXX)

# Disable in-source builds to prevent source tree corruption.
if( "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}" )
  message(FATAL_ERROR "FATAL: In-source builds disabled. Please create a separate build directory in which to run cmake, e.g. mkdir build; cd build; cmake ../" )
endif()

# Athena++ (and Kokkos) require C++17 standard as minimum
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force flag to denote C++ standard, as deprecated c++1z is not supported by nvcc.
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")

# Kokkos recommends NOT using default compiler extensions
set(CMAKE_CXX_EXTENSIONS OFF)

# following policy required to enable target_link_libraries() below 
cmake_policy(SET CMP0079 NEW)

# Use 'Release' as default build type. Use 'Debug' for testing.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release)
endif()

#------ default values for compile time options  -----------------------------------------

option(Athena_SINGLE_PRECISION "Compile for single precision" OFF)
option(Athena_ENABLE_MPI "Compile with MPI parallelism enabled" OFF)
option(Athena_ENABLE_OPENMP "Compile with OpenMP parallelism enabled" OFF)
set(PROBLEM built_in_pgens CACHE STRING "Name of problem generator function")

#------ set macros exported to config.hpp ------------------------------------------------

# set single precision macro (true/false)
if (Athena_SINGLE_PRECISION)
  set(SINGLE_PRECISION_ENABLED 1)
else()
  set(SINGLE_PRECISION_ENABLED 0)
endif()

# set MPI macro (true/false)
set(ENABLE_MPI OFF)
if (Athena_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI package required but could not be found.")
  endif()
  set(ENABLE_MPI ON)
  set(Kokkos_ENABLE_MPI ON CACHE BOOL "Compile Kokkos with MPI")
endif()
if (ENABLE_MPI)
  set(MPI_PARALLEL_ENABLED 1)
else ()
  set(MPI_PARALLEL_ENABLED 0)
endif()

# set OpenMP macro (true/false)
set(ENABLE_OPENMP OFF)
if (Athena_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS CXX)
  if (NOT OpenMP_FOUND)
    message(FATAL_ERROR "OpenMP package is required but could not be found.")
  endif()
  set(ENABLE_OPENMP ON)
  set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Compile Kokkos with OpenMP")
endif()
if (ENABLE_OPENMP)
  set(OPENMP_PARALLEL_ENABLED 1)
else()
  set(OPENMP_PARALLEL_ENABLED 0)
endif()

#set user problem generator flag
if (NOT ${PROBLEM} STREQUAL "built_in_pgens")
  message(STATUS "Including user-specified problem generator file: ${PROBLEM}")
  set(USER_PROBLEM_ENABLED 1)
else()
  set(USER_PROBLEM_ENABLED 0)
endif()

#------ set various Kokkos option --------------------------------------------------------

# Tell Kokkos to vectorize aggressively
# Kokkos prefers this capitalization for debugging reasons
SET (Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL
     "Kokkos aggressive vectorization")

# Tell Kokkos we need lambdas in Cuda.
if (Kokkos_ENABLE_CUDA)
  SET (Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable lambda expressions in CUDA")
endif()

# If this is a debug build, set kokkos debug on
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message(STATUS "Enabling Kokkos debug mode")
  set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Most general debug settings")
  set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL
     "Bounds checking on Kokkos views")
  set(Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK ON CACHE BOOL
      "Sanity checks on Kokkos DualView")
endif()

#------ create configuration header file from template in /src directory -----------------

# add subdirectories for source code and include files
option(Athena_ENABLE_CUBED_SPHERE_CONFORMAL_PROBE
       "Build mapped conformal research probe (not production support)" OFF)
if(Athena_ENABLE_CUBED_SPHERE_CONFORMAL_PROBE AND
   (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_OPENMP))
  message(FATAL_ERROR "Mapped conformal probe supports Serial or SYCL; one rank at runtime")
endif()
add_subdirectory(src)
add_subdirectory(kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

find_package(Threads REQUIRED)
target_link_libraries(athena PUBLIC Threads::Threads)

target_link_libraries(athena PUBLIC Kokkos::kokkos)
if (ENABLE_MPI)
  target_link_libraries(athena PUBLIC MPI::MPI_CXX)
endif()
if (ENABLE_OPENMP)
  target_link_libraries(athena PUBLIC OpenMP::OpenMP_CXX)
endif()
if (${PROBLEM} MATCHES "z4c/two_punctures/*")
	target_include_directories(athena PRIVATE ${CMAKE_SOURCE_DIR}/twopuncturesc/include)
	target_link_libraries(athena PUBLIC ${CMAKE_SOURCE_DIR}/twopuncturesc/lib/libTwoPunctures.a)
	#target_include_directories(athena PRIVATE ${CMAKE_SOURCE_DIR}/gsl/include)
	#target_link_libraries(athena PUBLIC ${CMAKE_SOURCE_DIR}/gsl/lib/libgsl.a)
        #target_link_libraries(athena PUBLIC ${CMAKE_SOURCE_DIR}/gsl/lib/libgslcblas.a)
	find_package(GSL REQUIRED)
	target_link_libraries(athena PUBLIC GSL::gsl GSL::gslcblas)
	set(TWO_PUNCTURES 1)
else()
	set(TWO_PUNCTURES 0)
endif()

if (${PROBLEM} MATCHES "z4c/spectre/*")
  # Load binary black hole initial data from the SpECTRE code:
  # https://spectre-code.org
  #
  # Uses the `BundledExporter` library from SpECTRE to read the initial data.
  # You can help CMake find the library by setting
  # `-D SPECTRE_ROOT=path/to/spectre/build` to the path where SpECTRE is
  # installed.
  find_library(
    SPECTRE_EXPORTER_LIB
    NAMES BundledExporter
    PATH_SUFFIXES lib
    PATHS ${SPECTRE_ROOT})
  find_path(
    SPECTRE_EXPORTER_INCLUDE_DIR
    NAMES spectre/Exporter.hpp
    PATH_SUFFIXES include
    PATHS ${SPECTRE_ROOT})
  add_library(spectre::Exporter UNKNOWN IMPORTED)
  set_target_properties(
    spectre::Exporter
    PROPERTIES IMPORTED_LOCATION ${SPECTRE_EXPORTER_LIB}
               INTERFACE_INCLUDE_DIRECTORIES ${SPECTRE_EXPORTER_INCLUDE_DIR})
  target_link_libraries(athena PRIVATE spectre::Exporter)
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/elliptica/*")
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/Elliptica_ID_Reader/include)
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/Elliptica_ID_Reader/lib/libelliptica_id_reader.a)
  if (Athena_ENABLE_OPENMP)
  else ()
    find_package(OpenMP COMPONENTS CXX)
    target_link_libraries(athena PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/rns/*")
  # Support for RNS library for rotating neutron stars. See
  # https://github.com/computationalrelativity/RNSC
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/rnsc/include)
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/rnsc/lib/libRNS.a)
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/sgrid/*")
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/sgrid/lib/libsgrid.a)
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/kadath/*")
  # Kadath (FUKa) BNS initial data reader.
  # Expects Kadath to be linked at ${CMAKE_SOURCE_DIR}/kadath
  # with libkadath.a in its lib/ directory.
  find_package(Boost REQUIRED)
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/kadath/include
      ${CMAKE_SOURCE_DIR}/kadath/include/Kadath_point_h
      #${CMAKE_SOURCE_DIR}/kadath/include/Solvers
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/kadath/lib/libkadath.a)
  # Kadath depends on Boost, FFTW3 (spectral transforms) and LAPACK (linear algebra)
  find_package(PkgConfig QUIET)
  if (PKG_CONFIG_FOUND)
    pkg_check_modules(FFTW3 fftw3)
  endif()
  if (FFTW3_FOUND)
    target_link_libraries(athena PUBLIC ${FFTW3_LIBRARIES})
  else()
    find_library(FFTW3_LIB NAMES fftw3)
    target_link_libraries(athena PUBLIC ${FFTW3_LIB})
  endif()
  find_package(LAPACK REQUIRED)
  target_link_libraries(athena PUBLIC ${LAPACK_LIBRARIES})
  find_package(GSL REQUIRED)
  target_link_libraries(athena PUBLIC GSL::gsl GSL::gslcblas)
  if (Athena_ENABLE_OPENMP)
  else ()
    find_package(OpenMP COMPONENTS CXX)
    target_link_libraries(athena PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

if (${PROBLEM} MATCHES "dyn_grmhd/lorene/*")
  # FIXME(user): The way the Fortran libraries here are linked is not great. Because these
  # are implicit libraries, we shouldn't have to add them manually, but at least on some
  # machines this seems to be necessary. We should find out if there's a better way to
  # do this.
  enable_language(Fortran)
  include(FortranCInterface)
  FortranCInterface_VERIFY(CXX)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  find_package(GSL REQUIRED)
  target_include_directories(athena PRIVATE
      ${CMAKE_SOURCE_DIR}/Lorene/Export/C++/Include
      ${CMAKE_SOURCE_DIR}/Lorene/C++/Include)
  target_link_libraries(athena PUBLIC
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorene_export.a
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorene.a
      ${CMAKE_SOURCE_DIR}/Lorene/Lib/liblorenef77.a
      ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
      GSL::gsl ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
endif()

if ("${PROBLEM}" STREQUAL "id_solve")
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_include_directories(athena PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(athena PRIVATE ${HDF5_LIBRARIES})
endif()

# enable include of header files created in build directories
target_include_directories(athena PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(config.hpp.in config.hpp)

install( TARGETS athena DESTINATION bin )

# Opt-in tests and standalone spherical experiment; not an AthenaK runtime mode.
option(Athena_ENABLE_HYPERBOLOIDAL_TESTS "Build experimental hyperboloidal tests" OFF)
if (Athena_ENABLE_HYPERBOLOIDAL_TESTS)
  enable_testing()
  add_executable(hyperboloidal_layer_constraint_tangent tst/hyperboloidal/test_layer_constraint_tangent.cpp)
  target_include_directories(hyperboloidal_layer_constraint_tangent PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_layer_constraint_tangent PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_layer_constraint_tangent COMMAND hyperboloidal_layer_constraint_tangent)
  add_executable(hyperboloidal_layer_wormhole_tests tst/hyperboloidal/test_layer_wormhole.cpp)
  target_include_directories(hyperboloidal_layer_wormhole_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_layer_wormhole_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_layer_wormhole_reference COMMAND hyperboloidal_layer_wormhole_tests)
  add_executable(hyperboloidal_symmetric_ghost_tests tst/hyperboloidal/test_symmetric_ghosts.cpp)
  target_include_directories(hyperboloidal_symmetric_ghost_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_symmetric_ghost_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_symmetric_ghosts COMMAND hyperboloidal_symmetric_ghost_tests)
  add_executable(hyperboloidal_physical_gauge_tests tst/hyperboloidal/test_physical_gauge.cpp)
  target_include_directories(hyperboloidal_physical_gauge_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_physical_gauge_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_physical_gauge_reference COMMAND hyperboloidal_physical_gauge_tests)
  add_executable(hyperboloidal_kernel_symbol tst/hyperboloidal/kernel_symbol.cpp)
  target_include_directories(hyperboloidal_kernel_symbol PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_kernel_symbol PRIVATE Kokkos::kokkos)
  add_executable(hyperboloidal_layer_tests tst/hyperboloidal/test_layer.cpp)
  target_include_directories(hyperboloidal_layer_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_layer_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_layer_reference COMMAND hyperboloidal_layer_tests)
  add_executable(hyperboloidal_tests tst/hyperboloidal/test_hyperboloidal.cpp)
  target_include_directories(hyperboloidal_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_building_blocks COMMAND hyperboloidal_tests)
  add_executable(hyperboloidal_constraints_tests tst/hyperboloidal/test_constraints.cpp)
  target_include_directories(hyperboloidal_constraints_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_constraints_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_constraints COMMAND hyperboloidal_constraints_tests)
  add_executable(hyperboloidal_rhs_tests tst/hyperboloidal/test_conformal_rhs.cpp)
  target_include_directories(hyperboloidal_rhs_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_rhs_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_rhs COMMAND hyperboloidal_rhs_tests)
  add_executable(hyperboloidal_mesh_tests tst/hyperboloidal/test_athenak_bridge.cpp)
  target_include_directories(hyperboloidal_mesh_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_mesh_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_mesh_adapter COMMAND hyperboloidal_mesh_tests)
  add_executable(hyperboloidal_ghost_tests tst/hyperboloidal/test_spherical_ghosts.cpp)
  target_include_directories(hyperboloidal_ghost_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_ghost_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_spherical_ghosts COMMAND hyperboloidal_ghost_tests)
  add_executable(hyperboloidal_wave_tests tst/hyperboloidal/test_conformal_wave.cpp)
  target_include_directories(hyperboloidal_wave_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_wave_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_conformal_wave COMMAND hyperboloidal_wave_tests)
  add_executable(hyperboloidal_dissipation_tests
      tst/hyperboloidal/test_interior_dissipation.cpp)
  target_include_directories(hyperboloidal_dissipation_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_dissipation_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_interior_dissipation COMMAND hyperboloidal_dissipation_tests)
  add_executable(hyperboloidal_cartesian_tests tst/hyperboloidal/test_cartesian_patch.cpp)
  target_include_directories(hyperboloidal_cartesian_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_cartesian_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_cartesian_patch COMMAND hyperboloidal_cartesian_tests)
  add_executable(hyperboloidal_radial_tests tst/hyperboloidal/test_cartesian_radial.cpp)
  target_include_directories(hyperboloidal_radial_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_radial_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_cartesian_radial COMMAND hyperboloidal_radial_tests)
  add_executable(hyperboloidal_hawking_tests tst/hyperboloidal/test_hawking_mass.cpp)
  target_include_directories(hyperboloidal_hawking_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(hyperboloidal_hawking_tests PRIVATE Kokkos::kokkos)
  add_test(NAME hyperboloidal_hawking_mass COMMAND hyperboloidal_hawking_tests)
  add_executable(hyperboloidal_spherical src/z4c/hyperboloidal/spherical_driver.cpp)
  target_include_directories(hyperboloidal_spherical PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(hyperboloidal_spherical PRIVATE Kokkos::kokkos)
  option(Athena_ENABLE_HYPERBOLOIDAL_ANALYSIS "Run Python layer symbolic and kernel gates (sympy/numpy/mpmath)" OFF)
  if (Athena_ENABLE_HYPERBOLOIDAL_ANALYSIS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME hyperboloidal_layer_wormhole_symbolic COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tst/hyperboloidal/check_layer_wormhole.py)
    add_test(NAME hyperboloidal_physical_gauge_poles COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tst/hyperboloidal/check_physical_gauge.py $<TARGET_FILE:hyperboloidal_physical_gauge_tests>)
    add_test(NAME hyperboloidal_layer_kernel_symbol COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tst/hyperboloidal/check_kernel_symbol.py $<TARGET_FILE:hyperboloidal_kernel_symbol>)
    add_test(NAME hyperboloidal_layer_cutoff COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tst/hyperboloidal/check_cutoff.py $<TARGET_FILE:hyperboloidal_layer_tests>)
    foreach(check layer_symbolic layer_geometry layer_residues)
      add_test(NAME hyperboloidal_${check} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tst/hyperboloidal/check_${check}.py)
    endforeach()
  endif()
  find_package(Python3 COMPONENTS Interpreter)
  if (Python3_Interpreter_FOUND)
    add_test(NAME hyperboloidal_evolution COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_SOURCE_DIR}/tst/hyperboloidal/check_evolution.py
      $<TARGET_FILE:hyperboloidal_spherical>)
  endif()
endif()

option(Athena_ENABLE_CUBED_SPHERE_TESTS "Build cubed sphere mapped-grid tests" OFF)
option(Athena_ENABLE_ELLIPTIC_SHIFT_3D_TESTS "Build exact metric-connection shift tests" OFF)
if(Athena_ENABLE_ELLIPTIC_SHIFT_3D_TESTS)
  enable_testing()
  add_executable(elliptic_shift_3d_operator_tests tst/elliptic_shift_3d/test_operator.cpp)
  target_include_directories(elliptic_shift_3d_operator_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_operator_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_exact_operator COMMAND elliptic_shift_3d_operator_tests)
  set_tests_properties(elliptic_shift_3d_exact_operator PROPERTIES TIMEOUT 30)
  add_executable(elliptic_shift_3d_tensor_scri_tests tst/elliptic_shift_3d/test_tensor_scri.cpp)
  target_include_directories(elliptic_shift_3d_tensor_scri_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_tensor_scri_tests PRIVATE Kokkos::kokkos)
  find_program(ELLIPTIC_SHIFT_3D_PYTHON NAMES python3 REQUIRED
      DOC "Python with NumPy for independent elliptic-shift radial audits")
  execute_process(COMMAND ${ELLIPTIC_SHIFT_3D_PYTHON} -c "import numpy"
      RESULT_VARIABLE ELLIPTIC_SHIFT_3D_NUMPY_STATUS OUTPUT_QUIET ERROR_QUIET)
  if(NOT ELLIPTIC_SHIFT_3D_NUMPY_STATUS EQUAL 0)
    message(FATAL_ERROR "3D shift reference tests require NumPy; set ELLIPTIC_SHIFT_3D_PYTHON to a Python with NumPy")
  endif()
  add_test(NAME elliptic_shift_3d_tensor_scri_radial
      COMMAND ${ELLIPTIC_SHIFT_3D_PYTHON} ${CMAKE_SOURCE_DIR}/tst/elliptic_shift_3d/test_scri_radial.py
      $<TARGET_FILE:elliptic_shift_3d_tensor_scri_tests>)
  set_tests_properties(elliptic_shift_3d_tensor_scri_radial PROPERTIES TIMEOUT 30)
  add_test(NAME elliptic_shift_3d_selected_telegraph_support COMMAND ${ELLIPTIC_SHIFT_3D_PYTHON}
      ${CMAKE_SOURCE_DIR}/tst/elliptic_shift_3d/test_telegraph_support.py $<TARGET_FILE:elliptic_shift_3d_operator_tests>)
  set_tests_properties(elliptic_shift_3d_selected_telegraph_support PROPERTIES TIMEOUT 30)
  add_executable(elliptic_shift_3d_wormhole_chart_tests tst/elliptic_shift_3d/test_wormhole_chart.cpp)
  target_include_directories(elliptic_shift_3d_wormhole_chart_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_wormhole_chart_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_wormhole_chart_radial
      COMMAND ${ELLIPTIC_SHIFT_3D_PYTHON} ${CMAKE_SOURCE_DIR}/tst/elliptic_shift_3d/test_wormhole_chart.py
      $<TARGET_FILE:elliptic_shift_3d_wormhole_chart_tests>)
  set_tests_properties(elliptic_shift_3d_wormhole_chart_radial PROPERTIES TIMEOUT 30)
  add_executable(elliptic_shift_3d_chart_equivalence_tests tst/elliptic_shift_3d/test_chart_equivalence.cpp)
  target_include_directories(elliptic_shift_3d_chart_equivalence_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_chart_equivalence_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_chart_physical_equivalence COMMAND elliptic_shift_3d_chart_equivalence_tests)
  set_tests_properties(elliptic_shift_3d_chart_physical_equivalence PROPERTIES TIMEOUT 30)
  add_library(elliptic_shift_3d_radial_kernel SHARED tst/elliptic_shift_1d/spherical_api.cpp)
  target_include_directories(elliptic_shift_3d_radial_kernel PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_radial_kernel PRIVATE Kokkos::kokkos)
  add_executable(elliptic_shift_3d_formal_scri_probe tst/elliptic_shift_1d/scri_chart_probe.cpp)
  target_include_directories(elliptic_shift_3d_formal_scri_probe PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_formal_scri_probe PRIVATE Kokkos::kokkos)
  add_executable(elliptic_shift_3d_native_fields_tests tst/elliptic_shift_3d/test_native_fields.cpp)
  target_include_directories(elliptic_shift_3d_native_fields_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_native_fields_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_native_field_layout COMMAND elliptic_shift_3d_native_fields_tests)
  set_tests_properties(elliptic_shift_3d_native_field_layout PROPERTIES TIMEOUT 30)
  add_executable(elliptic_shift_3d_wormhole_diagnostic tst/elliptic_shift_3d/diagnose_wormhole.cpp)
  target_include_directories(elliptic_shift_3d_wormhole_diagnostic PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_wormhole_diagnostic PRIVATE Kokkos::kokkos)
  add_executable(elliptic_shift_3d_scri_reconstruction_tests tst/elliptic_shift_3d/test_scri_reconstruction.cpp)
  target_include_directories(elliptic_shift_3d_scri_reconstruction_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_scri_reconstruction_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_scri_reconstruction COMMAND elliptic_shift_3d_scri_reconstruction_tests)
  set_tests_properties(elliptic_shift_3d_scri_reconstruction PROPERTIES TIMEOUT 30)
  foreach(component native_stage puncture_position puncture_vector positive_field_chart regular_penrose regular_kernel stationary_omega_product metric_divergence log_volume_norm enriched_log_derivative puncture_log_plans puncture_refinement composite_topology radial_line radial_line_contract topology_cycles shift_acceptance_audit odd_radial_log_derivative affine_shift_remainder enriched_log_value)
    add_executable(elliptic_shift_3d_${component}_tests tst/elliptic_shift_3d/test_${component}.cpp)
    target_include_directories(elliptic_shift_3d_${component}_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(elliptic_shift_3d_${component}_tests PRIVATE Kokkos::kokkos)
  endforeach()
  add_test(NAME elliptic_shift_3d_native_stage_state COMMAND elliptic_shift_3d_native_stage_tests)
  add_test(NAME elliptic_shift_3d_puncture_position COMMAND elliptic_shift_3d_puncture_position_tests)
  add_test(NAME elliptic_shift_3d_positive_field_chart COMMAND elliptic_shift_3d_positive_field_chart_tests)
  add_test(NAME elliptic_shift_3d_regular_penrose COMMAND elliptic_shift_3d_regular_penrose_tests --expect-regular-rhs)
  add_test(NAME elliptic_shift_3d_regular_kernel COMMAND elliptic_shift_3d_regular_kernel_tests)
  add_test(NAME elliptic_shift_3d_stationary_omega_product COMMAND elliptic_shift_3d_stationary_omega_product_tests)
  add_test(NAME elliptic_shift_3d_puncture_log_plans COMMAND elliptic_shift_3d_puncture_log_plans_tests)
  add_test(NAME elliptic_shift_3d_puncture_refinement COMMAND elliptic_shift_3d_puncture_refinement_tests)
  add_test(NAME elliptic_shift_3d_topology_transfers COMMAND elliptic_shift_3d_composite_topology_tests)
  add_test(NAME elliptic_shift_3d_radial_line_contract COMMAND elliptic_shift_3d_radial_line_contract_tests)
  add_test(NAME elliptic_shift_3d_topology_vector_cycle COMMAND elliptic_shift_3d_topology_cycles_tests --n 6 --shells 7 --equation vector)
  add_test(NAME elliptic_shift_3d_original_acceptance COMMAND elliptic_shift_3d_shift_acceptance_audit_tests)
  add_test(NAME elliptic_shift_3d_odd_radial_falsifier COMMAND elliptic_shift_3d_odd_radial_log_derivative_tests)
  add_test(NAME elliptic_shift_3d_affine_shift_remainder COMMAND elliptic_shift_3d_affine_shift_remainder_tests)
  add_test(NAME elliptic_shift_3d_enriched_log_value COMMAND elliptic_shift_3d_enriched_log_value_tests)
  set_tests_properties(elliptic_shift_3d_affine_shift_remainder elliptic_shift_3d_enriched_log_value PROPERTIES TIMEOUT 30)
  set_tests_properties(elliptic_shift_3d_puncture_refinement elliptic_shift_3d_topology_transfers elliptic_shift_3d_radial_line_contract elliptic_shift_3d_original_acceptance elliptic_shift_3d_odd_radial_falsifier PROPERTIES TIMEOUT 30)
  set_tests_properties(elliptic_shift_3d_topology_vector_cycle PROPERTIES TIMEOUT 90)
  set_tests_properties(elliptic_shift_3d_puncture_log_plans PROPERTIES TIMEOUT 30)
  add_test(NAME elliptic_shift_3d_log_volume_norm COMMAND elliptic_shift_3d_log_volume_norm_tests)
  add_test(NAME elliptic_shift_3d_enriched_log_derivative COMMAND elliptic_shift_3d_enriched_log_derivative_tests)
  set_tests_properties(elliptic_shift_3d_enriched_log_derivative PROPERTIES TIMEOUT 30)
  set_tests_properties(elliptic_shift_3d_log_volume_norm PROPERTIES TIMEOUT 30)
  set_tests_properties(elliptic_shift_3d_regular_penrose elliptic_shift_3d_regular_kernel elliptic_shift_3d_stationary_omega_product PROPERTIES TIMEOUT 30)
  set_tests_properties(elliptic_shift_3d_native_stage_state elliptic_shift_3d_puncture_position elliptic_shift_3d_positive_field_chart PROPERTIES TIMEOUT 30)
  add_executable(elliptic_shift_3d_composite_tests tst/elliptic_shift_3d/test_composite.cpp)
  target_include_directories(elliptic_shift_3d_composite_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_composite_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_global_composite
      COMMAND elliptic_shift_3d_composite_tests 6 1 1 1 0)
  add_test(NAME elliptic_shift_3d_radial_interface
      COMMAND elliptic_shift_3d_composite_tests 6 2 1 1 0 1e-10 4 1)
  set_tests_properties(elliptic_shift_3d_radial_interface PROPERTIES TIMEOUT 180)
  set_tests_properties(elliptic_shift_3d_global_composite PROPERTIES TIMEOUT 180)
  add_executable(elliptic_shift_3d_coupled_pilot tst/elliptic_shift_3d/coupled.cpp)
  target_include_directories(elliptic_shift_3d_coupled_pilot PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_coupled_pilot PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_coupled_minkowski
      COMMAND elliptic_shift_3d_coupled_pilot 6 0 0.0005 2 0 1)
  set_tests_properties(elliptic_shift_3d_coupled_minkowski PROPERTIES TIMEOUT 120)
  add_test(NAME elliptic_shift_3d_coupled_tensor_scri
      COMMAND elliptic_shift_3d_coupled_pilot 6 0 0.0005 2 0 1 1e-11 "" "" "" 4 1)
  set_tests_properties(elliptic_shift_3d_coupled_tensor_scri PROPERTIES TIMEOUT 120)
  add_executable(elliptic_shift_3d_lifecycle_tests tst/elliptic_shift_3d/test_lifecycle.cpp)
  target_include_directories(elliptic_shift_3d_lifecycle_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(elliptic_shift_3d_lifecycle_tests PRIVATE Kokkos::kokkos)
  add_test(NAME elliptic_shift_3d_regrid_restart
      COMMAND elliptic_shift_3d_lifecycle_tests ${CMAKE_BINARY_DIR}/composite-checkpoint-test.bin)
  add_test(NAME elliptic_shift_3d_radial_interface_regrid
      COMMAND elliptic_shift_3d_lifecycle_tests ${CMAKE_BINARY_DIR}/composite-radial-checkpoint-test.bin 1)
  set_tests_properties(elliptic_shift_3d_radial_interface_regrid PROPERTIES TIMEOUT 180)
  set_tests_properties(elliptic_shift_3d_regrid_restart PROPERTIES TIMEOUT 60)
  if(Athena_ENABLE_MPI)
    add_executable(elliptic_shift_3d_mpi_tests tst/elliptic_shift_3d/test_mpi.cpp)
    target_include_directories(elliptic_shift_3d_mpi_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(elliptic_shift_3d_mpi_tests PRIVATE Kokkos::kokkos MPI::MPI_CXX)
    add_test(NAME elliptic_shift_3d_mpi_reference
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:elliptic_shift_3d_mpi_tests>)
    set_tests_properties(elliptic_shift_3d_mpi_reference PROPERTIES TIMEOUT 120 PROCESSORS 2)
  endif()
endif()
if(Athena_ENABLE_CUBED_SPHERE_TESTS)
  enable_testing()
  add_executable(cubed_sphere_map_tests tst/cubed_sphere/test_map.cpp)
  target_include_directories(cubed_sphere_map_tests PRIVATE
      ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
  target_link_libraries(cubed_sphere_map_tests PRIVATE Kokkos::kokkos)
  add_test(NAME cubed_sphere_map COMMAND cubed_sphere_map_tests)
  if(Athena_ENABLE_CUBED_SPHERE_CONFORMAL_PROBE)
    add_executable(cubed_sphere_conformal_bridge_tests tst/cubed_sphere/test_conformal_bridge.cpp)
    target_include_directories(cubed_sphere_conformal_bridge_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(cubed_sphere_conformal_bridge_tests PRIVATE Kokkos::kokkos)
    add_test(NAME cubed_sphere_conformal_bridge COMMAND cubed_sphere_conformal_bridge_tests)
  endif()
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME cubed_sphere_regular_z4c
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tst/cubed_sphere/check_evolution.py
        $<TARGET_FILE:athena> --suite smoke --output ${CMAKE_BINARY_DIR}/Testing/cubed_sphere)
    set_tests_properties(cubed_sphere_regular_z4c PROPERTIES TIMEOUT 300)
    add_test(NAME cubed_sphere_scalar_transport
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tst/cubed_sphere/check_evolution.py
        $<TARGET_FILE:athena> --suite scalar --output ${CMAKE_BINARY_DIR}/Testing/cubed_sphere_scalar)
    set_tests_properties(cubed_sphere_scalar_transport PROPERTIES TIMEOUT 300)
    add_test(NAME cubed_sphere_failure_capture
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tst/cubed_sphere/check_evolution.py
        $<TARGET_FILE:athena> --suite failure --output ${CMAKE_BINARY_DIR}/Testing/cubed_sphere_failure)
    set_tests_properties(cubed_sphere_failure_capture PROPERTIES TIMEOUT 120)
  endif()
endif()
