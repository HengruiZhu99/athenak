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
option(Athena_BUILD_UNIT_TESTS "Build isolated C++ unit-test executables" OFF)
option(Athena_ENABLE_IRISK_INTERPOLATOR
       "Enable the input-selected IrisK spectral ADM importer" OFF)
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
if (${PROBLEM} STREQUAL "z4c_irisk_xcts")
  # Preserve the historical custom-PROBLEM spelling, but route it through the
  # same named entry point used by built_in_pgens.
  set(Athena_ENABLE_IRISK_INTERPOLATOR ON CACHE BOOL
      "Enable the input-selected IrisK spectral ADM importer" FORCE)
endif()
if (NOT ${PROBLEM} STREQUAL "built_in_pgens" AND
    NOT ${PROBLEM} STREQUAL "z4c_irisk_xcts")
  message(STATUS "Including user-specified problem generator file: ${PROBLEM}")
  set(USER_PROBLEM_ENABLED 1)
else()
  set(USER_PROBLEM_ENABLED 0)
endif()
if (Athena_ENABLE_IRISK_INTERPOLATOR)
  set(IRISK_INTERPOLATOR_ENABLED 1)
else()
  set(IRISK_INTERPOLATOR_ENABLED 0)
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
add_subdirectory(src)
add_subdirectory(kokkos)
include_directories(${Kokkos_INCLUDE_DIRS_RET})

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

if (Athena_ENABLE_IRISK_INTERPOLATOR)
  get_filename_component(_IRISK_DEFAULT_ROOT "${CMAKE_SOURCE_DIR}/../bbhk" ABSOLUTE)
  set(IRISK_ROOT "${_IRISK_DEFAULT_ROOT}" CACHE PATH
      "IrisK source tree providing the AthenaK spectral interpolator")
  set(IRISK_INTERPOLATOR_LIBRARY
      "${IRISK_ROOT}/build/serial-clang/src/libiris_athenak_interpolator.a"
      CACHE FILEPATH "Built IrisK AthenaK spectral interpolation library")
  set(_IRISK_INTERPOLATOR_HEADER
      "${IRISK_ROOT}/src/initial_data/handoff/irisk_athenak_spectral_interpolator.h")
  if (NOT EXISTS "${_IRISK_INTERPOLATOR_HEADER}" OR
      NOT EXISTS "${IRISK_INTERPOLATOR_LIBRARY}")
    message(FATAL_ERROR
      "Athena_ENABLE_IRISK_INTERPOLATOR requires IrisK's spectral "
      "interpolation library. "
      "Build target iris_athenak_interpolator in IrisK and set IRISK_ROOT / "
      "IRISK_INTERPOLATOR_LIBRARY if it is not in the default serial-clang build.")
  endif()
  target_include_directories(athena PRIVATE
      "${IRISK_ROOT}/src/initial_data/handoff")
  target_link_libraries(athena PUBLIC "${IRISK_INTERPOLATOR_LIBRARY}")
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

if (${PROBLEM} STREQUAL "id_solve")
  # --- HDF5 I/O (dynamic by default, static optional) --------------------------
  #   • Set STATIC_HDF5=ON at CMake configure time to bake every HDF5 object
  #     into the executable for easy archival or clusters that disallow .so’s.
  #   • Leave it OFF (default) to keep the binary small and enable VOL/plugins.
  #find_package(hdf5 COMPONENTS C HL REQUIRED)
  #target_include_directories(athena PRIVATE
  #    ${CMAKE_SOURCE_DIR}/hdf5/include)
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  target_link_libraries(athena PUBLIC
    hdf5::hdf5
    hdf5::hdf5_hl)
endif()

# enable include of header files created in build directories
target_include_directories(athena PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

configure_file(config.hpp.in config.hpp)

if (Athena_BUILD_UNIT_TESTS)
  # config.hpp exposes MPI_PARALLEL_ENABLED to every unit translation unit.
  # Link the imported MPI target directory-wide so tests that include Athena
  # headers receive mpi.h, and tests that compile production helpers also
  # resolve any enabled MPI calls.
  if (ENABLE_MPI)
    link_libraries(MPI::MPI_CXX)
  endif()
  target_sources(
      athena PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/cartoon_production_kernel_test.cpp)
  target_compile_definitions(athena PRIVATE ATHENA_Z4C_KERNEL_TESTS=1)

  enable_testing()
  find_package(PythonInterp 3 REQUIRED)
  # Low-frequency qualification only: each order has its own translation unit to
  # bound compiler memory, and the expensive grid convergence is driven by the
  # input-selected production MMS rather than routine CTest.
  add_executable(
      athena_cartoon_derivatives_unit_test EXCLUDE_FROM_ALL
      tst/unit/z4c/cartoon_derivatives_test.cpp
      tst/unit/z4c/cartoon_derivatives_order2_test.cpp
      tst/unit/z4c/cartoon_derivatives_order4_test.cpp
      tst/unit/z4c/cartoon_derivatives_order6_test.cpp)
  target_include_directories(
      athena_cartoon_derivatives_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_cartoon_derivatives_unit_test PRIVATE Kokkos::kokkos)

  add_executable(
      athena_cartoon_mms_structure_unit_test
      tst/unit/z4c/cartoon_mms_structure_test.cpp)
  target_include_directories(
      athena_cartoon_mms_structure_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_cartoon_mms_structure_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_cartoon_mms_structure
      COMMAND athena_cartoon_mms_structure_unit_test)
  find_program(UV_EXECUTABLE uv)
  if(UV_EXECUTABLE)
    add_test(
        NAME athena.z4c_cartoon_mms_static
        COMMAND
        ${UV_EXECUTABLE} run --no-project
        --with-requirements
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/requirements-z4c-cartoon-mms.txt
        python
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_mms_static_test.py
        --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    # The test deliberately fails closed when its exact SymPy dependency is
    # unavailable; the fallback preserves that useful error on minimal hosts.
    add_test(
        NAME athena.z4c_cartoon_mms_static
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_mms_static_test.py
        --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  endif()
  add_executable(
      athena_cartoon_mms_generated_reference_unit_test
      tst/unit/z4c/z4c_cartoon_mms_generated_reference_test.cpp)
  target_include_directories(
      athena_cartoon_mms_generated_reference_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_cartoon_mms_generated_reference_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_cartoon_mms_generated_reference
      COMMAND athena_cartoon_mms_generated_reference_unit_test
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_derivatives_reference.json)

  add_executable(
      athena_z4c_symmetry_dispatch_unit_test
      tst/unit/z4c/z4c_symmetry_dispatch_test.cpp
      src/z4c/z4c_symmetry.cpp)
  target_include_directories(
      athena_z4c_symmetry_dispatch_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_z4c_symmetry_dispatch_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_symmetry_dispatch
      COMMAND athena_z4c_symmetry_dispatch_unit_test)

  add_executable(
      athena_z4c_cartoon_axis_parity_unit_test
      tst/unit/z4c/cartoon_axis_parity_test.cpp)
  target_include_directories(
      athena_z4c_cartoon_axis_parity_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_axis_parity_unit_test PRIVATE Kokkos::kokkos)
  if (ENABLE_MPI)
    target_link_libraries(
        athena_z4c_cartoon_axis_parity_unit_test PRIVATE MPI::MPI_CXX)
  endif()
  add_test(
      NAME athena.z4c_cartoon_axis_parity
      COMMAND athena_z4c_cartoon_axis_parity_unit_test)

  add_executable(
      athena_z4c_cartoon_axis_boundary_unit_test
      tst/unit/z4c/cartoon_axis_boundary_test.cpp)
  target_include_directories(
      athena_z4c_cartoon_axis_boundary_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_axis_boundary_unit_test PRIVATE Kokkos::kokkos)
  if (ENABLE_MPI)
    target_link_libraries(
        athena_z4c_cartoon_axis_boundary_unit_test PRIVATE MPI::MPI_CXX)
  endif()
  add_test(
      NAME athena.z4c_cartoon_axis_boundary
      COMMAND athena_z4c_cartoon_axis_boundary_unit_test)

  add_executable(
      athena_z4c_cartoon_axis_centered_derivatives_unit_test
      tst/unit/z4c/cartoon_axis_centered_derivatives_test.cpp)
  target_include_directories(
      athena_z4c_cartoon_axis_centered_derivatives_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_axis_centered_derivatives_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_cartoon_axis_centered_derivatives
      COMMAND athena_z4c_cartoon_axis_centered_derivatives_unit_test)

  add_executable(
      athena_z4c_cartoon_regular_functionals_unit_test
      tst/unit/z4c/cartoon_regular_functionals_test.cpp)
  add_test(
      NAME athena.z4c_cartoon_regular_functionals
      COMMAND athena_z4c_cartoon_regular_functionals_unit_test)

  add_executable(
      athena_z4c_symmetry_validation_unit_test
      tst/unit/z4c/z4c_symmetry_validation_test.cpp
      src/z4c/z4c_symmetry.cpp)
  target_include_directories(
      athena_z4c_symmetry_validation_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_z4c_symmetry_validation_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_symmetry_validation
      COMMAND athena_z4c_symmetry_validation_unit_test)

  add_test(
      NAME athena.z4c_irisk_import_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_irisk_import_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})

  add_executable(
      athena_z4c_irisk_coordinate_map_unit_test
      tst/unit/z4c/z4c_irisk_coordinate_map_test.cpp)
  target_include_directories(
      athena_z4c_irisk_coordinate_map_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_irisk_coordinate_map_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_irisk_coordinate_map
      COMMAND athena_z4c_irisk_coordinate_map_unit_test)

  add_test(
      NAME athena.z4c_kerr_puncture_reference
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/kerr_puncture_reference_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
      NAME athena.z4c_kerr_half_plane_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_kerr_half_plane_static_test.py)

  add_test(
      NAME athena.z4c_kerr_half_plane_init
      COMMAND
      ${CMAKE_COMMAND}
      -DATHENA=$<TARGET_FILE:athena>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_kerr_half_plane_init.athinput
      -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/z4c_kerr_half_plane_init_test
      -DPYTHON=${PYTHON_EXECUTABLE}
      -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DHISTORY_VALIDATOR=${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_history_test.py
      -DFAILURE_REGION_EXTRACTOR=${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_failure_region_extract.py
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_kerr_half_plane_init_test.cmake)

  add_test(
      NAME athena.z4c_kerr_half_plane_amr_orders
      COMMAND
      ${CMAKE_COMMAND}
      -DATHENA=$<TARGET_FILE:athena>
      -DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_kerr_half_plane_amr.athinput
      -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/z4c_kerr_half_plane_amr_test
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_kerr_half_plane_amr_test.cmake)

  add_test(
      NAME athena.z4c_kerr_half_plane_restart
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_kerr_half_plane_restart_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_kerr_half_plane_restart.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_kerr_half_plane_restart_test
      --python ${PYTHON_EXECUTABLE}
      --history-validator
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_history_test.py)

  add_test(
      NAME athena.z4c_cartoon_frozen_operator_analysis
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cartoon_frozen_operator_analysis.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --output ${CMAKE_CURRENT_BINARY_DIR}/z4c_cartoon_frozen_operator_analysis.json)
  set_tests_properties(
      athena.z4c_cartoon_frozen_operator_analysis
      PROPERTIES TIMEOUT 120 LABELS "cartoon;stability;python")

  add_test(
      NAME athena.z4c_cartoon_operator_inventory
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/generate_z4c_cartoon_operator_table.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
      NAME athena.z4c_kerr_half_plane_convergence_analyzer
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/test_suite/z4c/cartoon_half_plane_kerr_convergence.py
      --self-test)
  add_test(
      NAME athena.z4c_kerr_half_plane_campaign
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/test_suite/z4c/cartoon_half_plane_kerr_campaign.py
      self-test
      --template
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_kerr_half_plane_convergence.athinput)

  add_executable(
      athena_stored_domain_bounds_unit_test
      tst/unit/z4c/stored_domain_bounds_test.cpp)
  target_include_directories(
      athena_stored_domain_bounds_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_stored_domain_bounds_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.stored_domain_bounds
      COMMAND athena_stored_domain_bounds_unit_test)

  add_executable(
      athena_z4c_shared_geometry_policy_unit_test
      tst/unit/z4c/shared_geometry_policy_test.cpp)
  target_include_directories(
      athena_z4c_shared_geometry_policy_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_shared_geometry_policy_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_shared_geometry_policy
      COMMAND athena_z4c_shared_geometry_policy_unit_test)

  add_executable(
      athena_z4c_chi_prolongation_unit_test
      tst/unit/z4c/z4c_chi_prolongation_test.cpp)
  target_include_directories(
      athena_z4c_chi_prolongation_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_chi_prolongation_unit_test PRIVATE Kokkos::kokkos)
  if (ENABLE_MPI)
    target_link_libraries(
        athena_z4c_chi_prolongation_unit_test PRIVATE MPI::MPI_CXX)
  endif()
  add_test(
      NAME athena.z4c_chi_prolongation
      COMMAND athena_z4c_chi_prolongation_unit_test)
  add_executable(
      athena_z4c_telegraph_damping_unit_test
      tst/unit/z4c/z4c_telegraph_damping_test.cpp)
  target_include_directories(
      athena_z4c_telegraph_damping_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_telegraph_damping_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_telegraph_damping
      COMMAND athena_z4c_telegraph_damping_unit_test)
  add_test(
      NAME athena.z4c_telegraph_damping_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_telegraph_damping_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.z4c_amr_chi_refresh_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_amr_chi_refresh_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.z4c_cartoon_amr_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/cartoon_amr_static_test.py)
  add_test(
      NAME athena.z4c_shock_avoiding_gauge_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_shock_avoiding_gauge_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --athena $<TARGET_FILE:athena>)

  add_executable(
      athena_z4c_cartoon_m0_fastflow_unit_test
      tst/unit/z4c/cartoon_m0_fastflow_test.cpp
      src/z4c/cartoon_m0_fastflow.cpp)
  target_compile_definitions(
      athena_z4c_cartoon_m0_fastflow_unit_test
      PRIVATE ATHENA_CARTOON_M0_MATH_ONLY=1)
  target_include_directories(
      athena_z4c_cartoon_m0_fastflow_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_m0_fastflow_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_cartoon_m0_fastflow
      COMMAND athena_z4c_cartoon_m0_fastflow_unit_test)

  add_test(
      NAME athena.z4c_policy_migration
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_policy_migration_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})

  # These byte-for-byte Cartesian goldens were independently reviewed only for
  # GNU Real64 Serial/OpenMP.  Do not register them on CUDA: device sin/cos and
  # fused arithmetic need not reproduce GNU host bytes, and this host regression
  # must never be mistaken for the dedicated CUDA Cartoon execution gate below.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     NOT Athena_SINGLE_PRECISION AND NOT Kokkos_ENABLE_CUDA AND
     (Kokkos_ENABLE_SERIAL OR Kokkos_ENABLE_OPENMP))
    add_test(
        NAME athena.z4c_rhs_policy_production_host_exact
        COMMAND
        ${CMAKE_COMMAND}
        -DATHENA=$<TARGET_FILE:athena>
        -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/z4c_rhs_policy_production
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_rhs_policy_production_test.cmake)
    set_tests_properties(
        athena.z4c_rhs_policy_production_host_exact
        PROPERTIES LABELS "host-exact-regression")
  endif()

  add_test(
      NAME athena.z4c_cartoon_production_kernels
      COMMAND $<TARGET_FILE:athena> --z4c-cartoon-kernel-test)
  add_test(
      NAME athena.z4c_cartoon_production_kernels_cuda_required
      COMMAND $<TARGET_FILE:athena> --z4c-cartoon-kernel-test --require-cuda)
  set_tests_properties(
      athena.z4c_cartoon_production_kernels_cuda_required
      PROPERTIES LABELS "cuda-required;cartoon-production-kernels")
  if(NOT Kokkos_ENABLE_CUDA)
    set_tests_properties(
        athena.z4c_cartoon_production_kernels_cuda_required
        PROPERTIES DISABLED TRUE)
    add_test(
        NAME athena.z4c_cartoon_cuda_requirement_rejects_host
        COMMAND $<TARGET_FILE:athena> --z4c-cartoon-kernel-test --require-cuda)
    set_tests_properties(
        athena.z4c_cartoon_cuda_requirement_rejects_host
        PROPERTIES WILL_FAIL TRUE LABELS "host-negative")
  endif()

  add_executable(
      athena_pdf_validation_unit_test
      tst/unit/outputs/pdf_validation_test.cpp)
  target_include_directories(
      athena_pdf_validation_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_pdf_validation_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.pdf_validation
      COMMAND athena_pdf_validation_unit_test)

  add_executable(
      athena_pdf_scatter_unit_test
      tst/unit/outputs/pdf_scatter_test.cpp)
  target_include_directories(
      athena_pdf_scatter_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_pdf_scatter_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.pdf_scatter_backend
      COMMAND athena_pdf_scatter_unit_test)
  if(Kokkos_ENABLE_OPENMP)
    foreach(pdf_threads 1 2 8)
      add_test(
          NAME athena.pdf_scatter_openmp_${pdf_threads}
          COMMAND athena_pdf_scatter_unit_test)
      set_tests_properties(
          athena.pdf_scatter_openmp_${pdf_threads}
          PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=${pdf_threads}")
    endforeach()
  endif()
  add_test(
      NAME athena.pdf_scatter_cuda_required
      COMMAND athena_pdf_scatter_unit_test --require-cuda)
  set_tests_properties(
      athena.pdf_scatter_cuda_required PROPERTIES LABELS "cuda-required")
  if(NOT Kokkos_ENABLE_CUDA)
    set_tests_properties(athena.pdf_scatter_cuda_required PROPERTIES DISABLED TRUE)
  endif()

  add_test(
      NAME athena.z4c_preallocation
      COMMAND
      ${CMAKE_COMMAND}
      -DATHENA=$<TARGET_FILE:athena>
      -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
      -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/z4c_preallocation_test
      -P ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_preallocation_test.cmake)

  add_test(
      NAME athena.z4c_restart_carrier
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_restart_carrier_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_restart_carrier.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_restart_carrier_test
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --legacy-base ea59e07ce7026b8b06720c7cb32bfe6788652978)
endif()

install( TARGETS athena DESTINATION bin )
