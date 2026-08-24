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

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE ATHENA_GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if (NOT ATHENA_GIT_COMMIT_HASH)
  set(ATHENA_GIT_COMMIT_HASH "unknown")
endif()
target_compile_definitions(
  athena PRIVATE ATHENA_GIT_COMMIT_HASH="${ATHENA_GIT_COMMIT_HASH}")

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
  # bound compiler memory. The explicit target covers the complete manufactured
  # SO(2) API at the evolved VC axis and first four radial layers.
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
      athena_z4c_grid_layout_unit_test
      tst/unit/z4c/z4c_grid_layout_test.cpp)
  target_include_directories(
      athena_z4c_grid_layout_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_z4c_grid_layout_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_vc_layout
      COMMAND athena_z4c_grid_layout_unit_test layout)
  add_test(
      NAME athena.z4c_vc_coordinates
      COMMAND athena_z4c_grid_layout_unit_test coordinates)
  add_test(
      NAME athena.z4c_vc_collapsed_dimension
      COMMAND athena_z4c_grid_layout_unit_test collapsed)

  add_executable(
      athena_z4c_vertex_derivatives_unit_test
      tst/unit/z4c/z4c_vertex_derivatives_test.cpp)
  target_include_directories(
      athena_z4c_vertex_derivatives_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_vertex_derivatives_unit_test PRIVATE Kokkos::kokkos)
  foreach(symmetry IN ITEMS cartesian cartoon)
    foreach(order IN ITEMS o2 o4 o6)
      add_test(
          NAME athena.z4c_vc_${symmetry}_derivatives_${order}
          COMMAND athena_z4c_vertex_derivatives_unit_test ${symmetry}_${order})
    endforeach()
  endforeach()

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
      athena_z4c_cartoon_vertex_axis_unit_test
      tst/unit/z4c/cartoon_vertex_axis_test.cpp)
  target_include_directories(
      athena_z4c_cartoon_vertex_axis_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_vertex_axis_unit_test PRIVATE Kokkos::kokkos)
  foreach(axis_contract IN ITEMS scalar vector tensor adm constraint rhs_regularity)
    add_test(
        NAME athena.z4c_vc_cartoon_axis_${axis_contract}
        COMMAND athena_z4c_cartoon_vertex_axis_unit_test ${axis_contract})
  endforeach()

  add_executable(
      athena_z4c_vertex_topology_unit_test
      tst/unit/z4c/z4c_vertex_topology_test.cpp)
  target_include_directories(
      athena_z4c_vertex_topology_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_vertex_topology_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_vc_vertex_topology
      COMMAND athena_z4c_vertex_topology_unit_test)

  add_executable(
      athena_z4c_vertex_boundary_indices_unit_test
      tst/unit/z4c/z4c_vertex_boundary_indices_test.cpp)
  target_include_directories(
      athena_z4c_vertex_boundary_indices_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_vertex_boundary_indices_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_vc_boundary_indices
      COMMAND athena_z4c_vertex_boundary_indices_unit_test)

  add_test(
      NAME athena.z4c_vc_task_order
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_task_order_static_test.py)

  add_test(
      NAME athena.z4c_vc_dynamic_amr
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_dynamic_amr_test)
  set_tests_properties(
      athena.z4c_vc_dynamic_amr PROPERTIES TIMEOUT 60 LABELS "z4c;vertex;amr")

  add_test(
      NAME athena.z4c_vc_multi_family_derefine
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_multi_family_derefine_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_multi_family_derefine_test
      --authority-map)
  set_tests_properties(
      athena.z4c_vc_multi_family_derefine PROPERTIES TIMEOUT 120
      LABELS "z4c;vertex;2d;cartesian;amr;derefine;regression")

  add_test(
      NAME athena.z4c_vc_derefine_writer_lifecycle
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_multi_family_derefine_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_derefine_writer_lifecycle
      --dimensions 2 --family-count 3 --order 4 --transfer-order 6
      --writer-lifecycle)
  set_tests_properties(
      athena.z4c_vc_derefine_writer_lifecycle PROPERTIES TIMEOUT 180
      LABELS "z4c;vertex;2d;cartesian;amr;derefine;diagnostic")

  function(add_z4c_vc_multi_family_case name input dimensions families order
           transfer_order)
    add_test(
        NAME ${name}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_multi_family_derefine_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
        --dimensions ${dimensions} --family-count ${families}
        --order ${order} --transfer-order ${transfer_order})
    set_tests_properties(
        ${name} PROPERTIES TIMEOUT 180
        LABELS "z4c;vertex;${dimensions}d;cartesian;amr;derefine;regression")
  endfunction()
  add_z4c_vc_multi_family_case(
      athena.z4c_vc_multi_family_derefine_2d_o4_q6
      z4c_vc_multi_family_derefine_2d_wide.athinput 2 3 4 6)
  add_z4c_vc_multi_family_case(
      athena.z4c_vc_multi_family_derefine_2d_o6_q8
      z4c_vc_multi_family_derefine_2d_wide.athinput 2 3 6 8)
  add_test(
      NAME athena.z4c_vc_multi_family_derefine_2d_mixed_refine
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_multi_family_derefine_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_multi_family_derefine_2d_mixed_refine
      --dimensions 2 --family-count 3 --order 4 --transfer-order 6
      --mixed-refine)
  set_tests_properties(
      athena.z4c_vc_multi_family_derefine_2d_mixed_refine PROPERTIES TIMEOUT 180
      LABELS "z4c;vertex;2d;cartesian;amr;refine;derefine;regression")
  add_z4c_vc_multi_family_case(
      athena.z4c_vc_multi_family_derefine_3d_o2_q4
      z4c_vc_multi_family_derefine_3d.athinput 3 2 2 4)
  add_z4c_vc_multi_family_case(
      athena.z4c_vc_multi_family_derefine_3d_o4_q6
      z4c_vc_multi_family_derefine_3d.athinput 3 2 4 6)
  add_z4c_vc_multi_family_case(
      athena.z4c_vc_multi_family_derefine_3d_o6_q8
      z4c_vc_multi_family_derefine_3d.athinput 3 2 6 8)
  add_test(
      NAME athena.z4c_vc_multi_family_derefine_2d_constant
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_multi_family_derefine_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_multi_family_derefine_2d_constant
      --dimensions 2 --family-count 3 --order 2 --transfer-order 4
      --amplitude 0)
  set_tests_properties(
      athena.z4c_vc_multi_family_derefine_2d_constant PROPERTIES TIMEOUT 180
      LABELS "z4c;vertex;2d;cartesian;amr;derefine;regression;constant")

  function(add_z4c_vc_dynamic_amr_case name input dimensions geometry order
           root_leaves expected_volume)
    add_test(
        NAME ${name}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
        --dimensions ${dimensions}
        --geometry ${geometry}
        --order ${order}
        --root-leaves ${root_leaves}
        --expected-volume ${expected_volume})
    set_tests_properties(
        ${name} PROPERTIES TIMEOUT 120
        LABELS "z4c;vertex;${dimensions}d;${geometry};amr")
  endfunction()

  foreach(order IN ITEMS 2 4 6)
    add_z4c_vc_dynamic_amr_case(
        athena.z4c_vc_dynamic_amr_2d_cartesian_o${order}
        z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
        2 cartesian ${order} 4 16.0)
    add_z4c_vc_dynamic_amr_case(
        athena.z4c_vc_dynamic_amr_2d_cartoon_o${order}
        z4c_vc_minkowski_dynamic_amr.athinput
        2 cartoon ${order} 4 201.06192982974676)
    add_z4c_vc_dynamic_amr_case(
        athena.z4c_vc_dynamic_amr_3d_cartesian_o${order}
        z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
        3 cartesian ${order} 1 64.0)
  endforeach()
  foreach(integrator IN ITEMS rk1 rk2 rk3)
    add_test(
        NAME athena.z4c_vc_dynamic_amr_2d_cartesian_o4_${integrator}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
        --athena $<TARGET_FILE:athena>
        --input
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
        --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_dynamic_amr_2d_cartesian_o4_${integrator}
        --dimensions 2 --geometry cartesian --order 4
        --integrator ${integrator} --root-leaves 4 --expected-volume 16.0)
    set_tests_properties(
        athena.z4c_vc_dynamic_amr_2d_cartesian_o4_${integrator}
        PROPERTIES TIMEOUT 120
        LABELS "z4c;vertex;2d;cartesian;amr;${integrator}")
  endforeach()
  function(add_z4c_cc_lifecycle_case name input dimensions geometry root_leaves
           expected_volume)
    add_test(
        NAME ${name}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
        --centering cell --dimensions ${dimensions} --geometry ${geometry}
        --order 6 --root-leaves ${root_leaves}
        --expected-volume ${expected_volume})
    set_tests_properties(
        ${name} PROPERTIES TIMEOUT 120
        LABELS "z4c;cell;regression;${dimensions}d;${geometry};amr;lifecycle")
  endfunction()
  add_z4c_cc_lifecycle_case(
      athena.z4c_cc_lifecycle_2d_cartesian
      z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
      2 cartesian 4 16.0)
  add_z4c_cc_lifecycle_case(
      athena.z4c_cc_lifecycle_2d_cartoon
      z4c_vc_minkowski_dynamic_amr.athinput
      2 cartoon 4 201.06192982974676)
  add_z4c_cc_lifecycle_case(
      athena.z4c_cc_lifecycle_3d_cartesian
      z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
      3 cartesian 1 64.0)
  add_test(
      NAME athena.z4c_vc_dynamic_amr_invalid_target
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_dynamic_amr_invalid_target
      --dimensions 3 --geometry cartesian --order 6 --root-leaves 1
      --expected-volume 64.0 --invalid-target-test)
  set_tests_properties(
      athena.z4c_vc_dynamic_amr_invalid_target
      PROPERTIES TIMEOUT 30 LABELS "z4c;vertex;3d;cartesian;amr;negative")
  add_test(
      NAME athena.z4c_vc_dynamic_amr_invalid_halo
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_dynamic_amr_invalid_halo
      --dimensions 3 --geometry cartesian --order 6 --root-leaves 8
      --expected-volume 64.0 --invalid-halo-test)
  set_tests_properties(
      athena.z4c_vc_dynamic_amr_invalid_halo
      PROPERTIES TIMEOUT 30 LABELS "z4c;vertex;3d;cartesian;amr;negative")
  if (Athena_ENABLE_MPI)
    foreach(ranks IN ITEMS 2 4)
      add_test(
          NAME athena.z4c_vc_split_family_derefine_mpi${ranks}
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_split_family_derefine_test.py
          --athena $<TARGET_FILE:athena>
          --input
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
          --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_split_family_derefine_mpi${ranks}
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG}
          --ranks ${ranks})
      set_tests_properties(
          athena.z4c_vc_split_family_derefine_mpi${ranks} PROPERTIES TIMEOUT 240
          PROCESSORS ${ranks}
          LABELS "z4c;vertex;2d;cartesian;amr;derefine;mpi;regression")
      add_test(
          NAME athena.z4c_vc_split_family_mixed_amr_mpi${ranks}
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_split_family_derefine_test.py
          --athena $<TARGET_FILE:athena>
          --input
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
          --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_split_family_mixed_amr_mpi${ranks}
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG}
          --ranks ${ranks} --mixed-refine)
      set_tests_properties(
          athena.z4c_vc_split_family_mixed_amr_mpi${ranks} PROPERTIES TIMEOUT 240
          PROCESSORS ${ranks}
          LABELS "z4c;vertex;2d;cartesian;amr;refine;derefine;mpi;regression")
      add_test(
          NAME athena.z4c_vc_split_family_remote_lower_mpi${ranks}
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_split_family_derefine_test.py
          --athena $<TARGET_FILE:athena>
          --input
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
          --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_split_family_remote_lower_mpi${ranks}
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG}
          --ranks ${ranks} --ownership-case remote_lower)
      set_tests_properties(
          athena.z4c_vc_split_family_remote_lower_mpi${ranks}
          PROPERTIES TIMEOUT 240 PROCESSORS ${ranks}
          LABELS "z4c;vertex;2d;cartesian;amr;derefine;mpi;regression;move-right")
    endforeach()
    add_test(
        NAME athena.z4c_vc_split_family_dual_target_mpi4
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_split_family_derefine_test.py
        --athena $<TARGET_FILE:athena>
        --input
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_multi_family_derefine_2d_wide.athinput
        --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_split_family_dual_target_mpi4
        --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG}
        --ranks 4 --ownership-case dual_split)
    set_tests_properties(
        athena.z4c_vc_split_family_dual_target_mpi4
        PROPERTIES TIMEOUT 240 PROCESSORS 4
        LABELS "z4c;vertex;2d;cartesian;amr;derefine;mpi;regression;dual-split")

    function(add_z4c_vc_dynamic_amr_mpi_case name input dimensions geometry
             ranks root_leaves expected_volume meshblocks_per_dim)
      add_test(
          NAME ${name}
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
          --athena $<TARGET_FILE:athena>
          --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
          --dimensions ${dimensions} --geometry ${geometry} --order 6
          --ranks ${ranks} --root-leaves ${root_leaves}
          --meshblocks-per-dim ${meshblocks_per_dim}
          --expected-volume ${expected_volume}
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG})
      set_tests_properties(
          ${name} PROPERTIES TIMEOUT 240
          LABELS "z4c;vertex;${dimensions}d;${geometry};amr;mpi")
    endfunction()
    add_z4c_vc_dynamic_amr_mpi_case(
        athena.z4c_vc_dynamic_amr_2d_cartesian_mpi2
        z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
        2 cartesian 2 4 16.0 1)
    foreach(transfer IN ITEMS 4 6)
      add_test(
          NAME athena.z4c_vc_dynamic_amr_2d_cartesian_o4_q${transfer}_mpi2
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
          --athena $<TARGET_FILE:athena>
          --input
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
          --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_dynamic_amr_2d_cartesian_o4_q${transfer}_mpi2
          --dimensions 2 --geometry cartesian --order 4
          --vertex-prolongation-order ${transfer}
          --ranks 2 --root-leaves 4 --expected-volume 16.0
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG})
      set_tests_properties(
          athena.z4c_vc_dynamic_amr_2d_cartesian_o4_q${transfer}_mpi2
          PROPERTIES TIMEOUT 240
          LABELS "z4c;vertex;2d;cartesian;amr;o4;q${transfer};mpi")
    endforeach()
    add_z4c_vc_dynamic_amr_mpi_case(
        athena.z4c_vc_dynamic_amr_2d_cartoon_mpi2
        z4c_vc_minkowski_dynamic_amr.athinput
        2 cartoon 2 4 201.06192982974676 1)
    add_z4c_vc_dynamic_amr_mpi_case(
        athena.z4c_vc_dynamic_amr_3d_cartesian_mpi2
        z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
        3 cartesian 2 8 64.0 2)
    add_z4c_vc_dynamic_amr_mpi_case(
        athena.z4c_vc_dynamic_amr_3d_cartesian_mpi4
        z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
        3 cartesian 4 8 64.0 2)
  endif()
  add_test(
      NAME athena.z4c_vc_linear_wave_convergence_o4
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_linear_wave_convergence_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_linear_wave_o4.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_linear_wave_o4_test)
  set_tests_properties(
      athena.z4c_vc_linear_wave_convergence_o4
      PROPERTIES TIMEOUT 900 LABELS "z4c;vertex;cartesian;convergence")
  add_test(
      NAME athena.z4c_vc_static_multilevel_wave
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_static_multilevel_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_static_multilevel_wave.athinput
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_static_multilevel_test)
  set_tests_properties(
      athena.z4c_vc_static_multilevel_wave
      PROPERTIES TIMEOUT 120 LABELS "z4c;vertex;cartesian;amr;interface")
  function(add_z4c_vc_static_multilevel_case name input expected_leaves
           interface_x1 labels)
    add_test(
        NAME ${name}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_static_multilevel_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
        --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
        --expected-leaves ${expected_leaves} --interface-x1 ${interface_x1})
    set_tests_properties(
        ${name} PROPERTIES TIMEOUT 180
        LABELS "z4c;vertex;amr;interface;${labels}")
  endfunction()
  add_z4c_vc_static_multilevel_case(
      athena.z4c_vc_static_multilevel_2d_cartesian_o4
      z4c_vc_static_multilevel_2d_cartesian.athinput 5 0.0 "2d;cartesian")
  add_test(
      NAME athena.z4c_vc_static_multilevel_2d_cartoon_axis_o4
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_static_multilevel_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_static_multilevel_2d_cartoon_axis.athinput
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_static_multilevel_2d_cartoon_axis_o4
      --expected-leaves 5 --interface-x1 1.0 --cartoon-axis)
  set_tests_properties(
      athena.z4c_vc_static_multilevel_2d_cartoon_axis_o4
      PROPERTIES TIMEOUT 180 LABELS "z4c;vertex;amr;interface;2d;cartoon")
  add_z4c_vc_static_multilevel_case(
      athena.z4c_vc_static_multilevel_3d_cartesian_o4
      z4c_vc_static_multilevel_wave.athinput 9 0.0 "3d;cartesian")
  function(add_z4c_gauge_wave_case name input centering order dimensions)
    add_test(
        NAME ${name}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_gauge_wave_convergence_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
        --centering ${centering}
        --order ${order})
    set_tests_properties(
        ${name} PROPERTIES TIMEOUT 300
        LABELS "z4c;gauge_wave;analytic;${centering};${dimensions}d;o${order}")
  endfunction()
  foreach(centering IN ITEMS cell vertex)
    foreach(order IN ITEMS 2 4 6)
      add_z4c_gauge_wave_case(
          athena.z4c_${centering}_gauge_wave_2d_cartesian_o${order}
          z4c_gauge_wave_2d_cartesian.athinput ${centering} ${order} 2)
      add_z4c_gauge_wave_case(
          athena.z4c_${centering}_gauge_wave_3d_cartesian_o${order}
          z4c_gauge_wave_3d_cartesian.athinput ${centering} ${order} 3)
    endforeach()
  endforeach()
  if (Athena_ENABLE_MPI)
    foreach(ranks IN ITEMS 2 4)
      add_test(
          NAME athena.z4c_vc_static_multilevel_3d_cartesian_mpi${ranks}
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_static_multilevel_test.py
          --athena $<TARGET_FILE:athena>
          --input
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_static_multilevel_wave.athinput
          --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
          --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_static_multilevel_3d_cartesian_mpi${ranks}
          --expected-leaves 9 --interface-x1 0.0
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG}
          --ranks ${ranks})
      set_tests_properties(
          athena.z4c_vc_static_multilevel_3d_cartesian_mpi${ranks}
          PROPERTIES TIMEOUT 240
          LABELS "z4c;vertex;amr;interface;3d;cartesian;mpi")
    endforeach()
  endif()
  add_test(
      NAME athena.z4c_vc_amr_history
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_amr_history_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_amr_history_test)
  set_tests_properties(
      athena.z4c_vc_amr_history
      PROPERTIES TIMEOUT 120 LABELS "z4c;vertex;amr;history;mpi")
  add_test(
      NAME athena.z4c_vc_amr_history_3d_cartesian
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_amr_history_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_amr_history_3d_cartesian
      --dimensions 3)
  set_tests_properties(
      athena.z4c_vc_amr_history_3d_cartesian
      PROPERTIES TIMEOUT 600 LABELS "z4c;vertex;3d;cartesian;amr;history")
  if (Athena_ENABLE_MPI)
    add_test(
        NAME athena.z4c_vc_amr_history_mpi
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_amr_history_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_amr_history_mpi_test
        --mpiexec ${MPIEXEC_EXECUTABLE}
        --np-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(
        athena.z4c_vc_amr_history_mpi
        PROPERTIES TIMEOUT 150 LABELS "z4c;vertex;amr;history;mpi")
    add_test(
        NAME athena.z4c_vc_history_mpi
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_dynamic_amr_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_history_mpi_test
        --mpiexec ${MPIEXEC_EXECUTABLE}
        --np-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(
        athena.z4c_vc_history_mpi
        PROPERTIES TIMEOUT 90 LABELS "z4c;vertex;history;mpi")
    add_test(
        NAME athena.z4c_vc_amr_history_3d_cartesian_mpi
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_amr_history_test.py
        --athena $<TARGET_FILE:athena>
        --input
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
        --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_amr_history_3d_cartesian_mpi
        --dimensions 3 --mpiexec ${MPIEXEC_EXECUTABLE}
        --np-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(
        athena.z4c_vc_amr_history_3d_cartesian_mpi
        PROPERTIES TIMEOUT 300 LABELS "z4c;vertex;3d;cartesian;amr;history;mpi")
  endif()

  add_test(
      NAME athena.z4c_vc_restart
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_restart_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_restart_test)
  set_tests_properties(
      athena.z4c_vc_restart PROPERTIES TIMEOUT 90 LABELS "z4c;vertex;restart")
  function(add_z4c_vc_restart_case name input dimensions active_n1 active_n2
           active_n3 labels)
    add_test(
        NAME ${name}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_restart_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/${input}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}
        --dimensions ${dimensions}
        --active-n1 ${active_n1} --active-n2 ${active_n2} --active-n3 ${active_n3})
    set_tests_properties(
        ${name} PROPERTIES TIMEOUT 180 LABELS "z4c;vertex;restart;${labels}")
  endfunction()
  add_z4c_vc_restart_case(
      athena.z4c_vc_restart_2d_cartesian
      z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput 2 17 17 1
      "2d;cartesian")
  foreach(transfer IN ITEMS 4 6)
    add_test(
        NAME athena.z4c_vc_restart_2d_cartesian_o4_q${transfer}
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_restart_test.py
        --athena $<TARGET_FILE:athena>
        --input
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
        --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_restart_2d_cartesian_o4_q${transfer}
        --dimensions 2 --active-n1 17 --active-n2 17 --active-n3 1
        --spatial-order 4 --vertex-prolongation-order ${transfer})
    set_tests_properties(
        athena.z4c_vc_restart_2d_cartesian_o4_q${transfer}
        PROPERTIES TIMEOUT 180
        LABELS "z4c;vertex;restart;2d;cartesian;o4;q${transfer}")
  endforeach()
  add_z4c_vc_restart_case(
      athena.z4c_vc_restart_2d_cartoon
      z4c_vc_minkowski_dynamic_amr.athinput 2 17 17 1 "2d;cartoon")
  add_z4c_vc_restart_case(
      athena.z4c_vc_restart_3d_cartesian
      z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput 3 17 17 17
      "3d;cartesian")
  if (Athena_ENABLE_MPI)
    foreach(transfer IN ITEMS 4 6)
      add_test(
          NAME athena.z4c_vc_restart_rank_change_o4_q${transfer}
          COMMAND
          ${PYTHON_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_restart_test.py
          --athena $<TARGET_FILE:athena>
          --input
          ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_2d_cartesian.athinput
          --work-dir
          ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_restart_rank_change_o4_q${transfer}
          --dimensions 2 --active-n1 17 --active-n2 17 --active-n3 1
          --spatial-order 4 --vertex-prolongation-order ${transfer}
          --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG})
      set_tests_properties(
          athena.z4c_vc_restart_rank_change_o4_q${transfer}
          PROPERTIES TIMEOUT 180
          LABELS "z4c;vertex;restart;2d;cartesian;o4;q${transfer};mpi")
    endforeach()
    add_test(
        NAME athena.z4c_vc_restart_rank_change
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_restart_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr.athinput
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_restart_rank_change_test
        --mpiexec ${MPIEXEC_EXECUTABLE}
        --np-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(
        athena.z4c_vc_restart_rank_change
        PROPERTIES TIMEOUT 120 LABELS "z4c;vertex;restart;mpi")
    add_test(
        NAME athena.z4c_vc_restart_3d_cartesian_rank_change
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_restart_test.py
        --athena $<TARGET_FILE:athena>
        --input
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_minkowski_dynamic_amr_3d_cartesian.athinput
        --work-dir
        ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_restart_3d_cartesian_rank_change
        --dimensions 3 --active-n1 17 --active-n2 17 --active-n3 17
        --mpiexec ${MPIEXEC_EXECUTABLE} --np-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(
        athena.z4c_vc_restart_3d_cartesian_rank_change
        PROPERTIES TIMEOUT 240 LABELS "z4c;vertex;restart;3d;cartesian;mpi")
  endif()

  add_executable(
      athena_z4c_vertex_amr_unit_test
      tst/unit/z4c/z4c_vertex_amr_test.cpp)
  target_include_directories(
      athena_z4c_vertex_amr_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_vertex_amr_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_vc_amr_transfer
      COMMAND athena_z4c_vertex_amr_unit_test)

  add_executable(
      athena_z4c_vertex_history_quadrature_unit_test
      tst/unit/z4c/z4c_vertex_history_quadrature_test.cpp)
  target_include_directories(
      athena_z4c_vertex_history_quadrature_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_vertex_history_quadrature_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_vc_history_quadrature
      COMMAND athena_z4c_vertex_history_quadrature_unit_test)

  add_test(
      NAME athena.z4c_vc_output
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_output_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_output.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_output_test)
  set_tests_properties(
      athena.z4c_vc_output PROPERTIES TIMEOUT 90 LABELS "z4c;vertex;output")
  add_test(
      NAME athena.z4c_vc_output_2d_cartoon
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_output_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_output.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_output_2d_cartoon)
  set_tests_properties(
      athena.z4c_vc_output_2d_cartoon
      PROPERTIES TIMEOUT 90 LABELS "z4c;vertex;output;2d;cartoon")
  add_test(
      NAME athena.z4c_vc_output_3d_cartesian
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vertex_output_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_output_3d_cartesian.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_output_3d_cartesian
      --dimensions 3 --active-n1 9 --active-n2 9 --active-n3 9
      --multiblock-nx1 16)
  set_tests_properties(
      athena.z4c_vc_output_3d_cartesian
      PROPERTIES TIMEOUT 120 LABELS "z4c;vertex;output;3d;cartesian")

  add_executable(
      athena_z4c_vertex_to_cell_unit_test
      tst/unit/z4c/z4c_vertex_to_cell_test.cpp)
  target_include_directories(
      athena_z4c_vertex_to_cell_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_vertex_to_cell_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_vc_to_cc_adm
      COMMAND athena_z4c_vertex_to_cell_unit_test)

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
      athena_z4c_sommerfeld_derivatives_unit_test
      tst/unit/z4c/sommerfeld_derivatives_test.cpp)
  target_include_directories(
      athena_z4c_sommerfeld_derivatives_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_sommerfeld_derivatives_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_sommerfeld_derivatives
      COMMAND athena_z4c_sommerfeld_derivatives_unit_test)

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
  if (Athena_ENABLE_IRISK_INTERPOLATOR)
    add_test(
        NAME athena.z4c_vc_brill_direct_init
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vc_brill_direct_init_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_brill_direct_fixed.athinput
        --coefficients
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/investigations/brill_figure3_five_run_domain64_20260815/evidence/domain64/bundle/brill_global_48x32.coefficients
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_vc_brill_direct_init_test)
    set_tests_properties(
        athena.z4c_vc_brill_direct_init
        PROPERTIES TIMEOUT 120 LABELS "z4c;vertex;cartoon;brill;initialization")
  endif()

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
      NAME athena.z4c_vc_kerr_puncture_rejection
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_vc_kerr_puncture_rejection_test.py
      --athena $<TARGET_FILE:athena>
      --input
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_vc_kerr_half_plane_amr.athinput
      --work-dir
      ${CMAKE_CURRENT_BINARY_DIR}/athena.z4c_vc_kerr_puncture_rejection)
  set_tests_properties(
      athena.z4c_vc_kerr_puncture_rejection
      PROPERTIES TIMEOUT 30 LABELS "z4c;vertex;cartoon;kerr;negative")

  # The frozen one-cycle Kerr half-plane case is the authoritative CC selector
  # contract.  Keep this exact byte/payload comparison on deterministic GNU
  # host builds without MPI startup; accelerator arithmetic has its own gates.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
     NOT Athena_SINGLE_PRECISION AND NOT Athena_ENABLE_MPI AND
     NOT Kokkos_ENABLE_CUDA AND NOT Kokkos_ENABLE_HIP AND
     NOT Kokkos_ENABLE_SYCL)
    add_test(
        NAME athena.z4c_cc_selector_equivalence
        COMMAND
        ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_cc_selector_equivalence_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/z4c_kerr_half_plane_init.athinput
        --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/z4c_cc_selector_equivalence_test)
    set_tests_properties(
        athena.z4c_cc_selector_equivalence
        PROPERTIES TIMEOUT 120 LABELS "z4c;cell;regression;host-exact-regression")
  endif()

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
      athena_z4c_cartoon_amr_transfer_qualification_unit_test
      tst/unit/z4c/z4c_cartoon_amr_transfer_qualification_test.cpp)
  target_include_directories(
      athena_z4c_cartoon_amr_transfer_qualification_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_amr_transfer_qualification_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_cartoon_amr_transfer_qualification
      COMMAND athena_z4c_cartoon_amr_transfer_qualification_unit_test)

  add_executable(
      athena_z4c_cartoon_lattice_index_unit_test
      tst/unit/z4c/z4c_cartoon_lattice_index_test.cpp)
  target_include_directories(
      athena_z4c_cartoon_lattice_index_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_cartoon_lattice_index_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_cartoon_lattice_index
      COMMAND athena_z4c_cartoon_lattice_index_unit_test)
  add_executable(
      athena_z4c_amr_radius_unit_test
      tst/unit/z4c/z4c_amr_radius_test.cpp)
  target_include_directories(
      athena_z4c_amr_radius_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_z4c_amr_radius_unit_test PRIVATE Kokkos::kokkos)
  add_test(NAME athena.z4c_amr_radius COMMAND athena_z4c_amr_radius_unit_test)
  add_executable(
      athena_z4c_coarse_cache_ownership_unit_test
      tst/unit/z4c/z4c_coarse_cache_ownership_test.cpp)
  target_include_directories(
      athena_z4c_coarse_cache_ownership_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_coarse_cache_ownership_unit_test PRIVATE Kokkos::kokkos)
  if (ENABLE_MPI)
    target_link_libraries(
        athena_z4c_coarse_cache_ownership_unit_test PRIVATE MPI::MPI_CXX)
  endif()
  add_test(
      NAME athena.z4c_coarse_cache_ownership
      COMMAND athena_z4c_coarse_cache_ownership_unit_test)
  add_executable(
      athena_z4c_chi_preupdate_diagnostic_unit_test
      tst/unit/z4c/chi_preupdate_diagnostic_test.cpp)
  target_include_directories(
      athena_z4c_chi_preupdate_diagnostic_unit_test
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_chi_preupdate_diagnostic_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_chi_preupdate_diagnostic
      COMMAND athena_z4c_chi_preupdate_diagnostic_unit_test)
  add_test(
      NAME athena.z4c_chi_preupdate_diagnostic_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/chi_preupdate_diagnostic_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(
      athena_z4c_state_admissibility_unit_test
      tst/unit/z4c/state_admissibility_test.cpp)
  target_include_directories(
      athena_z4c_state_admissibility_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_state_admissibility_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_state_admissibility
      COMMAND athena_z4c_state_admissibility_unit_test)
  add_test(
      NAME athena.z4c_state_admissibility_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/state_admissibility_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(
      athena_z4c_timestep_contract_unit_test
      tst/unit/z4c/timestep_contract_test.cpp)
  target_include_directories(
      athena_z4c_timestep_contract_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_timestep_contract_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_timestep_contract
      COMMAND athena_z4c_timestep_contract_unit_test)
  add_test(
      NAME athena.z4c_timestep_contract_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/timestep_contract_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.z4c_shift_control_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_shift_control_static_test.py)
  if (ENABLE_MPI)
    add_test(
        NAME athena.z4c_coarse_cache_ownership_mpi2
        COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
        ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:athena_z4c_coarse_cache_ownership_unit_test>
        ${MPIEXEC_POSTFLAGS})
  endif()
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
      NAME athena.z4c_o4_symmetric_contract_static
      COMMAND
      ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_o4_symmetric_contract_static_test.py
      --source-dir ${CMAKE_CURRENT_SOURCE_DIR})
  add_executable(
      athena_z4c_amr_shadow_sensor_unit_test
      tst/unit/z4c/z4c_amr_shadow_sensor_test.cpp)
  target_include_directories(
      athena_z4c_amr_shadow_sensor_unit_test
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_amr_shadow_sensor_unit_test PRIVATE Kokkos::kokkos)
  add_test(
      NAME athena.z4c_amr_shadow_sensor
      COMMAND athena_z4c_amr_shadow_sensor_unit_test)
  add_executable(
      athena_z4c_amr_jump_diagnostic_unit_test
      tst/unit/z4c/amr_jump_diagnostic_test.cpp
      src/globals.cpp
      src/parameter_input.cpp
      src/outputs/io_wrapper.cpp)
  target_include_directories(
      athena_z4c_amr_jump_diagnostic_unit_test
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
      athena_z4c_amr_jump_diagnostic_unit_test PRIVATE Kokkos::kokkos)
  if (ENABLE_MPI)
    target_link_libraries(
        athena_z4c_amr_jump_diagnostic_unit_test PRIVATE MPI::MPI_CXX)
  endif()
  add_test(
      NAME athena.z4c_amr_jump_diagnostic
      COMMAND athena_z4c_amr_jump_diagnostic_unit_test)
  add_executable(
      athena_amr_history_format_unit_test
      tst/unit/mesh/amr_history_format_test.cpp
      src/mesh/amr_history_format.cpp)
  target_include_directories(
      athena_amr_history_format_unit_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  add_test(NAME athena.amr_history_format COMMAND athena_amr_history_format_unit_test)
  add_executable(
      athena_amr_cadence_unit_test
      tst/unit/mesh/amr_cadence_test.cpp)
  target_include_directories(
      athena_amr_cadence_unit_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(athena_amr_cadence_unit_test PRIVATE Kokkos::kokkos)
  add_test(NAME athena.amr_cadence COMMAND athena_amr_cadence_unit_test)
  add_test(
      NAME athena.z4c_amr_configuration_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_amr_configuration_static_test.py
      --source-root ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.amr_history_extension_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/mesh/amr_history_extension_static_test.py
      --source-root ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.amr_history_shadow_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/mesh/amr_history_shadow_static_test.py
      --source-root ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.amr_history_integration
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/mesh/amr_history_integration_test.py
      --athena $<TARGET_FILE:athena>
      --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/amr_history_lwave.athinput
      --work-dir ${CMAKE_CURRENT_BINARY_DIR}/amr_history_integration_test)
  set_tests_properties(athena.amr_history_integration PROPERTIES
      TIMEOUT 600 LABELS "amr;history;serial")
  add_test(
      NAME athena.z4c_amr_derefine_factor_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_amr_derefine_factor_static_test.py
      --source-root ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
      NAME athena.z4c_history_deterministic_reduction_static
      COMMAND ${PYTHON_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/z4c/z4c_history_deterministic_reduction_static_test.py
      --source-root ${CMAKE_CURRENT_SOURCE_DIR})
  if (Athena_ENABLE_MPI)
    add_test(
        NAME athena.amr_history_mpi2
        COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tst/unit/mesh/amr_history_integration_test.py
        --athena $<TARGET_FILE:athena>
        --input ${CMAKE_CURRENT_SOURCE_DIR}/tst/inputs/amr_history_lwave.athinput
        --work-dir ${CMAKE_CURRENT_BINARY_DIR}/amr_history_mpi2_test
        --mpi-only --mpiexec ${MPIEXEC_EXECUTABLE}
        --np-flag=${MPIEXEC_NUMPROC_FLAG})
    set_tests_properties(athena.amr_history_mpi2 PROPERTIES
        TIMEOUT 180 LABELS "amr;history;mpi")
  endif()
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
