add_executable(multirate_local_ownership_tests local_block_ownership.cpp)
target_include_directories(multirate_local_ownership_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_options(multirate_local_ownership_tests PRIVATE -UNDEBUG)
add_test(NAME multirate_local_ownership COMMAND multirate_local_ownership_tests)

add_executable(multirate_mapped_ownership_tests mapped_ownership.cpp
    ${CMAKE_SOURCE_DIR}/src/bvals/mapped_exchange.cpp)
target_include_directories(multirate_mapped_ownership_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(multirate_mapped_ownership_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(multirate_mapped_ownership_tests PRIVATE MPI::MPI_CXX)
  foreach(ranks 2 3)
    add_test(NAME multirate_mapped_ownership_${ranks} COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:multirate_mapped_ownership_tests> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(multirate_mapped_ownership_${ranks} PROPERTIES TIMEOUT 60)
  endforeach()
endif()
add_test(NAME multirate_mapped_ownership COMMAND multirate_mapped_ownership_tests)
set_tests_properties(multirate_mapped_ownership PROPERTIES TIMEOUT 60)

add_executable(multirate_owner_header_tests parameter_header.cpp
    ${CMAKE_SOURCE_DIR}/src/parameter_input.cpp
    ${CMAKE_SOURCE_DIR}/src/globals.cpp
    ${CMAKE_SOURCE_DIR}/src/outputs/io_wrapper.cpp)
target_include_directories(multirate_owner_header_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(multirate_owner_header_tests PRIVATE Kokkos::kokkos)
if(Athena_ENABLE_MPI)
  target_link_libraries(multirate_owner_header_tests PRIVATE MPI::MPI_CXX)
endif()
add_test(NAME multirate_owner_header COMMAND multirate_owner_header_tests)

# The production executable currently owns its source objects. Reuse those
# objects for a native lifecycle qualification rather than a second physics build.
# Other generators still run the map, MPI exchange, and header tests above.
if(CMAKE_GENERATOR MATCHES "Makefiles")
  set(_migration_launcher)
  set(_migration_ranks 1)
  if(Athena_ENABLE_MPI)
    set(_migration_launcher --launcher
        "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} {ranks}")
    set(_migration_ranks 2 3)
  endif()
  foreach(ranks IN LISTS _migration_ranks)
    add_test(NAME multirate_accepted_migration_${ranks} COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/qualify_migration.py
        --build ${CMAKE_BINARY_DIR} --cxx ${CMAKE_CXX_COMPILER}
        --fixture ${CMAKE_SOURCE_DIR}/tst/multirate_native/small.athinput
        --ranks ${ranks} ${_migration_launcher})
    set_tests_properties(multirate_accepted_migration_${ranks} PROPERTIES
        TIMEOUT 900 RUN_SERIAL TRUE LABELS "multirate;native;qualification")
  endforeach()
endif()
