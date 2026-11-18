set(_native_launcher)
if(Athena_ENABLE_MPI)
  set(_native_launcher --launcher "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} {ranks}")
  set(_native_ranks 2)
else()
  set(_native_ranks 1)
endif()
add_test(NAME multirate_native_qualification COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/qualify.py --binary $<TARGET_FILE:athena>
  --output ${CMAKE_CURRENT_BINARY_DIR}/results --ranks ${_native_ranks}
  ${_native_launcher})
set_tests_properties(multirate_native_qualification PROPERTIES TIMEOUT 1800
  LABELS "multirate;native;qualification" RUN_SERIAL TRUE)

set(_adaptive_balance)
if(NOT Athena_ENABLE_METIS)
  set(_adaptive_balance --fixed-partition)
endif()
add_test(NAME multirate_adaptive_endpoint COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/adaptive.py --binary $<TARGET_FILE:athena>
  --output ${CMAKE_CURRENT_BINARY_DIR}/adaptive --ranks ${_native_ranks}
  ${_native_launcher} ${_adaptive_balance})
set_tests_properties(multirate_adaptive_endpoint PROPERTIES TIMEOUT 1800
  LABELS "multirate;native;qualification" RUN_SERIAL TRUE)
