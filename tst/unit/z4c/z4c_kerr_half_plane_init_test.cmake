if(NOT DEFINED ATHENA OR NOT DEFINED INPUT OR NOT DEFINED TEST_DIR OR
   NOT DEFINED PYTHON OR NOT DEFINED HISTORY_VALIDATOR OR
   NOT DEFINED FAILURE_REGION_EXTRACTOR OR NOT DEFINED SOURCE_DIR)
  message(FATAL_ERROR
    "ATHENA, INPUT, TEST_DIR, PYTHON, HISTORY_VALIDATOR, FAILURE_REGION_EXTRACTOR, and SOURCE_DIR are required")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
execute_process(
  COMMAND "${ATHENA}" -i "${INPUT}"
  WORKING_DIRECTORY "${TEST_DIR}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "half-plane Kerr initialization failed (${result})\nstdout:\n${stdout}\nstderr:\n${stderr}")
endif()
if(NOT stdout MATCHES "Initialized arXiv:1001.4077 Kerr puncture")
  message(FATAL_ERROR "half-plane Kerr initialization did not reach the pgen contract")
endif()

set(history "${TEST_DIR}/z4c_kerr_half_plane_init.z4c.user.hst")
if(NOT EXISTS "${history}")
  message(FATAL_ERROR "half-plane Kerr initialization omitted its history evidence")
endif()
file(READ "${history}" history_text)
string(TOLOWER "${history_text}" history_lower)
if(history_lower MATCHES "(^|[^a-z])(nan|[+-]?inf)([^a-z]|$)")
  message(FATAL_ERROR "half-plane Kerr initial constraints contain non-finite values")
endif()
string(REGEX MATCHALL "\n[ ]*0\\.00000e\\+00[^\n]*" history_rows "${history_text}")
list(LENGTH history_rows row_count)
if(row_count LESS 1)
  message(FATAL_ERROR "half-plane Kerr initialization emitted no t=0 history row")
endif()
if(NOT history_text MATCHES "\n[ ]*1\\.00000e-04[^\n]*1\\.00000e\\+00")
  message(FATAL_ERROR "half-plane Kerr regression did not complete one RK cycle")
endif()
execute_process(
  COMMAND "${PYTHON}" "${HISTORY_VALIDATOR}" --history "${history}"
  RESULT_VARIABLE history_result
  OUTPUT_VARIABLE history_stdout
  ERROR_VARIABLE history_stderr)
if(NOT history_result EQUAL 0)
  message(FATAL_ERROR
    "half-plane history contract failed (${history_result})\n"
    "stdout:\n${history_stdout}\nstderr:\n${history_stderr}")
endif()

file(GLOB_RECURSE z4c_files
  "${TEST_DIR}/bin/*z4c_kerr_half_plane_init.z4c.00001.bin")
file(GLOB_RECURSE constraint_files
  "${TEST_DIR}/bin/*z4c_kerr_half_plane_init.con.00001.bin")
list(LENGTH z4c_files z4c_count)
list(LENGTH constraint_files constraint_count)
if(z4c_count LESS 1 OR NOT z4c_count EQUAL constraint_count)
  message(FATAL_ERROR
    "half-plane Kerr diagnostic carrier inventory is incomplete: z4c=${z4c_count}, con=${constraint_count}")
endif()
execute_process(
  COMMAND "${PYTHON}" "${FAILURE_REGION_EXTRACTOR}"
          --source-dir "${SOURCE_DIR}"
          --z4c ${z4c_files}
          --constraints ${constraint_files}
          --output "${TEST_DIR}/former_failure_region.json"
  RESULT_VARIABLE diagnostic_result
  OUTPUT_VARIABLE diagnostic_stdout
  ERROR_VARIABLE diagnostic_stderr)
if(NOT diagnostic_result EQUAL 0)
  message(FATAL_ERROR
    "half-plane former-failure-region extraction failed (${diagnostic_result})\n"
    "stdout:\n${diagnostic_stdout}\nstderr:\n${diagnostic_stderr}")
endif()

message(STATUS
  "half-plane Kerr initialization, RK parity tasks, diagnostic history, and failure-region extraction passed")
