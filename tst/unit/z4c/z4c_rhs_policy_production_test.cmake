# Execute the real AthenaK task graph for each supported Cartesian stencil.  This
# complements the collapsed SO(2) manufactured-policy oracle while the universal
# Cartoon problem-generator gate remains intentionally closed.

if(NOT DEFINED ATHENA OR NOT DEFINED SOURCE_DIR OR NOT DEFINED TEST_DIR)
  message(FATAL_ERROR "ATHENA, SOURCE_DIR, and TEST_DIR are required")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
file(READ "${SOURCE_DIR}/tst/inputs/z4c_rhs_policy.athinput" base_input)

foreach(stencil 2 4 6)
  math(EXPR nghost "${stencil} / 2 + 1")
  if(stencil EQUAL 4)
    # Production O4 deliberately retains a fourth allocated/communicated
    # ghost layer as buffer headroom while fd_stencil remains three.
    set(nghost 4)
  endif()
  set(case_dir "${TEST_DIR}/order${stencil}")
  file(MAKE_DIRECTORY "${case_dir}")
  string(REPLACE "nghost = 2" "nghost = ${nghost}" input_text "${base_input}")
  string(REPLACE "spatial_order = 2" "spatial_order = ${stencil}"
                 input_text "${input_text}")
  string(REPLACE "basename = z4c_rhs_policy"
                 "basename = z4c_rhs_policy_o${stencil}" input_text "${input_text}")
  if(stencil EQUAL 4 OR stencil EQUAL 6)
    string(REPLACE "spatial_order = ${stencil}"
                   "spatial_order = ${stencil}\nhistory_kretschmann = true"
                   input_text "${input_text}")
  endif()
  set(input_file "${case_dir}/input.athinput")
  file(WRITE "${input_file}" "${input_text}")
  execute_process(
      COMMAND "${ATHENA}" -i "${input_file}"
      WORKING_DIRECTORY "${case_dir}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE standard_output
      ERROR_VARIABLE standard_error
      TIMEOUT 60)
  set(output "${standard_output}\n${standard_error}")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "order ${stencil} production path failed (${result}):\n${output}")
  endif()
  foreach(marker "Setup complete, executing task list(s)" "cycle=1")
    string(FIND "${output}" "${marker}" marker_found)
    if(marker_found EQUAL -1)
      message(FATAL_ERROR "order ${stencil} did not reach '${marker}':\n${output}")
    endif()
  endforeach()
  file(GLOB histories "${case_dir}/z4c_rhs_policy_o${stencil}*.hst")
  file(GLOB diagnostics
       "${case_dir}/bin/z4c_rhs_policy_o${stencil}.diagnostics.*.bin")
  file(GLOB states
       "${case_dir}/bin/z4c_rhs_policy_o${stencil}.state.*.bin")
  file(GLOB waveforms "${case_dir}/waveforms/*")
  if(NOT histories OR NOT diagnostics OR NOT states OR NOT waveforms)
    message(FATAL_ERROR
        "order ${stencil} missed history/derived/Weyl production artifacts: "
        "history='${histories}' diagnostics='${diagnostics}' states='${states}' "
        "waveforms='${waveforms}'")
  endif()
  foreach(path IN LISTS histories diagnostics states waveforms)
    file(SIZE "${path}" artifact_size)
    if(artifact_size EQUAL 0)
      message(FATAL_ERROR "order ${stencil} produced empty artifact ${path}")
    endif()
  endforeach()
  file(READ "${histories}" history_text)
  file(STRINGS "${histories}" history_rows REGEX "^[ ]*[0-9+-]")
  list(GET history_rows -1 final_history_row)
  string(TOLOWER "${final_history_row}" final_history_lower)
  if(final_history_lower MATCHES "(^|[^a-z])(nan|inf)([^a-z]|$)")
    message(FATAL_ERROR "order ${stencil} final history contains a nonfinite value")
  endif()
  if(stencil EQUAL 4 OR stencil EQUAL 6)
    string(FIND "${history_text}" "maxAbsKret" kretschmann_found)
    if(kretschmann_found EQUAL -1)
      message(FATAL_ERROR "order ${stencil} history did not exercise curvature reduction")
    endif()
    # History is emitted once before ADM scratch has been initialized.  The exact
    # pre-migration binary therefore also records maxAbsKret=inf at t=0.  Make the
    # known lifecycle seam visible instead of accidentally claiming every row is
    # finite; all post-initialization rows remain covered by the final-row check.
    list(GET history_rows 0 initial_history_row)
    separate_arguments(initial_fields UNIX_COMMAND "${initial_history_row}")
    list(GET initial_fields 13 initial_kretschmann)
    if(NOT initial_kretschmann STREQUAL "inf")
      message(FATAL_ERROR
          "order ${stencil} baseline lifecycle sentinel changed: expected t=0 "
          "maxAbsKret=inf before ADM initialization, got '${initial_kretschmann}'")
    endif()
    message(STATUS
        "order ${stencil} lifecycle sentinel: t=0 maxAbsKret=inf (pre-ADM initialization); "
        "post-initialization history is finite")
  endif()
  list(GET states -1 final_state)
  list(GET diagnostics -1 final_diagnostics)
  list(GET waveforms 0 first_waveform)
  file(SHA256 "${final_state}" state_sha256)
  file(SHA256 "${final_diagnostics}" diagnostic_sha256)
  file(SHA256 "${first_waveform}" waveform_sha256)
  file(SHA256 "${histories}" history_sha256)
  message(STATUS
      "order ${stencil} numeric fingerprints: state=${state_sha256} "
      "diagnostic=${diagnostic_sha256} history=${history_sha256} "
      "waveform=${waveform_sha256}")

  # Reproduce the reviewer's exact-base comparison deck as a separate run.  Its
  # stable basename is part of the binary/text bytes, and the low-order derived
  # output is deliberately absent because the base implementation reads outside
  # its valid stencil at orders 2 and 4.  This makes state/history/Weyl raw-file
  # equality a real gate instead of weakening it to decoded-value equivalence.
  set(base_case_dir "${TEST_DIR}/exact_base_order${stencil}")
  file(MAKE_DIRECTORY "${base_case_dir}")
  string(REPLACE "basename = z4c_rhs_policy_o${stencil}"
                 "basename = cmp_o${stencil}" base_comparison_input "${input_text}")
  string(REGEX REPLACE
      "<output2>\nfile_type = bin\nvariable = z4c_diag\ndcycle = 1\nid = diagnostics\n"
      "" base_comparison_input "${base_comparison_input}")
  # The exact-base state/history/Weyl comparison did not enable the curvature
  # history column; keep the independent order-6 diagnostic run above unchanged.
  string(REPLACE "history_kretschmann = true\n" ""
                 base_comparison_input "${base_comparison_input}")
  set(base_comparison_file "${base_case_dir}/input.athinput")
  file(WRITE "${base_comparison_file}" "${base_comparison_input}")
  execute_process(
      COMMAND "${ATHENA}" -i "${base_comparison_file}"
      WORKING_DIRECTORY "${base_case_dir}"
      RESULT_VARIABLE base_comparison_result
      OUTPUT_VARIABLE base_comparison_stdout
      ERROR_VARIABLE base_comparison_stderr
      TIMEOUT 60)
  if(NOT base_comparison_result EQUAL 0)
    message(FATAL_ERROR
        "order ${stencil} exact-base comparison run failed "
        "(${base_comparison_result}):\n${base_comparison_stdout}\n"
        "${base_comparison_stderr}")
  endif()

  if(stencil EQUAL 6)
    # The sixth-order base diagnostic is valid and was byte-identical in the
    # independent base-vs-tip review.  Re-run with that review's exact basename
    # so serialization bytes, not only decoded values, are frozen.
    set(base_diagnostic_dir "${TEST_DIR}/exact_base_diagnostic_order6")
    file(MAKE_DIRECTORY "${base_diagnostic_dir}")
    string(REPLACE "basename = z4c_rhs_policy_o6"
                   "basename = cmp_diag_o6" base_diagnostic_input "${input_text}")
    set(base_diagnostic_file "${base_diagnostic_dir}/input.athinput")
    file(WRITE "${base_diagnostic_file}" "${base_diagnostic_input}")
    execute_process(
        COMMAND "${ATHENA}" -i "${base_diagnostic_file}"
        WORKING_DIRECTORY "${base_diagnostic_dir}"
        RESULT_VARIABLE base_diagnostic_result
        OUTPUT_VARIABLE base_diagnostic_stdout
        ERROR_VARIABLE base_diagnostic_stderr
        TIMEOUT 60)
    if(NOT base_diagnostic_result EQUAL 0)
      message(FATAL_ERROR
          "order 6 exact-base diagnostic run failed (${base_diagnostic_result}):\n"
          "${base_diagnostic_stdout}\n${base_diagnostic_stderr}")
    endif()
  endif()
endforeach()

execute_process(
    COMMAND "${PYTHON_EXECUTABLE}"
            "${SOURCE_DIR}/tst/unit/z4c/z4c_rhs_policy_regression.py"
            --athena "${ATHENA}"
            --source-dir "${SOURCE_DIR}"
            --test-dir "${TEST_DIR}"
    RESULT_VARIABLE regression_result
    OUTPUT_VARIABLE regression_output
    ERROR_VARIABLE regression_error)
if(NOT regression_result EQUAL 0)
  message(FATAL_ERROR
      "Cartesian production numeric regression failed (${regression_result}):\n"
      "${regression_output}\n${regression_error}")
endif()
message(STATUS "${regression_output}")

message(STATUS "Cartesian production RHS/constraints/Sbc/diagnostics/Weyl orders passed")
