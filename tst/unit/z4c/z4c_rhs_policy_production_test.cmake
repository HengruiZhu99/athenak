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
  set(case_dir "${TEST_DIR}/order${stencil}")
  file(MAKE_DIRECTORY "${case_dir}")
  string(REPLACE "nghost = 2" "nghost = ${nghost}" input_text "${base_input}")
  string(REPLACE "spatial_order = 2" "spatial_order = ${stencil}"
                 input_text "${input_text}")
  string(REPLACE "basename = z4c_rhs_policy"
                 "basename = z4c_rhs_policy_o${stencil}" input_text "${input_text}")
  if(stencil EQUAL 6)
    string(REPLACE "spatial_order = 6"
                   "spatial_order = 6\nhistory_kretschmann = true"
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
  if(stencil EQUAL 6)
    string(FIND "${history_text}" "maxAbsKret" kretschmann_found)
    if(kretschmann_found EQUAL -1)
      message(FATAL_ERROR "sixth-order history did not exercise curvature reduction")
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
          "sixth-order baseline lifecycle sentinel changed: expected t=0 "
          "maxAbsKret=inf before ADM initialization, got '${initial_kretschmann}'")
    endif()
    message(STATUS
        "order 6 lifecycle sentinel: t=0 maxAbsKret=inf (pre-ADM initialization); "
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
endforeach()

message(STATUS "Cartesian production RHS/constraints/Sbc/diagnostics/Weyl orders passed")
