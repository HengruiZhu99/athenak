# Process-level regression coverage for the production
# ParameterInput -> MeshBlockPack::AddPhysics preallocation seam.

if(NOT DEFINED ATHENA OR NOT DEFINED SOURCE_DIR OR NOT DEFINED TEST_DIR)
  message(FATAL_ERROR "ATHENA, SOURCE_DIR, and TEST_DIR are required")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

function(run_input case_name input_text expected_success expected_diagnostic)
  set(input_file "${TEST_DIR}/${case_name}.athinput")
  file(WRITE "${input_file}" "${input_text}")
  execute_process(
      COMMAND "${ATHENA}" -i "${input_file}"
      WORKING_DIRECTORY "${TEST_DIR}"
      RESULT_VARIABLE result
      OUTPUT_VARIABLE standard_output
      ERROR_VARIABLE standard_error
      TIMEOUT 15)
  set(output "${standard_output}\n${standard_error}")

  if(expected_success)
    if(NOT "${result}" STREQUAL "0")
      message(FATAL_ERROR "${case_name}: expected success, got ${result}:\n${output}")
    endif()
    string(FIND "${output}" "AssembleZ4cTasks" allocation_reached)
    if(allocation_reached EQUAL -1)
      message(FATAL_ERROR "${case_name}: Z4c allocation/task assembly was not reached")
    endif()
    string(FIND "${output}" "Z4c preallocation validation failed" prealloc_failure)
    if(NOT prealloc_failure EQUAL -1)
      message(FATAL_ERROR "${case_name}: unexpected preallocation failure:\n${output}")
    endif()
  else()
    if("${result}" STREQUAL "0")
      message(FATAL_ERROR "${case_name}: expected a preallocation rejection")
    endif()
    set(full_diagnostic
        "Z4c preallocation validation failed: ${expected_diagnostic}")
    string(FIND "${output}" "${full_diagnostic}" diagnostic_found)
    if(diagnostic_found EQUAL -1)
      message(FATAL_ERROR
          "${case_name}: expected diagnostic '${full_diagnostic}', got:\n${output}")
    endif()

    # These strings occur only after the first statement of AddPhysics has returned.
    # Their absence freezes failure ordering before Z4c allocation/task assembly,
    # problem-generator dispatch, driver/output construction, or execution.
    foreach(forbidden
            "AssembleZ4cTasks"
            "telegraph_tau must be positive"
            "Problem generator name could not be found"
            "Setup complete, executing task list(s)")
      string(FIND "${output}" "${forbidden}" forbidden_found)
      if(NOT forbidden_found EQUAL -1)
        message(FATAL_ERROR
            "${case_name}: later constructor side effect '${forbidden}' was observed")
      endif()
    endforeach()
    file(GLOB output_side_effects "${TEST_DIR}/prealloc_${case_name}*")
    if(output_side_effects)
      message(FATAL_ERROR
          "${case_name}: output construction left side effects: ${output_side_effects}")
    endif()
  endif()
endfunction()

file(READ "${SOURCE_DIR}/tst/inputs/z4c_cartesian_preallocation.athinput"
     cartesian_base)

function(run_cartesian case_name nghost requested_order)
  set(input_text "${cartesian_base}")
  string(REPLACE "nghost = 2" "nghost = ${nghost}" input_text "${input_text}")
  if("${requested_order}" STREQUAL "absent")
    string(REPLACE "spatial_order = -1" "" input_text "${input_text}")
  else()
    string(REPLACE "spatial_order = -1" "spatial_order = ${requested_order}"
                   input_text "${input_text}")
  endif()
  string(REPLACE "basename = z4c_preallocation"
                 "basename = prealloc_${case_name}" input_text "${input_text}")
  run_input("${case_name}" "${input_text}" TRUE "")
endfunction()

function(run_cartesian_reject case_name nghost requested_order expected_diagnostic)
  set(input_text "${cartesian_base}")
  string(REPLACE "nghost = 2" "nghost = ${nghost}" input_text "${input_text}")
  string(REPLACE "spatial_order = -1" "spatial_order = ${requested_order}"
                 input_text "${input_text}")
  string(REPLACE "basename = z4c_preallocation"
                 "basename = prealloc_${case_name}" input_text "${input_text}")
  run_input("${case_name}" "${input_text}" FALSE "${expected_diagnostic}")
endfunction()

# Preserve the default Cartesian requested<=0 convention on the production path.
foreach(nghost 2 3 4)
  run_cartesian("cart_absent_ng${nghost}" "${nghost}" absent)
  run_cartesian("cart_zero_ng${nghost}" "${nghost}" 0)
  run_cartesian("cart_negative_ng${nghost}" "${nghost}" -1)
endforeach()
run_cartesian(cart_explicit_2 2 2)
run_cartesian(cart_explicit_4 3 4)
run_cartesian(cart_explicit_6 4 6)

# Construct real one- and two-axis PDFs through the production output factory on
# collapsed Cartesian storage.  Directory creation is a constructor sentinel;
# the nbin2=1 case freezes the shared second-axis predicate and staging count.
foreach(pdf_dimension 1 2)
  set(input_text "${cartesian_base}")
  string(REPLACE "basename = z4c_preallocation"
                 "basename = prealloc_cart_pdf_${pdf_dimension}d"
                 input_text "${input_text}")
  set(pdf_block
      "\n<output1>\nfile_type = pdf\ndcycle = 1\nid = production_${pdf_dimension}d\nvariable = z4c_chi\nnbin = 4\nbin_min = 0.1\nbin_max = 1.1\nlogscale = false")
  if(pdf_dimension EQUAL 2)
    string(APPEND pdf_block
           "\nvariable_2 = z4c_alpha\nnbin2 = 1\nbin2_min = 0.1\nbin2_max = 1.1\nlogscale2 = false")
  endif()
  string(APPEND input_text "${pdf_block}\n")
  run_input("cart_pdf_${pdf_dimension}d" "${input_text}" TRUE "")
  set(pdf_directory "${TEST_DIR}/pdf_production_${pdf_dimension}d")
  if(pdf_dimension EQUAL 2)
    string(APPEND pdf_directory "_z4c_alpha")
  endif()
  if(NOT IS_DIRECTORY "${pdf_directory}")
    message(FATAL_ERROR
        "cart_pdf_${pdf_dimension}d: production PDF constructor was not reached")
  endif()
  if(NOT EXISTS
     "${pdf_directory}/prealloc_cart_pdf_${pdf_dimension}d.bins.pdf")
    message(FATAL_ERROR
        "cart_pdf_${pdf_dimension}d: collapsed PDF staging/write was not reached")
  endif()
endforeach()

run_cartesian_reject(
    cart_invalid_positive 4 8 "<z4c>/spatial_order must be 2, 4, or 6")
run_cartesian_reject(
    cart_insufficient_ghosts 2 6
    "effective <z4c>/spatial_order=6 requires at least 4 ghost cells, but <mesh>/nghost=2")

# The shipped deck must pass preallocation even though this built-in-pgen executable
# intentionally stops later because it does not contain the custom one-puncture pgen.
execute_process(
    COMMAND "${ATHENA}" -i
            "${SOURCE_DIR}/inputs/z4c/onepuncture/z4c_onepuncture.athinput"
    WORKING_DIRECTORY "${TEST_DIR}"
    RESULT_VARIABLE shipped_result
    OUTPUT_VARIABLE shipped_stdout
    ERROR_VARIABLE shipped_stderr
    TIMEOUT 15)
set(shipped_output "${shipped_stdout}\n${shipped_stderr}")
string(FIND "${shipped_output}" "AssembleZ4cTasks" shipped_allocation)
string(FIND "${shipped_output}" "Z4c preallocation validation failed" shipped_reject)
string(FIND "${shipped_output}" "Problem generator name could not be found" shipped_pgen)
if(shipped_result EQUAL 0 OR shipped_allocation EQUAL -1 OR
   NOT shipped_reject EQUAL -1 OR shipped_pgen EQUAL -1)
  message(FATAL_ERROR "shipped one-puncture compatibility path failed:\n${shipped_output}")
endif()

set(cartoon_template [=[
<job>
basename = prealloc_@CASE_NAME@

<mesh>
nghost = 2
nx1 = 8
x1min = -1.0
x1max = 1.0
ix1_bc = outflow
ox1_bc = outflow
nx2 = 4
x2min = -0.5
x2max = 0.5
ix2_bc = outflow
ox2_bc = outflow
nx3 = @NX3@
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = 4
nx2 = 4
nx3 = @NX3@

<coord>
minkowski = true

<time>
evolution = dynamic
integrator = rk2
cfl_number = 0.1
nlim = 0
tlim = 0.0
ndiag = 1

<z4c>
symmetry = cartoon_so2
coordinate_map = signed_rho_z_suppressed_y_v1
symmetry_schema = 1
spatial_order = -1
telegraph_tau = -1
@Z4C_EXTRA@

<problem>
pgen_name = constructor_side_effect_sentinel

@EXTRA_BLOCKS@
]=])

function(run_cartoon_reject case_name expected_diagnostic z4c_extra extra_blocks)
  set(CASE_NAME "${case_name}")
  set(NX3 1)
  set(Z4C_EXTRA "${z4c_extra}")
  set(EXTRA_BLOCKS "${extra_blocks}")
  string(CONFIGURE "${cartoon_template}" input_text @ONLY)
  run_input("${case_name}" "${input_text}" FALSE "${expected_diagnostic}")
endfunction()

function(run_cartoon_pdf_reject case_name expected_diagnostic pdf_parameters)
  set(pdf_block
      "<output1>\nfile_type = pdf\ndcycle = 1\nid = reject_${case_name}\n${pdf_parameters}")
  run_cartoon_reject("${case_name}" "${expected_diagnostic}" "" "${pdf_block}")
  file(GLOB pdf_side_effects "${TEST_DIR}/pdf_reject_${case_name}*")
  if(pdf_side_effects)
    message(FATAL_ERROR
        "${case_name}: rejected PDF reached allocation/output construction: ${pdf_side_effects}")
  endif()
endfunction()

# Production collection of common configuration and every incompatible physics block.
run_cartoon_reject(
    bad_mode "<z4c>/symmetry must be cartesian3d or cartoon_so2, not 'helical'"
    "symmetry = helical" "")
run_cartoon_reject(
    bad_map "cartoon_so2 requires coordinate_map=signed_rho_z_suppressed_y_v1"
    "coordinate_map = cartesian_xyz" "")
run_cartoon_reject(
    bad_schema "unsupported <z4c>/symmetry_schema for cartoon_so2"
    "symmetry_schema = 99" "")
set(CASE_NAME bad_nx3)
set(NX3 4)
set(Z4C_EXTRA "")
set(EXTRA_BLOCKS "")
string(CONFIGURE "${cartoon_template}" bad_nx3_input @ONLY)
run_input(
    bad_nx3 "${bad_nx3_input}" FALSE
    "cartoon_so2 requires mesh/nx3=meshblock/nx3=1")

foreach(physics hydro mhd ion-neutral radiation turb_driving particles)
  run_cartoon_reject(
      "physics_${physics}" "cartoon_so2 vacuum Z4c forbids <${physics}> physics"
      "" "<${physics}>\nsentinel = true")
endforeach()

# Actual optional-consumer and FastFlow parameter spellings.
run_cartoon_reject(
    tracker "cartoon_so2 does not support compact-object tracker co_0_type"
    "co_0_type = BH" "")
run_cartoon_reject(
    horizon_dump "cartoon_so2 does not support horizon dump dump_horizon_0"
    "dump_horizon_0 = true" "")
run_cartoon_reject(
    wave "cartoon_so2 does not support Z4c wave extraction"
    "nrad_wave_extraction = 1" "")
run_cartoon_reject(
    cce "cartoon_so2 does not support CCE extraction"
    "" "<cce>\nnum_radii = 1")
run_cartoon_reject(
    fastflow "cartoon_so2 does not support FastFlow before the m=0 Cartoon adapter is integrated"
    "" "<fastflow>\nnum_horizons = 1")
foreach(legacy_key
        center_x_0
        use_puncture_0
        wait_until_punc_are_close_0
        use_puncture_massweighted_center_0)
  run_cartoon_reject(
      "fastflow_${legacy_key}"
      "cartoon_so2 does not support legacy FastFlow key ${legacy_key}"
      "" "<fastflow>\n${legacy_key} = 1")
endforeach()

# Every rejected output token, the unknown fallback, PDF options, and active-state rules.
foreach(file_type cart sph cbin pvtk trk)
  run_cartoon_reject(
      "output_${file_type}"
      "cartoon_so2 rejects file_type=${file_type} in <output1> before output construction"
      "" "<output1>\nfile_type = ${file_type}\ndcycle = 1")
endforeach()
run_cartoon_reject(
    output_unknown
    "unknown file_type='mystery' in <output1>; supported Cartoon types are tab,hst,log,vtk,pdf,bin,rst"
    "" "<output1>\nfile_type = mystery\ndt = 1.0")
run_cartoon_reject(
    output_missing
    "unknown file_type='' in <output1>; supported Cartoon types are tab,hst,log,vtk,pdf,bin,rst"
    "" "<output1>\ndcycle = 1")
run_cartoon_reject(
    pdf_mass_weighted
    "<output1> PDF mass_weighted=true is unsupported for vacuum Z4c"
    "" "<output1>\nfile_type = pdf\ndcycle = 1\nvariable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nmass_weighted = true")
run_cartoon_reject(
    pdf_second_axis_without_variable
    "<output1> PDF second-axis keys require variable_2"
    "" "<output1>\nfile_type = pdf\ndcycle = 1\nvariable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nnbin2 = 2")
run_cartoon_reject(
    pdf_variable_without_nbin
    "<output1> PDF variable_2 requires explicit nbin2"
    "" "<output1>\nfile_type = pdf\ndcycle = 1\nvariable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_chi")
run_cartoon_reject(
    pdf_variable_zero_nbin
    "<output1> PDF variable_2 requires nbin2 in [1,4094]"
    "" "<output1>\nfile_type = pdf\ndcycle = 1\nvariable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_chi\nnbin2 = 0")

# Symmetric primary/second-axis validation must fail in AddPhysics before the
# pgen or PDF constructor.  Unique IDs make output side effects mechanically visible.
run_cartoon_pdf_reject(
    pdf_primary_missing_min "<output1> PDF requires bin_min"
    "variable = z4c_chi\nnbin = 8\nbin_max = 1.0")
run_cartoon_pdf_reject(
    pdf_primary_missing_max "<output1> PDF requires bin_max"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1")
run_cartoon_pdf_reject(
    pdf_second_missing_min "<output1> PDF variable_2 requires explicit bin2_min"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_max = 1.0")
run_cartoon_pdf_reject(
    pdf_second_missing_max "<output1> PDF variable_2 requires explicit bin2_max"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_min = 0.1")
run_cartoon_pdf_reject(
    pdf_primary_equal "<output1> primary axis requires bin_min < bin_max"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 0.1")
run_cartoon_pdf_reject(
    pdf_primary_reversed "<output1> primary axis requires bin_min < bin_max"
    "variable = z4c_chi\nnbin = 8\nbin_min = 1.0\nbin_max = 0.1")
run_cartoon_pdf_reject(
    pdf_second_equal "<output1> second axis requires bin_min < bin_max"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_min = 0.1\nbin2_max = 0.1")
run_cartoon_pdf_reject(
    pdf_second_reversed "<output1> second axis requires bin_min < bin_max"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_min = 1.0\nbin2_max = 0.1")
run_cartoon_pdf_reject(
    pdf_primary_negative_log "<output1> primary axis logarithmic bin_min must be positive"
    "variable = z4c_chi\nnbin = 8\nbin_min = -0.1\nbin_max = 1.0")
run_cartoon_pdf_reject(
    pdf_second_negative_log "<output1> second axis logarithmic bin_min must be positive"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_min = -0.1\nbin2_max = 1.0")
run_cartoon_pdf_reject(
    pdf_primary_nonfinite_step "<output1> primary axis bin step must be finite and positive"
    "variable = z4c_chi\nnbin = 8\nbin_min = -1.7976931348623157e308\nbin_max = 1.7976931348623157e308\nlogscale = false")
run_cartoon_pdf_reject(
    pdf_second_nonfinite_step "<output1> second axis bin step must be finite and positive"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_min = -1.7976931348623157e308\nbin2_max = 1.7976931348623157e308\nlogscale2 = false")
run_cartoon_pdf_reject(
    pdf_primary_zero_log_step "<output1> primary axis bin step must be finite and positive"
    "variable = z4c_chi\nnbin = 8\nbin_min = 1.7976931348623155e308\nbin_max = 1.7976931348623157e308")
run_cartoon_pdf_reject(
    pdf_second_zero_log_step "<output1> second axis bin step must be finite and positive"
    "variable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 8\nbin2_min = 1.7976931348623155e308\nbin2_max = 1.7976931348623157e308")
run_cartoon_reject(
    pdf_maximum_checked_without_allocation
    "problem generator 'constructor_side_effect_sentinel' has no audited cartoon_so2 adapter"
    "" "<output1>\nfile_type = pdf\ndcycle = 1\nid = max_no_alloc\nvariable = z4c_chi\nnbin = 4094\nbin_min = 0.1\nbin_max = 1.0\nvariable_2 = z4c_alpha\nnbin2 = 4094\nbin2_min = 0.1\nbin2_max = 1.0")
if(IS_DIRECTORY "${TEST_DIR}/pdf_max_no_alloc_z4c_alpha")
  message(FATAL_ERROR
      "pdf_maximum_checked_without_allocation: PDF constructor allocated storage")
endif()

# Supported and disabled outputs must survive collection and reach the universal pgen gate.
foreach(file_type tab hst log vtk pdf bin rst)
  if(file_type STREQUAL "pdf")
    set(output_options
        "<output1>\nfile_type = pdf\ndcycle = 1\nvariable = z4c_chi\nnbin = 8\nbin_min = 0.1\nbin_max = 1.0")
  else()
    set(output_options "<output1>\nfile_type = ${file_type}\ndcycle = 1")
  endif()
  run_cartoon_reject(
      "output_supported_${file_type}"
      "problem generator 'constructor_side_effect_sentinel' has no audited cartoon_so2 adapter"
      "" "${output_options}")
endforeach()
run_cartoon_reject(
    output_disabled
    "problem generator 'constructor_side_effect_sentinel' has no audited cartoon_so2 adapter"
    "" "<output1>\nfile_type = cart\ndcycle = 0\ndt = 1.0")
run_cartoon_reject(
    output_disabled_dt
    "problem generator 'constructor_side_effect_sentinel' has no audited cartoon_so2 adapter"
    "" "<output1>\nfile_type = cart\ndt = 0.0")

# Restart carrier spellings are collected now, while matching metadata remains staged at pgen.
run_cartoon_reject(
    restart_mismatch
    "restart symmetry/map/schema metadata conflicts with cartoon_so2"
    "restart_symmetry = cartesian3d\nrestart_coordinate_map = cartesian_xyz\nrestart_symmetry_schema = 1"
    "")
run_cartoon_reject(
    restart_partial
    "restart symmetry/map/schema metadata conflicts with cartoon_so2"
    "restart_symmetry = cartoon_so2" "")
run_cartoon_reject(
    restart_match
    "problem generator 'constructor_side_effect_sentinel' has no audited cartoon_so2 adapter"
    "restart_symmetry = cartoon_so2\nrestart_coordinate_map = signed_rho_z_suppressed_y_v1\nrestart_symmetry_schema = 1"
    "")
run_cartoon_reject(
    pgen_gate
    "problem generator 'constructor_side_effect_sentinel' has no audited cartoon_so2 adapter"
    "" "")

message(STATUS "Z4c production preallocation process tests passed")
