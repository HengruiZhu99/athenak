if(NOT DEFINED ATHENA OR NOT DEFINED INPUT OR NOT DEFINED TEST_DIR)
  message(FATAL_ERROR "ATHENA, INPUT, and TEST_DIR are required")
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

# AthenaK AMR allocates an even ghost width.  O4 therefore intentionally uses
# four stored ghost layers while all mathematical Z4c operators dispatch from
# the configured three-point half-width stencil.
foreach(order IN ITEMS 2 4 6)
  if(order EQUAL 2)
    set(nghost 2)
  else()
    set(nghost 4)
  endif()
  set(case_name "z4c_kerr_half_plane_amr_o${order}")
  set(case_dir "${TEST_DIR}/o${order}")
  file(MAKE_DIRECTORY "${case_dir}")
  execute_process(
    COMMAND "${ATHENA}" -i "${INPUT}"
            "mesh/nghost=${nghost}"
            "z4c/spatial_order=${order}"
            "job/basename=${case_name}"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  file(WRITE "${case_dir}/stdout.log" "${stdout}")
  file(WRITE "${case_dir}/stderr.log" "${stderr}")
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "half-plane Kerr O${order}/NGHOST=${nghost} AMR run failed (${result})\n"
      "stdout:\n${stdout}\nstderr:\n${stderr}")
  endif()
  if(NOT stdout MATCHES "Initialized arXiv:1001.4077 Kerr puncture")
    message(FATAL_ERROR "O${order} AMR run did not initialize Kerr data")
  endif()
  if(NOT stdout MATCHES "AMR_Z4C_CHI_PROLONGATION cycle=1")
    message(FATAL_ERROR "O${order} AMR run omitted chi-prolongation evidence")
  endif()
  if(NOT stdout MATCHES "Current number of MeshBlocks = 14")
    message(FATAL_ERROR "O${order} AMR run did not produce the exact refined tree")
  endif()
  if(NOT stdout MATCHES "6 MeshBlocks created, 0 deleted by AMR")
    message(FATAL_ERROR "O${order} AMR run changed the exact refinement inventory")
  endif()

  set(history "${case_dir}/${case_name}.z4c.user.hst")
  if(NOT EXISTS "${history}")
    message(FATAL_ERROR "O${order} AMR run omitted history evidence")
  endif()
  file(READ "${history}" history_text)
  string(TOLOWER "${history_text}" history_lower)
  if(history_lower MATCHES "(^|[^a-z])(nan|[+-]?inf)([^a-z]|$)")
    message(FATAL_ERROR "O${order} AMR history contains non-finite values")
  endif()
  if(NOT history_text MATCHES
      "\n[ ]*1\\.00000e-03[^\n]*1\\.40000e\\+01[ ]+1\\.00000e\\+00")
    message(FATAL_ERROR
      "O${order} AMR history omits the cycle-one 14-block/level-one state")
  endif()
endforeach()

# Repeat O6 with four times as many blocks in each active area unit.  This
# exercises axis blocks, off-axis neighbors, and axis/coarse-fine corners under
# a substantially different decomposition without changing the physical grid.
set(case_name "z4c_kerr_half_plane_amr_manyblocks")
set(case_dir "${TEST_DIR}/manyblocks")
file(MAKE_DIRECTORY "${case_dir}")
execute_process(
  COMMAND "${ATHENA}" -i "${INPUT}"
          "meshblock/nx1=8" "meshblock/nx2=8"
          "mesh_refinement/max_nmb_per_rank=256"
          "job/basename=${case_name}"
  WORKING_DIRECTORY "${case_dir}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
file(WRITE "${case_dir}/stdout.log" "${stdout}")
file(WRITE "${case_dir}/stderr.log" "${stderr}")
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "many-block half-plane Kerr AMR run failed (${result})\n"
    "stdout:\n${stdout}\nstderr:\n${stderr}")
endif()
if(NOT stdout MATCHES "Root grid = 4 x 8 x 1 MeshBlocks" OR
   NOT stdout MATCHES "Current number of MeshBlocks = 56" OR
   NOT stdout MATCHES "24 MeshBlocks created, 0 deleted by AMR")
  message(FATAL_ERROR "many-block AMR tree inventory changed:\n${stdout}")
endif()
set(history "${case_dir}/${case_name}.z4c.user.hst")
if(NOT EXISTS "${history}")
  message(FATAL_ERROR "many-block AMR run omitted history evidence")
endif()
file(READ "${history}" history_text)
string(TOLOWER "${history_text}" history_lower)
if(history_lower MATCHES "(^|[^a-z])(nan|[+-]?inf)([^a-z]|$)")
  message(FATAL_ERROR "many-block AMR history contains non-finite values")
endif()

message(STATUS
  "half-plane Kerr O2/O4/O6 axis-touching and multi-block AMR runs passed")
