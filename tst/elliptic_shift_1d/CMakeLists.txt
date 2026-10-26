cmake_minimum_required(VERSION 3.16)
project(elliptic_shift_1d LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(spherical_fast SHARED fast_integrator.cpp)
target_compile_options(spherical_fast PRIVATE -Wall -Wextra -ffp-contract=off)
get_filename_component(HYP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_subdirectory("${HYP_ROOT}/kokkos" kokkos)
add_library(spherical_z4c SHARED spherical_api.cpp)
target_include_directories(spherical_z4c PRIVATE "${HYP_ROOT}/src")
target_link_libraries(spherical_z4c PRIVATE Kokkos::kokkos)
target_compile_options(spherical_z4c PRIVATE -Wall -Wextra)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME nonlinear_elliptic_shift
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_elliptic.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME hyperboloidal_wormhole_initial_data
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_wormhole_data.py --library $<TARGET_FILE:spherical_z4c>)
add_executable(scri_chart_probe scri_chart_probe.cpp)
target_include_directories(scri_chart_probe PRIVATE "${HYP_ROOT}/src")
target_link_libraries(scri_chart_probe PRIVATE Kokkos::kokkos)
add_test(NAME spherical_g1_tensor_agreement
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_g1_rhs.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME scalar_nonlinear_geometry_and_analytic_wave
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_rhs.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME analytic_regular_center_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_origin_limits.py)
add_test(NAME compiled_spherical_engine_agreement
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fast.py
    --library $<TARGET_FILE:spherical_z4c> --fast-library $<TARGET_FILE:spherical_fast>)
add_test(NAME initial_data_arithmetic_isolation
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_initial_precision.py)
add_test(NAME analytic_scri_numerator_derivatives
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scri_limits.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME biased_transport_consistency_and_symbol
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_advection.py)
add_test(NAME initial_lapse_mass_jet_compatibility
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_initial_lapse.py)
add_test(NAME isotropic_hyperboloidal_wormhole_geometry
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_wormhole_isotropic.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME puncture_asymptotic_coefficient_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_limits.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME puncture_chart_independent_native_tensor_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME reference_lapse_damping_scalar_geometry
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_rhs.py --library $<TARGET_FILE:spherical_z4c> --damping-form reference-lapse)
add_test(NAME reference_lapse_damping_AF_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_limits.py --library $<TARGET_FILE:spherical_z4c> --damping-form reference-lapse)
add_test(NAME reference_lapse_damping_AF_tensor_agreement
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c> --damping-form reference-lapse)
add_test(NAME rational_lapse_AF_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_limits.py --library $<TARGET_FILE:spherical_z4c> --slicing-family rational)
add_test(NAME rational_lapse_AF_tensor_agreement
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c> --slicing-family rational)
add_test(NAME independent_physical_ADM_nonlinear_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_adm.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME Bondi_frame_independent_Christoffel_audit
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_bondi.py)
add_test(NAME SBP_radial_algebra_and_compatible_D2
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_sbp.py)
add_test(NAME SBP_AF_chart_native_tensor_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c> --radial-operator sbp63)
add_test(NAME incoming_puncture_gauge_principal_boundary
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gauge_boundary.py)
add_test(NAME collapsed_source_AF_native_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c> --lapse-background-source collapsed-inner)
add_test(NAME collapsed_source_AF_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_limits.py --library $<TARGET_FILE:spherical_z4c> --lapse-background-source collapsed-inner)
add_test(NAME stationary_CMC_trumpet_native_geometry
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_trumpet_data.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME positive_trumpet_lapse_chart_native_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_trumpet_lapse_chart.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME positive_regular_scalar_and_native_geometry
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_positive_regular.py --library $<TARGET_FILE:spherical_z4c>)
add_test(NAME compiled_positive_fields_and_stage_monitors
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_fast_positive.py --library $<TARGET_FILE:spherical_z4c> --fast-library $<TARGET_FILE:spherical_fast>)
add_test(NAME high_order_invariant_horizon_extraction
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_horizon.py)
add_test(NAME blended_log_AF_native_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c> --slicing-family blended-log)
add_test(NAME blended_log_AF_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_limits.py --library $<TARGET_FILE:spherical_z4c> --slicing-family blended-log)
add_test(NAME banded_shift_ODE_dense_native_and_compiled_controls
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_shift_ode.py --library $<TARGET_FILE:spherical_z4c> --fast-library $<TARGET_FILE:spherical_fast>)
add_test(NAME long_integer_step_clock_and_fractional_epochs
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_time_lattice.py)
add_test(NAME scalar_clock_cadence_and_long_aligned_restart
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_clock_evolution.py --library $<TARGET_FILE:spherical_z4c> --fast-library $<TARGET_FILE:spherical_fast>)
add_test(NAME physical_log_AF_native_rates
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_chart.py --library $<TARGET_FILE:spherical_z4c> --lapse-restoring-profile physical-log --lapse-background-source collapsed-inner)
add_test(NAME physical_log_AF_coefficient_limits
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_puncture_limits.py --library $<TARGET_FILE:spherical_z4c> --lapse-restoring-profile physical-log --lapse-background-source collapsed-inner)
add_test(NAME physical_log_finite_Khat_native_collapsed_limit
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_physical_log_source.py --library $<TARGET_FILE:spherical_z4c>)
