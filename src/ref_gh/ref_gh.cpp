//========================================================================================
//! \file ref_gh.cpp
//! \brief Construction and storage for the separate 50-field reference-frame GH module.
//========================================================================================
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/feedback_continuation.hpp"
#include "ref_gh/puncture_exponent.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_analytic_radial_q.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"
#include "ref_gh/reference_provider_cache.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace ref_gh {

namespace {

void RunFeedbackContinuationSelfTest() {
  const FeedbackContinuationParameters parameters{
      0.25, 0.5, 0.90, 0.70, 1.0, 8.0, 0.5, 3.0, 0.20};
  const FeedbackContinuationObservables safe{1.0, 1.0, 1.0, 0.0};
  const FeedbackContinuationObservables unsafe{8.0, 0.5, 3.0, 0.20};
  constexpr Real dt = 1.0e-4;

  // Verify the activation two-jet directly, including the exact constant
  // endpoint branches required by the continuation contract.
  const ReferenceJet left_endpoint = QuinticSmoothstep(
      ControllerJet(0.0, 0.125, -0.25));
  const ReferenceJet right_endpoint = QuinticSmoothstep(
      ControllerJet(1.0, 0.125, -0.25));
  if (left_endpoint.value != 0.0 || left_endpoint.d[0] != 0.0
      || left_endpoint.dd[0][0] != 0.0 || right_endpoint.value != 1.0
      || right_endpoint.d[0] != 0.0 || right_endpoint.dd[0][0] != 0.0) {
    std::cout << "### FATAL ERROR: continuation activation endpoint jets are not "
                 "exact constants."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int sample = 1; sample < 1000; ++sample) {
    const Real value = static_cast<Real>(sample)/1000.0;
    const Real first = 0.05 + 0.10*value;
    const Real second = -0.02 + 0.04*value;
    const ReferenceJet actual = QuinticSmoothstep(
        ControllerJet(value, first, second));
    const Real derivative = 30.0*value*value*(1.0 - value)*(1.0 - value);
    const Real second_derivative =
        60.0*value*(1.0 - value)*(1.0 - 2.0*value);
    const Real expected_value = ScalarQuinticSmoothstep(value);
    const Real expected_first = derivative*first;
    const Real expected_second =
        second_derivative*first*first + derivative*second;
    const Real tolerance = 256.0*std::numeric_limits<Real>::epsilon();
    if (std::abs(actual.value - expected_value) > tolerance
        || std::abs(actual.d[0] - expected_first) > tolerance
        || std::abs(actual.dd[0][0] - expected_second) > tolerance) {
      std::cout << "### FATAL ERROR: continuation activation two-jet chain rule "
                   "failed at xi=" << value << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  constexpr Real endpoint_probe = 1.0e-6;
  const ReferenceJet near_left = QuinticSmoothstep(
      ControllerJet(endpoint_probe, 0.2, 0.1));
  const ReferenceJet near_right = QuinticSmoothstep(
      ControllerJet(1.0 - endpoint_probe, 0.2, 0.1));
  if (std::abs(near_left.d[0]) > 2.0e-11
      || std::abs(near_right.d[0]) > 2.0e-11
      || std::abs(near_left.dd[0][0]) > 3.0e-6
      || std::abs(near_right.dd[0][0]) > 3.0e-6) {
    std::cout << "### FATAL ERROR: continuation activation jet does not approach "
                 "its C2 endpoint continuously."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real xi = 0.0;
  Real xi_dot = 0.0;
  Real previous_xi = xi;
  bool reached_endpoint = false;
  for (int step = 0; step < 100000; ++step) {
    const auto command = EvaluateFeedbackContinuation(
        parameters, safe, xi, xi_dot, false, false);
    xi += dt*command.xi_rhs;
    xi_dot += dt*command.xi_dot_rhs;
    if (xi < previous_xi || xi_dot < 0.0 || command.v_cmd < 0.0
        || command.v_cmd > parameters.v_max) {
      std::cout << "### FATAL ERROR: always-safe continuation self-test lost "
                   "monotonicity or a command bound."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    previous_xi = xi;
    if (xi >= 1.0) {
      reached_endpoint = true;
      break;
    }
  }
  if (!reached_endpoint) {
    std::cout << "### FATAL ERROR: always-safe continuation self-test did not "
                 "reach the endpoint."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const auto endpoint = EvaluateFeedbackContinuation(
      parameters, safe, 1.0, 0.125, false, true);
  if (endpoint.v_cmd != 0.0 || endpoint.xi_rhs != 0.0
      || endpoint.xi_dot_rhs != 0.0 || endpoint.endpoint_factor != 0.0) {
    std::cout << "### FATAL ERROR: continuation endpoint is not exactly stationary."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real previous_command = parameters.v_max;
  for (int sample = 0; sample <= 1000; ++sample) {
    const Real fraction = static_cast<Real>(sample)/1000.0;
    const Real condition = Kokkos::exp(
        fraction*1.2*Kokkos::log(parameters.condition_stop));
    const FeedbackContinuationObservables approach{
        condition, 1.0, 1.0, 0.0};
    const auto command = EvaluateFeedbackContinuation(
        parameters, approach, 0.25, 0.1, false, false);
    if (command.v_cmd > previous_command + 32.0*std::numeric_limits<Real>::epsilon()
        || command.v_cmd < 0.0) {
      std::cout << "### FATAL ERROR: approach-to-stop command is not monotone."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    previous_command = command.v_cmd;
  }
  if (previous_command != 0.0) {
    std::cout << "### FATAL ERROR: stop risk does not produce zero command."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Evolve an actual temporary excursion.  The command must freeze, xi must
  // remain monotone while the rate relaxes, and the rate must resume smoothly
  // once the safe history returns.
  xi = 0.25;
  xi_dot = 0.10;
  previous_xi = xi;
  Real previous_xi_dot = xi_dot;
  for (int step = 0; step < 10000; ++step) {
    const auto command = EvaluateFeedbackContinuation(
        parameters, unsafe, xi, xi_dot, false, false);
    if (command.v_cmd != 0.0) {
      std::cout << "### FATAL ERROR: temporary unsafe history did not freeze the "
                   "continuation command."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    xi += dt*command.xi_rhs;
    xi_dot += dt*command.xi_dot_rhs;
    if (xi < previous_xi || xi_dot < 0.0
        || std::abs(xi_dot - previous_xi_dot) > 6.0e-5) {
      std::cout << "### FATAL ERROR: temporary unsafe history lost monotonicity "
                   "or rate smoothness."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    previous_xi = xi;
    previous_xi_dot = xi_dot;
  }
  const Real frozen_rate = xi_dot;
  for (int step = 0; step < 10000; ++step) {
    const auto command = EvaluateFeedbackContinuation(
        parameters, safe, xi, xi_dot, false, false);
    xi += dt*command.xi_rhs;
    xi_dot += dt*command.xi_dot_rhs;
    if (xi < previous_xi || xi_dot < 0.0
        || std::abs(xi_dot - previous_xi_dot) > 6.0e-5) {
      std::cout << "### FATAL ERROR: recovered safe history lost monotonicity or "
                   "rate smoothness."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    previous_xi = xi;
    previous_xi_dot = xi_dot;
  }
  if (!(xi_dot > frozen_rate)) {
    std::cout << "### FATAL ERROR: continuation rate did not resume after a "
                 "temporary unsafe history."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // A permanently unsafe history must never reverse xi and must asymptotically
  // remove the residual forward rate without producing a nonzero command.
  xi = 0.40;
  xi_dot = 0.10;
  previous_xi = xi;
  for (int step = 0; step < 50000; ++step) {
    const auto command = EvaluateFeedbackContinuation(
        parameters, unsafe, xi, xi_dot, false, false);
    xi += dt*command.xi_rhs;
    xi_dot += dt*command.xi_dot_rhs;
    if (command.v_cmd != 0.0 || xi < previous_xi || xi_dot < 0.0) {
      std::cout << "### FATAL ERROR: permanently unsafe history violated freeze "
                   "or monotonicity."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    previous_xi = xi;
  }
  if (xi_dot > 5.0e-6 || xi >= 0.46) {
    std::cout << "### FATAL ERROR: permanently unsafe continuation did not settle."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const auto vetoed = EvaluateFeedbackContinuation(
      parameters, safe, 0.4, 0.1, true, false);
  const auto resumed = EvaluateFeedbackContinuation(
      parameters, safe, 0.4, 0.0, false, false);
  const auto permanent = EvaluateFeedbackContinuation(
      parameters, unsafe, 0.0, 0.0, false, false);
  if (vetoed.v_cmd != 0.0 || resumed.v_cmd != parameters.v_max
      || permanent.v_cmd != 0.0 || permanent.xi_rhs != 0.0) {
    std::cout << "### FATAL ERROR: continuation freeze/resume manufactured "
                 "history failed."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH feedback continuation manufactured histories passed: "
               "safe, approach-stop, excursion-recovery, permanent-unsafe, endpoint"
            << std::endl;
}

}  // namespace

char const * const RefGh::StateNames[RefGh::nref_gh] = {
  "ref_gh_Psi00", "ref_gh_Psi01", "ref_gh_Psi02", "ref_gh_Psi03",
  "ref_gh_Psi11", "ref_gh_Psi12", "ref_gh_Psi13", "ref_gh_Psi22",
  "ref_gh_Psi23", "ref_gh_Psi33",
  "ref_gh_Pi00", "ref_gh_Pi01", "ref_gh_Pi02", "ref_gh_Pi03",
  "ref_gh_Pi11", "ref_gh_Pi12", "ref_gh_Pi13", "ref_gh_Pi22",
  "ref_gh_Pi23", "ref_gh_Pi33",
  "ref_gh_Phi100", "ref_gh_Phi101", "ref_gh_Phi102", "ref_gh_Phi103",
  "ref_gh_Phi111", "ref_gh_Phi112", "ref_gh_Phi113", "ref_gh_Phi122",
  "ref_gh_Phi123", "ref_gh_Phi133",
  "ref_gh_Phi200", "ref_gh_Phi201", "ref_gh_Phi202", "ref_gh_Phi203",
  "ref_gh_Phi211", "ref_gh_Phi212", "ref_gh_Phi213", "ref_gh_Phi222",
  "ref_gh_Phi223", "ref_gh_Phi233",
  "ref_gh_Phi300", "ref_gh_Phi301", "ref_gh_Phi302", "ref_gh_Phi303",
  "ref_gh_Phi311", "ref_gh_Phi312", "ref_gh_Phi313", "ref_gh_Phi322",
  "ref_gh_Phi323", "ref_gh_Phi333",
  "ref_gh_Hhat0", "ref_gh_Hhat1", "ref_gh_Hhat2", "ref_gh_Hhat3",
  "ref_gh_theta0", "ref_gh_theta1", "ref_gh_theta2", "ref_gh_theta3",
  "ref_gh_Upsilon1", "ref_gh_Upsilon2", "ref_gh_Upsilon3"
};

char const * const RefGh::ConstraintNames[RefGh::ncon] = {
  "ref_gh_C0", "ref_gh_C1", "ref_gh_C2", "ref_gh_C3",
  "ref_gh_reduction", "ref_gh_curl",
  "ref_gh_Q", "ref_gh_Delta", "ref_gh_frame_Ricci",
  "ref_gh_coordinate_Ricci", "ref_gh_source_curvature",
  "ref_gh_source_QQ", "ref_gh_source_DeltaDelta",
  "ref_gh_source_damping", "ref_gh_source_frame_correction",
  "ref_gh_metric_condition"
};

RefGh::RefGh(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("u0 ref_gh", 1, 1, 1, 1, 1),
    u1("u1 ref_gh", 1, 1, 1, 1, 1),
    u_rhs("u_rhs ref_gh", 1, 1, 1, 1, 1),
    u_con("u_con ref_gh", 1, 1, 1, 1, 1),
    coarse_u0("coarse u0 ref_gh", 1, 1, 1, 1, 1),
    reference_provider(), reference_workspace(), reference_evolution(),
    reference_diagnostic(), reference_static(), reference_stage(),
    q_sample_cells(), q_sample_weights(), q_sample_count(0),
    reference_table("ref_gh reference table", 1, 1),
    reference_cache_time(NAN), reference_diagnostic_time(NAN),
    max_location_diagnostic_time(NAN), max_location_diagnostic_cycle(-1),
    controller_generation(0), q_controller_generation(0),
    reference_cache_generation(0),
    reference_diagnostic_generation(0),
    controller{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    controller_base{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    controller_rhs{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    controller_diagnostics{},
    q_controller{1.0, 0.0}, q_controller_base{1.0, 0.0},
    q_controller_rhs{0.0, 0.0},
    continuation_constraint_veto(false), continuation_frozen(false),
    continuation_completed(false), q_controller_frozen(false),
    continuation_veto_start_time(-1.0),
    continuation_veto_start_level(-1.0), continuation_veto_last_level(-1.0),
    reference_cache_oracle_validated(false),
    reference_diagnostic_oracle_validated(false),
    analytic_static_initialized(false),
    dtnew(0.0), max_char_speed(0.0), pmy_pack(ppack), pinput(pin) {
  opt.fd_order = pin->GetOrAddInteger("ref_gh", "fd_order", 4);
  opt.extrap_order = pin->GetOrAddInteger("ref_gh", "extrap_order", 2);
  const std::string reference_name =
      pin->GetOrAddString("ref_gh", "reference", "minkowski");
  if (reference_name == "minkowski") {
    opt.reference_kind = 0;
  } else if (reference_name == "trumpet") {
    opt.reference_kind = 1;
  } else if (reference_name == "time_dependent_lapse_test") {
    opt.reference_kind = 2;
  } else if (reference_name == "time_dependent_spatial_test") {
    opt.reference_kind = 3;
  } else if (reference_name == "wormhole") {
    opt.reference_kind = 4;
  } else if (reference_name == "controlled_transition") {
    opt.reference_kind = 5;
  } else if (reference_name == "generic_singular") {
    opt.reference_kind = 6;
  } else if (reference_name == "trumpet_q_controlled") {
    opt.reference_kind = 7;
  } else {
    std::cout << "### FATAL ERROR: ref_gh reference must be minkowski, trumpet, "
                 "time_dependent_lapse_test, time_dependent_spatial_test, "
                 "wormhole, controlled_transition, generic_singular, or "
                 "trumpet_q_controlled."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.reference_time_dependent =
      GetReferenceProviderMetadata(opt.reference_kind).time_dependent;
  opt.reference_controlled = opt.reference_kind == 5;
  opt.reference_q_controlled = opt.reference_kind == 7;
  const std::string reference_backend = pin->GetOrAddString(
      "ref_gh", "reference_backend", "generic_cache_oracle");
  if (reference_backend == "generic_cache_oracle") {
    opt.reference_backend = 0;
  } else if (reference_backend == "analytic_radial_q") {
    opt.reference_backend = 1;
  } else {
    std::cout << "### FATAL ERROR: ref_gh reference_backend must be "
                 "generic_cache_oracle or analytic_radial_q."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.reference_backend == 1 && !opt.reference_q_controlled) {
    std::cout << "### FATAL ERROR: analytic_radial_q requires "
                 "reference=trumpet_q_controlled."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.controller_enabled =
      pin->GetOrAddBoolean("ref_gh", "controller_enabled", false);
  opt.q_controller_enabled =
      pin->GetOrAddBoolean("ref_gh", "q_controller_enabled", false);
  opt.q_prescribed_enabled =
      pin->GetOrAddBoolean("ref_gh", "q_prescribed_enabled", false);
  if (opt.reference_kind == 7
      && !opt.q_controller_enabled && !opt.q_prescribed_enabled) {
    opt.reference_time_dependent = false;
  }
  const bool continuation_self_test =
      pin->GetOrAddBoolean("ref_gh", "continuation_self_test", false);
  const std::string continuation_mode =
      pin->GetOrAddString("ref_gh", "continuation_mode", "legacy_time");
  if (continuation_mode == "legacy_time") {
    opt.continuation_mode = 0;
  } else if (continuation_mode == "prescribed") {
    opt.continuation_mode = 1;
  } else if (continuation_mode == "feedback") {
    opt.continuation_mode = 2;
  } else {
    std::cout << "### FATAL ERROR: ref_gh continuation_mode must be "
                 "legacy_time, prescribed, or feedback."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string source_name =
      pin->GetOrAddString("ref_gh", "source", "covariant");
  if (source_name == "covariant") {
    opt.source_kind = 0;
  } else if (source_name == "coordinate_oracle") {
    opt.source_kind = 1;
  } else {
    std::cout << "### FATAL ERROR: ref_gh source must be covariant or coordinate_oracle."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.reference_backend == 1 && opt.source_kind != 0) {
    std::cout << "### FATAL ERROR: analytic_radial_q is a covariant-source "
                 "production backend; coordinate_oracle retains the generic "
                 "cache backend."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.debug_task_fences =
      pin->GetOrAddBoolean("ref_gh", "debug_task_fences", false);
  opt.validate_reference_cache =
      pin->GetOrAddBoolean("ref_gh", "validate_reference_cache", false);
  opt.max_location_diagnostics =
      pin->GetOrAddBoolean("ref_gh", "max_location_diagnostics", false);
  const std::string phi_ordering =
      pin->GetOrAddString("ref_gh", "phi_ordering", "compatible");
  if (phi_ordering == "compatible") {
    opt.phi_ordering = 0;
  } else if (phi_ordering == "standard") {
    opt.phi_ordering = 1;
  } else {
    std::cout << "### FATAL ERROR: ref_gh phi_ordering must be compatible or standard."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string transition_path =
      pin->GetOrAddString("ref_gh", "transition_path", "shrinking_width");
  if (transition_path == "shrinking_width") {
    opt.transition_path = 0;
  } else if (transition_path == "fixed_core") {
    opt.transition_path = 1;
  } else if (transition_path == "fixed_width") {
    opt.transition_path = 2;
  } else {
    std::cout << "### FATAL ERROR: ref_gh transition_path must be "
                 "shrinking_width, fixed_core, or fixed_width."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.gamma0 = pin->GetOrAddReal("ref_gh", "gamma0", 1.0);
  opt.gamma2 = pin->GetOrAddReal("ref_gh", "gamma2", 0.0);
  opt.gauge_driver_enabled =
      pin->GetOrAddBoolean("ref_gh", "gauge_driver_enabled", false);
  opt.gauge_reference_subtraction = pin->GetOrAddBoolean(
      "ref_gh", "gauge_reference_subtraction", false);
  opt.gauge_mu = pin->GetOrAddReal("ref_gh", "gauge_mu", 1.0);
  opt.gauge_eta = pin->GetOrAddReal("ref_gh", "gauge_eta", 1.0);
  opt.shift_nu = pin->GetOrAddReal("ref_gh", "shift_nu", 0.75);
  opt.shift_eta = pin->GetOrAddReal("ref_gh", "shift_eta", 1.0);
  opt.exclude_puncture_stencil_diagnostics = pin->GetOrAddBoolean(
      "ref_gh", "exclude_puncture_stencil_diagnostics", false);
  opt.diss = pin->GetOrAddReal("ref_gh", "diss", 0.02);
  opt.fail_closed_dt = pin->GetOrAddReal("ref_gh", "fail_closed_dt", 0.0);
  opt.reference_mass = pin->GetOrAddReal("ref_gh", "reference_mass", 1.0);
  opt.reference_center[0] = pin->GetOrAddReal("ref_gh", "reference_x", 0.0);
  opt.reference_center[1] = pin->GetOrAddReal("ref_gh", "reference_y", 0.0);
  opt.reference_center[2] = pin->GetOrAddReal("ref_gh", "reference_z", 0.0);
  opt.generic_gaussian_width =
      pin->GetOrAddReal("ref_gh", "generic_gaussian_width", 3.0);
  opt.generic_q_initial =
      pin->GetOrAddReal("ref_gh", "generic_q_initial", 2.0);
  opt.generic_q_final = pin->GetOrAddReal("ref_gh", "generic_q_final", 1.0);
  opt.generic_transition_time =
      pin->GetOrAddReal("ref_gh", "generic_transition_time", 8.0);
  opt.q_gaussian_width =
      pin->GetOrAddReal("ref_gh", "q_gaussian_width", 3.0);
  opt.q_initial = pin->GetOrAddReal("ref_gh", "q_initial", 1.0);
  opt.q_min = pin->GetOrAddReal("ref_gh", "q_min", 0.5);
  opt.q_max = pin->GetOrAddReal("ref_gh", "q_max", 2.5);
  opt.q_rate_limit = pin->GetOrAddReal("ref_gh", "q_rate_limit", 0.25);
  opt.q_acceleration_limit =
      pin->GetOrAddReal("ref_gh", "q_acceleration_limit", 0.25);
  opt.q_omega = pin->GetOrAddReal("ref_gh", "q_omega", 0.5);
  opt.q_zeta = pin->GetOrAddReal("ref_gh", "q_zeta", 1.0);
  opt.q_prescribed_target =
      pin->GetOrAddReal("ref_gh", "q_prescribed_target", 0.9);
  opt.q_prescribed_duration =
      pin->GetOrAddReal("ref_gh", "q_prescribed_duration", 8.0);
  opt.r_core0 = pin->GetOrAddReal("ref_gh", "r_core0", 0.30);
  opt.tau_core = pin->GetOrAddReal("ref_gh", "tau_core", 1.5);
  opt.kappa_core = pin->GetOrAddReal("ref_gh", "kappa_core", 1.0);
  opt.transition_width =
      pin->GetOrAddReal("ref_gh", "transition_width", 0.20);
  opt.tau_transition = pin->GetOrAddReal("ref_gh", "tau_transition", 4.0);
  opt.r_fit_min = pin->GetOrAddReal("ref_gh", "r_fit_min", 0.15);
  opt.r_fit_max = pin->GetOrAddReal("ref_gh", "r_fit_max", 0.40);
  opt.controller_fit_buffer_cells =
      pin->GetOrAddReal("ref_gh", "controller_fit_buffer_cells", 4.0);
  opt.regularization_outer_start =
      pin->GetOrAddReal("ref_gh", "regularization_outer_start", 0.50);
  opt.regularization_outer_end =
      pin->GetOrAddReal("ref_gh", "regularization_outer_end", 0.60);
  opt.controller_zeta = pin->GetOrAddReal("ref_gh", "controller_zeta", 1.0);
  opt.controller_omega_q =
      pin->GetOrAddReal("ref_gh", "controller_omega_q", 0.25);
  opt.controller_omega_p =
      pin->GetOrAddReal("ref_gh", "controller_omega_p", 0.25);
  opt.controller_acceleration_limit =
      pin->GetOrAddReal("ref_gh", "controller_acceleration_limit", 0.05);
  opt.controller_delta_bound =
      pin->GetOrAddReal("ref_gh", "controller_delta_bound", 0.25);
  opt.controller_rate_bound =
      pin->GetOrAddReal("ref_gh", "controller_rate_bound", 0.10);
  opt.continuation_v_max =
      pin->GetOrAddReal("ref_gh", "continuation_v_max", 0.25);
  opt.continuation_tau_v =
      pin->GetOrAddReal("ref_gh", "continuation_tau_v", 0.5);
  opt.continuation_xi_end_start =
      pin->GetOrAddReal("ref_gh", "continuation_xi_end_start", 0.90);
  opt.continuation_risk_slow =
      pin->GetOrAddReal("ref_gh", "continuation_risk_slow", 0.70);
  opt.continuation_risk_stop =
      pin->GetOrAddReal("ref_gh", "continuation_risk_stop", 1.0);
  opt.continuation_condition_stop =
      pin->GetOrAddReal("ref_gh", "continuation_condition_stop", 8.0);
  opt.continuation_lapse_min_stop =
      pin->GetOrAddReal("ref_gh", "continuation_lapse_min_stop", 0.5);
  opt.continuation_lapse_max_stop =
      pin->GetOrAddReal("ref_gh", "continuation_lapse_max_stop", 3.0);
  opt.continuation_v2_stop =
      pin->GetOrAddReal("ref_gh", "continuation_v2_stop", 0.20);
  opt.continuation_gh_warning =
      pin->GetOrAddReal("ref_gh", "continuation_gh_warning", 2.0e-2);
  opt.continuation_reduction_warning =
      pin->GetOrAddReal("ref_gh", "continuation_reduction_warning", 5.0e-3);
  opt.continuation_curl_warning =
      pin->GetOrAddReal("ref_gh", "continuation_curl_warning", 8.0e-2);
  opt.continuation_growth_time =
      pin->GetOrAddReal("ref_gh", "continuation_growth_time", 0.5);
  controller.delta_q =
      pin->GetOrAddReal("ref_gh", "controller_delta_q", 0.0);
  controller.delta_q_dot =
      pin->GetOrAddReal("ref_gh", "controller_delta_q_dot", 0.0);
  controller.delta_p =
      pin->GetOrAddReal("ref_gh", "controller_delta_p", 0.0);
  controller.delta_p_dot =
      pin->GetOrAddReal("ref_gh", "controller_delta_p_dot", 0.0);
  controller.xi = pin->GetOrAddReal("ref_gh", "continuation_xi", 0.0);
  controller.xi_dot =
      pin->GetOrAddReal("ref_gh", "continuation_xi_dot", 0.0);
  q_controller.q = pin->GetOrAddReal("ref_gh", "q", opt.q_initial);
  q_controller.q_dot = pin->GetOrAddReal("ref_gh", "q_dot", 0.0);
  continuation_constraint_veto =
      pin->GetOrAddBoolean("ref_gh", "continuation_constraint_veto", false);
  continuation_frozen =
      pin->GetOrAddBoolean("ref_gh", "continuation_frozen", false);
  continuation_completed =
      pin->GetOrAddBoolean("ref_gh", "continuation_completed", false);
  q_controller_frozen =
      pin->GetOrAddBoolean("ref_gh", "q_controller_frozen", false);
  continuation_veto_start_time =
      pin->GetOrAddReal("ref_gh", "continuation_veto_start_time", -1.0);
  continuation_veto_start_level =
      pin->GetOrAddReal("ref_gh", "continuation_veto_start_level", -1.0);
  continuation_veto_last_level =
      pin->GetOrAddReal("ref_gh", "continuation_veto_last_level", -1.0);
  const Real stored_generation =
      pin->GetOrAddReal("ref_gh", "controller_generation", 0.0);
  if (!(stored_generation >= 0.0) || !std::isfinite(stored_generation)) {
    std::cout << "### FATAL ERROR: invalid stored Ref-GH controller generation."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  controller_generation = static_cast<std::uint64_t>(stored_generation);
  const Real stored_q_generation =
      pin->GetOrAddReal("ref_gh", "q_controller_generation", 0.0);
  if (!(stored_q_generation >= 0.0)
      || !std::isfinite(stored_q_generation)) {
    std::cout << "### FATAL ERROR: invalid stored Ref-GH q-controller generation."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  q_controller_generation =
      static_cast<std::uint64_t>(stored_q_generation);
  controller_base = controller;
  q_controller_base = q_controller;
  const int derivative_radius = opt.fd_order/2;
  if ((opt.fd_order != 2 && opt.fd_order != 4 && opt.fd_order != 6)
      || ppack->pmesh->mb_indcs.ng < 2*derivative_radius) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh fd_order must be 2, 4, or 6, with at least "
              << "fd_order ghost cells for its compatible two-pass update." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.gamma0 <= 0.0 || opt.gamma2 < 0.0 || !std::isfinite(opt.gamma2)
      || opt.gauge_mu <= 0.0 || !std::isfinite(opt.gauge_mu)
      || opt.gauge_eta <= 0.0 || !std::isfinite(opt.gauge_eta)
      || opt.shift_nu <= 0.0 || !std::isfinite(opt.shift_nu)
      || opt.shift_eta <= 0.0 || !std::isfinite(opt.shift_eta)
      || opt.diss < 0.0 || opt.fail_closed_dt < 0.0
      || opt.reference_mass <= 0.0
      || opt.generic_gaussian_width <= 0.0
      || !std::isfinite(opt.generic_q_initial)
      || !std::isfinite(opt.generic_q_final)
      || opt.generic_q_initial < 0.5 || opt.generic_q_initial > 2.5
      || opt.generic_q_final < 0.5 || opt.generic_q_final > 2.5
      || opt.generic_transition_time <= 0.0
      || opt.q_gaussian_width <= 0.0 || !std::isfinite(opt.q_initial)
      || opt.q_min < 0.5 || opt.q_max > 2.5 || opt.q_max <= opt.q_min
      || q_controller.q < opt.q_min || q_controller.q > opt.q_max
      || !std::isfinite(q_controller.q_dot)
      || opt.q_rate_limit <= 0.0 || opt.q_acceleration_limit <= 0.0
      || opt.q_omega <= 0.0 || opt.q_zeta <= 0.0
      || !std::isfinite(opt.q_prescribed_target)
      || opt.q_prescribed_target < opt.q_min
      || opt.q_prescribed_target > opt.q_max
      || !(opt.q_prescribed_duration > 0.0)
      || opt.r_core0 <= 0.0 || opt.tau_core <= 0.0
      || opt.kappa_core <= 0.0 || opt.transition_width <= 0.0
      || opt.tau_transition <= 0.0
      || opt.r_fit_min <= 0.0 || opt.r_fit_max <= opt.r_fit_min
      || opt.controller_fit_buffer_cells <= 0.0
      || opt.regularization_outer_start <= opt.r_fit_max
      || opt.regularization_outer_end <= opt.regularization_outer_start
      || opt.controller_zeta <= 0.0 || opt.controller_omega_q <= 0.0
      || opt.controller_omega_p <= 0.0
      || opt.controller_acceleration_limit <= 0.0
      || opt.controller_delta_bound <= 0.0 || opt.controller_rate_bound <= 0.0
      || opt.continuation_v_max <= 0.0 || opt.continuation_tau_v <= 0.0
      || opt.continuation_xi_end_start <= 0.0
      || opt.continuation_xi_end_start >= 1.0
      || opt.continuation_risk_slow < 0.0
      || opt.continuation_risk_stop <= opt.continuation_risk_slow
      || opt.continuation_condition_stop <= 1.0
      || opt.continuation_lapse_min_stop <= 0.0
      || opt.continuation_lapse_min_stop >= 1.0
      || opt.continuation_lapse_max_stop <= 1.0
      || opt.continuation_v2_stop <= 0.0
      || opt.continuation_gh_warning < 2.0e-2
      || opt.continuation_reduction_warning < 5.0e-3
      || opt.continuation_curl_warning < 8.0e-2
      || opt.continuation_growth_time <= 0.0
      || opt.extrap_order < 2 || opt.extrap_order > 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh requires gamma0>0, gamma2>=0, positive "
              << "finite gauge-driver parameters, diss>=0, "
              << "fail_closed_dt>=0, "
              << "valid positive reference/controller scales, and extrap_order "
                 "in [2,4]." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.gauge_reference_subtraction && !opt.gauge_driver_enabled) {
    std::cout << "### FATAL ERROR: ref_gh gauge_reference_subtraction requires "
                 "gauge_driver_enabled=true." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.q_controller_enabled && !opt.reference_q_controlled) {
    std::cout << "### FATAL ERROR: q_controller_enabled requires "
                 "reference=trumpet_q_controlled." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.q_prescribed_enabled && !opt.reference_q_controlled) {
    std::cout << "### FATAL ERROR: q_prescribed_enabled requires "
                 "reference=trumpet_q_controlled." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.q_controller_enabled && opt.q_prescribed_enabled) {
    std::cout << "### FATAL ERROR: closed-loop and prescribed q modes are "
                 "mutually exclusive." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.q_prescribed_enabled && opt.q_initial != 1.0) {
    std::cout << "### FATAL ERROR: the prescribed q pulse requires "
                 "q_initial=1." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.reference_q_controlled && ppack->pmesh->multilevel) {
    std::cout << "### FATAL ERROR: trumpet_q_controlled qualification is "
                 "uniform-grid only; SMR is out of scope." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!opt.q_controller_enabled && !opt.q_prescribed_enabled
      && q_controller.q_dot != 0.0) {
    std::cout << "### FATAL ERROR: a static q-controlled reference requires "
                 "q_dot=0 when q_controller_enabled=false." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.reference_controlled && opt.continuation_mode != 0
      && (opt.transition_path != kFixedCorePath || opt.phi_ordering != 0
          || opt.controller_enabled || controller.delta_q != 0.0
          || controller.delta_q_dot != 0.0 || controller.delta_p != 0.0
          || controller.delta_p_dot != 0.0 || controller.xi < 0.0
          || controller.xi > 1.0 || controller.xi_dot < 0.0)) {
    std::cout << "### FATAL ERROR: continuation requires fixed_core, compatible "
                 "Phi ordering, the exponent controller disabled, exact "
                 "delta_q=delta_p=0, and admissible nonnegative xi state."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.continuation_mode == 2
      && (opt.continuation_condition_stop > 8.0
          || opt.continuation_lapse_max_stop > 3.0
          || opt.continuation_lapse_min_stop < 0.5
          || opt.continuation_v2_stop > 0.20)) {
    std::cout << "### FATAL ERROR: feedback continuation stop thresholds exceed "
                 "the fixed fail-closed safety caps."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (continuation_completed) {
    controller.xi = 1.0;
    controller.xi_dot = 0.0;
  }
  if (continuation_self_test) RunFeedbackContinuationSelfTest();
  if (ppack->pmesh->multilevel && opt.fd_order == 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh fd_order=6 lacks matching AthenaK AMR transfer."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->GetString("time", "evolution") != "static" && !ppack->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh evolution requires a three-dimensional mesh."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(u0, nmb, nref_gh, n3, n2, n1);
  Kokkos::realloc(u1, nmb, nref_gh, n3, n2, n1);
  Kokkos::realloc(u_rhs, nmb, nref_gh, n3, n2, n1);
  Kokkos::realloc(u_con, nmb, ncon, n3, n2, n1);
  if (opt.reference_backend == 1) {
    Kokkos::realloc(reference_static, nmb, kAnalyticRadialQStaticSize,
                    n3, n2, n1);
    Kokkos::realloc(reference_stage, nmb, kAnalyticRadialQStageSize,
                    n3, n2, n1);
  } else {
    Kokkos::realloc(
        reference_provider, nmb, kReferenceProviderSize, n3, n2, n1);
    Kokkos::realloc(
        reference_workspace, nmb, kReferenceWorkspaceSize, n3, n2, n1);
    Kokkos::realloc(
        reference_evolution, nmb, kReferenceEvolutionSize, n3, n2, n1);
    Kokkos::realloc(
        reference_diagnostic, nmb, kReferenceDiagnosticSize, n3, n2, n1);
  }
  if (global_variable::my_rank == 0) {
    const std::size_t generic_bytes = sizeof(Real)*(
        reference_provider.size() + reference_workspace.size()
        + reference_evolution.size() + reference_diagnostic.size());
    const std::size_t analytic_static_bytes =
        sizeof(Real)*reference_static.size();
    const std::size_t analytic_stage_bytes =
        sizeof(Real)*reference_stage.size();
    std::cout << "ref_gh reference allocation backend="
              << (opt.reference_backend == 1
                      ? "analytic_radial_q" : "generic_cache_oracle")
              << " generic_bytes=" << generic_bytes
              << " analytic_static_components="
              << (opt.reference_backend == 1 ? kAnalyticRadialQStaticSize : 0)
              << " analytic_stage_components="
              << (opt.reference_backend == 1 ? kAnalyticRadialQStageSize : 0)
              << " analytic_static_bytes=" << analytic_static_bytes
              << " analytic_stage_bytes=" << analytic_stage_bytes
              << std::endl;
  }
  if (opt.reference_q_controlled && opt.q_controller_enabled) {
    Real h = std::numeric_limits<Real>::max();
    for (int m = 0; m < ppack->nmb_thispack; ++m) {
      h = std::min(h, std::min(
          ppack->pmb->mb_size.h_view(m).dx1,
          std::min(ppack->pmb->mb_size.h_view(m).dx2,
                   ppack->pmb->mb_size.h_view(m).dx3)));
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_ATHENA_REAL, MPI_MIN,
                  MPI_COMM_WORLD);
#endif
    const int stencil_radius = PunctureEvolutionStencilRadius(
        opt.fd_order, opt.diss);
    std::vector<int> cells;
    std::vector<Real> weights;
    for (int m = 0; m < ppack->nmb_thispack; ++m) {
      for (int kk = 0; kk < indcs.nx3; ++kk) {
        for (int jj = 0; jj < indcs.nx2; ++jj) {
          for (int ii = 0; ii < indcs.nx1; ++ii) {
            const Real x = CellCenterX(
                ii, indcs.nx1, ppack->pmb->mb_size.h_view(m).x1min,
                ppack->pmb->mb_size.h_view(m).x1max);
            const Real y = CellCenterX(
                jj, indcs.nx2, ppack->pmb->mb_size.h_view(m).x2min,
                ppack->pmb->mb_size.h_view(m).x2max);
            const Real z = CellCenterX(
                kk, indcs.nx3, ppack->pmb->mb_size.h_view(m).x3min,
                ppack->pmb->mb_size.h_view(m).x3max);
            const Real displacement[3] = {
                x - opt.reference_center[0], y - opt.reference_center[1],
                z - opt.reference_center[2]};
            const Real radius = std::sqrt(
                displacement[0]*displacement[0]
                + displacement[1]*displacement[1]
                + displacement[2]*displacement[2]);
            if (!InPunctureEstimatorShell(
                    radius, h, opt.q_gaussian_width*opt.reference_mass)) {
              continue;
            }
            const Real spacing[3] = {
                ppack->pmb->mb_size.h_view(m).dx1,
                ppack->pmb->mb_size.h_view(m).dx2,
                ppack->pmb->mb_size.h_view(m).dx3};
            if (!PunctureStencilIsClear(
                    displacement, spacing, stencil_radius)) continue;
            const int packed = ((m*indcs.nx3 + kk)*indcs.nx2 + jj)
                               *indcs.nx1 + ii;
            cells.push_back(packed);
            weights.push_back(PunctureEstimatorWeight(radius, h));
          }
        }
      }
    }
    q_sample_count = static_cast<int>(cells.size());
    q_sample_cells = DvceArray1D<int>("ref_gh q sample cells", q_sample_count);
    q_sample_weights =
        DvceArray1D<Real>("ref_gh q sample weights", q_sample_count);
    auto host_cells = Kokkos::create_mirror_view(q_sample_cells);
    auto host_weights = Kokkos::create_mirror_view(q_sample_weights);
    for (int n = 0; n < q_sample_count; ++n) {
      host_cells(n) = cells[n];
      host_weights(n) = weights[n];
    }
    Kokkos::deep_copy(q_sample_cells, host_cells);
    Kokkos::deep_copy(q_sample_weights, host_weights);
    if (global_variable::my_rank == 0) {
      std::cout << "ref_gh compact q sample list local_count="
                << q_sample_count << " bytes="
                << sizeof(int)*q_sample_cells.size()
                       + sizeof(Real)*q_sample_weights.size()
                << std::endl;
    }
  }
  Kokkos::deep_copy(u0, 0.0);
  Kokkos::deep_copy(u1, 0.0);
  Kokkos::deep_copy(u_rhs, 0.0);
  Kokkos::deep_copy(u_con, 0.0);
  if (ppack->pmesh->multilevel) {
    const int cn1 = indcs.cnx1 + 2*indcs.ng;
    const int cn2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    const int cn3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(coarse_u0, nmb, nref_gh, cn3, cn2, cn1);
    Kokkos::deep_copy(coarse_u0, 0.0);
  }
  if (opt.reference_kind == 1 || opt.reference_kind == 5
      || opt.reference_kind == 7) {
    Kokkos::realloc(reference_table, kTrumpetProfiles, kTrumpetTableSize);
    auto host_table = Kokkos::create_mirror_view(reference_table);
    for (int i = 0; i < kTrumpetTableSize; ++i) {
      host_table(kProfileAlpha, i) = kTrumpetAlpha[i];
      host_table(kProfileAlphaDy, i) = kTrumpetAlphaDy[i];
      host_table(kProfileAlphaDyy, i) = kTrumpetAlphaDyy[i];
      host_table(kProfileArealRadius, i) = kTrumpetArealRadius[i];
      host_table(kProfileArealRadiusDy, i) = kTrumpetArealRadiusDy[i];
      host_table(kProfileArealRadiusDyy, i) = kTrumpetArealRadiusDyy[i];
      host_table(kProfileShiftQ, i) = kTrumpetShiftQ[i];
      host_table(kProfileShiftQDy, i) = kTrumpetShiftQDy[i];
      host_table(kProfileShiftQDyy, i) = kTrumpetShiftQDyy[i];
      host_table(kCoeffAlpha, i) = kTrumpetAlphaA0[i];
      host_table(kCoeffAlpha + 1, i) = kTrumpetAlphaA1[i];
      host_table(kCoeffAlpha + 2, i) = kTrumpetAlphaA2[i];
      host_table(kCoeffAlpha + 3, i) = kTrumpetAlphaA3[i];
      host_table(kCoeffAlpha + 4, i) = kTrumpetAlphaA4[i];
      host_table(kCoeffAlpha + 5, i) = kTrumpetAlphaA5[i];
      host_table(kCoeffArealRadius, i) = kTrumpetArealRadiusA0[i];
      host_table(kCoeffArealRadius + 1, i) = kTrumpetArealRadiusA1[i];
      host_table(kCoeffArealRadius + 2, i) = kTrumpetArealRadiusA2[i];
      host_table(kCoeffArealRadius + 3, i) = kTrumpetArealRadiusA3[i];
      host_table(kCoeffArealRadius + 4, i) = kTrumpetArealRadiusA4[i];
      host_table(kCoeffArealRadius + 5, i) = kTrumpetArealRadiusA5[i];
      host_table(kCoeffShiftQ, i) = kTrumpetShiftQA0[i];
      host_table(kCoeffShiftQ + 1, i) = kTrumpetShiftQA1[i];
      host_table(kCoeffShiftQ + 2, i) = kTrumpetShiftQA2[i];
      host_table(kCoeffShiftQ + 3, i) = kTrumpetShiftQA3[i];
      host_table(kCoeffShiftQ + 4, i) = kTrumpetShiftQA4[i];
      host_table(kCoeffShiftQ + 5, i) = kTrumpetShiftQA5[i];
    }
    Kokkos::deep_copy(reference_table, host_table);
  }
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(nref_gh);
  if (ppack->padm != nullptr) ppack->padm->SetADMVariables = &RefGh::SetADMVariables;
}

RefGh::~RefGh() { delete pbval_u; }

}  // namespace ref_gh
