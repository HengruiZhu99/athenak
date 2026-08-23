//========================================================================================
//! \file feedback_continuation.hpp
//! \brief Equation-preserving closed-loop rate governor for Ref-GH continuation.
//========================================================================================
#ifndef REF_GH_FEEDBACK_CONTINUATION_HPP_
#define REF_GH_FEEDBACK_CONTINUATION_HPP_

#include "athena.hpp"

namespace ref_gh {

struct FeedbackContinuationParameters {
  Real v_max;
  Real tau_v;
  Real xi_end_start;
  Real risk_slow;
  Real risk_stop;
  Real condition_stop;
  Real relative_lapse_min_stop;
  Real relative_lapse_max_stop;
  Real v2_stop;
};

struct FeedbackContinuationObservables {
  Real condition_max;
  Real relative_lapse_min;
  Real relative_lapse_max;
  Real v2_max;
};

struct FeedbackContinuationCommand {
  Real risk_condition;
  Real risk_lapse_min;
  Real risk_lapse_max;
  Real risk_v2;
  Real risk;
  Real risk_factor;
  Real endpoint_factor;
  Real v_cmd;
  Real xi_rhs;
  Real xi_dot_rhs;
};

KOKKOS_INLINE_FUNCTION
Real ScalarQuinticSmoothstep(const Real x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return x*x*x*(10.0 + x*(-15.0 + 6.0*x));
}

KOKKOS_INLINE_FUNCTION
Real DecreasingQuinticRamp(const Real value, const Real one_at,
                           const Real zero_at) {
  if (value <= one_at) return 1.0;
  if (value >= zero_at) return 0.0;
  return 1.0 - ScalarQuinticSmoothstep((value - one_at)/(zero_at - one_at));
}

KOKKOS_INLINE_FUNCTION
FeedbackContinuationCommand EvaluateFeedbackContinuation(
    const FeedbackContinuationParameters &parameters,
    const FeedbackContinuationObservables &observables,
    const Real xi, const Real xi_dot, const bool constraint_veto,
    const bool completed) {
  FeedbackContinuationCommand command{};
  if (observables.condition_max > 1.0) {
    command.risk_condition = Kokkos::log(observables.condition_max)
                             /Kokkos::log(parameters.condition_stop);
  }
  if (observables.relative_lapse_min > 0.0
      && observables.relative_lapse_min < 1.0) {
    command.risk_lapse_min =
        Kokkos::log(1.0/observables.relative_lapse_min)
        /Kokkos::log(1.0/parameters.relative_lapse_min_stop);
  }
  if (observables.relative_lapse_max > 1.0) {
    command.risk_lapse_max = Kokkos::log(observables.relative_lapse_max)
                             /Kokkos::log(parameters.relative_lapse_max_stop);
  }
  if (observables.v2_max > 0.0) {
    command.risk_v2 = Kokkos::sqrt(observables.v2_max/parameters.v2_stop);
  }
  command.risk = command.risk_condition;
  if (command.risk_lapse_min > command.risk) command.risk = command.risk_lapse_min;
  if (command.risk_lapse_max > command.risk) command.risk = command.risk_lapse_max;
  if (command.risk_v2 > command.risk) command.risk = command.risk_v2;
  command.risk_factor = DecreasingQuinticRamp(
      command.risk, parameters.risk_slow, parameters.risk_stop);
  command.endpoint_factor = DecreasingQuinticRamp(
      xi, parameters.xi_end_start, 1.0);
  command.v_cmd = parameters.v_max*command.risk_factor*command.endpoint_factor;
  if (constraint_veto || completed || xi >= 1.0) command.v_cmd = 0.0;
  command.xi_rhs = (completed || xi >= 1.0) ? 0.0 : xi_dot;
  command.xi_dot_rhs = (completed || xi >= 1.0)
      ? 0.0 : (command.v_cmd - xi_dot)/parameters.tau_v;
  return command;
}

}  // namespace ref_gh

#endif  // REF_GH_FEEDBACK_CONTINUATION_HPP_
