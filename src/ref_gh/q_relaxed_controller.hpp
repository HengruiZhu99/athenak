//========================================================================================
//! \file q_relaxed_controller.hpp
//! \brief One-parameter finite-resolution-relaxation controller algebra.
//========================================================================================
#ifndef REF_GH_Q_RELAXED_CONTROLLER_HPP_
#define REF_GH_Q_RELAXED_CONTROLLER_HPP_

#include "athena.hpp"

namespace ref_gh {

struct QRelaxedControllerRhs {
  Real q;
  Real q_dot;
};

struct PrescribedQTrajectory {
  Real q;
  Real q_dot;
  Real q_ddot;
};

// Compact C2 pulse used only to isolate moving-reference algebra from
// feedback. It starts and ends at q=1 with vanishing first and second time
// derivatives, and reaches target at half duration.
KOKKOS_INLINE_FUNCTION
PrescribedQTrajectory EvaluatePrescribedQTrajectory(
    const Real time, const Real target, const Real duration) {
  if (time <= 0.0 || time >= duration) return {1.0, 0.0, 0.0};
  constexpr Real pi = 3.141592653589793238462643383279502884;
  const Real frequency = pi/duration;
  const Real sine = Kokkos::sin(frequency*time);
  const Real cosine = Kokkos::cos(frequency*time);
  const Real amplitude = target - 1.0;
  const Real sine2 = sine*sine;
  const Real sine3 = sine2*sine;
  const Real sine4 = sine2*sine2;
  return {
      1.0 + amplitude*sine4,
      4.0*amplitude*frequency*sine3*cosine,
      4.0*amplitude*frequency*frequency
          *(3.0*sine2*cosine*cosine - sine4)};
}

KOKKOS_INLINE_FUNCTION
QRelaxedControllerRhs EvaluateQRelaxedControllerRhs(
    const Real q, const Real q_dot, const Real q_est, const Real omega,
    const Real zeta, const Real acceleration_limit) {
  const Real raw_acceleration =
      -2.0*zeta*omega*q_dot + omega*omega*(q_est - q);
  return {q_dot,
          acceleration_limit*Kokkos::tanh(
              raw_acceleration/acceleration_limit)};
}

}  // namespace ref_gh

#endif  // REF_GH_Q_RELAXED_CONTROLLER_HPP_
