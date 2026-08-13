#ifndef Z4C_Z4C_SHIFT_GAUGE_HPP_
#define Z4C_Z4C_SHIFT_GAUGE_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_shift_gauge.hpp
//! \brief Device-safe pointwise source terms for opt-in one-equation shift gauges.

#include "athena.hpp"

namespace z4c {

enum class ShiftGaugeProfile : int {
  standard = 0,
  candidate_a = 1,
  candidate_c = 2,
};

struct ShiftGaugeForces {
  Real gamma = 0.0;
  Real chi_gradient = 0.0;
  Real lapse_gradient = 0.0;

  KOKKOS_INLINE_FUNCTION
  Real total() const {
    return gamma + chi_gradient + lapse_gradient;
  }
};

//! Candidate C uses Q=alpha^2 chi and
//! G_C(Q)=Q+(1-Q)/(1+Q).  The latter form avoids explicitly forming Q^2.
KOKKOS_INLINE_FUNCTION
Real CandidateCGammaCoefficient(const Real alpha, const Real chi) {
  const Real q = alpha * alpha * chi;
  return q + (1.0 - q) / (1.0 + q);
}

//! Return the non-advective, non-damping force in one shift component.
//!
//! The caller supplies one row of the inverse conformal metric and the raw
//! accepted/stage alpha and chi gradients.  Candidate A and Candidate C must
//! not substitute chi_guarded for raw chi in this formula.
KOKKOS_INLINE_FUNCTION
ShiftGaugeForces EvaluateModifiedShiftGaugeForces(
    const ShiftGaugeProfile profile, const Real alpha, const Real chi,
    const Real gamma_component, const Real inverse_metric_row[3],
    const Real lapse_gradient[3], const Real chi_gradient[3]) {
  ShiftGaugeForces result;

  if (profile == ShiftGaugeProfile::candidate_a) {
    result.gamma = alpha * alpha * chi * gamma_component;
    return result;
  }

  if (profile == ShiftGaugeProfile::candidate_c) {
    result.gamma = CandidateCGammaCoefficient(alpha, chi) * gamma_component;
    for (int j = 0; j < 3; ++j) {
      result.chi_gradient +=
          0.5 * alpha * alpha * inverse_metric_row[j] * chi_gradient[j];
      result.lapse_gradient -=
          alpha * chi * inverse_metric_row[j] * lapse_gradient[j];
    }
  }

  return result;
}

}  // namespace z4c

#endif  // Z4C_Z4C_SHIFT_GAUGE_HPP_
