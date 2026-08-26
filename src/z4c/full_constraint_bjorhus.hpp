//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file full_constraint_bjorhus.hpp
//! \brief Local algebra for the full-field, constraint-only Z4c Bjorhus closure.

#ifndef Z4C_FULL_CONSTRAINT_BJORHUS_HPP_
#define Z4C_FULL_CONSTRAINT_BJORHUS_HPP_

#include <cmath>

#include "athena.hpp"

namespace z4c {

//! The two scalar and three Cartesian components below represent four independent
//! incoming rows: the two scalar light rows and the two tangential vector light rows.
//! `vector_covector` is projected tangentially by SolveFullConstraintBjorhusCorrection.
struct FullConstraintBjorhusRates {
  Real theta = 0.0;
  Real z_normal = 0.0;
  Real vector_covector[3] = {0.0, 0.0, 0.0};
};

struct FullConstraintBjorhusCorrection {
  Real theta = 0.0;
  Real gamma_u[3] = {0.0, 0.0, 0.0};
};

//! Change induced in the opposite-sign (outgoing) constraint-rate rows by the
//! Theta/Gamma-only incoming correction.  This is intentionally exposed because the
//! sparse compatibility treatment cannot generally make these values zero.
struct FullConstraintBjorhusOutgoingRateChange {
  Real theta = 0.0;
  Real z_normal = 0.0;
  Real vector_covector[3] = {0.0, 0.0, 0.0};
};

struct FullConstraintBjorhusFrame {
  Real metric_dd[3][3] = {};
  Real metric_uu[3][3] = {};
  Real normal_d[3] = {};
  Real normal_u[3] = {};
};

enum class FullConstraintBjorhusStatus {
  valid,
  invalid_metric,
  invalid_normal,
  invalid_coefficient,
  invalid_scalar_map,
};

//! Construct the outward conformal-unit normal from all incident coordinate faces.
//!
//! Sylvester's criterion is used rather than a determinant-only test so a symmetric
//! matrix with two negative eigenvalues cannot enter the characteristic map.
KOKKOS_INLINE_FUNCTION
FullConstraintBjorhusStatus MakeFullConstraintBjorhusFrame(
    const Real metric_dd[3][3], const int side[3],
    FullConstraintBjorhusFrame *frame) {
  const Real minor1 = metric_dd[0][0];
  const Real minor2 =
      metric_dd[0][0] * metric_dd[1][1] - metric_dd[0][1] * metric_dd[0][1];
  const Real determinant =
      metric_dd[0][0] * (metric_dd[1][1] * metric_dd[2][2] -
                         metric_dd[1][2] * metric_dd[1][2]) -
      metric_dd[0][1] * (metric_dd[0][1] * metric_dd[2][2] -
                         metric_dd[0][2] * metric_dd[1][2]) +
      metric_dd[0][2] * (metric_dd[0][1] * metric_dd[1][2] -
                         metric_dd[0][2] * metric_dd[1][1]);
  if (!(Kokkos::isfinite(minor1) && Kokkos::isfinite(minor2) &&
        Kokkos::isfinite(determinant)) ||
      minor1 <= 0.0 || minor2 <= 0.0 || determinant <= 0.0) {
    return FullConstraintBjorhusStatus::invalid_metric;
  }

  const Real inverse_determinant = 1.0 / determinant;
  Real inverse[3][3];
  inverse[0][0] =
      (metric_dd[1][1] * metric_dd[2][2] -
       metric_dd[1][2] * metric_dd[1][2]) * inverse_determinant;
  inverse[0][1] = inverse[1][0] =
      (metric_dd[0][2] * metric_dd[1][2] -
       metric_dd[0][1] * metric_dd[2][2]) * inverse_determinant;
  inverse[0][2] = inverse[2][0] =
      (metric_dd[0][1] * metric_dd[1][2] -
       metric_dd[0][2] * metric_dd[1][1]) * inverse_determinant;
  inverse[1][1] =
      (metric_dd[0][0] * metric_dd[2][2] -
       metric_dd[0][2] * metric_dd[0][2]) * inverse_determinant;
  inverse[1][2] = inverse[2][1] =
      (metric_dd[0][1] * metric_dd[0][2] -
       metric_dd[0][0] * metric_dd[1][2]) * inverse_determinant;
  inverse[2][2] =
      (metric_dd[0][0] * metric_dd[1][1] -
       metric_dd[0][1] * metric_dd[0][1]) * inverse_determinant;

  Real normal_squared = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      normal_squared += side[a] * inverse[a][b] * side[b];
    }
  }
  if (!(Kokkos::isfinite(normal_squared)) || normal_squared <= 0.0) {
    return FullConstraintBjorhusStatus::invalid_normal;
  }
  const Real inverse_norm = 1.0 / Kokkos::sqrt(normal_squared);
  for (int a = 0; a < 3; ++a) {
    frame->normal_d[a] = side[a] * inverse_norm;
    frame->normal_u[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      frame->normal_u[a] += inverse[a][b] * frame->normal_d[b];
    }
    for (int b = 0; b < 3; ++b) {
      frame->metric_dd[a][b] = metric_dd[a][b];
      frame->metric_uu[a][b] = inverse[a][b];
    }
  }
  return FullConstraintBjorhusStatus::valid;
}

//! Solve the sparse four-by-four compatibility map for Theta/Gamma RHS corrections.
//!
//! For frozen chi C and outward conformal-unit normal s, the incoming constraint rows
//! have correction dependence
//!
//!   delta dot(w_P1+) = sqrt(C) delta Theta + C/2 delta Gamma_s,
//!   delta dot(w_P2+) = 2/(3 sqrt(C)) delta Theta - delta Gamma_s,
//!   delta dot(w_PA+) = -delta Gamma_A.
//!
//! The target is zero incoming rate.  The map determinant is -4 sqrt(C)/3.
KOKKOS_INLINE_FUNCTION
FullConstraintBjorhusStatus SolveFullConstraintBjorhusCorrection(
    const Real chi, const FullConstraintBjorhusFrame &frame,
    const FullConstraintBjorhusRates &volume_rates,
    FullConstraintBjorhusCorrection *correction) {
  if (!(Kokkos::isfinite(chi)) || chi <= 0.0) {
    return FullConstraintBjorhusStatus::invalid_coefficient;
  }
  const Real sqrt_chi = Kokkos::sqrt(chi);
  const Real scalar_map_determinant = -4.0 * sqrt_chi / 3.0;
  if (!(Kokkos::isfinite(scalar_map_determinant)) ||
      scalar_map_determinant == 0.0) {
    return FullConstraintBjorhusStatus::invalid_scalar_map;
  }

  correction->theta =
      -3.0 * (volume_rates.theta + 0.5 * chi * volume_rates.z_normal) /
      (4.0 * sqrt_chi);
  const Real gamma_normal =
      0.75 * volume_rates.z_normal - 0.5 * volume_rates.theta / chi;

  // The tangential vector-row rate is represented as a covector.  Raise only its
  // tangential projection and add the independently solved normal component.
  Real vector_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    vector_normal += frame.normal_u[a] * volume_rates.vector_covector[a];
  }
  for (int a = 0; a < 3; ++a) {
    correction->gamma_u[a] = gamma_normal * frame.normal_u[a];
    for (int b = 0; b < 3; ++b) {
      const Real tangential_covector =
          volume_rates.vector_covector[b] - frame.normal_d[b] * vector_normal;
      correction->gamma_u[a] +=
          frame.metric_uu[a][b] * tangential_covector;
    }
  }
  return FullConstraintBjorhusStatus::valid;
}

//! Re-evaluate only the correction-dependent part of the four incoming rates.
//! This is shared by the production assertion and manufactured unit tests.
KOKKOS_INLINE_FUNCTION
FullConstraintBjorhusRates ApplyFullConstraintBjorhusCorrectionToRates(
    const Real chi, const FullConstraintBjorhusFrame &frame,
    const FullConstraintBjorhusRates &volume_rates,
    const FullConstraintBjorhusCorrection &correction) {
  FullConstraintBjorhusRates corrected = volume_rates;
  const Real sqrt_chi = Kokkos::sqrt(chi);
  Real gamma_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    gamma_normal += frame.normal_d[a] * correction.gamma_u[a];
  }
  corrected.theta += sqrt_chi * correction.theta + 0.5 * chi * gamma_normal;
  corrected.z_normal +=
      2.0 * correction.theta / (3.0 * sqrt_chi) - gamma_normal;

  for (int a = 0; a < 3; ++a) {
    Real lowered_gamma = 0.0;
    for (int b = 0; b < 3; ++b) {
      lowered_gamma += frame.metric_dd[a][b] * correction.gamma_u[b];
    }
    corrected.vector_covector[a] -= lowered_gamma;
  }
  return corrected;
}

//! Project the sparse correction onto the paired outgoing characteristic-rate rows.
//!
//! These are changes, not target values.  A nonzero result is expected for generic
//! incoming data and is the precise MOL compatibility limitation of this closure.
KOKKOS_INLINE_FUNCTION
FullConstraintBjorhusOutgoingRateChange
FullConstraintBjorhusInducedOutgoingRateChange(
    const Real chi, const FullConstraintBjorhusFrame &frame,
    const FullConstraintBjorhusCorrection &correction) {
  FullConstraintBjorhusOutgoingRateChange change;
  const Real sqrt_chi = Kokkos::sqrt(chi);
  Real gamma_normal = 0.0;
  for (int a = 0; a < 3; ++a) {
    gamma_normal += frame.normal_d[a] * correction.gamma_u[a];
  }
  change.theta = -sqrt_chi * correction.theta + 0.5 * chi * gamma_normal;
  change.z_normal =
      -2.0 * correction.theta / (3.0 * sqrt_chi) - gamma_normal;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      change.vector_covector[a] -=
          frame.metric_dd[a][b] * correction.gamma_u[b];
    }
  }
  return change;
}

//! Deterministic face owner: x1 precedes x2, which precedes x3.  The Cartoon axis
//! endpoint is never owned by CPBC, including its intersection with a z boundary.
KOKKOS_INLINE_FUNCTION
constexpr bool FullConstraintBjorhusOwnsPoint(const int direction,
                                               const int side[3],
                                               const bool cartoon_axis_point) {
  if (cartoon_axis_point || direction < 0 || direction > 2 || side[direction] == 0) {
    return false;
  }
  for (int prior = 0; prior < direction; ++prior) {
    if (side[prior] != 0) return false;
  }
  return true;
}

}  // namespace z4c

#endif  // Z4C_FULL_CONSTRAINT_BJORHUS_HPP_
