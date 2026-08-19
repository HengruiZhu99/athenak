//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file timestep_contract.hpp
//! \brief Pure helpers for the explicit Z4c timestep contract.

#ifndef Z4C_TIMESTEP_CONTRACT_HPP_
#define Z4C_TIMESTEP_CONTRACT_HPP_

#include <array>
#include <cmath>
#include <limits>

#include "athena.hpp"

namespace z4c {

//! Coefficients of AthenaK's actual explicit 2S update, indexed by stage.
struct ExplicitRKMethod {
  int stages = 0;
  std::array<Real, 4> gam0{};
  std::array<Real, 4> gam1{};
  std::array<Real, 4> beta{};
  std::array<Real, 4> delta{};
};

//! Evaluate the linear stability polynomial produced by the implemented 2S update.
//! This intentionally models CopyU and ExpRKUpdate rather than assuming a textbook RK.
inline Real ExplicitRKStabilityPolynomial(const ExplicitRKMethod &method, const Real z) {
  Real state = 1.0;
  Real accumulator = 0.0;
  for (int stage = 0; stage < method.stages; ++stage) {
    if (stage == 0) {
      accumulator = state;
    } else {
      accumulator += method.delta[stage - 1] * state;
    }
    state = method.gam0[stage] * state + method.gam1[stage] * accumulator +
            method.beta[stage] * z * state;
  }
  return state;
}

//! Return the contiguous negative-real stability radius of the actual update.
//! A fixed bisection is sufficient here because all supported explicit methods have one
//! interval [0,r] containing the origin on the negative real axis.
inline Real ExplicitRKNegativeRealStabilityRadius(const ExplicitRKMethod &method) {
  if (method.stages <= 0) return std::numeric_limits<Real>::quiet_NaN();
  Real low = 0.0;
  Real high = 1.0;
  for (int expand = 0; expand < 32; ++expand) {
    const Real value = ExplicitRKStabilityPolynomial(method, -high);
    if (!std::isfinite(value)) return std::numeric_limits<Real>::quiet_NaN();
    if (std::fabs(value) > 1.0) break;
    high *= 2.0;
  }
  if (std::fabs(ExplicitRKStabilityPolynomial(method, -high)) <= 1.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  for (int iteration = 0; iteration < 96; ++iteration) {
    const Real middle = 0.5 * (low + high);
    const Real value = ExplicitRKStabilityPolynomial(method, -middle);
    if (!std::isfinite(value)) return std::numeric_limits<Real>::quiet_NaN();
    if (std::fabs(value) <= 1.0) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return low;
}

inline Real SourceTimestepCeiling(const Real source_safety, const Real negative_radius,
                                  const Real max_rate) {
  if (!std::isfinite(source_safety) || !std::isfinite(negative_radius) ||
      !std::isfinite(max_rate) || source_safety <= 0.0 || negative_radius <= 0.0 ||
      max_rate < 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  if (max_rate == 0.0) return std::numeric_limits<Real>::max();
  return source_safety * negative_radius / max_rate;
}

//! Select the two Z4c limits without applying the ordinary spatial CFL to the source cap.
inline Real SelectZ4cTimestep(const Real spatial_cfl, const Real dt_spatial,
                              const Real dt_source) {
  if (!std::isfinite(spatial_cfl) || !std::isfinite(dt_spatial) ||
      !std::isfinite(dt_source) || spatial_cfl <= 0.0 || dt_spatial <= 0.0 ||
      dt_source <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  return std::fmin(spatial_cfl * dt_spatial, dt_source);
}

inline Real BonaMassoCoordinateSpeed(const Real alpha, const Real lapse_f,
                                     const Real physical_inverse_diagonal) {
  if (!std::isfinite(alpha) || !std::isfinite(lapse_f) ||
      !std::isfinite(physical_inverse_diagonal) || alpha <= 0.0 || lapse_f < 0.0 ||
      physical_inverse_diagonal <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  return alpha * std::sqrt(lapse_f * physical_inverse_diagonal);
}

inline Real TelegraphCoordinateSpeed(const Real chi, const Real gradient,
                                     const Real conformal_inverse_diagonal) {
  if (!std::isfinite(chi) || !std::isfinite(gradient) ||
      !std::isfinite(conformal_inverse_diagonal) || chi <= 0.0 || gradient < 0.0 ||
      conformal_inverse_diagonal <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  return std::sqrt(chi * gradient * conformal_inverse_diagonal);
}

inline Real GammaDriverCoordinateSpeed(const Real gamma_driver_coefficient,
                                       const Real conformal_inverse_diagonal) {
  if (!std::isfinite(gamma_driver_coefficient) ||
      !std::isfinite(conformal_inverse_diagonal) || gamma_driver_coefficient < 0.0 ||
      conformal_inverse_diagonal <= 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  // Longitudinal Gamma-driver mode of Gamma_t = Delta beta + grad div(beta)/3.
  return std::sqrt((4.0 / 3.0) * gamma_driver_coefficient *
                   conformal_inverse_diagonal);
}

KOKKOS_INLINE_FUNCTION
Real CoordinateCharacteristicSpeed(const Real beta_normal, const Real light_speed,
                                   const Real lapse_speed, const Real telegraph_speed,
                                   const Real gamma_driver_speed) {
  if (!Kokkos::isfinite(beta_normal) || !Kokkos::isfinite(light_speed) ||
      !Kokkos::isfinite(lapse_speed) || !Kokkos::isfinite(telegraph_speed) ||
      !Kokkos::isfinite(gamma_driver_speed) || light_speed < 0.0 || lapse_speed < 0.0 ||
      telegraph_speed < 0.0 || gamma_driver_speed < 0.0) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  return Kokkos::fabs(beta_normal) +
         fmax(light_speed, fmax(lapse_speed, fmax(telegraph_speed, gamma_driver_speed)));
}

}  // namespace z4c

#endif  // Z4C_TIMESTEP_CONTRACT_HPP_
