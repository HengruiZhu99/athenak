//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file telegraph_damping.hpp
//! \brief Scale-invariant local damping helpers for the telegrapher lapse.

#ifndef Z4C_TELEGRAPH_DAMPING_HPP_
#define Z4C_TELEGRAPH_DAMPING_HPP_

#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace z4c {

enum class TelegraphDampingPrescription : int {
  fixed = 0,
  max_domain_abs_K = 1,
  local_abs_K = 2,
  local_extrinsic_curvature_norm = 3,
  local_chi_gradient_norm = 4,
};

struct TelegraphCoefficients {
  Real damping;
  Real gradient;
};

//! Scale tau and kappa by the same domain curvature scale:
//! Q=mu/max|K|, tau_eff=tau/max|K|, kappa_eff=kappa/max|K|.
//! The cancelled form avoids a divide by zero on a time-symmetric slice.
KOKKOS_INLINE_FUNCTION
TelegraphCoefficients ScaleInvariantTelegraphCoefficients(
    const Real mu, const Real max_abs_K, const Real tau, const Real kappa) {
  (void)max_abs_K;
  return {mu / tau, kappa / tau};
}

//! Clamp only a roundoff-sized negative contraction. A materially negative
//! contraction remains fail-visible through sqrt(NaN), rather than a floor.
KOKKOS_INLINE_FUNCTION
Real RoundoffSafeNonnegativeSqrt(const Real value, const Real absolute_term_sum) {
  if (value >= 0.0) return Kokkos::sqrt(value);
  constexpr Real kRoundoffMultiplier = 64.0;
  const Real tolerance = kRoundoffMultiplier * std::numeric_limits<Real>::epsilon() *
                         absolute_term_sum;
  if (absolute_term_sum > 0.0 && -value <= tolerance) return 0.0;
  return Kokkos::sqrt(value);
}

KOKKOS_INLINE_FUNCTION
Real LocalAbsKTelegraphMu(const Real K) { return Kokkos::fabs(K); }

//! Compute sqrt(K_ij K^ij). With gamma_ij=chi^-1 gtilde_ij and
//! K_ij=chi^-1(Atilde_ij+K gtilde_ij/3), the conformal factors cancel:
//! K_ij K^ij=Atilde_ij Atilde^ij+K^2/3.
KOKKOS_INLINE_FUNCTION
Real LocalExtrinsicCurvatureNormTelegraphMu(
    const Real K,
    const Real gu_xx, const Real gu_xy, const Real gu_xz,
    const Real gu_yy, const Real gu_yz, const Real gu_zz,
    const Real A_xx, const Real A_xy, const Real A_xz,
    const Real A_yy, const Real A_yz, const Real A_zz) {
  const Real gu[3][3] = {
      {gu_xx, gu_xy, gu_xz},
      {gu_xy, gu_yy, gu_yz},
      {gu_xz, gu_yz, gu_zz}};
  const Real A[3][3] = {
      {A_xx, A_xy, A_xz},
      {A_xy, A_yy, A_yz},
      {A_xz, A_yz, A_zz}};
  Real norm_squared = K * K / 3.0;
  Real absolute_term_sum = Kokkos::fabs(K * K / 3.0);
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          const Real term = gu[a][b] * gu[c][d] * A[a][c] * A[d][b];
          norm_squared += term;
          absolute_term_sum += Kokkos::fabs(term);
        }
      }
    }
  }
  return RoundoffSafeNonnegativeSqrt(norm_squared, absolute_term_sum);
}

//! Compute sqrt(gamma^ij d_i chi d_j chi), using the physical inverse metric
//! gamma^ij=chi^(-4/chi_psi_power) gtilde^ij.
KOKKOS_INLINE_FUNCTION
Real LocalChiGradientNormTelegraphMu(
    const Real chi, const Real chi_psi_power,
    const Real gu_xx, const Real gu_xy, const Real gu_xz,
    const Real gu_yy, const Real gu_yz, const Real gu_zz,
    const Real dchi_x, const Real dchi_y, const Real dchi_z) {
  const Real gu[3][3] = {
      {gu_xx, gu_xy, gu_xz},
      {gu_xy, gu_yy, gu_yz},
      {gu_xz, gu_yz, gu_zz}};
  const Real dchi[3] = {dchi_x, dchi_y, dchi_z};
  const Real physical_inverse_factor =
      Kokkos::pow(chi, -4.0 / chi_psi_power);
  Real conformal_norm_squared = 0.0;
  Real absolute_term_sum = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      const Real term = gu[a][b] * dchi[a] * dchi[b];
      conformal_norm_squared += term;
      absolute_term_sum += Kokkos::fabs(term);
    }
  }
  const Real norm_squared = physical_inverse_factor * conformal_norm_squared;
  absolute_term_sum = Kokkos::fabs(physical_inverse_factor) * absolute_term_sum;
  return RoundoffSafeNonnegativeSqrt(norm_squared, absolute_term_sum);
}

}  // namespace z4c

#endif  // Z4C_TELEGRAPH_DAMPING_HPP_
