//========================================================================================
//! \file puncture_exponent.hpp
//! \brief Reference-independent local puncture-exponent diagnostics.
//========================================================================================
#ifndef REF_GH_PUNCTURE_EXPONENT_HPP_
#define REF_GH_PUNCTURE_EXPONENT_HPP_

#include "athena.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct LocalPunctureExponents {
  Real q;
  Real p;
  bool spatial_valid;
  bool lapse_valid;
};

// Evaluate the local physical exponents directly at one Cartesian point:
//
//   q_loc = -(1/6) X^k gamma^{ij} partial_k gamma_ij,
//   p_loc =          X^k partial_k ln(alpha).
//
// The four-metric and its derivatives are physical coordinate components.
// In production they are reconstructed from Psi/Phi and the reference jets, so
// q_loc requires no extra spatial finite difference.  The lapse expression is
// obtained from alpha=(-g^{00})^{-1/2} and is diagnostic-only for this phase.
KOKKOS_INLINE_FUNCTION
LocalPunctureExponents ComputeLocalPunctureExponents(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const Real displacement[3]) {
  LocalPunctureExponents result{NAN, NAN, false, false};

  Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
  Real spatial_determinant = 0.0;
  if (InvertSpatial3(metric, spatial_inverse, spatial_determinant)) {
    Real radial_log_determinant_derivative = 0.0;
    for (int k = 0; k < 3; ++k) {
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          radial_log_determinant_derivative +=
              displacement[k]*spatial_inverse[i][j]
              *d_metric[k + 1][i + 1][j + 1];
        }
      }
    }
    result.q = -radial_log_determinant_derivative/6.0;
    result.spatial_valid = Kokkos::isfinite(result.q);
  }

  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (Invert4(metric, inverse, determinant) && inverse[0][0] < 0.0) {
    Real radial_d_inverse_00 = 0.0;
    for (int k = 0; k < 3; ++k) {
      Real d_inverse_00 = 0.0;
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          d_inverse_00 -= inverse[0][a]*inverse[0][b]
                          *d_metric[k + 1][a][b];
        }
      }
      radial_d_inverse_00 += displacement[k]*d_inverse_00;
    }
    result.p = -0.5*radial_d_inverse_00/inverse[0][0];
    result.lapse_valid = Kokkos::isfinite(result.p);
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
Real PunctureEstimatorWeight(const Real radius, const Real h) {
  const Real ratio = 2.0*h/radius;
  return ratio*ratio*ratio;
}

KOKKOS_INLINE_FUNCTION
bool InPunctureEstimatorShell(const Real radius, const Real h,
                              const Real gaussian_width) {
  return radius >= 2.0*h && radius < 8.0*h
         && 8.0*h < 0.5*gaussian_width;
}

}  // namespace ref_gh

#endif  // REF_GH_PUNCTURE_EXPONENT_HPP_
