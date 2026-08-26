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

// Evaluate the physical/reference spatial-exponent mismatch directly from the
// first-order Ref-GH state,
//
//   epsilon_G = -(1/6) X^k (G^-1)^IJ bar_theta^K_k Phi_KIJ.
//
// Here G_IJ is the spatial block of Psi_AB.  This is a diagnostic independent
// of the coordinate-metric reconstruction used by q_loc; no coordinate-space
// finite difference or interpolation is involved.
KOKKOS_INLINE_FUNCTION
bool ComputeRelativeSpatialExponentMismatch(
    const Real relative_metric[4][4], const Real phi[3][4][4],
    const Real reference_spatial_coframe[3][3],
    const Real displacement[3], Real &epsilon_g) {
  Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
  Real spatial_determinant = 0.0;
  if (!InvertSpatial3(relative_metric, spatial_inverse,
                      spatial_determinant)) {
    epsilon_g = NAN;
    return false;
  }
  Real radial_log_determinant_derivative = 0.0;
  for (int k = 0; k < 3; ++k) {
    for (int I = 0; I < 3; ++I) {
      for (int J = 0; J < 3; ++J) {
        for (int K = 0; K < 3; ++K) {
          radial_log_determinant_derivative +=
              displacement[k]*spatial_inverse[I][J]
              *reference_spatial_coframe[K][k]
              *phi[K][I + 1][J + 1];
        }
      }
    }
  }
  epsilon_g = -radial_log_determinant_derivative/6.0;
  return Kokkos::isfinite(epsilon_g);
}

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
                              const Real /*gaussian_width*/) {
  return radius >= 2.0*h && radius < 8.0*h;
}

// Treat the union of centered directional stencils by its axis-aligned support
// box.  A sample is puncture-overlapping only when the puncture lies inside
// that complete box, i.e. every coordinate displacement is within the stencil
// reach.  In particular, a point far from the puncture is not rejected merely
// because it lies close to one coordinate plane through the puncture.
KOKKOS_INLINE_FUNCTION
bool PunctureStencilIsClear(const Real displacement[3], const Real spacing[3],
                            const int stencil_radius) {
  for (int p = 0; p < 3; ++p) {
    const Real reach = static_cast<Real>(stencil_radius)*spacing[p];
    if (Kokkos::abs(displacement[p]) > reach) return true;
  }
  return false;
}

KOKKOS_INLINE_FUNCTION
bool PunctureStencilIsClear(const Real displacement[3], const Real h,
                            const int stencil_radius) {
  const Real spacing[3] = {h, h, h};
  return PunctureStencilIsClear(displacement, spacing, stencil_radius);
}

// The centered first derivative for order 2p reaches p cells.  AthenaK's
// matching Kreiss-Oliger operator Diss<p+1> reaches one cell farther whenever
// it is enabled.  A diagnostic that promises to exclude every point whose
// evolved domain of dependence touches the puncture must use this larger
// footprint, not just the derivative stencil.
KOKKOS_INLINE_FUNCTION
int PunctureEvolutionStencilRadius(const int fd_order,
                                   const Real dissipation) {
  return fd_order/2 + ((dissipation > 0.0) ? 1 : 0);
}

}  // namespace ref_gh

#endif  // REF_GH_PUNCTURE_EXPONENT_HPP_
