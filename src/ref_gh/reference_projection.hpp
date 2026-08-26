//========================================================================================
//! \file reference_projection.hpp
//! \brief Project one physical coordinate geometry into a Ref-GH frame.
//========================================================================================
#ifndef REF_GH_REFERENCE_PROJECTION_HPP_
#define REF_GH_REFERENCE_PROJECTION_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct ProjectedFirstOrderMetric {
  Real psi[4][4];       // NOLINT(runtime/arrays)
  Real pi[4][4];        // NOLINT(runtime/arrays)
  Real phi[3][4][4];    // NOLINT(runtime/arrays)
  Real d_psi[4][4][4]; // NOLINT(runtime/arrays)
  bool valid;
};

struct ProjectedStationaryGaugeState {
  Real hhat[4];   // NOLINT(runtime/arrays)
  Real theta[4];  // NOLINT(runtime/arrays)
  bool valid;
};

// Project the same physical coordinate metric and coordinate first derivative
// into an arbitrary current reference tetrad.  This is a change of variables,
// not a change of physical initial data.
KOKKOS_INLINE_FUNCTION
ProjectedFirstOrderMetric ProjectPhysicalMetricToReference(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const ReferenceGeometry &reference) {
  ProjectedFirstOrderMetric result{};
  result.valid = false;
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
    return result;
  }
  const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real shift[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    shift[i] = lapse*lapse*inverse[0][i + 1];
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          result.psi[A][B] += reference.frame[A][a]
                              *reference.frame[B][b]*metric[a][b];
          for (int p = 0; p < 4; ++p) {
            result.d_psi[p][A][B] +=
                (reference.d_frame[p][A][a]*reference.frame[B][b]
                 + reference.frame[A][a]*reference.d_frame[p][B][b])
                    *metric[a][b]
                + reference.frame[A][a]*reference.frame[B][b]
                    *d_metric[p][a][b];
          }
        }
      }
      for (int I = 0; I < 3; ++I) {
        for (int i = 0; i < 3; ++i) {
          result.phi[I][A][B] +=
              reference.spatial_frame[I][i]
              *result.d_psi[i + 1][A][B];
        }
      }
      result.pi[A][B] = -result.d_psi[0][A][B]/lapse;
      for (int i = 0; i < 3; ++i) {
        result.pi[A][B] += shift[i]*result.d_psi[i + 1][A][B]/lapse;
      }
    }
  }
  result.valid = true;
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      result.valid = result.valid && Kokkos::isfinite(result.psi[A][B])
                     && Kokkos::isfinite(result.pi[A][B]);
      for (int I = 0; I < 3; ++I) {
        result.valid = result.valid
                       && Kokkos::isfinite(result.phi[I][A][B]);
      }
    }
  }
  return result;
}

// Project the stationary physical ordinary-GH source and improved-driver
// auxiliary into the current frame.  In coordinate components the stationary
// auxiliary is theta_a=-beta^i partial_i Hhat_a.  Treating both objects as
// physical covectors makes the initialization independent of the chosen
// reference representation, including a time-dependent current frame.
KOKKOS_INLINE_FUNCTION
ProjectedStationaryGaugeState ProjectStationaryPhysicalGaugeToReference(
    const ReferenceGeometry &physical,
    const ReferenceGeometry &current_reference) {
  ProjectedStationaryGaugeState result{};
  result.valid = false;
  CoordinateGhGeometry geometry;
  Real determinant = 0.0;
  if (!ComputeCoordinateGhGeometry(
          physical.metric, physical.d_metric, physical, geometry,
          determinant)) {
    return result;
  }
  Real d_hhat_coordinate[4][4] = {};  // NOLINT(runtime/arrays)
  ImplicitGaugeSourceDerivative(
      physical.metric, physical.d_metric, physical, geometry,
      d_hhat_coordinate);
  Real theta_coordinate[4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int i = 0; i < 3; ++i) {
      theta_coordinate[a] -=
          geometry.shift[i]*d_hhat_coordinate[i + 1][a];
    }
    for (int A = 0; A < 4; ++A) {
      result.hhat[A] += current_reference.frame[A][a]
                          *geometry.gauge_source[a];
      result.theta[A] += current_reference.frame[A][a]
                           *theta_coordinate[a];
    }
  }
  result.valid = true;
  for (int A = 0; A < 4; ++A) {
    result.valid = result.valid && Kokkos::isfinite(result.hhat[A])
                   && Kokkos::isfinite(result.theta[A]);
  }
  return result;
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_PROJECTION_HPP_
