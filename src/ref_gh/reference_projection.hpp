//========================================================================================
//! \file reference_projection.hpp
//! \brief Project one physical coordinate geometry into a Ref-GH frame.
//========================================================================================
#ifndef REF_GH_REFERENCE_PROJECTION_HPP_
#define REF_GH_REFERENCE_PROJECTION_HPP_

#include "athena.hpp"
#include "ref_gh/generated/analytic_radial_q_gauge.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct ProjectedFirstOrderMetric {
  Real psi[4][4];       // NOLINT(runtime/arrays)
  Real pi[4][4];        // NOLINT(runtime/arrays)
  Real phi[3][4][4];    // NOLINT(runtime/arrays)
  bool valid;
};

struct ProjectedStationaryGaugeState {
  Real hhat[4];   // NOLINT(runtime/arrays)
  Real theta[4];  // NOLINT(runtime/arrays)
  bool valid;
};

// Project the same physical coordinate metric and coordinate first derivative
// into an arbitrary current reference tetrad.  This is a change of variables,
// not a change of physical initial data.  dPsi is deliberately kept as one
// four-component point scratch rather than as part of the returned object: no
// caller consumes it, and retaining all 64 components materially increases
// accelerator private memory in the physical-boundary kernel.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
ProjectedFirstOrderMetric ProjectPhysicalMetricToReferenceImpl(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const Reference &reference, const Real lapse, const Real shift[3]) {
  ProjectedFirstOrderMetric result{};
  result.valid = false;
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      Real d_psi[4] = {};  // NOLINT(runtime/arrays)
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          result.psi[A][B] += ReferenceFrame(reference, A, a)
                              *ReferenceFrame(reference, B, b)*metric[a][b];
          for (int p = 0; p < 4; ++p) {
            d_psi[p] +=
                (ReferenceDFrame(reference, p, A, a)
                   *ReferenceFrame(reference, B, b)
                 + ReferenceFrame(reference, A, a)
                   *ReferenceDFrame(reference, p, B, b))
                    *metric[a][b]
                + ReferenceFrame(reference, A, a)
                    *ReferenceFrame(reference, B, b)
                    *d_metric[p][a][b];
          }
        }
      }
      for (int I = 0; I < 3; ++I) {
        for (int i = 0; i < 3; ++i) {
          result.phi[I][A][B] +=
              ReferenceSpatialFrame(reference, I, i)*d_psi[i + 1];
        }
      }
      result.pi[A][B] = -d_psi[0]/lapse;
      for (int i = 0; i < 3; ++i) {
        result.pi[A][B] += shift[i]*d_psi[i + 1]/lapse;
      }
      if (A != B) {
        result.psi[B][A] = result.psi[A][B];
        result.pi[B][A] = result.pi[A][B];
        for (int I = 0; I < 3; ++I) {
          result.phi[I][B][A] = result.phi[I][A][B];
        }
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

KOKKOS_INLINE_FUNCTION
ProjectedFirstOrderMetric ProjectPhysicalMetricToReference(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const ReferenceGeometry &reference) {
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  ProjectedFirstOrderMetric invalid{};
  invalid.valid = false;
  if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
    return invalid;
  }
  const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real shift[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    shift[i] = lapse*lapse*inverse[0][i + 1];
  }
  return ProjectPhysicalMetricToReferenceImpl(
      metric, d_metric, reference, lapse, shift);
}

// Compact production projection for the q-controlled stationary physical
// trumpet.  The physical coordinate metric is evaluated directly from its
// radial two-jet; neither this path nor its caller materializes a
// ReferenceGeometry, spin, or Riemann object.
KOKKOS_INLINE_FUNCTION
ProjectedFirstOrderMetric ProjectAnalyticPhysicalMetricToReference(
    const AnalyticRadialQPoint &physical,
    const AnalyticRadialQPoint &current_reference) {
  Real metric[4][4];        // NOLINT(runtime/arrays)
  Real d_metric[4][4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = AnalyticMetric(physical, a, b);
      for (int p = 0; p < 4; ++p) {
        d_metric[p][a][b] = AnalyticDMetric(physical, p, a, b);
      }
    }
  }
  const Real lapse = physical.alpha.value;
  const Real shift[3] = {
      physical.b.value*physical.displacement[0],
      physical.b.value*physical.displacement[1],
      physical.b.value*physical.displacement[2]};
  return ProjectPhysicalMetricToReferenceImpl(
      metric, d_metric, current_reference, lapse, shift);
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

// Equation-identical compact form of the stationary gauge projection.  The
// generated physical baseline is first converted back to the coordinate
// covector and then projected into the current frame, exactly matching the
// generic ReferenceGeometry path above.
KOKKOS_INLINE_FUNCTION
ProjectedStationaryGaugeState
ProjectAnalyticStationaryPhysicalGaugeToReference(
    const AnalyticRadialQPoint &physical,
    const AnalyticRadialQPoint &current_reference) {
  ProjectedStationaryGaugeState result{};
  result.valid = false;
  const AnalyticRadialQGaugeBaseline physical_gauge =
      PopulateGeneratedAnalyticRadialQGauge(
          physical.alpha, physical.l, physical.b,
          physical.displacement, physical.radius);
  if (!physical_gauge.valid) return result;
  for (int a = 0; a < 4; ++a) {
    Real hhat_coordinate = 0.0;
    Real theta_coordinate = 0.0;
    for (int P = 0; P < 4; ++P) {
      hhat_coordinate += ReferenceCoframe(physical, P, a)
                         *physical_gauge.hhat[P];
      theta_coordinate += ReferenceCoframe(physical, P, a)
                          *physical_gauge.theta[P];
    }
    for (int A = 0; A < 4; ++A) {
      result.hhat[A] += ReferenceFrame(current_reference, A, a)
                        *hhat_coordinate;
      result.theta[A] += ReferenceFrame(current_reference, A, a)
                         *theta_coordinate;
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
