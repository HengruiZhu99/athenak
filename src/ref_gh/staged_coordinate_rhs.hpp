//========================================================================================
//! \file staged_coordinate_rhs.hpp
//! \brief Componentized standard-coordinate GH source and analytic frame transform.
//========================================================================================
#ifndef REF_GH_STAGED_COORDINATE_RHS_HPP_
#define REF_GH_STAGED_COORDINATE_RHS_HPP_

#include "athena.hpp"
#include "ref_gh/analytic_radial_q_source.hpp"
#include "ref_gh/ref_gh_state.hpp"
#include "ref_gh/staged_covariant_rhs.hpp"

namespace ref_gh {

// The coordinate discriminator deliberately uses the already-qualified
// standard-GH partial-wave equation and analytic frame transformation.  It is
// an equation-preserving alternative to the covariant contraction, not a new
// formulation.  Metric symmetry is packed throughout.
enum StagedCoordinateComponent : int {
  kStagedCoordinateMetric = 0,                    // sym(4) = 10
  kStagedCoordinateDMetric = kStagedCoordinateMetric + 10,  // 4 x sym(4) = 40
  kStagedCoordinateDReferenceGauge = kStagedCoordinateDMetric + 40,  // 4 x 4
  kStagedCoordinateGaugeSource = kStagedCoordinateDReferenceGauge + 16,  // 4
  kStagedCoordinateGaugeConstraint = kStagedCoordinateGaugeSource + 4,   // 4
  kStagedCoordinateFinalSource = kStagedCoordinateGaugeConstraint + 4,   // sym(4)
  kStagedCoordinateSize = kStagedCoordinateFinalSource + 10
};

static_assert(kStagedCoordinateSize == 84,
              "staged coordinate source must remain exactly 84 Reals/cell");
static_assert(kStagedPhysicalSize + kStagedCoordinateSize
                  + kSymmetric4Size == 126,
              "coordinate discriminator must stay below 128 transient Reals/cell");

struct LocalStagedCoordinatePoint {
  Real metric[10];                    // NOLINT(runtime/arrays)
  Real inverse[10];                   // NOLINT(runtime/arrays)
  Real d_metric[40];                  // NOLINT(runtime/arrays)
  Real d_reference_gauge[16];         // NOLINT(runtime/arrays)
  Real gauge_source[4];               // NOLINT(runtime/arrays)
  Real gauge_constraint[4];           // NOLINT(runtime/arrays)
  Real normal_upper[4];               // NOLINT(runtime/arrays)
  Real normal_lower[4];               // NOLINT(runtime/arrays)
  Real coordinate_source[10];         // NOLINT(runtime/arrays)

  KOKKOS_INLINE_FUNCTION Real Metric(const int a, const int b) const {
    return metric[Symmetric4Index(a, b)];
  }
  KOKKOS_INLINE_FUNCTION Real Inverse(const int a, const int b) const {
    return inverse[Symmetric4Index(a, b)];
  }
  KOKKOS_INLINE_FUNCTION Real DMetric(const int p, const int a,
                                      const int b) const {
    return d_metric[10*p + Symmetric4Index(a, b)];
  }
  KOKKOS_INLINE_FUNCTION Real DReferenceGauge(const int p, const int a) const {
    return d_reference_gauge[4*p + a];
  }
  KOKKOS_INLINE_FUNCTION Real GaugeSource(const int a) const {
    return gauge_source[a];
  }
  KOKKOS_INLINE_FUNCTION Real GaugeConstraint(const int a) const {
    return gauge_constraint[a];
  }
  KOKKOS_INLINE_FUNCTION Real NormalUpper(const int a) const {
    return normal_upper[a];
  }
  KOKKOS_INLINE_FUNCTION Real NormalLower(const int a) const {
    return normal_lower[a];
  }
  KOKKOS_INLINE_FUNCTION Real CoordinateSource(const int a, const int b) const {
    return coordinate_source[Symmetric4Index(a, b)];
  }
  KOKKOS_INLINE_FUNCTION void SetCoordinateSource(const int a, const int b,
                                                   const Real value) {
    coordinate_source[Symmetric4Index(a, b)] = value;
  }
};

struct DeviceStagedCoordinatePoint {
  DvceArray5D<Real> coordinate;
  DvceArray5D<Real> physical;
  DvceArray5D<Real> coordinate_source;
  int m;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real Metric(const int a, const int b) const {
    return coordinate(
        m, kStagedCoordinateMetric + Symmetric4Index(a, b), k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real Inverse(const int a, const int b) const {
    const Real lapse = physical(m, kStagedLapse, k, j, i);
    const Real inverse_lapse_sq = 1.0/(lapse*lapse);
    if (a == 0 && b == 0) return -inverse_lapse_sq;
    if (a == 0 || b == 0) {
      const int p = (a == 0 ? b : a) - 1;
      return physical(m, kStagedShift + p, k, j, i)*inverse_lapse_sq;
    }
    const Real beta_a = physical(m, kStagedShift + a - 1, k, j, i);
    const Real beta_b = physical(m, kStagedShift + b - 1, k, j, i);
    return physical(
        m, kStagedInverseSpatial + Symmetric3Index(a - 1, b - 1), k, j, i)
        - beta_a*beta_b*inverse_lapse_sq;
  }
  KOKKOS_INLINE_FUNCTION Real DMetric(const int p, const int a,
                                      const int b) const {
    return coordinate(
        m, kStagedCoordinateDMetric + 10*p + Symmetric4Index(a, b), k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real DReferenceGauge(const int p, const int a) const {
    return coordinate(
        m, kStagedCoordinateDReferenceGauge + 4*p + a, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real GaugeSource(const int a) const {
    return coordinate(m, kStagedCoordinateGaugeSource + a, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real GaugeConstraint(const int a) const {
    return coordinate(m, kStagedCoordinateGaugeConstraint + a, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real NormalUpper(const int a) const {
    const Real lapse = physical(m, kStagedLapse, k, j, i);
    return a == 0 ? 1.0/lapse
                  : -physical(m, kStagedShift + a - 1, k, j, i)/lapse;
  }
  KOKKOS_INLINE_FUNCTION Real NormalLower(const int a) const {
    return a == 0 ? -physical(m, kStagedLapse, k, j, i) : 0.0;
  }
  KOKKOS_INLINE_FUNCTION Real CoordinateSource(const int a, const int b) const {
    return coordinate_source(m, Symmetric4Index(a, b), k, j, i);
  }
  KOKKOS_INLINE_FUNCTION void SetFinalSource(const int A, const int B,
                                              const Real value) const {
    coordinate(
        m, kStagedCoordinateFinalSource + Symmetric4Index(A, B), k, j, i)
        = value;
  }
  KOKKOS_INLINE_FUNCTION Real FinalSource(const int A, const int B) const {
    return coordinate(
        m, kStagedCoordinateFinalSource + Symmetric4Index(A, B), k, j, i);
  }
};

KOKKOS_INLINE_FUNCTION
bool PrepareLocalStagedCoordinatePoint(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const CompactAnalyticCoordinateGeometry &compact,
    LocalStagedCoordinatePoint &point) {
  Real determinant = 0.0;
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  if (!Invert4(metric, inverse, determinant)) return false;
  for (int a = 0; a < 4; ++a) {
    point.normal_upper[a] = compact.geometry.normal_upper[a];
    point.normal_lower[a] = compact.geometry.normal_lower[a];
    point.gauge_source[a] = compact.geometry.gauge_source[a];
    point.gauge_constraint[a] = compact.geometry.gauge_constraint[a];
    for (int b = a; b < 4; ++b) {
      point.metric[Symmetric4Index(a, b)] = metric[a][b];
      point.inverse[Symmetric4Index(a, b)] = inverse[a][b];
    }
    for (int p = 0; p < 4; ++p) {
      point.d_reference_gauge[4*p + a] = compact.d_reference_gauge[p][a];
    }
  }
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        point.d_metric[10*p + Symmetric4Index(a, b)] = d_metric[p][a][b];
      }
    }
  }
  return true;
}

template <typename Point>
KOKKOS_INLINE_FUNCTION
Real StagedCoordinateChristoffelFirst(
    const Point &point, const int a, const int b, const int c) {
  return 0.5*(point.DMetric(b, a, c) + point.DMetric(c, a, b)
              - point.DMetric(a, b, c));
}

template <typename Point>
KOKKOS_INLINE_FUNCTION
Real StagedCoordinateChristoffelUpper(
    const Point &point, const int a, const int b, const int c) {
  Real value = 0.0;
  for (int d = 0; d < 4; ++d) {
    value += point.Inverse(a, d)
             *StagedCoordinateChristoffelFirst(point, d, b, c);
  }
  return value;
}

template <typename Point>
KOKKOS_INLINE_FUNCTION
Real StagedCoordinateContractedUpper(const Point &point, const int a) {
  Real value = 0.0;
  for (int b = 0; b < 4; ++b) {
    for (int c = 0; c < 4; ++c) {
      value += point.Inverse(b, c)
               *StagedCoordinateChristoffelUpper(point, a, b, c);
    }
  }
  return value;
}

// One work item owns one coordinate symmetric pair ab.  The loop ordering and
// terms are the component form of StandardGhPartialWaveSource.
template <typename Point>
KOKKOS_INLINE_FUNCTION
Real StagedCoordinateStandardSourceComponent(
    const Point &point, const int a, const int b, const Real gamma0) {
  Real nabla_h_ab = point.DReferenceGauge(a, b);
  Real nabla_h_ba = point.DReferenceGauge(b, a);
  for (int c = 0; c < 4; ++c) {
    const Real christoffel = StagedCoordinateChristoffelUpper(point, c, a, b);
    nabla_h_ab -= christoffel*point.GaugeSource(c);
    nabla_h_ba -= christoffel*point.GaugeSource(c);
  }
  Real value = -nabla_h_ab - nabla_h_ba;
  for (int c = 0; c < 4; ++c) {
    for (int d = 0; d < 4; ++d) {
      for (int e = 0; e < 4; ++e) {
        for (int f = 0; f < 4; ++f) {
          value += 2.0*point.Inverse(c, d)*point.Inverse(e, f)
                   *(point.DMetric(e, c, a)*point.DMetric(f, d, b)
                     - StagedCoordinateChristoffelFirst(point, a, c, e)
                       *StagedCoordinateChristoffelFirst(point, b, d, f));
        }
      }
    }
  }
  for (int c = 0; c < 4; ++c) {
    const Real projector = ((c == a) ? point.NormalLower(b) : 0.0)
                           + ((c == b) ? point.NormalLower(a) : 0.0)
                           - point.Metric(a, b)*point.NormalUpper(c);
    value += gamma0*projector*point.GaugeConstraint(c);
  }
  return value;
}

// One work item owns one reference-frame symmetric pair AB.  The coordinate
// partial source was completed by the preceding flat kernel and is read-only.
template <typename Point, typename Reference>
KOKKOS_INLINE_FUNCTION
Real StagedCoordinateTransformComponent(
    const Point &point, const Reference &reference,
    const Real d_psi[4], const int A, const int B) {
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      const Real frame_A = ReferenceFrame(reference, A, a);
      const Real frame_B = ReferenceFrame(reference, B, b);
      value += frame_A*frame_B*point.CoordinateSource(a, b);
      for (int c = 0; c < 4; ++c) {
        const Real d_frame_A = ReferenceDFrame(reference, c, A, a);
        const Real d_frame_B = ReferenceDFrame(reference, c, B, b);
        const Real d_tensor = d_frame_A*frame_B + frame_A*d_frame_B;
        for (int d = 0; d < 4; ++d) {
          const Real dd_tensor =
              ReferenceDDFrame(reference, c, d, A, a)*frame_B
              + d_frame_A*ReferenceDFrame(reference, d, B, b)
              + ReferenceDFrame(reference, d, A, a)*d_frame_B
              + frame_A*ReferenceDDFrame(reference, c, d, B, b);
          value += 2.0*point.Inverse(c, d)*d_tensor*point.DMetric(d, a, b)
                   + point.Inverse(c, d)*dd_tensor*point.Metric(a, b);
        }
      }
    }
  }
  for (int c = 0; c < 4; ++c) {
    value -= StagedCoordinateContractedUpper(point, c)*d_psi[c];
  }
  return value;
}

// Point-local oracle for the two production flat kernels.  The retained
// covariant source remains the independent algebraic oracle.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool StagedCoordinateScalarWaveSource(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const Real d_psi[4][4][4], const Reference &reference,
    const CompactAnalyticCoordinateGeometry &compact, const Real gamma0,
    Real source[4][4]) {
  LocalStagedCoordinatePoint point{};
  if (!PrepareLocalStagedCoordinatePoint(metric, d_metric, compact, point)) {
    return false;
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      point.SetCoordinateSource(
          a, b, StagedCoordinateStandardSourceComponent(point, a, b, gamma0));
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      Real derivative[4];  // NOLINT(runtime/arrays)
      for (int c = 0; c < 4; ++c) derivative[c] = d_psi[c][A][B];
      const Real value = StagedCoordinateTransformComponent(
          point, reference, derivative, A, B);
      source[A][B] = source[B][A] = value;
    }
  }
  return true;
}

}  // namespace ref_gh

#endif  // REF_GH_STAGED_COORDINATE_RHS_HPP_
