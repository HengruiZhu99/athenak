//========================================================================================
//! \file analytic_radial_q_production.hpp
//! \brief Production adapter for compact analytic radial-q reference data.
//========================================================================================
#ifndef REF_GH_ANALYTIC_RADIAL_Q_PRODUCTION_HPP_
#define REF_GH_ANALYTIC_RADIAL_Q_PRODUCTION_HPP_

#include "athena.hpp"
#include "ref_gh/analytic_radial_q_source.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/generated/analytic_radial_q_gauge.hpp"
#include "ref_gh/reference_analytic_hot.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_gauge_baseline.hpp"
#include "ref_gh/ref_gh_geometry.hpp"

namespace ref_gh {

// One compile-time-independent point handle keeps the established generic
// cache path available as the oracle while allowing production task code to
// dispatch to the 12+8 analytic coefficient views and the selective 141-Real
// hot view.  The analytic branch never reads the generic views; those views
// have zero allocation in that mode.
struct ProductionReferencePoint {
  int backend;
  ReferenceCachePoint generic;
  AnalyticRadialQPoint analytic;
  AnalyticRadialQHotPoint analytic_hot;
};

template <bool Analytic>
KOKKOS_INLINE_FUNCTION
auto MakeTypedProductionReferencePoint(
    const DvceArray5D<Real> &evolution,
    const DvceArray5D<Real> &diagnostic,
    const DvceArray5D<Real> &reference_static,
    const DvceArray5D<Real> &reference_stage,
    const DvceArray5D<Real> &reference_hot,
    const int m, const int k, const int j, const int i,
    const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z) {
  if constexpr (Analytic) {
    return MakeAnalyticRadialQHotPoint(
        reference_static, reference_stage, reference_hot, m, k, j, i, x, y, z,
        center_x, center_y, center_z);
  } else {
    return ReferenceCachePoint{evolution, diagnostic, m, k, j, i};
  }
}

// Compute one coordinate metric derivative without materializing the full
// dPsi/dg/CoordinateGhGeometry working set.  This is the same frame-derivative
// contraction used by LoadProductionPointGeometry, specialized to the single
// component needed by compact diagnostic kernels.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real ProductionCoordinateMetricDerivative(
    const DvceArray5D<Real> &state, const Reference &reference,
    const int m, const int k, const int j, const int i,
    const Real metric[4][4], const Real lapse, const Real shift[3],
    const int p, const int a, const int b) {
  Real derivative = 0.0;
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      Real d_psi = 0.0;
      if (p == 0) {
        d_psi = -lapse*state(m, PiIndex(A, B), k, j, i);
        for (int q = 0; q < 3; ++q) {
          Real spatial_d_psi = 0.0;
          for (int I = 0; I < 3; ++I) {
            spatial_d_psi += ReferenceSpatialCoframe(reference, I, q)
                *state(m, PhiIndex(I, A, B), k, j, i);
          }
          d_psi += shift[q]*spatial_d_psi;
        }
      } else {
        for (int I = 0; I < 3; ++I) {
          d_psi += ReferenceSpatialCoframe(reference, I, p - 1)
              *state(m, PhiIndex(I, A, B), k, j, i);
        }
      }
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          d_psi -=
              (ReferenceDFrame(reference, p, A, c)
                 *ReferenceFrame(reference, B, d)
               + ReferenceFrame(reference, A, c)
                 *ReferenceDFrame(reference, p, B, d))*metric[c][d];
        }
      }
      derivative += ReferenceCoframe(reference, A, a)
                    *ReferenceCoframe(reference, B, b)*d_psi;
    }
  }
  return derivative;
}

// Exact overload selected by the analytic CalcRHS instantiation.  This keeps
// the recursive frame-derivative implementation out of the PVC kernel image.
KOKKOS_INLINE_FUNCTION
Real ReferenceFrameMotion(const AnalyticRadialQPoint &point, const int A,
                          const int lambda, const int B) {
  return GeneratedAnalyticRadialQFrameMotion(
      point.alpha, point.l, point.b, point.displacement, point.radius,
      A, lambda, B);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceFrameMotion(const AnalyticRadialQHotPoint &point, const int A,
                          const int lambda, const int B) {
  return ReferenceFrameMotion(point.analytic, A, lambda, B);
}

KOKKOS_INLINE_FUNCTION
Real ProductionReferenceDtTheta(const AnalyticRadialQPoint &point,
                                const int A) {
  return PopulateGeneratedAnalyticRadialQGauge(
      point.alpha, point.l, point.b, point.displacement,
      point.radius).dt_theta[A];
}

KOKKOS_INLINE_FUNCTION
Real ProductionReferenceDtTheta(const AnalyticRadialQHotPoint &point,
                                const int A) {
  return ProductionReferenceDtTheta(point.analytic, A);
}

KOKKOS_INLINE_FUNCTION
Real ProductionReferenceDtTheta(const ReferenceCachePoint &point,
                                const int A) {
  return ReferenceDtTheta(point, A);
}

KOKKOS_INLINE_FUNCTION
ReferenceGaugeBaseline ComputeProductionReferenceGaugeBaseline(
    const AnalyticRadialQPoint &point) {
  const AnalyticRadialQGaugeBaseline generated =
      PopulateGeneratedAnalyticRadialQGauge(
          point.alpha, point.l, point.b, point.displacement, point.radius);
  ReferenceGaugeBaseline result{};
  result.valid = generated.valid;
  for (int A = 0; A < 4; ++A) {
    result.hhat[A] = generated.hhat[A];
    result.theta[A] = generated.theta[A];
    for (int p = 0; p < 4; ++p) {
      result.d_hhat[p][A] = generated.d_hhat[p][A];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
ReferenceGaugeBaseline ComputeProductionReferenceGaugeBaseline(
    const AnalyticRadialQHotPoint &point) {
  return ComputeProductionReferenceGaugeBaseline(point.analytic);
}

KOKKOS_INLINE_FUNCTION
ReferenceGaugeBaseline ComputeProductionReferenceGaugeBaseline(
    const ReferenceCachePoint &point) {
  return ComputeReferenceGaugeBaseline(point);
}

KOKKOS_INLINE_FUNCTION
ProductionReferencePoint MakeProductionReferencePoint(
    const int backend, const DvceArray5D<Real> &evolution,
    const DvceArray5D<Real> &diagnostic,
    const DvceArray5D<Real> &reference_static,
    const DvceArray5D<Real> &reference_stage,
    const DvceArray5D<Real> &reference_hot,
    const int m, const int k, const int j, const int i,
    const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z) {
  ProductionReferencePoint point{
      backend, {evolution, diagnostic, m, k, j, i}, {}, {}};
  if (backend == 1) {
    point.analytic = MakeAnalyticRadialQPoint(
        reference_static, reference_stage, m, k, j, i, x, y, z,
        center_x, center_y, center_z);
    point.analytic_hot = {
        point.analytic, reference_hot, m, k, j, i};
  }
  return point;
}

KOKKOS_INLINE_FUNCTION
Real ReferenceCoframe(const ProductionReferencePoint &point,
                      const int A, const int a) {
  return point.backend == 1 ? ReferenceCoframe(point.analytic, A, a)
                            : ReferenceCoframe(point.generic, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceFrame(const ProductionReferencePoint &point,
                    const int A, const int a) {
  return point.backend == 1 ? ReferenceFrame(point.analytic, A, a)
                            : ReferenceFrame(point.generic, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDFrame(const ProductionReferencePoint &point, const int p,
                     const int A, const int a) {
  return point.backend == 1 ? ReferenceDFrame(point.analytic, p, A, a)
                            : ReferenceDFrame(point.generic, p, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialFrame(const ProductionReferencePoint &point,
                           const int I, const int i) {
  return point.backend == 1 ? ReferenceSpatialFrame(point.analytic, I, i)
                            : ReferenceSpatialFrame(point.generic, I, i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialCoframe(const ProductionReferencePoint &point,
                             const int I, const int i) {
  return point.backend == 1 ? ReferenceSpatialCoframe(point.analytic, I, i)
                            : ReferenceSpatialCoframe(point.generic, I, i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDtSpatialFrame(const ProductionReferencePoint &point,
                             const int I, const int i) {
  return point.backend == 1 ? ReferenceDtSpatialFrame(point.analytic, I, i)
                            : ReferenceDtSpatialFrame(point.generic, I, i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceStructure(const ProductionReferencePoint &point,
                        const int I, const int J, const int K) {
  return point.backend == 1 ? ReferenceStructure(point.analytic, I, J, K)
                            : ReferenceStructure(point.generic, I, J, K);
}

// Explicit overload: the analytic production path consumes the generated
// frame-motion contraction instead of rebuilding it from derivative matrices.
KOKKOS_INLINE_FUNCTION
Real ReferenceFrameMotion(const ProductionReferencePoint &point, const int A,
                          const int lambda, const int B) {
  if (point.backend == 1) {
    return GeneratedAnalyticRadialQFrameMotion(
        point.analytic.alpha, point.analytic.l, point.analytic.b,
        point.analytic.displacement, point.analytic.radius, A, lambda, B);
  }
  Real value = 0.0;
  for (int a = 0; a < 4; ++a) {
    value += ReferenceDFrame(point.generic, lambda, A, a)
             *ReferenceCoframe(point.generic, B, a);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real ProductionReferenceDtTheta(const ProductionReferencePoint &point,
                                const int A) {
  if (point.backend == 1) {
    return PopulateGeneratedAnalyticRadialQGauge(
        point.analytic.alpha, point.analytic.l, point.analytic.b,
        point.analytic.displacement, point.analytic.radius).dt_theta[A];
  }
  return ReferenceDtTheta(point.generic, A);
}

KOKKOS_INLINE_FUNCTION
ReferenceGaugeBaseline ComputeProductionReferenceGaugeBaseline(
    const ProductionReferencePoint &point) {
  if (point.backend == 0) return ComputeReferenceGaugeBaseline(point.generic);
  const AnalyticRadialQGaugeBaseline generated =
      PopulateGeneratedAnalyticRadialQGauge(
          point.analytic.alpha, point.analytic.l, point.analytic.b,
          point.analytic.displacement, point.analytic.radius);
  ReferenceGaugeBaseline result{};
  result.valid = generated.valid;
  for (int A = 0; A < 4; ++A) {
    result.hhat[A] = generated.hhat[A];
    result.theta[A] = generated.theta[A];
    for (int p = 0; p < 4; ++p) {
      result.d_hhat[p][A] = generated.d_hhat[p][A];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
bool LoadProductionPointGeometry(
    const DvceArray5D<Real> &state, const ProductionReferencePoint &reference,
    const int m, const int k, const int j, const int i,
    Real psi[4][4], Real pi[4][4], Real phi[3][4][4],
    Real d_psi[4][4][4], Real metric[4][4], Real d_metric[4][4][4],
    CoordinateGhGeometry &geometry, Real &determinant) {
  if (reference.backend == 0) {
    return LoadPointGeometry(state, reference.generic, m, k, j, i, psi, pi,
                             phi, d_psi, metric, d_metric, geometry,
                             determinant);
  }
  LoadSymmetric(state, kPsiOffset, m, k, j, i, psi);
  LoadSymmetric(state, kPiOffset, m, k, j, i, pi);
  for (int p = 0; p < 3; ++p) {
    for (int A = 0; A < 4; ++A) {
      for (int B = A; B < 4; ++B) {
        phi[p][A][B] = phi[p][B][A] =
            state(m, PhiIndex(p, A, B), k, j, i);
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = 0.0;
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          metric[a][b] += ReferenceCoframe(reference.analytic, A, a)
                          *ReferenceCoframe(reference.analytic, B, b)
                          *psi[A][B];
        }
      }
    }
  }
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
    return false;
  }
  const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real shift[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) shift[p] = lapse*lapse*inverse[0][p + 1];
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int p = 0; p < 3; ++p) {
        d_psi[p + 1][A][B] = 0.0;
        for (int I = 0; I < 3; ++I) {
          d_psi[p + 1][A][B] +=
              ReferenceSpatialCoframe(reference.analytic, I, p)*phi[I][A][B];
        }
      }
      d_psi[0][A][B] = -lapse*pi[A][B];
      for (int p = 0; p < 3; ++p) {
        d_psi[0][A][B] += shift[p]*d_psi[p + 1][A][B];
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    Real frame_corrected[4][4];  // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        frame_corrected[A][B] = d_psi[p][A][B];
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            frame_corrected[A][B] -=
                (ReferenceDFrame(reference.analytic, p, A, a)
                   *ReferenceFrame(reference.analytic, B, b)
                 + ReferenceFrame(reference.analytic, A, a)
                   *ReferenceDFrame(reference.analytic, p, B, b))*metric[a][b];
          }
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_metric[p][a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            d_metric[p][a][b] +=
                ReferenceCoframe(reference.analytic, A, a)
                *ReferenceCoframe(reference.analytic, B, b)
                *frame_corrected[A][B];
          }
        }
      }
    }
  }
  CompactAnalyticCoordinateGeometry compact;
  if (!ComputeCompactAnalyticCoordinateGeometry(
          metric, d_metric, reference.analytic, compact, determinant)) {
    return false;
  }
  geometry = compact.geometry;
  return true;
}

KOKKOS_INLINE_FUNCTION
bool LoadProductionPointGeometry(
    const DvceArray5D<Real> &state, const ReferenceCachePoint &reference,
    const int m, const int k, const int j, const int i,
    Real psi[4][4], Real pi[4][4], Real phi[3][4][4],
    Real d_psi[4][4][4], Real metric[4][4], Real d_metric[4][4][4],
    CoordinateGhGeometry &geometry, Real &determinant) {
  return LoadPointGeometry(state, reference, m, k, j, i, psi, pi, phi,
                           d_psi, metric, d_metric, geometry, determinant);
}

KOKKOS_INLINE_FUNCTION
bool LoadProductionPointGeometry(
    const DvceArray5D<Real> &state, const AnalyticRadialQPoint &reference,
    const int m, const int k, const int j, const int i,
    Real psi[4][4], Real pi[4][4], Real phi[3][4][4],
    Real d_psi[4][4][4], Real metric[4][4], Real d_metric[4][4][4],
    CoordinateGhGeometry &geometry, Real &determinant) {
  const ProductionReferencePoint wrapper{1, {}, reference};
  return LoadProductionPointGeometry(
      state, wrapper, m, k, j, i, psi, pi, phi, d_psi, metric, d_metric,
      geometry, determinant);
}

KOKKOS_INLINE_FUNCTION
bool LoadProductionPointGeometry(
    const DvceArray5D<Real> &state, const AnalyticRadialQHotPoint &reference,
    const int m, const int k, const int j, const int i,
    Real psi[4][4], Real pi[4][4], Real phi[3][4][4],
    Real d_psi[4][4][4], Real metric[4][4], Real d_metric[4][4][4],
    CoordinateGhGeometry &geometry, Real &determinant) {
  return LoadProductionPointGeometry(
      state, reference.analytic, m, k, j, i, psi, pi, phi, d_psi, metric,
      d_metric, geometry, determinant);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const ProductionReferencePoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4]) {
  return reference.backend == 1
      ? CovariantGhScalarWaveSourceProduction(
            psi, pi, phi, reference.analytic_hot, geometry, gamma0, source)
      : CovariantGhScalarWaveSourceProduction(
            psi, pi, phi, reference.generic, geometry, gamma0, source);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const AnalyticRadialQPoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4]) {
  // Oracle/debug overload only; production task dispatch uses the hot point.
  return CompactAnalyticRadialQScalarWaveSource(
      psi, pi, phi, reference, geometry, gamma0, source);
}

// Qualified loop-form production source.  The 141-Real point supplies only
// cached Cartan coefficients; frame/coframe data remain analytic 12+8 values.
// The monolithic joint-CSE source above remains an independent oracle.
KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const AnalyticRadialQHotPoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4]) {
  return CovariantGhScalarWaveSourceProduction(
      psi, pi, phi, reference, geometry, gamma0, source);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const ReferenceCachePoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4]) {
  return CovariantGhScalarWaveSourceProduction(
      psi, pi, phi, reference, geometry, gamma0, source);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSourceDiagnostics(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const ProductionReferencePoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4], CovariantSourceSectors &sectors) {
  return reference.backend == 1
      ? CovariantGhScalarWaveSource(
            psi, pi, phi, reference.analytic_hot, geometry, gamma0, source,
            sectors)
      : CovariantGhScalarWaveSource(
            psi, pi, phi, reference.generic, geometry, gamma0, source,
            sectors);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSourceDiagnostics(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const AnalyticRadialQPoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4], CovariantSourceSectors &sectors) {
  // Oracle/debug overload only.  Production task dispatch has the hot-point
  // type and cannot select this monolithic generated contraction.
  return CompactAnalyticRadialQScalarWaveSource(
      psi, pi, phi, reference, geometry, gamma0, source, &sectors);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSourceDiagnostics(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const AnalyticRadialQHotPoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4], CovariantSourceSectors &sectors) {
  return CovariantGhScalarWaveSource(
      psi, pi, phi, reference, geometry, gamma0, source, sectors);
}

KOKKOS_INLINE_FUNCTION
bool ProductionCovariantScalarWaveSourceDiagnostics(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const ReferenceCachePoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4], CovariantSourceSectors &sectors) {
  return CovariantGhScalarWaveSource(
      psi, pi, phi, reference, geometry, gamma0, source, sectors);
}

KOKKOS_INLINE_FUNCTION
void AddProductionOrdinaryGaugeSource(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const ProductionReferencePoint &reference,
    const CoordinateGhGeometry &geometry, const Real hhat[4],
    const Real d_hhat[4][4], const Real gamma0, Real source[4][4]) {
  if (reference.backend == 0) {
    AddOrdinaryGaugePartialWaveSource(metric, d_metric, reference.generic,
                                      geometry, hhat, d_hhat, gamma0, source);
    return;
  }
  CompactAnalyticCoordinateGeometry compact;
  Real determinant = 0.0;
  if (!ComputeCompactAnalyticCoordinateGeometry(
          metric, d_metric, reference.analytic, compact, determinant)) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) source[a][b] = NAN;
    }
    return;
  }
  AddCompactAnalyticOrdinaryGaugeSource(
      metric, d_metric, reference.analytic, compact, hhat, d_hhat, gamma0,
      source);
}

KOKKOS_INLINE_FUNCTION
void AddProductionOrdinaryGaugeSource(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const AnalyticRadialQPoint &reference,
    const CoordinateGhGeometry &geometry, const Real hhat[4],
    const Real d_hhat[4][4], const Real gamma0, Real source[4][4]) {
  CompactAnalyticCoordinateGeometry compact;
  Real determinant = 0.0;
  if (!ComputeCompactAnalyticCoordinateGeometry(
          metric, d_metric, reference, compact, determinant)) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) source[a][b] = NAN;
    }
    return;
  }
  AddCompactAnalyticOrdinaryGaugeSource(
      metric, d_metric, reference, compact, hhat, d_hhat, gamma0, source);
}

KOKKOS_INLINE_FUNCTION
void AddProductionOrdinaryGaugeSource(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const AnalyticRadialQHotPoint &reference,
    const CoordinateGhGeometry &geometry, const Real hhat[4],
    const Real d_hhat[4][4], const Real gamma0, Real source[4][4]) {
  AddProductionOrdinaryGaugeSource(
      metric, d_metric, reference.analytic, geometry, hhat, d_hhat, gamma0,
      source);
}

KOKKOS_INLINE_FUNCTION
void AddProductionOrdinaryGaugeSource(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const ReferenceCachePoint &reference,
    const CoordinateGhGeometry &geometry, const Real hhat[4],
    const Real d_hhat[4][4], const Real gamma0, Real source[4][4]) {
  AddOrdinaryGaugePartialWaveSource(metric, d_metric, reference, geometry,
                                    hhat, d_hhat, gamma0, source);
}

}  // namespace ref_gh

#endif  // REF_GH_ANALYTIC_RADIAL_Q_PRODUCTION_HPP_
