//========================================================================================
//! \file reference_analytic_hot.hpp
//! \brief Symmetry-reduced analytic radial-q coefficients consumed by the GH source.
//========================================================================================
#ifndef REF_GH_REFERENCE_ANALYTIC_HOT_HPP_
#define REF_GH_REFERENCE_ANALYTIC_HOT_HPP_

#include "athena.hpp"
#include "ref_gh/reference_analytic_radial_q.hpp"
#include "ref_gh/reference_cache.hpp"

namespace ref_gh {

// This view deliberately contains only the three Cartan objects repeatedly
// consumed by the covariant scalar-wave source.  Frame/coframe/motion data stay
// in the accepted 12-static/8-stage analytic representation.  In particular,
// this is not a partial resurrection of the 1171-Real generic cache pipeline.
enum ReferenceAnalyticHotComponent : int {
  kAnalyticHotSpin = 0,                       // antisym_eta(4,4) x 4
  kAnalyticHotSpinDerivative = kAnalyticHotSpin + 24,
                                                   // 4 x antisym_eta(4,4) x 4
  kAnalyticHotRiemann = kAnalyticHotSpinDerivative + 96,
                                                   // sym(bivector,bivector)
  kReferenceAnalyticHotSize = kAnalyticHotRiemann + 21
};

static_assert(kReferenceAnalyticHotSize == 141,
              "analytic radial-q hot-reference layout changed");
static_assert(kReferenceAnalyticHotSize < 160,
              "analytic radial-q hot-reference budget exceeded");

struct AnalyticRadialQHotPoint {
  AnalyticRadialQPoint analytic;
  DvceArray5D<Real> hot;
  int m;
  int k;
  int j;
  int i;
};

KOKKOS_INLINE_FUNCTION
AnalyticRadialQHotPoint MakeAnalyticRadialQHotPoint(
    const DvceArray5D<Real> &reference_static,
    const DvceArray5D<Real> &reference_stage,
    const DvceArray5D<Real> &reference_hot,
    const int m, const int k, const int j, const int i,
    const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z) {
  return {MakeAnalyticRadialQPoint(
              reference_static, reference_stage, m, k, j, i, x, y, z,
              center_x, center_y, center_z),
          reference_hot, m, k, j, i};
}

KOKKOS_INLINE_FUNCTION
Real ReferenceCoframe(const AnalyticRadialQHotPoint &point,
                      const int A, const int a) {
  return ReferenceCoframe(point.analytic, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceFrame(const AnalyticRadialQHotPoint &point,
                    const int A, const int a) {
  return ReferenceFrame(point.analytic, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDFrame(const AnalyticRadialQHotPoint &point, const int p,
                     const int A, const int a) {
  return ReferenceDFrame(point.analytic, p, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDDFrame(const AnalyticRadialQHotPoint &point, const int p,
                      const int q, const int A, const int a) {
  return ReferenceDDFrame(point.analytic, p, q, A, a);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialFrame(const AnalyticRadialQHotPoint &point,
                           const int I, const int i) {
  return ReferenceSpatialFrame(point.analytic, I, i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialCoframe(const AnalyticRadialQHotPoint &point,
                             const int I, const int i) {
  return ReferenceSpatialCoframe(point.analytic, I, i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceDtSpatialFrame(const AnalyticRadialQHotPoint &point,
                             const int I, const int i) {
  return ReferenceDtSpatialFrame(point.analytic, I, i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceStructure(const AnalyticRadialQHotPoint &point,
                        const int I, const int J, const int K) {
  return ReferenceStructure(point.analytic, I, J, K);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpin(const AnalyticRadialQHotPoint &point, const int A,
                   const int B, const int C) {
  if (A == B) return 0.0;
  const int lower = (A < B) ? A : B;
  const int upper = (A < B) ? B : A;
  const Real eta_lower = (lower == 0) ? -1.0 : 1.0;
  const Real eta_upper = (upper == 0) ? -1.0 : 1.0;
  const Real orientation = (A < B) ? 1.0 : -eta_upper/eta_lower;
  return orientation*point.hot(
      point.m, kAnalyticHotSpin + 4*RefAntisymmetricPair4(A, B) + C,
      point.k, point.j, point.i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceSpinDerivative(const AnalyticRadialQHotPoint &point,
                             const int D, const int A, const int B,
                             const int C) {
  if (A == B) return 0.0;
  const int lower = (A < B) ? A : B;
  const int upper = (A < B) ? B : A;
  const Real eta_lower = (lower == 0) ? -1.0 : 1.0;
  const Real eta_upper = (upper == 0) ? -1.0 : 1.0;
  const Real orientation = (A < B) ? 1.0 : -eta_upper/eta_lower;
  return orientation*point.hot(
      point.m, kAnalyticHotSpinDerivative + 24*D
                   + 4*RefAntisymmetricPair4(A, B) + C,
      point.k, point.j, point.i);
}

KOKKOS_INLINE_FUNCTION
Real ReferenceRiemann(const AnalyticRadialQHotPoint &point, const int A,
                      const int B, const int C, const int D) {
  if (A == B || C == D) return 0.0;
  const Real first_orientation = (A < B) ? 1.0 : -1.0;
  const Real second_orientation = (C < D) ? 1.0 : -1.0;
  const Real eta_A = (A == 0) ? -1.0 : 1.0;
  const int first_pair = RefAntisymmetricPair4(A, B);
  const int second_pair = RefAntisymmetricPair4(C, D);
  const Real lower = point.hot(
      point.m,
      kAnalyticHotRiemann + RefSymmetricPair6(first_pair, second_pair),
      point.k, point.j, point.i);
  return eta_A*first_orientation*second_orientation*lower;
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_ANALYTIC_HOT_HPP_
