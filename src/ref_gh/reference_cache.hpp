//========================================================================================
//! \file reference_cache.hpp
//! \brief Compact device-side SoA cache for Ref-GH reference geometry.
//========================================================================================
#ifndef REF_GH_REFERENCE_CACHE_HPP_
#define REF_GH_REFERENCE_CACHE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"

namespace ref_gh {

// Components consumed by every RHS evaluation. Tensor symmetries are part of
// the storage contract; accessors below reconstruct arbitrary index orderings.
enum ReferenceEvolutionComponent : int {
  kRefCoframe = 0,                         // 4 x 4
  kRefFrame = kRefCoframe + 16,            // 4 x 4
  kRefDFrame = kRefFrame + 16,             // 4 x 4 x 4
  kRefChristoffel = kRefDFrame + 64,       // 4 x sym(4,4)
  kRefSpatialFrame = kRefChristoffel + 40, // 3 x 3
  kRefSpatialCoframe = kRefSpatialFrame + 9,
  kRefDtSpatialFrame = kRefSpatialCoframe + 9,
  kRefStructure = kRefDtSpatialFrame + 9,  // antisym(3,3) x 3
  kRefSpin = kRefStructure + 9,            // antisym_eta(4,4) x 4
  kRefSpinDerivative = kRefSpin + 24,      // 4 x antisym_eta(4,4) x 4
  kRefRiemann = kRefSpinDerivative + 96,   // sym(bivector(4),bivector(4))
  kReferenceEvolutionSize = kRefRiemann + 21
};

// Second derivatives are required while constructing spin derivatives and by
// the optional coordinate-source oracle, but are not read by the production
// covariant RHS. They therefore live outside the hot evolution cache.
enum ReferenceDiagnosticComponent : int {
  kRefDDFrame = 0,                         // sym(4,4) x 4 x 4
  kRefDChristoffel = kRefDDFrame + 160,    // 4 x 4 x sym(4,4)
  kRefRicci = kRefDChristoffel + 160,      // 4 x 4
  kRefDtTheta = kRefRicci + 16,            // 4 final + 8 update scratch
  kReferenceDiagnosticSize = kRefDtTheta + 12
};

// Provider/profile data are evaluated once per cell per RK stage. Each scalar
// jet is laid out as value, four first derivatives, sixteen second derivatives,
// and the twelve mixed third derivatives d_t d_i d_q needed by time-dependent
// gauge-reference subtraction.
enum ReferenceProviderComponent : int {
  kRefProviderAlpha = 0,
  kRefProviderPsi2 = kRefProviderAlpha + 33,
  kRefProviderShiftQ = kRefProviderPsi2 + 33,
  kRefProviderArealRadius = kRefProviderShiftQ + 33,
  kReferenceProviderSize = kRefProviderArealRadius + 1
};

// The first half of this array holds metric jets during connection assembly.
// Once the connection is complete, the same storage is reused for coframe and
// coordinate spin derivatives, so no separate per-stage scratch allocation is
// required.
enum ReferenceWorkspaceComponent : int {
  kRefWorkspaceMetricJet = 0,                 // sym(4,4) x 33
  kRefWorkspaceInverseMetricJet = kRefWorkspaceMetricJet + 330,
                                                // sym(4,4) x (1 + 4 + 3)
  kRefWorkspaceCoframeDerivative = 0,         // 4 x 4 x 4 (reuse)
  kRefWorkspaceSpinCoordinateDerivative = 64,// 4 x 6 x 4 (reuse)
  kReferenceWorkspaceSize = kRefWorkspaceInverseMetricJet + 80
};

static_assert(kReferenceEvolutionSize == 313,
              "Ref-GH production reference-cache layout changed");
static_assert(kReferenceDiagnosticSize == 348,
              "Ref-GH derivative/diagnostic cache layout changed");
static_assert(kReferenceProviderSize == 100,
              "Ref-GH provider cache layout changed");
static_assert(kReferenceWorkspaceSize == 410,
              "Ref-GH update workspace layout changed");

KOKKOS_INLINE_FUNCTION constexpr int RefMatrix4(const int offset,
                                                 const int a, const int b) {
  return offset + 4*a + b;
}
KOKKOS_INLINE_FUNCTION constexpr int RefRank3(const int offset, const int a,
                                               const int b, const int c) {
  return offset + 16*a + 4*b + c;
}
KOKKOS_INLINE_FUNCTION constexpr int RefMatrix3(const int offset,
                                                 const int i, const int j) {
  return offset + 3*i + j;
}
KOKKOS_INLINE_FUNCTION constexpr int RefSymmetricPair4(int a, int b) {
  if (a > b) { const int temporary = a; a = b; b = temporary; }
  return a*4 - a*(a - 1)/2 + b - a;
}
KOKKOS_INLINE_FUNCTION constexpr int RefAntisymmetricPair4(int a, int b) {
  if (a > b) { const int temporary = a; a = b; b = temporary; }
  return a*(7 - a)/2 + b - a - 1;
}
KOKKOS_INLINE_FUNCTION constexpr int RefAntisymmetricPair3(int a, int b) {
  if (a > b) { const int temporary = a; a = b; b = temporary; }
  return a*(5 - a)/2 + b - a - 1;
}
KOKKOS_INLINE_FUNCTION constexpr int RefSymmetricPair6(int a, int b) {
  if (a > b) { const int temporary = a; a = b; b = temporary; }
  return a*6 - a*(a - 1)/2 + b - a;
}
KOKKOS_INLINE_FUNCTION void RefDecodeSymmetricPair4(const int pair,
                                                     int &a, int &b) {
  constexpr int first[10] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
  constexpr int second[10] = {0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
  a = first[pair];
  b = second[pair];
}
KOKKOS_INLINE_FUNCTION void RefDecodeAntisymmetricPair4(const int pair,
                                                         int &a, int &b) {
  constexpr int first[6] = {0, 0, 0, 1, 1, 2};
  constexpr int second[6] = {1, 2, 3, 2, 3, 3};
  a = first[pair];
  b = second[pair];
}
KOKKOS_INLINE_FUNCTION void RefDecodeAntisymmetricPair3(const int pair,
                                                         int &a, int &b) {
  constexpr int first[3] = {0, 0, 1};
  constexpr int second[3] = {1, 2, 2};
  a = first[pair];
  b = second[pair];
}
KOKKOS_INLINE_FUNCTION void RefDecodeSymmetricPair6(const int pair,
                                                     int &a, int &b) {
  int remaining = pair;
  a = 0;
  while (remaining >= 6 - a) {
    remaining -= 6 - a;
    ++a;
  }
  b = a + remaining;
}
KOKKOS_INLINE_FUNCTION constexpr int RefJetDerivative(const int offset,
                                                       const int p) {
  return offset + 1 + p;
}
KOKKOS_INLINE_FUNCTION constexpr int RefJetSecondDerivative(const int offset,
                                                             const int p,
                                                             const int q) {
  return offset + 5 + 4*p + q;
}
KOKKOS_INLINE_FUNCTION constexpr int RefJetMixedTimeThirdDerivative(
    const int offset, const int i, const int q) {
  return offset + 21 + 4*i + q;
}

struct ReferenceCachePoint {
  DvceArray5D<Real> evolution;
  DvceArray5D<Real> diagnostic;
  int m;
  int k;
  int j;
  int i;
};

struct ReferenceProviderPoint {
  DvceArray5D<Real> provider;
  int m;
  int k;
  int j;
  int i;
};

struct ReferenceWorkspacePoint {
  DvceArray5D<Real> workspace;
  int m;
  int k;
  int j;
  int i;
};

KOKKOS_INLINE_FUNCTION
Real ReferenceCoframe(const ReferenceGeometry &r, const int A, const int a) {
  return r.coframe[A][a];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceCoframe(const ReferenceCachePoint &r, const int A, const int a) {
  return r.evolution(r.m, RefMatrix4(kRefCoframe, A, a), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceFrame(const ReferenceGeometry &r, const int A, const int a) {
  return r.frame[A][a];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceFrame(const ReferenceCachePoint &r, const int A, const int a) {
  return r.evolution(r.m, RefMatrix4(kRefFrame, A, a), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDFrame(const ReferenceGeometry &r, const int p, const int A,
                     const int a) {
  return r.d_frame[p][A][a];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDFrame(const ReferenceCachePoint &r, const int p, const int A,
                     const int a) {
  return r.evolution(r.m, RefRank3(kRefDFrame, p, A, a), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDDFrame(const ReferenceGeometry &r, const int p, const int q,
                      const int A, const int a) {
  return r.dd_frame[p][q][A][a];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDDFrame(const ReferenceCachePoint &r, const int p, const int q,
                      const int A, const int a) {
  return r.diagnostic(
      r.m, kRefDDFrame + 16*RefSymmetricPair4(p, q) + 4*A + a,
      r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceChristoffel(const ReferenceGeometry &r, const int a, const int b,
                          const int c) {
  return r.christoffel[a][b][c];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceChristoffel(const ReferenceCachePoint &r, const int a, const int b,
                          const int c) {
  return r.evolution(r.m, kRefChristoffel + 10*a + RefSymmetricPair4(b, c),
                     r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDChristoffel(const ReferenceGeometry &r, const int p, const int a,
                           const int b, const int c) {
  return r.d_christoffel[p][a][b][c];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDChristoffel(const ReferenceCachePoint &r, const int p, const int a,
                           const int b, const int c) {
  return r.diagnostic(
      r.m, kRefDChristoffel + 40*p + 10*a + RefSymmetricPair4(b, c),
      r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialFrame(const ReferenceGeometry &r, const int I, const int i) {
  return r.spatial_frame[I][i];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialFrame(const ReferenceCachePoint &r, const int I, const int i) {
  return r.evolution(r.m, RefMatrix3(kRefSpatialFrame, I, i), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialCoframe(const ReferenceGeometry &r, const int I, const int i) {
  return r.spatial_coframe[I][i];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpatialCoframe(const ReferenceCachePoint &r, const int I, const int i) {
  return r.evolution(r.m, RefMatrix3(kRefSpatialCoframe, I, i), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDtSpatialFrame(const ReferenceGeometry &r, const int I, const int i) {
  return r.dt_spatial_frame[I][i];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDtSpatialFrame(const ReferenceCachePoint &r, const int I, const int i) {
  return r.evolution(r.m, RefMatrix3(kRefDtSpatialFrame, I, i), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceStructure(const ReferenceGeometry &r, const int I, const int J,
                        const int K) {
  return r.structure[I][J][K];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceStructure(const ReferenceCachePoint &r, const int I, const int J,
                        const int K) {
  if (I == J) return 0.0;
  const Real orientation = (I < J) ? 1.0 : -1.0;
  return orientation*r.evolution(
      r.m, kRefStructure + 3*RefAntisymmetricPair3(I, J) + K,
      r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpin(const ReferenceGeometry &r, const int A, const int B,
                   const int C) {
  return r.spin[A][B][C];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpin(const ReferenceCachePoint &r, const int A, const int B,
                   const int C) {
  if (A == B) return 0.0;
  const int lower = (A < B) ? A : B;
  const int upper = (A < B) ? B : A;
  const Real eta_lower = (lower == 0) ? -1.0 : 1.0;
  const Real eta_upper = (upper == 0) ? -1.0 : 1.0;
  const Real orientation = (A < B) ? 1.0 : -eta_upper/eta_lower;
  return orientation*r.evolution(
      r.m, kRefSpin + 4*RefAntisymmetricPair4(A, B) + C, r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpinDerivative(const ReferenceGeometry &r, const int D,
                             const int A, const int B, const int C) {
  return r.spin_derivative[D][A][B][C];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpinDerivative(const ReferenceCachePoint &r, const int D,
                             const int A, const int B, const int C) {
  if (A == B) return 0.0;
  const int lower = (A < B) ? A : B;
  const int upper = (A < B) ? B : A;
  const Real eta_lower = (lower == 0) ? -1.0 : 1.0;
  const Real eta_upper = (upper == 0) ? -1.0 : 1.0;
  const Real orientation = (A < B) ? 1.0 : -eta_upper/eta_lower;
  return orientation*r.evolution(
      r.m, kRefSpinDerivative + 24*D
             + 4*RefAntisymmetricPair4(A, B) + C,
      r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceRiemann(const ReferenceGeometry &r, const int A, const int B,
                      const int C, const int D) {
  return r.riemann_frame[A][B][C][D];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceRiemann(const ReferenceCachePoint &r, const int A, const int B,
                      const int C, const int D) {
  if (A == B || C == D) return 0.0;
  const Real first_orientation = (A < B) ? 1.0 : -1.0;
  const Real second_orientation = (C < D) ? 1.0 : -1.0;
  const Real eta_A = (A == 0) ? -1.0 : 1.0;
  const int first_pair = RefAntisymmetricPair4(A, B);
  const int second_pair = RefAntisymmetricPair4(C, D);
  const Real lower = r.evolution(
      r.m, kRefRiemann + RefSymmetricPair6(first_pair, second_pair),
      r.k, r.j, r.i);
  return eta_A*first_orientation*second_orientation*lower;
}
KOKKOS_INLINE_FUNCTION
Real ReferenceRicci(const ReferenceGeometry &r, const int A, const int B) {
  return r.ricci_frame[A][B];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceRicci(const ReferenceCachePoint &r, const int A, const int B) {
  return r.diagnostic(r.m, RefMatrix4(kRefRicci, A, B), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDtTheta(const ReferenceCachePoint &r, const int A) {
  return r.diagnostic(r.m, kRefDtTheta + A, r.k, r.j, r.i);
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_CACHE_HPP_
