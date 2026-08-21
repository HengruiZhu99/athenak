//========================================================================================
//! \file reference_cache.hpp
//! \brief Device-side structure-of-arrays cache for Ref-GH reference geometry.
//========================================================================================
#ifndef REF_GH_REFERENCE_CACHE_HPP_
#define REF_GH_REFERENCE_CACHE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"

namespace ref_gh {

// Quantities used on every RHS evaluation.  Components are flattened into the
// second index of a normal AthenaK cell-centered device array.
enum ReferenceEvolutionComponent : int {
  kRefCoframe = 0,
  kRefFrame = kRefCoframe + 16,
  kRefDFrame = kRefFrame + 16,
  kRefChristoffel = kRefDFrame + 64,
  kRefSpatialFrame = kRefChristoffel + 64,
  kRefSpatialCoframe = kRefSpatialFrame + 9,
  kRefDtSpatialFrame = kRefSpatialCoframe + 9,
  kRefStructure = kRefDtSpatialFrame + 9,
  kRefSpin = kRefStructure + 27,
  kRefSpinDerivative = kRefSpin + 64,
  kRefRiemann = kRefSpinDerivative + 256,
  kReferenceEvolutionSize = kRefRiemann + 256
};

// Second derivatives needed only by the coordinate-source oracle and native
// reference-curvature diagnostics stay out of the hot production cache.
enum ReferenceDiagnosticComponent : int {
  kRefDDFrame = 0,
  kRefDChristoffel = kRefDDFrame + 256,
  kRefRicci = kRefDChristoffel + 256,
  kReferenceDiagnosticSize = kRefRicci + 16
};

static_assert(kReferenceEvolutionSize == 790,
              "Ref-GH production reference-cache layout changed");
static_assert(kReferenceDiagnosticSize == 528,
              "Ref-GH diagnostic reference-cache layout changed");

KOKKOS_INLINE_FUNCTION constexpr int RefMatrix4(const int offset,
                                                 const int a, const int b) {
  return offset + 4*a + b;
}
KOKKOS_INLINE_FUNCTION constexpr int RefRank3(const int offset, const int a,
                                               const int b, const int c) {
  return offset + 16*a + 4*b + c;
}
KOKKOS_INLINE_FUNCTION constexpr int RefRank4(const int offset, const int a,
                                               const int b, const int c,
                                               const int d) {
  return offset + 64*a + 16*b + 4*c + d;
}
KOKKOS_INLINE_FUNCTION constexpr int RefMatrix3(const int offset,
                                                 const int i, const int j) {
  return offset + 3*i + j;
}
KOKKOS_INLINE_FUNCTION constexpr int RefRank3Spatial(const int offset,
                                                      const int i, const int j,
                                                      const int k) {
  return offset + 9*i + 3*j + k;
}

struct ReferenceCachePoint {
  DvceArray5D<Real> evolution;
  DvceArray5D<Real> diagnostic;
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
  return r.diagnostic(r.m, RefRank4(kRefDDFrame, p, q, A, a), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceChristoffel(const ReferenceGeometry &r, const int a, const int b,
                          const int c) {
  return r.christoffel[a][b][c];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceChristoffel(const ReferenceCachePoint &r, const int a, const int b,
                          const int c) {
  return r.evolution(r.m, RefRank3(kRefChristoffel, a, b, c), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDChristoffel(const ReferenceGeometry &r, const int p, const int a,
                           const int b, const int c) {
  return r.d_christoffel[p][a][b][c];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceDChristoffel(const ReferenceCachePoint &r, const int p, const int a,
                           const int b, const int c) {
  return r.diagnostic(r.m, RefRank4(kRefDChristoffel, p, a, b, c), r.k, r.j, r.i);
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
  return r.evolution(r.m, RefRank3Spatial(kRefStructure, I, J, K),
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
  return r.evolution(r.m, RefRank3(kRefSpin, A, B, C), r.k, r.j, r.i);
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpinDerivative(const ReferenceGeometry &r, const int D,
                             const int A, const int B, const int C) {
  return r.spin_derivative[D][A][B][C];
}
KOKKOS_INLINE_FUNCTION
Real ReferenceSpinDerivative(const ReferenceCachePoint &r, const int D,
                             const int A, const int B, const int C) {
  return r.evolution(r.m, RefRank4(kRefSpinDerivative, D, A, B, C),
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
  return r.evolution(r.m, RefRank4(kRefRiemann, A, B, C, D), r.k, r.j, r.i);
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
void StoreReferenceEvolution(const ReferenceGeometry &r,
                             const DvceArray5D<Real> &cache,
                             const int m, const int k, const int j, const int i) {
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      cache(m, RefMatrix4(kRefCoframe, A, a), k, j, i) = r.coframe[A][a];
      cache(m, RefMatrix4(kRefFrame, A, a), k, j, i) = r.frame[A][a];
      for (int p = 0; p < 4; ++p) {
        cache(m, RefRank3(kRefDFrame, p, A, a), k, j, i) = r.d_frame[p][A][a];
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        cache(m, RefRank3(kRefChristoffel, a, b, c), k, j, i) =
            r.christoffel[a][b][c];
        cache(m, RefRank3(kRefSpin, a, b, c), k, j, i) = r.spin[a][b][c];
        for (int d = 0; d < 4; ++d) {
          cache(m, RefRank4(kRefSpinDerivative, a, b, c, d), k, j, i) =
              r.spin_derivative[a][b][c][d];
          cache(m, RefRank4(kRefRiemann, a, b, c, d), k, j, i) =
              r.riemann_frame[a][b][c][d];
        }
      }
    }
  }
  for (int I = 0; I < 3; ++I) {
    for (int p = 0; p < 3; ++p) {
      cache(m, RefMatrix3(kRefSpatialFrame, I, p), k, j, i) =
          r.spatial_frame[I][p];
      cache(m, RefMatrix3(kRefSpatialCoframe, I, p), k, j, i) =
          r.spatial_coframe[I][p];
      cache(m, RefMatrix3(kRefDtSpatialFrame, I, p), k, j, i) =
          r.dt_spatial_frame[I][p];
      for (int J = 0; J < 3; ++J) {
        cache(m, RefRank3Spatial(kRefStructure, I, p, J), k, j, i) =
            r.structure[I][p][J];
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void StoreReferenceDiagnostic(const ReferenceGeometry &r,
                              const DvceArray5D<Real> &cache,
                              const int m, const int k, const int j, const int i) {
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          cache(m, RefRank4(kRefDDFrame, p, a, b, c), k, j, i) =
              r.dd_frame[p][a][b][c];
          cache(m, RefRank4(kRefDChristoffel, p, a, b, c), k, j, i) =
              r.d_christoffel[p][a][b][c];
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      cache(m, RefMatrix4(kRefRicci, A, B), k, j, i) = r.ricci_frame[A][B];
    }
  }
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_CACHE_HPP_
