//========================================================================================
//! \file staged_covariant_rhs.hpp
//! \brief Packed active-cell algebra for the flat-kernel Ref-GH source path.
//========================================================================================
#ifndef REF_GH_STAGED_COVARIANT_RHS_HPP_
#define REF_GH_STAGED_COVARIANT_RHS_HPP_

#include "athena.hpp"
#include "ref_gh/ref_gh_state.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

// Persistent reference storage is separate.  These two layouts are transient
// and active-cell only: 32 physical Reals plus 64 covariant-preparation Reals.
enum StagedPhysicalComponent : int {
  kStagedLapse = 0,
  kStagedShift = kStagedLapse + 1,                    // 3
  kStagedInverseSpatial = kStagedShift + 3,           // sym(3) = 6
  kStagedSpatialConnection = kStagedInverseSpatial + 6,  // 3 x sym(3) = 18
  kStagedTraceK = kStagedSpatialConnection + 18,
  kStagedDAlpha = kStagedTraceK + 1,                  // 3
  kStagedPhysicalSize = kStagedDAlpha + 3
};

enum StagedCovariantComponent : int {
  kStagedInversePsi = 0,                              // sym(4) = 10
  kStagedQ = kStagedInversePsi + 10,                  // 4 x sym(4) = 40
  kStagedDelta = kStagedQ + 40,                       // 4
  kStagedGaugeSource = kStagedDelta + 4,              // sym(4) = 10
  kStagedCovariantSize = kStagedGaugeSource + 10
};

static_assert(kStagedPhysicalSize == 32,
              "staged physical geometry must remain exactly 32 Reals/cell");
static_assert(kStagedCovariantSize == 64,
              "staged covariant preparation must remain exactly 64 Reals/cell");
static_assert(kStagedPhysicalSize + kStagedCovariantSize
                  + kSymmetric4Size == 106,
              "total transient Ref-GH RHS storage must remain 106 Reals/cell");

KOKKOS_INLINE_FUNCTION
constexpr int Symmetric3Index(const int a, const int b) {
  const int lo = (a < b) ? a : b;
  const int hi = (a < b) ? b : a;
  return lo*3 - lo*(lo - 1)/2 + (hi - lo);
}

KOKKOS_INLINE_FUNCTION
void Symmetric4Pair(const int component, int &a, int &b) {
  int remaining = component;
  for (a = 0; a < 4; ++a) {
    const int row = 4 - a;
    if (remaining < row) {
      b = a + remaining;
      return;
    }
    remaining -= row;
  }
  a = 3;
  b = 3;
}

struct LocalStagedCovariantPoint {
  Real inverse[10];  // NOLINT(runtime/arrays)
  Real q_values[40]; // NOLINT(runtime/arrays)
  Real delta_values[4]; // NOLINT(runtime/arrays)

  KOKKOS_INLINE_FUNCTION Real Inverse(const int A, const int B) const {
    return inverse[Symmetric4Index(A, B)];
  }
  KOKKOS_INLINE_FUNCTION Real Q(const int C, const int A, const int B) const {
    return q_values[10*C + Symmetric4Index(A, B)];
  }
  KOKKOS_INLINE_FUNCTION Real Delta(const int A) const {
    return delta_values[A];
  }
  KOKKOS_INLINE_FUNCTION void SetInverse(const int A, const int B,
                                         const Real value) {
    inverse[Symmetric4Index(A, B)] = value;
  }
  KOKKOS_INLINE_FUNCTION void SetQ(const int C, const int A, const int B,
                                   const Real value) {
    q_values[10*C + Symmetric4Index(A, B)] = value;
  }
  KOKKOS_INLINE_FUNCTION void SetDelta(const int A, const Real value) {
    delta_values[A] = value;
  }
};

struct DeviceStagedCovariantPoint {
  DvceArray5D<Real> values;
  int m;
  int k;
  int j;
  int i;

  KOKKOS_INLINE_FUNCTION Real Inverse(const int A, const int B) const {
    return values(m, kStagedInversePsi + Symmetric4Index(A, B), k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real Q(const int C, const int A, const int B) const {
    return values(m, kStagedQ + 10*C + Symmetric4Index(A, B), k, j, i);
  }
  KOKKOS_INLINE_FUNCTION Real Delta(const int A) const {
    return values(m, kStagedDelta + A, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION void SetInverse(const int A, const int B,
                                         const Real value) const {
    values(m, kStagedInversePsi + Symmetric4Index(A, B), k, j, i) = value;
  }
  KOKKOS_INLINE_FUNCTION void SetQ(const int C, const int A, const int B,
                                   const Real value) const {
    values(m, kStagedQ + 10*C + Symmetric4Index(A, B), k, j, i) = value;
  }
  KOKKOS_INLINE_FUNCTION void SetDelta(const int A, const Real value) const {
    values(m, kStagedDelta + A, k, j, i) = value;
  }
};

template <typename Packed>
KOKKOS_INLINE_FUNCTION
Real StagedDeltaLower(const Packed &packed, const int A, const int B,
                      const int C) {
  return 0.5*(packed.Q(B, A, C) + packed.Q(C, A, B)
              - packed.Q(A, B, C));
}

template <typename Packed>
KOKKOS_INLINE_FUNCTION
Real StagedDeltaUpper(const Packed &packed, const int A, const int B,
                      const int C) {
  Real value = 0.0;
  for (int D = 0; D < 4; ++D) {
    value += packed.Inverse(A, D)*StagedDeltaLower(packed, D, B, C);
  }
  return value;
}

template <typename Reference, typename Packed>
KOKKOS_INLINE_FUNCTION
Real StagedP(const Real psi[4][4], const Reference &reference,
             const Packed &packed, const int C, const int A, const int B) {
  Real value = packed.Q(C, A, B);
  for (int D = 0; D < 4; ++D) {
    value += ReferenceSpin(reference, D, A, C)*psi[D][B]
             + ReferenceSpin(reference, D, B, C)*psi[A][D];
  }
  return value;
}

// Construct the packed inverse, q, and contracted Delta without ever owning
// redundant 4x4x4 p/q/connection arrays.  The caller supplies either a local
// oracle object or an active-cell device-view point.
template <typename Reference, typename Packed>
KOKKOS_INLINE_FUNCTION
bool PrepareStagedCovariantPoint(
    const Real psi[4][4], const Real pi[4][4], const Real phi[3][4][4],
    const Reference &reference, const Real normal[4], Packed &packed) {
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!Invert4(psi, inverse, determinant)) return false;
  if (!(normal[0] > 0.0) || !Kokkos::isfinite(normal[0])) return false;
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) packed.SetInverse(A, B, inverse[A][B]);
  }
  for (int C = 0; C < 4; ++C) {
    for (int A = 0; A < 4; ++A) {
      for (int B = A; B < 4; ++B) {
        Real p = C == 0 ? -pi[A][B] : phi[C - 1][A][B];
        if (C == 0) {
          for (int I = 0; I < 3; ++I) p -= normal[I + 1]*phi[I][A][B];
          p /= normal[0];
        }
        Real q = p;
        for (int D = 0; D < 4; ++D) {
          q -= ReferenceSpin(reference, D, A, C)*psi[D][B]
               + ReferenceSpin(reference, D, B, C)*psi[A][D];
        }
        packed.SetQ(C, A, B, q);
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    Real delta = 0.0;
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        delta += packed.Inverse(B, C)*StagedDeltaLower(packed, A, B, C);
      }
    }
    packed.SetDelta(A, delta);
  }
  return true;
}

// One flat work item owns one symmetric pair AB.  It carries only the physical
// Psi matrix, a four-vector normal, and scalar contraction accumulators.
template <typename Reference, typename Packed>
KOKKOS_INLINE_FUNCTION
Real StagedCovariantSourceComponent(
    const Real psi[4][4], const Reference &reference, const Real normal[4],
    const Packed &packed, const int A, const int B, const Real gamma0) {
  Real curvature = 0.0;
  Real qq = 0.0;
  Real delta_product = 0.0;
  Real damping = 0.0;
  Real frame_correction = 0.0;
  for (int C = 0; C < 4; ++C) {
    for (int D = 0; D < 4; ++D) {
      for (int E = 0; E < 4; ++E) {
        curvature -= packed.Inverse(C, D)*(
            ReferenceRiemann(reference, E, C, D, A)*psi[B][E]
            + ReferenceRiemann(reference, E, C, D, B)*psi[A][E]);
        for (int F = 0; F < 4; ++F) {
          qq += 2.0*packed.Inverse(C, D)*packed.Inverse(E, F)
                *packed.Q(E, C, A)*packed.Q(F, D, B);
          delta_product -=
              2.0*packed.Inverse(C, D)*packed.Inverse(E, F)
              *StagedDeltaLower(packed, A, C, E)
              *StagedDeltaLower(packed, B, D, F);
        }
      }
    }
    Real normal_lower_A = 0.0;
    Real normal_lower_B = 0.0;
    for (int D = 0; D < 4; ++D) {
      normal_lower_A += psi[A][D]*normal[D];
      normal_lower_B += psi[B][D]*normal[D];
    }
    const Real frame_projector = ((C == A) ? normal_lower_B : 0.0)
                                 + ((C == B) ? normal_lower_A : 0.0)
                                 - psi[A][B]*normal[C];
    damping += gamma0*frame_projector*packed.Delta(C);
  }
  for (int C = 0; C < 4; ++C) {
    for (int D = 0; D < 4; ++D) {
      Real f_cdab = 0.0;
      for (int E = 0; E < 4; ++E) {
        f_cdab -= (ReferenceSpin(reference, E, D, C)
                   + StagedDeltaUpper(packed, E, D, C))
                  *StagedP(psi, reference, packed, E, A, B);
        f_cdab += ReferenceSpinDerivative(reference, C, E, A, D)*psi[E][B]
                  + ReferenceSpin(reference, E, A, D)
                    *StagedP(psi, reference, packed, C, E, B)
                  + ReferenceSpinDerivative(reference, C, E, B, D)*psi[A][E]
                  + ReferenceSpin(reference, E, B, D)
                    *StagedP(psi, reference, packed, C, A, E)
                  + ReferenceSpin(reference, E, D, C)*packed.Q(E, A, B)
                  + ReferenceSpin(reference, E, A, C)*packed.Q(D, E, B)
                  + ReferenceSpin(reference, E, B, C)*packed.Q(D, A, E);
      }
      frame_correction += packed.Inverse(C, D)*f_cdab;
    }
  }
  return curvature + qq + delta_product + damping + frame_correction;
}

}  // namespace ref_gh

#endif  // REF_GH_STAGED_COVARIANT_RHS_HPP_
