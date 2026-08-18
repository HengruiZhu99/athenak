//========================================================================================
// AthenaK reference-frame first-order generalized harmonic state layout.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REF_GH_STATE_HPP_
#define REF_GH_REF_GH_STATE_HPP_

#include "athena.hpp"

namespace ref_gh {

// The independent symmetric tetrad pairs are ordered
// 00, 01, 02, 03, 11, 12, 13, 22, 23, 33.
constexpr int kSymmetric4Size = 10;
constexpr int kSpatialDimension = 3;
constexpr int kPsiOffset = 0;
constexpr int kPiOffset = kPsiOffset + kSymmetric4Size;
constexpr int kPhiOffset = kPiOffset + kSymmetric4Size;
constexpr int nvar = kPhiOffset + kSpatialDimension*kSymmetric4Size;

static_assert(nvar == 50, "Reference-frame FO-GH must evolve exactly 50 fields.");

KOKKOS_INLINE_FUNCTION
constexpr int Symmetric4Index(const int a, const int b) {
  const int lo = (a < b) ? a : b;
  const int hi = (a < b) ? b : a;
  return lo*4 - lo*(lo - 1)/2 + (hi - lo);
}

KOKKOS_INLINE_FUNCTION
constexpr int PsiIndex(const int a, const int b) {
  return kPsiOffset + Symmetric4Index(a, b);
}

KOKKOS_INLINE_FUNCTION
constexpr int PiIndex(const int a, const int b) {
  return kPiOffset + Symmetric4Index(a, b);
}

KOKKOS_INLINE_FUNCTION
constexpr int PhiIndex(const int i, const int a, const int b) {
  return kPhiOffset + i*kSymmetric4Size + Symmetric4Index(a, b);
}

// Small fixed-size point state suitable for device-local algebra.  Coordinate metric
// components are deliberately absent: they are reconstructed as cell-local intermediates.
struct PointState {
  Real psi[kSymmetric4Size];  // NOLINT(runtime/arrays)
  Real pi[kSymmetric4Size];   // NOLINT(runtime/arrays)
  Real phi[kSpatialDimension][kSymmetric4Size];  // NOLINT(runtime/arrays)
};

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_STATE_HPP_
