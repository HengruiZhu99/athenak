//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file vertex_amr.hpp
//! \brief Native point-value restriction/prolongation for vertex-centered fields.

#ifndef MESH_VERTEX_AMR_HPP_
#define MESH_VERTEX_AMR_HPP_

#include "athena.hpp"

namespace vertex_amr {

template <int ORDER>
struct MidpointRule;

template <>
struct MidpointRule<2> {
  static constexpr int points = 2;
  static constexpr int left_offset = 0;
  KOKKOS_INLINE_FUNCTION static constexpr Real weight(const int p) {
    return p == 0 || p == 1 ? 0.5 : 0.0;
  }
};

template <>
struct MidpointRule<4> {
  static constexpr int points = 4;
  static constexpr int left_offset = -1;
  KOKKOS_INLINE_FUNCTION static constexpr Real weight(const int p) {
    return p == 0 ? -1.0 / 16.0
         : p == 1 ?  9.0 / 16.0
         : p == 2 ?  9.0 / 16.0
         : p == 3 ? -1.0 / 16.0 : 0.0;
  }
};

template <>
struct MidpointRule<6> {
  static constexpr int points = 6;
  static constexpr int left_offset = -2;
  KOKKOS_INLINE_FUNCTION static constexpr Real weight(const int p) {
    return p == 0 ?   3.0 / 256.0
         : p == 1 ? -25.0 / 256.0
         : p == 2 ? 150.0 / 256.0
         : p == 3 ? 150.0 / 256.0
         : p == 4 ? -25.0 / 256.0
         : p == 5 ?   3.0 / 256.0 : 0.0;
  }
};

struct DirectionStencil {
  int count = 1;
  int index[6] = {0, 0, 0, 0, 0, 0};
  Real weight[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

template <int ORDER>
KOKKOS_INLINE_FUNCTION DirectionStencil MakeDirectionStencil(
    const int fine_index, const int fine_start, const int coarse_start,
    const bool collapsed) {
  DirectionStencil stencil;
  if (collapsed) {
    stencil.index[0] = 0;
    return stencil;
  }
  const int offset = fine_index - fine_start;
  const int coarse_left = coarse_start + offset / 2;
  if ((offset & 1) == 0) {
    stencil.index[0] = coarse_left;
    return stencil;
  }
  stencil.count = MidpointRule<ORDER>::points;
  for (int p = 0; p < stencil.count; ++p) {
    stencil.index[p] = coarse_left + MidpointRule<ORDER>::left_offset + p;
    stencil.weight[p] = MidpointRule<ORDER>::weight(p);
  }
  return stencil;
}

template <typename FineView, typename CoarseView>
KOKKOS_INLINE_FUNCTION void InjectRestrictVCPoint(
    const int m, const int v, const int ck, const int cj, const int ci,
    const int fine_is, const int fine_js, const int fine_ks,
    const int coarse_is, const int coarse_js, const int coarse_ks,
    const bool collapse_x2, const bool collapse_x3,
    FineView fine, CoarseView coarse) {
  const int fi = fine_is + 2 * (ci - coarse_is);
  const int fj = collapse_x2 ? 0 : fine_js + 2 * (cj - coarse_js);
  const int fk = collapse_x3 ? 0 : fine_ks + 2 * (ck - coarse_ks);
  coarse(m, v, ck, cj, ci) = fine(m, v, fk, fj, fi);
}

template <int ORDER, typename CoarseView, typename FineView>
KOKKOS_INLINE_FUNCTION Real ProlongVCPoint(
    const int m, const int v, const int fk, const int fj, const int fi,
    const int fine_is, const int fine_js, const int fine_ks,
    const int coarse_is, const int coarse_js, const int coarse_ks,
    const bool collapse_x2, const bool collapse_x3,
    CoarseView coarse, FineView fine) {
  const DirectionStencil x =
      MakeDirectionStencil<ORDER>(fi, fine_is, coarse_is, false);
  const DirectionStencil y =
      MakeDirectionStencil<ORDER>(fj, fine_js, coarse_js, collapse_x2);
  const DirectionStencil z =
      MakeDirectionStencil<ORDER>(fk, fine_ks, coarse_ks, collapse_x3);
  Real value = 0.0;
  for (int pz = 0; pz < z.count; ++pz) {
    for (int py = 0; py < y.count; ++py) {
      for (int px = 0; px < x.count; ++px) {
        value += z.weight[pz] * y.weight[py] * x.weight[px] *
                 coarse(m, v, z.index[pz], y.index[py], x.index[px]);
      }
    }
  }
  fine(m, v, fk, fj, fi) = value;
  return value;
}

template <int ORDER>
KOKKOS_INLINE_FUNCTION constexpr int RequiredCoarseGhostWidth() {
  return ORDER == 2 ? 1 : ORDER == 4 ? 1 : 2;
}

}  // namespace vertex_amr

#endif  // MESH_VERTEX_AMR_HPP_
