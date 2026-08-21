//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file vertex_to_cell.hpp
//! \brief Symmetric point interpolation from native VC nodes to CC adapter points.

#ifndef Z4C_VERTEX_TO_CELL_HPP_
#define Z4C_VERTEX_TO_CELL_HPP_

#include "athena.hpp"
#include "mesh/vertex_amr.hpp"

namespace z4c {

template <int ORDER>
KOKKOS_INLINE_FUNCTION vertex_amr::DirectionStencil VertexToCellStencil(
    const int cell_index, const bool collapsed) {
  vertex_amr::DirectionStencil stencil;
  if (collapsed) {
    stencil.index[0] = 0;
    return stencil;
  }
  stencil.count = vertex_amr::MidpointRule<ORDER>::points;
  for (int p = 0; p < stencil.count; ++p) {
    stencil.index[p] =
        cell_index + vertex_amr::MidpointRule<ORDER>::left_offset + p;
    stencil.weight[p] = vertex_amr::MidpointRule<ORDER>::weight(p);
  }
  return stencil;
}

template <int ORDER, typename VertexView>
KOKKOS_INLINE_FUNCTION Real InterpolateVertexToCellPoint(
    const VertexView &vertex, const int m, const int v, const int k,
    const int j, const int i, const bool collapse_x2,
    const bool collapse_x3) {
  const auto sx = VertexToCellStencil<ORDER>(i, false);
  const auto sy = VertexToCellStencil<ORDER>(j, collapse_x2);
  const auto sz = VertexToCellStencil<ORDER>(k, collapse_x3);
  Real value = 0.0;
  for (int pz = 0; pz < sz.count; ++pz) {
    for (int py = 0; py < sy.count; ++py) {
      for (int px = 0; px < sx.count; ++px) {
        value += sz.weight[pz] * sy.weight[py] * sx.weight[px] *
                 vertex(m, v, sz.index[pz], sy.index[py], sx.index[px]);
      }
    }
  }
  return value;
}

}  // namespace z4c

#endif  // Z4C_VERTEX_TO_CELL_HPP_
