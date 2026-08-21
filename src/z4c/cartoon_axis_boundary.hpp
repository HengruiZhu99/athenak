//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_axis_boundary.hpp
//! \brief Exact cell-centered parity extension across the half-plane SO(2) axis.

#ifndef Z4C_CARTOON_AXIS_BOUNDARY_HPP_
#define Z4C_CARTOON_AXIS_BOUNDARY_HPP_

#include <Kokkos_Macros.hpp>
#include <type_traits>

#include "z4c/cartoon_axis_parity.hpp"
#include "z4c/z4c_grid.hpp"

namespace z4c {

KOKKOS_INLINE_FUNCTION constexpr int AxisGhostIndex(const int active_start,
                                                    const int depth) {
  return active_start - depth - 1;
}

KOKKOS_INLINE_FUNCTION constexpr int AxisMirrorActiveIndex(
    const int active_start, const int depth) {
  return active_start + depth;
}

//! VC rho=0 is an evolved node, so negative-rho ghost depth q mirrors from +q;
//! the axis node itself is never copied into a negative-rho ghost.
KOKKOS_INLINE_FUNCTION constexpr int VertexAxisMirrorActiveIndex(
    const int axis_index, const int depth) {
  return axis_index + depth + 1;
}

template <typename Centering>
KOKKOS_INLINE_FUNCTION constexpr int CenteredAxisMirrorActiveIndex(
    const int active_start, const int depth) {
  if constexpr (std::is_same_v<Centering, VertexCenteredZ4c>) {
    return VertexAxisMirrorActiveIndex(active_start, depth);
  } else {
    static_assert(std::is_same_v<Centering, CellCenteredZ4c>,
                  "Unknown Z4c centering policy tag");
    return AxisMirrorActiveIndex(active_start, depth);
  }
}

//! Fill one (meshblock,component,k,j) line.  The caller owns topology validation and
//! guarantees that `active_start ... active_start+ghost_depth-1` is active storage.
template <typename Array5D>
KOKKOS_INLINE_FUNCTION bool FillAxisGhostLine(
    const Array5D &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth,
    const int parity_sign) {
  if ((parity_sign != -1 && parity_sign != 1) || ghost_depth < 0) return false;
  for (int depth = 0; depth < ghost_depth; ++depth) {
    const int source = AxisMirrorActiveIndex(active_start, depth);
    const int target = AxisGhostIndex(active_start, depth);
    state(meshblock, component, k, j, target) =
        static_cast<typename Array5D::value_type>(parity_sign) *
        state(meshblock, component, k, j, source);
  }
  return true;
}

template <typename Centering, typename Array5D>
KOKKOS_INLINE_FUNCTION bool FillCenteredAxisGhostLine(
    const Array5D &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth,
    const int parity_sign) {
  if ((parity_sign != -1 && parity_sign != 1) || ghost_depth < 0) return false;
  for (int depth = 0; depth < ghost_depth; ++depth) {
    const int source = CenteredAxisMirrorActiveIndex<Centering>(active_start, depth);
    const int target = AxisGhostIndex(active_start, depth);
    state(meshblock, component, k, j, target) =
        static_cast<typename Array5D::value_type>(parity_sign) *
        state(meshblock, component, k, j, source);
  }
  return true;
}

template <typename Array5D>
KOKKOS_INLINE_FUNCTION bool FillZ4cAxisGhostLine(
    const Array5D &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth) {
  return FillAxisGhostLine(
      state, meshblock, component, k, j, active_start, ghost_depth,
      Z4cStateAxisParitySignFromPackedIndex(component));
}

template <typename Centering, typename Array5D>
KOKKOS_INLINE_FUNCTION bool FillCenteredZ4cAxisGhostLine(
    const Array5D &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth) {
  return FillCenteredAxisGhostLine<Centering>(
      state, meshblock, component, k, j, active_start, ghost_depth,
      Z4cStateAxisParitySignFromPackedIndex(component));
}

template <typename Array5D>
KOKKOS_INLINE_FUNCTION bool FillAdmAxisGhostLine(
    const Array5D &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth) {
  return FillAxisGhostLine(
      state, meshblock, component, k, j, active_start, ghost_depth,
      AdmStateAxisParitySignFromPackedIndex(component));
}

template <typename Array5D>
KOKKOS_INLINE_FUNCTION bool FillConstraintAxisGhostLine(
    const Array5D &state, const int meshblock, const int component,
    const int k, const int j, const int active_start, const int ghost_depth) {
  return FillAxisGhostLine(
      state, meshblock, component, k, j, active_start, ghost_depth,
      ConstraintAxisParitySignFromPackedIndex(component));
}

}  // namespace z4c

#endif  // Z4C_CARTOON_AXIS_BOUNDARY_HPP_
