//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_grid.hpp
//! \brief Immutable Z4c centering tags and index geometry.

#ifndef Z4C_Z4C_GRID_HPP_
#define Z4C_Z4C_GRID_HPP_

#include <cstdint>
#include <type_traits>

namespace z4c {

enum class Z4cGridCentering : std::uint8_t { cell, vertex };

struct CellCenteredZ4c {};
struct VertexCenteredZ4c {};

//! Z4c-owned index geometry. RegionIndcs remains cell-centered and is never reinterpreted.
//!
//! `nx*` and `cnx*` are physical interval counts. Active bounds include one more point
//! in each noncollapsed direction for vertex-centered storage. Collapsed directions
//! contain one point and no ghost storage.
struct Z4cGridLayout {
  static constexpr int kCenteringSchema = 1;

  Z4cGridCentering centering = Z4cGridCentering::cell;
  int centering_schema = kCenteringSchema;
  int ng = 0;
  int coarse_ng = 0;

  int nx1 = 1, nx2 = 1, nx3 = 1;
  int is = 0, ie = 0, js = 0, je = 0, ks = 0, ke = 0;
  int n1 = 1, n2 = 1, n3 = 1;

  int cnx1 = 1, cnx2 = 1, cnx3 = 1;
  int cis = 0, cie = 0, cjs = 0, cje = 0, cks = 0, cke = 0;
  int cn1 = 1, cn2 = 1, cn3 = 1;
};

static_assert(std::is_trivially_copyable_v<Z4cGridLayout>);

namespace detail {

struct Z4cDirectionLayout {
  int start;
  int end;
  int stored_count;
};

constexpr Z4cDirectionLayout MakeZ4cDirectionLayout(
    const Z4cGridCentering centering, const int intervals, const int ghost_width) {
  if (intervals <= 1) return {0, 0, 1};
  const int active_count =
      intervals + (centering == Z4cGridCentering::vertex ? 1 : 0);
  return {ghost_width, ghost_width + active_count - 1,
          active_count + 2 * ghost_width};
}

constexpr int CoarseIntervals(const int intervals) {
  return intervals > 1 ? intervals / 2 : 1;
}

}  // namespace detail

template <typename RegionIndices>
constexpr Z4cGridLayout MakeZ4cGridLayout(
    const Z4cGridCentering centering, const RegionIndices &indcs,
    const int coarse_ghost_width) {
  const auto x1 = detail::MakeZ4cDirectionLayout(centering, indcs.nx1, indcs.ng);
  const auto x2 = detail::MakeZ4cDirectionLayout(centering, indcs.nx2, indcs.ng);
  const auto x3 = detail::MakeZ4cDirectionLayout(centering, indcs.nx3, indcs.ng);

  const int cnx1 = detail::CoarseIntervals(indcs.nx1);
  const int cnx2 = detail::CoarseIntervals(indcs.nx2);
  const int cnx3 = detail::CoarseIntervals(indcs.nx3);
  const auto cx1 = detail::MakeZ4cDirectionLayout(centering, cnx1,
                                                  coarse_ghost_width);
  const auto cx2 = detail::MakeZ4cDirectionLayout(centering, cnx2,
                                                  coarse_ghost_width);
  const auto cx3 = detail::MakeZ4cDirectionLayout(centering, cnx3,
                                                  coarse_ghost_width);

  return {centering,
          Z4cGridLayout::kCenteringSchema,
          indcs.ng,
          coarse_ghost_width,
          indcs.nx1,
          indcs.nx2,
          indcs.nx3,
          x1.start,
          x1.end,
          x2.start,
          x2.end,
          x3.start,
          x3.end,
          x1.stored_count,
          x2.stored_count,
          x3.stored_count,
          cnx1,
          cnx2,
          cnx3,
          cx1.start,
          cx1.end,
          cx2.start,
          cx2.end,
          cx3.start,
          cx3.end,
          cx1.stored_count,
          cx2.stored_count,
          cx3.stored_count};
}

template <typename RegionIndices>
constexpr Z4cGridLayout MakeZ4cGridLayout(
    const Z4cGridCentering centering, const RegionIndices &indcs) {
  return MakeZ4cGridLayout(centering, indcs, indcs.ng);
}

}  // namespace z4c

#endif  // Z4C_Z4C_GRID_HPP_
