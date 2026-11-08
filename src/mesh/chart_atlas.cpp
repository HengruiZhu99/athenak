//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file chart_atlas.cpp
//! \brief Host-side atlas metadata and native logical-location map construction.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "mesh/chart_atlas.hpp"
#include "mesh/logical_location.hpp"

namespace chart_atlas {

Atlas::Atlas(Real half_width, std::vector<Real> radial_knots,
             std::array<int, 3> core_blocks, std::array<int, 2> angular_blocks,
             Real spherical_inner_radius, bool equiangular,
             int evolved_overlap, bool outer_endpoint_grid)
    : half_width_(half_width), spherical_inner_radius_(spherical_inner_radius),
      radial_knots_(std::move(radial_knots)), core_blocks_(core_blocks),
      angular_blocks_(angular_blocks), evolved_overlap_(evolved_overlap),
      equiangular_(equiangular || spherical_inner_radius > 0),
      outer_endpoint_grid_(outer_endpoint_grid) {
  if (!std::isfinite(half_width_) || half_width_ <= 0 || radial_knots_.empty()
      || !std::isfinite(spherical_inner_radius_) || spherical_inner_radius_ < 0
      || spherical_inner_radius_ >= half_width_ || evolved_overlap_ < 0
      || (equiangular_ && spherical_inner_radius_ == 0)) {
    throw std::invalid_argument("invalid Cartesian/shell chart geometry");
  }
  Real lower = spherical_inner_radius_ > 0
      ? spherical_inner_radius_ : std::sqrt(Real(3))*half_width_;
  for (Real radius : radial_knots_) {
    if (!std::isfinite(radius) || radius <= lower) {
      throw std::invalid_argument("shell radii must be finite and strictly increasing");
    }
    lower = radius;
  }
  if (radial_knots_.size() > std::numeric_limits<std::int32_t>::max()) {
    throw std::invalid_argument("too many radial intervals for logical locations");
  }
  int max_blocks = static_cast<int>(radial_knots_.size());
  for (int n : core_blocks_) {
    if (n < 1) throw std::invalid_argument("invalid core root grid");
    max_blocks = std::max(max_blocks, n);
  }
  for (int n : angular_blocks_) {
    if (n < 1) throw std::invalid_argument("invalid shell angular grid");
    max_blocks = std::max(max_blocks, n);
  }
  while ((1LL << root_level_) < max_blocks) ++root_level_;
}

std::array<int, 3> Atlas::RootBlocks(int chart_id) const {
  if (chart_id < 0 || chart_id >= NumCharts()) throw std::out_of_range("unknown chart");
  return chart_id == 0 ? core_blocks_
      : std::array<int, 3>{static_cast<int>(radial_knots_.size()),
                           angular_blocks_[0], angular_blocks_[1]};
}

Seam Atlas::AngularSeam(int chart_id, int axis, int side) {
  if (chart_id < 1 || chart_id > 6 || (axis != 1 && axis != 2)
      || (side != -1 && side != 1)) {
    throw std::invalid_argument("angular seam requires a shell chart, angular axis, and side");
  }
  const int face = chart_id-1;
  const int normal_axis = face/2;
  const int normal_sign = face%2 == 0 ? 1 : -1;
  const int neighbor_normal = (normal_axis+axis)%3;
  const int neighbor_chart = 2*neighbor_normal + (side < 0 ? 2 : 1);
  // For the cyclic bases (a,b,c), crossing u reaches a b-normal face and
  // crossing v reaches a c-normal face. The source normal becomes the other
  // face's v or u edge, respectively; the third Cartesian coordinate is shared.
  return {neighbor_chart, axis, side, 3-axis, normal_sign, 3-axis, axis};
}

IndexBounds Atlas::ShellRefinementParents(int chart_id, int logical_level,
                                         const Bounds &radius_angles) const {
  const auto roots = RootBlocks(chart_id);
  if (chart_id == 0 || logical_level <= root_level_ || logical_level > 31) {
    throw std::invalid_argument("shell SMR requires chart_id 1..6 and a positive refinement level");
  }
  if (spherical_inner_radius_ == 0 && radial_knots_.size() == 1) {
    throw std::invalid_argument("radius_angles SMR requires a spherical radial layer");
  }
  const Real radial_min = spherical_inner_radius_ > 0
      ? spherical_inner_radius_ : radial_knots_.front();
  for (int a=0; a<3; ++a) {
    const Real lower = a == 0 ? radial_min : -1;
    const Real upper = a == 0 ? radial_knots_.back() : 1;
    if (!std::isfinite(radius_angles.lo[a]) || !std::isfinite(radius_angles.hi[a])
        || radius_angles.lo[a] > radius_angles.hi[a]
        || radius_angles.lo[a] < lower || radius_angles.hi[a] > upper) {
      throw std::invalid_argument(a == 0 && spherical_inner_radius_ == 0
          ? "radius_angles SMR excludes the non-spherical cube-to-sphere first layer"
          : "radius_angles SMR region must lie within the spherical shell chart");
    }
  }
  const std::int64_t factor = 1LL << (logical_level-root_level_-1);
  IndexBounds selected;
  for (int a=0; a<3; ++a) {
    const std::int64_t count = roots[a]*factor;
    const auto edge = [&](std::int64_t index) {
      if (a != 0) return -1 + Real(2)*index/count;
      if (index == count) return radial_knots_.back();
      const int layer = static_cast<int>(index/factor);
      const Real lower = layer == 0 ? spherical_inner_radius_ : radial_knots_[layer-1];
      return lower + Real(index%factor)*(radial_knots_[layer]-lower)/factor;
    };
    // Binary search actual physical edges, preserving strict lower and inclusive
    // upper selection at radial knots without floating-point index rounding.
    std::int64_t first = 0, last = count;
    while (first < last) {
      const auto mid = first+(last-first)/2;
      if (edge(mid+1) <= radius_angles.lo[a]) first = mid+1;
      else last = mid;
    }
    selected.first[a] = std::min(first, count-1);
    first = 0;
    last = count;
    while (first < last) {
      const auto mid = first+(last-first)/2;
      if (edge(mid) < radius_angles.hi[a]) first = mid+1;
      else last = mid;
    }
    selected.last[a] = std::max(first, selected.first[a]+1);
  }
  return selected;
}

std::string Atlas::Identity() const {
  // Bump the convention version whenever chart orientation, logical-to-radial
  // indexing, evolved-region rules, or endpoint placement changes. Hex floats
  // preserve every stored binary digit and ignore the process numeric locale.
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "athenak-chart-atlas-v2;half_width=" << std::hexfloat << half_width_
      << ";inner_radius=" << spherical_inner_radius_ << ";radii=";
  for (Real radius : radial_knots_) out << radius << ',';
  out << ";core=";
  for (int n : core_blocks_) out << n << ',';
  out << ";angular=";
  for (int n : angular_blocks_) out << n << ',';
  out << ";overlap=" << evolved_overlap_ << ";equiangular=" << equiangular_
      << ";outer_endpoint=" << outer_endpoint_grid_;
  return out.str();
}

BlockMap Atlas::GetBlockMap(const LogicalLocation &loc) const {
  const auto root_blocks = RootBlocks(loc.chart_id);
  if (loc.level < root_level_ || loc.level > 31) {
    throw std::out_of_range("atlas requires a physical leaf logical location");
  }
  const std::int64_t factor = 1LL << (loc.level-root_level_);
  const std::int32_t index[3] = {loc.lx1, loc.lx2, loc.lx3};
  std::int64_t extent[3];
  for (int a=0; a<3; ++a) {
    extent[a] = root_blocks[a]*factor;
    if (index[a] < 0 || index[a] >= extent[a]) {
      throw std::out_of_range("logical location is outside chart root grid");
    }
  }

  BlockMap map;
  map.chart_id = loc.chart_id;
  map.half_width = half_width_;
  if (loc.chart_id == 0) {
    for (int a=0; a<3; ++a) {
      // Preserve the reference's parent-to-child midpoint construction. A
      // direct refined-grid fraction is mathematically identical but rounds
      // differently near the puncture, where derivatives amplify those bits.
      const std::int64_t root_index=index[a]/factor;
      map.lo[a] = -half_width_ + 2*half_width_*root_index/root_blocks[a];
      map.hi[a] = -half_width_ + 2*half_width_*(root_index+1)/root_blocks[a];
      for (std::int64_t child=factor/2;child>0;child/=2) {
        const Real middle=(map.lo[a]+map.hi[a])/2;
        if (index[a]&child) map.lo[a]=middle;
        else map.hi[a]=middle;
      }
      if (index[a] == 0) map.overlap_sides[2*a] = evolved_overlap_;
      if (index[a]+1LL == extent[a]) map.overlap_sides[2*a+1] = evolved_overlap_;
    }
    return map;
  }

  map.face = loc.chart_id-1;
  map.layer = static_cast<int>(loc.lx1/factor);
  map.inner_radius = map.layer == 0 ? spherical_inner_radius_ : radial_knots_[map.layer-1];
  map.outer_radius = radial_knots_[map.layer];
  map.cube_transition = map.layer == 0 && spherical_inner_radius_ == 0;
  map.equiangular = equiangular_;
  map.lo[0] = Real(loc.lx1%factor)/factor;
  map.hi[0] = Real(loc.lx1%factor+1)/factor;
  for (int a=1; a<3; ++a) {
    map.lo[a] = -1 + Real(2)*index[a]/extent[a];
    map.hi[a] = -1 + Real(2)*(index[a]+1LL)/extent[a];
  }
  map.outer_surface = loc.lx1+1LL == extent[0];
  if (outer_endpoint_grid_ && map.layer+1 == static_cast<int>(radial_knots_.size())) {
    map.node_offset[0] = 0;
  }
  if (map.layer == 0) {
    // Extend the evolved chart region, not internal boundaries introduced by
    // refinement. First-layer outer radial overlap remains independently evolved.
    if (loc.lx1 == 0 && spherical_inner_radius_ == 0) {
      map.overlap_sides[0] = evolved_overlap_;
    }
    if (loc.lx1+1LL == factor) map.overlap_sides[1] = evolved_overlap_;
    for (int a=1; a<3; ++a) {
      if (index[a] == 0) map.overlap_sides[2*a] = evolved_overlap_;
      if (index[a]+1LL == extent[a]) map.overlap_sides[2*a+1] = evolved_overlap_;
    }
  }
  return map;
}

Bounds Atlas::PhysicalBounds(const LogicalLocation &loc) const {
  const auto map = GetBlockMap(loc);
  Bounds bounds;
  for (int a=0; a<3; ++a) {
    bounds.lo[a] = std::numeric_limits<Real>::infinity();
    bounds.hi[a] = -std::numeric_limits<Real>::infinity();
  }
  // A shell coordinate is monotone in radial position. Its angular extrema
  // occur at an edge or at zero in the other angular coordinate; corners alone
  // miss the extrema of a spherical face. The same candidates bound the linear
  // cube-to-sphere transition. This is an AABB of nominal active block bounds.
  Real candidate[3][3] = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
  for (int a=1; a<3; ++a) {
    candidate[a][2] = std::max(Real(0), std::min(Real(1), -map.lo[a]/(map.hi[a]-map.lo[a])));
  }
  for (int k=0; k<3; ++k) {
    for (int j=0; j<3; ++j) {
      for (int i=0; i<2; ++i) {
        const Real local[3] = {candidate[0][i], candidate[1][j], candidate[2][k]};
        Real x[3];
        map.Forward(local, x);
        for (int a=0; a<3; ++a) {
          bounds.lo[a] = std::min(bounds.lo[a], x[a]);
          bounds.hi[a] = std::max(bounds.hi[a], x[a]);
        }
      }
    }
  }
  return bounds;
}

}  // namespace chart_atlas
