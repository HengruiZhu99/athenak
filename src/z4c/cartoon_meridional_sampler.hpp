//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_meridional_sampler.hpp
//! \brief Shared geometry and scalar sampling on the half-plane Cartoon mesh.

#ifndef Z4C_CARTOON_MERIDIONAL_SAMPLER_HPP_
#define Z4C_CARTOON_MERIDIONAL_SAMPLER_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/curvature_diagnostics.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_restart.hpp"
#include "z4c/z4c_symmetry.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

inline constexpr Real kCartoonTwoPi =
    6.2831853071795864769252867665590057683943387987502;

//! Cell measure for Z4c integral diagnostics. Cartesian behavior is unchanged;
//! Cartoon uses the physical nonnegative-rho meridional half-plane.
KOKKOS_INLINE_FUNCTION
Real Z4cDiagnosticCellMeasure(const Z4cSymmetryMode mode, const Real rho,
                              const Real dx1, const Real dx2, const Real dx3,
                              const Real spatial_determinant) {
  const Real proper_factor = Kokkos::sqrt(Kokkos::fabs(spatial_determinant));
  if (mode == Z4cSymmetryMode::cartoon_so2) {
    return rho > 0.0
               ? kCartoonTwoPi * rho * dx1 * dx2 * proper_factor
               : 0.0;
  }
  return dx1 * dx2 * dx3 * proper_factor;
}

//! con.C is AthenaK's complete constraint inventory:
//! H^2 + gamma_ij M^i M^j + Theta^2 + 4 gamma_ij Z^i Z^j.
KOKKOS_INLINE_FUNCTION
Real Z4cAggregateConstraintNorm(const Real aggregate_squared) {
  return Kokkos::sqrt(Kokkos::fmax(aggregate_squared, 0.0));
}

//! A bilinear cell-centered stencil owned by exactly one global leaf MeshBlock.
//! Component reconstruction remains the caller's responsibility, allowing the
//! later m=0 FastFlow adapter to reuse this scalar interpolation primitive.
struct CartoonMeridionalStencil {
  bool valid = false;
  int gid = -1;
  int level = -1;
  int owner_rank = -1;
  int local_block = -1;
  int k = 0;
  int i0 = 0;
  int j0 = 0;
  Real wi = 0.0;
  Real wj = 0.0;
};

template <typename Array>
KOKKOS_INLINE_FUNCTION Real SampleCartoonMeridionalScalar(
    const Array &field, const int component,
    const CartoonMeridionalStencil &stencil) {
  const int m = stencil.local_block;
  const int k = stencil.k;
  const int i = stencil.i0;
  const int j = stencil.j0;
  const Real wi = stencil.wi;
  const Real wj = stencil.wj;
  const Real lower = (1.0 - wi) * field(m, component, k, j, i) +
                     wi * field(m, component, k, j, i + 1);
  const Real upper = (1.0 - wi) * field(m, component, k, j + 1, i) +
                     wi * field(m, component, k, j + 1, i + 1);
  return (1.0 - wj) * lower + wj * upper;
}

namespace meridional_detail {

inline void LogicalEdges(const Mesh &mesh, const LogicalLocation &location,
                         Real *x1min, Real *x1max, Real *x2min, Real *x2max) {
  const int level_offset = location.level - mesh.root_level;
  const std::int64_t blocks_x1 =
      static_cast<std::int64_t>(mesh.nmb_rootx1) << level_offset;
  const std::int64_t blocks_x2 =
      static_cast<std::int64_t>(mesh.nmb_rootx2) << level_offset;
  const Real fraction_x1 = 1.0 / static_cast<Real>(blocks_x1);
  const Real fraction_x2 = 1.0 / static_cast<Real>(blocks_x2);
  const Real length_x1 = mesh.mesh_size.x1max - mesh.mesh_size.x1min;
  const Real length_x2 = mesh.mesh_size.x2max - mesh.mesh_size.x2min;
  *x1min = mesh.mesh_size.x1min + location.lx1 * fraction_x1 * length_x1;
  *x1max = mesh.mesh_size.x1min + (location.lx1 + 1) * fraction_x1 * length_x1;
  *x2min = mesh.mesh_size.x2min + location.lx2 * fraction_x2 * length_x2;
  *x2max = mesh.mesh_size.x2min + (location.lx2 + 1) * fraction_x2 * length_x2;
}

inline bool ContainsClosed(const Real value, const Real lower, const Real upper) {
  const Real scale = std::max({Real(1.0), std::fabs(lower), std::fabs(upper)});
  const Real tolerance = 32.0 * std::numeric_limits<Real>::epsilon() * scale;
  return value >= lower - tolerance && value <= upper + tolerance;
}

}  // namespace meridional_detail

//! Select the finest leaf touching a physical point. At equal level, prefer
//! positive rho and positive z at symmetry interfaces, then the lowest gid.
inline CartoonMeridionalStencil LocateCartoonMeridionalPointOnGrid(
    Mesh *mesh, const Real rho, const Real z,
    const Z4cGridCentering centering) {
  CartoonMeridionalStencil stencil;
  bool selected_positive_rho = false;
  bool selected_positive_z = false;
  Real selected_x1min = 0.0;
  Real selected_x1max = 0.0;
  Real selected_x2min = 0.0;
  Real selected_x2max = 0.0;
  for (int gid = 0; gid < mesh->nmb_total; ++gid) {
    Real x1min = 0.0;
    Real x1max = 0.0;
    Real x2min = 0.0;
    Real x2max = 0.0;
    const LogicalLocation &location = mesh->lloc_eachmb[gid];
    meridional_detail::LogicalEdges(*mesh, location, &x1min, &x1max, &x2min,
                                    &x2max);
    if (!meridional_detail::ContainsClosed(rho, x1min, x1max) ||
        !meridional_detail::ContainsClosed(z, x2min, x2max)) {
      continue;
    }
    const bool positive_rho = x1min >= 0.0;
    const bool positive_z = x2min >= 0.0;
    const bool preferable =
        !stencil.valid || location.level > stencil.level ||
        (location.level == stencil.level &&
         (positive_rho > selected_positive_rho ||
          (positive_rho == selected_positive_rho &&
           (positive_z > selected_positive_z ||
            (positive_z == selected_positive_z && gid < stencil.gid)))));
    if (!preferable) continue;
    stencil.valid = true;
    stencil.gid = gid;
    stencil.level = location.level;
    selected_positive_rho = positive_rho;
    selected_positive_z = positive_z;
    selected_x1min = x1min;
    selected_x1max = x1max;
    selected_x2min = x2min;
    selected_x2max = x2max;
  }
  if (!stencil.valid) return stencil;

  stencil.owner_rank = mesh->rank_eachmb[stencil.gid];
  if (stencil.owner_rank != global_variable::my_rank) return stencil;
  stencil.local_block = stencil.gid - mesh->gids_eachrank[stencil.owner_rank];
  if (stencil.local_block < 0 ||
      stencil.local_block >= mesh->pmb_pack->nmb_thispack) {
    stencil.valid = false;
    return stencil;
  }
  const auto &indices = mesh->mb_indcs;
  const bool vertex = centering == Z4cGridCentering::vertex;
  const auto &layout = mesh->pmb_pack->pz4c->layout;
  const Real dx1 = (selected_x1max - selected_x1min) / indices.nx1;
  const Real dx2 = (selected_x2max - selected_x2min) / indices.nx2;
  const Real offset_i = (rho - selected_x1min) / dx1 - (vertex ? 0.0 : 0.5);
  const Real offset_j = (z - selected_x2min) / dx2 - (vertex ? 0.0 : 0.5);
  int lower_i = static_cast<int>(std::floor(offset_i));
  int lower_j = static_cast<int>(std::floor(offset_j));
  if (vertex) {
    // A query on an upper block vertex belongs to the last interpolation
    // interval.  This keeps both points active while preserving an exact
    // weight of one at the endpoint.
    lower_i = std::max(0, std::min(indices.nx1 - 1, lower_i));
    lower_j = std::max(0, std::min(indices.nx2 - 1, lower_j));
  }
  stencil.i0 = (vertex ? layout.is : indices.is) + lower_i;
  stencil.j0 = (vertex ? layout.js : indices.js) + lower_j;
  stencil.k = vertex ? layout.ks : indices.ks;
  stencil.wi = offset_i - lower_i;
  stencil.wj = offset_j - lower_j;
  const int total_i = vertex ? layout.n1 : indices.nx1 + 2 * indices.ng;
  const int total_j = vertex ? layout.n2 : indices.nx2 + 2 * indices.ng;
  if (stencil.i0 < 0 || stencil.i0 + 1 >= total_i || stencil.j0 < 0 ||
      stencil.j0 + 1 >= total_j) {
    stencil.valid = false;
  }
  return stencil;
}

//! Locate a query on the legacy cell-centred ADM/Z4c adapter.
inline CartoonMeridionalStencil LocateCartoonMeridionalPoint(
    Mesh *mesh, const Real rho, const Real z) {
  return LocateCartoonMeridionalPointOnGrid(
      mesh, rho, z, Z4cGridCentering::cell);
}

//! Locate a query on the authoritative native Z4c grid.
inline CartoonMeridionalStencil LocateNativeCartoonMeridionalPoint(
    Mesh *mesh, const Real rho, const Real z) {
  return LocateCartoonMeridionalPointOnGrid(
      mesh, rho, z, mesh->pmb_pack->pz4c->layout.centering);
}

struct CartoonCentralSample {
  bool valid = false;
  Real lapse = 0.0;
  Real constraint_norm = 0.0;
  Real abs_kretschmann = 0.0;
  int gid = -1;
  int level = -1;
  enum class Status {
    valid,
    missing_center_leaf,
    invalid_common_lattice,
    unsupported_level_gap,
    missing_support,
    duplicate_support,
    invalid_owner,
    insufficient_derivative_halo,
    nonfinite_support
  } status = Status::missing_center_leaf;
};

inline constexpr int kCartoonCentralQuadrants = 4;
inline constexpr int kCartoonCentralSourcesPerQuadrant = 4;
inline constexpr int kCartoonCentralMaxSources =
    kCartoonCentralQuadrants * kCartoonCentralSourcesPerQuadrant;

struct CartoonCentralSupport {
  bool expected = false;
  int matches = 0;
  int gid = -1;
  int level = -1;
  int owner_rank = -1;
  int local_block = -1;
  int k = 0;
  int i = 0;
  int j = 0;
  Real rho = 0.0;
  Real z = 0.0;
  Real restriction_weight = 0.0;
  Real final_weight = 0.0;
};

struct CartoonCentralSupportSet {
  CartoonCentralSupport point[kCartoonCentralMaxSources];
  int source_count = 0;
  int gid = -1;
  int level = -1;
  int common_level = -1;
  unsigned int refined_mask = 0;
  Real common_dx1 = 0.0;
  Real common_dx2 = 0.0;
  bool half_plane = false;
  CartoonCentralSample::Status construction_status =
      CartoonCentralSample::Status::valid;
};

KOKKOS_INLINE_FUNCTION
int CartoonCentralSourceSlot(const int quadrant, const int child) {
  return quadrant * kCartoonCentralSourcesPerQuadrant + child;
}

inline void InitializeCartoonCentralSupportGeometry(
    CartoonCentralSupportSet *supports, const int common_level,
    const Real common_dx1, const Real common_dx2,
    const unsigned int refined_mask, const bool half_plane = false) {
  supports->source_count = 0;
  supports->common_level = common_level;
  supports->common_dx1 = common_dx1;
  supports->common_dx2 = common_dx2;
  supports->refined_mask = refined_mask;
  supports->half_plane = half_plane;
  for (int slot = 0; slot < kCartoonCentralMaxSources; ++slot) {
    supports->point[slot] = CartoonCentralSupport{};
  }
  for (int quadrant = 0; quadrant < kCartoonCentralQuadrants; ++quadrant) {
    if (half_plane && (quadrant & 1) == 0) continue;
    const Real rho_sign = (quadrant & 1) == 0 ? -1.0 : 1.0;
    const Real z_sign = (quadrant & 2) == 0 ? -1.0 : 1.0;
    const Real center_rho = 0.5 * rho_sign * common_dx1;
    const Real center_z = 0.5 * z_sign * common_dx2;
    const bool refined = (refined_mask & (1U << quadrant)) != 0;
    const int children = refined ? kCartoonCentralSourcesPerQuadrant : 1;
    for (int child = 0; child < children; ++child) {
      CartoonCentralSupport &point =
          supports->point[CartoonCentralSourceSlot(quadrant, child)];
      const Real child_rho_sign = (child & 1) == 0 ? -1.0 : 1.0;
      const Real child_z_sign = (child & 2) == 0 ? -1.0 : 1.0;
      point.expected = true;
      point.level = common_level + (refined ? 1 : 0);
      point.rho = center_rho +
                  (refined ? 0.25 * child_rho_sign * common_dx1 : 0.0);
      point.z = center_z +
                (refined ? 0.25 * child_z_sign * common_dx2 : 0.0);
      point.restriction_weight = refined ? 0.25 : 1.0;
      point.final_weight = half_plane
                               ? (refined ? 0.125 : 0.5)
                               : (refined ? 0.0625 : 0.25);
      ++supports->source_count;
    }
  }
}

template <int NGHOST>
inline bool CartoonCentralActiveCellHasStoredDerivativeHalo(
    const RegionIndcs &indices, const int i, const int j) {
  constexpr int radius = NGHOST - 1;
  const int total_i = indices.nx1 + 2 * indices.ng;
  const int total_j = indices.nx2 + 2 * indices.ng;
  return indices.ng >= NGHOST && i >= indices.is && i <= indices.ie &&
         j >= indices.js && j <= indices.je && i - radius >= 0 &&
         i + radius < total_i && j - radius >= 0 && j + radius < total_j;
}

template <int NGHOST>
inline CartoonCentralSample::Status ValidateCartoonCentralSupportSet(
    const CartoonCentralSupportSet &supports, const RegionIndcs &indices,
    const int nranks) {
  if (supports.gid < 0 || supports.level < 0) {
    return CartoonCentralSample::Status::missing_center_leaf;
  }
  if (supports.construction_status != CartoonCentralSample::Status::valid) {
    return supports.construction_status;
  }
  if (supports.common_level < 0 || !(supports.common_dx1 > 0.0) ||
      !(supports.common_dx2 > 0.0) || !std::isfinite(supports.common_dx1) ||
      !std::isfinite(supports.common_dx2) ||
      (supports.refined_mask & ~0xFU) != 0 ||
      (supports.half_plane && (supports.refined_mask & 0x5U) != 0)) {
    return CartoonCentralSample::Status::invalid_common_lattice;
  }
  int expected_count = 0;
  for (int s = 0; s < kCartoonCentralMaxSources; ++s) {
    const CartoonCentralSupport &point = supports.point[s];
    if (!point.expected) continue;
    ++expected_count;
    if (point.matches == 0) {
      return CartoonCentralSample::Status::missing_support;
    }
    if (point.matches != 1) {
      return CartoonCentralSample::Status::duplicate_support;
    }
    const int quadrant = s / kCartoonCentralSourcesPerQuadrant;
    const int child = s % kCartoonCentralSourcesPerQuadrant;
    const bool refined = (supports.refined_mask & (1U << quadrant)) != 0;
    const int expected_level = supports.common_level + (refined ? 1 : 0);
    const Real rho_sign = (quadrant & 1) == 0 ? -1.0 : 1.0;
    const Real z_sign = (quadrant & 2) == 0 ? -1.0 : 1.0;
    const Real child_rho_sign = (child & 1) == 0 ? -1.0 : 1.0;
    const Real child_z_sign = (child & 2) == 0 ? -1.0 : 1.0;
    const Real expected_rho = 0.5 * rho_sign * supports.common_dx1 +
        (refined ? 0.25 * child_rho_sign * supports.common_dx1 : 0.0);
    const Real expected_z = 0.5 * z_sign * supports.common_dx2 +
        (refined ? 0.25 * child_z_sign * supports.common_dx2 : 0.0);
    const Real rho_scale = std::max(Real(1.0), std::fabs(expected_rho));
    const Real z_scale = std::max(Real(1.0), std::fabs(expected_z));
    const Real tolerance = 128.0 * std::numeric_limits<Real>::epsilon();
    if (point.level != expected_level ||
        std::fabs(point.rho - expected_rho) > tolerance * rho_scale ||
        std::fabs(point.z - expected_z) > tolerance * z_scale ||
        point.restriction_weight != (refined ? 0.25 : 1.0) ||
        point.final_weight != (supports.half_plane
                                   ? (refined ? 0.125 : 0.5)
                                   : (refined ? 0.0625 : 0.25))) {
      return CartoonCentralSample::Status::invalid_common_lattice;
    }
    if (point.owner_rank < 0 || point.owner_rank >= nranks) {
      return CartoonCentralSample::Status::invalid_owner;
    }
    if (!CartoonCentralActiveCellHasStoredDerivativeHalo<NGHOST>(
            indices, point.i, point.j)) {
      return CartoonCentralSample::Status::insufficient_derivative_halo;
    }
    for (int previous = 0; previous < s; ++previous) {
      const CartoonCentralSupport &other = supports.point[previous];
      if (other.expected && point.level == other.level && point.gid == other.gid &&
          point.k == other.k && point.j == other.j && point.i == other.i) {
        return CartoonCentralSample::Status::duplicate_support;
      }
    }
  }
  if (expected_count != supports.source_count) {
    return CartoonCentralSample::Status::invalid_common_lattice;
  }
  Real final_weight_sum = 0.0;
  for (int s = 0; s < kCartoonCentralMaxSources; ++s) {
    if (supports.point[s].expected) final_weight_sum += supports.point[s].final_weight;
  }
  if (std::fabs(final_weight_sum - 1.0) >
      64.0 * std::numeric_limits<Real>::epsilon()) {
    return CartoonCentralSample::Status::invalid_common_lattice;
  }
  return CartoonCentralSample::Status::valid;
}

inline CartoonCentralSample ReconstructCartoonCentralSupportValues(
    const CartoonCentralSupportSet &supports,
    const Real lapse[kCartoonCentralMaxSources],
    const Real constraint_squared[kCartoonCentralMaxSources],
    const Real kretschmann[kCartoonCentralMaxSources]) {
  CartoonCentralSample sample;
  Real lapse_sum = 0.0;
  Real constraint_sum = 0.0;
  Real kretschmann_sum = 0.0;
  for (int slot = 0; slot < kCartoonCentralMaxSources; ++slot) {
    if (!supports.point[slot].expected) continue;
    if (!std::isfinite(lapse[slot]) || lapse[slot] < 0.0 ||
        !std::isfinite(constraint_squared[slot]) ||
        constraint_squared[slot] < 0.0 || !std::isfinite(kretschmann[slot])) {
      sample.status = CartoonCentralSample::Status::nonfinite_support;
      return sample;
    }
    const Real weight = supports.point[slot].final_weight;
    lapse_sum += weight * lapse[slot];
    constraint_sum += weight * constraint_squared[slot];
    kretschmann_sum += weight * kretschmann[slot];
  }
  sample.lapse = lapse_sum;
  sample.constraint_norm = Z4cAggregateConstraintNorm(constraint_sum);
  sample.abs_kretschmann = std::fabs(kretschmann_sum);
  sample.valid = std::isfinite(sample.lapse) &&
                 std::isfinite(sample.constraint_norm) &&
                 std::isfinite(sample.abs_kretschmann);
  sample.status = sample.valid ? CartoonCentralSample::Status::valid
                               : CartoonCentralSample::Status::nonfinite_support;
  return sample;
}

inline CartoonCentralSample ReconstructCartoonCentralFourPoint(
    const Real lapse[4], const Real constraint_squared[4],
    const Real kretschmann[4]) {
  CartoonCentralSupportSet supports;
  InitializeCartoonCentralSupportGeometry(&supports, 0, 1.0, 1.0, 0U);
  Real expanded_lapse[kCartoonCentralMaxSources] = {};
  Real expanded_constraint[kCartoonCentralMaxSources] = {};
  Real expanded_kretschmann[kCartoonCentralMaxSources] = {};
  for (int quadrant = 0; quadrant < kCartoonCentralQuadrants; ++quadrant) {
    const int slot = CartoonCentralSourceSlot(quadrant, 0);
    expanded_lapse[slot] = lapse[quadrant];
    expanded_constraint[slot] = constraint_squared[quadrant];
    expanded_kretschmann[slot] = kretschmann[quadrant];
  }
  return ReconstructCartoonCentralSupportValues(
      supports, expanded_lapse, expanded_constraint, expanded_kretschmann);
}

namespace meridional_detail {

inline bool ContainsOpen(const Real value, const Real lower, const Real upper) {
  const Real scale = std::max({Real(1.0), std::fabs(lower), std::fabs(upper)});
  const Real tolerance = 32.0 * std::numeric_limits<Real>::epsilon() * scale;
  return value > lower + tolerance && value < upper - tolerance;
}

inline bool ResolveActiveCellCenter(const Real value, const Real lower,
                                    const Real upper, const int ncells,
                                    const int active_start, int *index) {
  if (ncells <= 0 || !(upper > lower)) return false;
  const Real dx = (upper - lower) / static_cast<Real>(ncells);
  const Real offset = (value - lower) / dx - 0.5;
  const long long nearest = std::llround(offset);
  const Real scale = std::max(Real(1.0), std::fabs(offset));
  if (std::fabs(offset - static_cast<Real>(nearest)) >
          128.0 * std::numeric_limits<Real>::epsilon() * scale ||
      nearest < 0 || nearest >= ncells) {
    return false;
  }
  *index = active_start + static_cast<int>(nearest);
  return true;
}

template <int NGHOST>
inline CartoonCentralSupportSet BuildCartoonCentralSupportSet(
    Mesh *mesh, const CartoonMeridionalStencil &center) {
  CartoonCentralSupportSet supports;
  supports.gid = center.gid;
  supports.level = center.level;
  if (center.gid < 0 || center.level < mesh->root_level ||
      center.gid >= mesh->nmb_total) {
    supports.construction_status =
        CartoonCentralSample::Status::missing_center_leaf;
    return supports;
  }
  const bool half_plane = mesh->mesh_size.x1min == 0.0;

  int quadrant_level[kCartoonCentralQuadrants] = {-1, -1, -1, -1};
  int quadrant_matches[kCartoonCentralQuadrants] = {};
  bool aligned = true;
  for (int gid = 0; gid < mesh->nmb_total; ++gid) {
    Real x1min = 0.0;
    Real x1max = 0.0;
    Real x2min = 0.0;
    Real x2max = 0.0;
    const LogicalLocation &location = mesh->lloc_eachmb[gid];
    LogicalEdges(*mesh, location, &x1min, &x1max, &x2min, &x2max);
    const Real dx1 = (x1max - x1min) / mesh->mb_indcs.nx1;
    const Real dx2 = (x2max - x2min) / mesh->mb_indcs.nx2;
    if (!(dx1 > 0.0) || !(dx2 > 0.0) || !std::isfinite(dx1) ||
        !std::isfinite(dx2)) {
      aligned = false;
      continue;
    }
    const Real scale =
        std::max({Real(1.0), std::fabs(x1min), std::fabs(x1max),
                  std::fabs(x2min), std::fabs(x2max)});
    const Real tolerance =
        128.0 * std::numeric_limits<Real>::epsilon() * scale;
    for (int quadrant = 0; quadrant < kCartoonCentralQuadrants; ++quadrant) {
      if (half_plane && (quadrant & 1) == 0) continue;
      const bool positive_rho = (quadrant & 1) != 0;
      const bool positive_z = (quadrant & 2) != 0;
      const bool enters_rho = positive_rho
          ? (x1max > tolerance && x1min <= tolerance)
          : (x1min < -tolerance && x1max >= -tolerance);
      const bool enters_z = positive_z
          ? (x2max > tolerance && x2min <= tolerance)
          : (x2min < -tolerance && x2max >= -tolerance);
      if (!enters_rho || !enters_z) continue;
      const Real face_i = -x1min / dx1;
      const Real face_j = -x2min / dx2;
      const long long nearest_i = std::llround(face_i);
      const long long nearest_j = std::llround(face_j);
      const bool face_aligned =
          std::fabs(face_i - static_cast<Real>(nearest_i)) <=
              128.0 * std::numeric_limits<Real>::epsilon() *
                  std::max(Real(1.0), std::fabs(face_i)) &&
          std::fabs(face_j - static_cast<Real>(nearest_j)) <=
              128.0 * std::numeric_limits<Real>::epsilon() *
                  std::max(Real(1.0), std::fabs(face_j)) &&
          nearest_i >= 0 && nearest_i <= mesh->mb_indcs.nx1 &&
          nearest_j >= 0 && nearest_j <= mesh->mb_indcs.nx2;
      aligned = aligned && face_aligned;
      ++quadrant_matches[quadrant];
      quadrant_level[quadrant] = location.level;
    }
  }
  for (int quadrant = 0; quadrant < kCartoonCentralQuadrants; ++quadrant) {
    if (half_plane && (quadrant & 1) == 0) continue;
    if (quadrant_matches[quadrant] == 0) {
      supports.construction_status =
          CartoonCentralSample::Status::missing_support;
      return supports;
    }
    if (quadrant_matches[quadrant] != 1) {
      supports.construction_status =
          CartoonCentralSample::Status::duplicate_support;
      return supports;
    }
  }
  if (!aligned) {
    supports.construction_status =
        CartoonCentralSample::Status::invalid_common_lattice;
    return supports;
  }
  int common_level = quadrant_level[half_plane ? 1 : 0];
  for (int quadrant = 0; quadrant < kCartoonCentralQuadrants; ++quadrant) {
    if (half_plane && (quadrant & 1) == 0) continue;
    common_level = std::min(common_level, quadrant_level[quadrant]);
  }
  unsigned int refined_mask = 0U;
  for (int quadrant = 0; quadrant < kCartoonCentralQuadrants; ++quadrant) {
    if (half_plane && (quadrant & 1) == 0) continue;
    if (quadrant_level[quadrant] == common_level + 1) {
      refined_mask |= 1U << quadrant;
    } else if (quadrant_level[quadrant] != common_level) {
      supports.construction_status =
          CartoonCentralSample::Status::unsupported_level_gap;
      return supports;
    }
  }

  const int level_offset = common_level - mesh->root_level;
  if (level_offset < 0 || level_offset >= 62) {
    supports.construction_status =
        CartoonCentralSample::Status::invalid_common_lattice;
    return supports;
  }
  const std::int64_t blocks_x1 =
      static_cast<std::int64_t>(mesh->nmb_rootx1) << level_offset;
  const std::int64_t blocks_x2 =
      static_cast<std::int64_t>(mesh->nmb_rootx2) << level_offset;
  if (blocks_x1 <= 0 || blocks_x2 <= 0 || mesh->mb_indcs.nx1 <= 0 ||
      mesh->mb_indcs.nx2 <= 0) {
    supports.construction_status =
        CartoonCentralSample::Status::invalid_common_lattice;
    return supports;
  }
  const Real dx1 = (mesh->mesh_size.x1max - mesh->mesh_size.x1min) /
                   (static_cast<Real>(blocks_x1) * mesh->mb_indcs.nx1);
  const Real dx2 = (mesh->mesh_size.x2max - mesh->mesh_size.x2min) /
                   (static_cast<Real>(blocks_x2) * mesh->mb_indcs.nx2);
  if (!(dx1 > 0.0) || !(dx2 > 0.0) || !std::isfinite(dx1) ||
      !std::isfinite(dx2)) {
    supports.construction_status =
        CartoonCentralSample::Status::invalid_common_lattice;
    return supports;
  }
  InitializeCartoonCentralSupportGeometry(
      &supports, common_level, dx1, dx2, refined_mask, half_plane);

  for (int s = 0; s < kCartoonCentralMaxSources; ++s) {
    CartoonCentralSupport &point = supports.point[s];
    if (!point.expected) continue;
    for (int gid = 0; gid < mesh->nmb_total; ++gid) {
      Real x1min = 0.0;
      Real x1max = 0.0;
      Real x2min = 0.0;
      Real x2max = 0.0;
      const LogicalLocation &location = mesh->lloc_eachmb[gid];
      LogicalEdges(*mesh, location, &x1min, &x1max, &x2min, &x2max);
      if (location.level != point.level ||
          !ContainsOpen(point.rho, x1min, x1max) ||
          !ContainsOpen(point.z, x2min, x2max)) {
        continue;
      }
      int i = 0;
      int j = 0;
      if (!ResolveActiveCellCenter(point.rho, x1min, x1max,
                                   mesh->mb_indcs.nx1, mesh->mb_indcs.is, &i) ||
          !ResolveActiveCellCenter(point.z, x2min, x2max,
                                   mesh->mb_indcs.nx2, mesh->mb_indcs.js, &j)) {
        continue;
      }
      ++point.matches;
      if (point.matches != 1) continue;
      point.gid = gid;
      point.owner_rank = mesh->rank_eachmb[gid];
      point.k = mesh->mb_indcs.ks;
      point.i = i;
      point.j = j;
      if (point.owner_rank == global_variable::my_rank &&
          point.owner_rank >= 0 && point.owner_rank < global_variable::nranks) {
        point.local_block = gid - mesh->gids_eachrank[point.owner_rank];
      }
    }
  }
  return supports;
}

}  // namespace meridional_detail

template <int NGHOST>
inline CartoonCentralSample SampleCartoonCentralDiagnostics(Mesh *mesh) {
  CartoonCentralSample sample;
  const CartoonMeridionalStencil center =
      LocateCartoonMeridionalPoint(mesh, 0.0, 0.0);
  sample.gid = center.gid;
  sample.level = center.level;
  const CartoonCentralSupportSet supports =
      meridional_detail::BuildCartoonCentralSupportSet<NGHOST>(mesh, center);
  CartoonCentralSample::Status local_status =
      ValidateCartoonCentralSupportSet<NGHOST>(
          supports, mesh->mb_indcs, global_variable::nranks);
  if (local_status == CartoonCentralSample::Status::valid) {
    for (int s = 0; s < kCartoonCentralMaxSources; ++s) {
      const CartoonCentralSupport &point = supports.point[s];
      if (!point.expected) continue;
      if (point.owner_rank == global_variable::my_rank &&
          (point.local_block < 0 ||
           point.local_block >= mesh->pmb_pack->nmb_thispack)) {
        local_status = CartoonCentralSample::Status::invalid_owner;
      }
    }
  }

  Kokkos::View<Real *> local_values("Cartoon central scalar slots",
                                     3 * kCartoonCentralMaxSources);
  Kokkos::View<int *> local_flags("Cartoon central scalar flags",
                                   2 * kCartoonCentralMaxSources);
  Kokkos::deep_copy(local_values, 0.0);
  Kokkos::deep_copy(local_flags, 0);
  if (local_status == CartoonCentralSample::Status::valid) {
    auto u0 = mesh->pmb_pack->pz4c->u0;
    auto constraints = mesh->pmb_pack->pz4c->u_con;
    auto adm = mesh->pmb_pack->padm->adm;
    auto size = mesh->pmb_pack->pmb->mb_size.d_view;
    const int alpha = mesh->pmb_pack->pz4c->I_Z4C_ALPHA;
    const int nx1 = mesh->mb_indcs.nx1;
    const int is = mesh->mb_indcs.is;
    Kokkos::parallel_for(
        "Cartoon axis-central physical support diagnostics",
        Kokkos::RangePolicy<DevExeSpace>(0, kCartoonCentralMaxSources),
        KOKKOS_LAMBDA(const int s) {
          const CartoonCentralSupport point = supports.point[s];
          if (!point.expected || point.local_block < 0) return;
          local_flags(s) = 1;
          const Real lapse = u0(point.local_block, alpha, point.k, point.j, point.i);
          const Real constraint_squared =
              constraints(point.local_block, 0, point.k, point.j, point.i);

          const Real inverse_spacing[3] = {
              1.0 / size(point.local_block).dx1,
              1.0 / size(point.local_block).dx2,
              1.0 / size(point.local_block).dx3};
          auto derivatives = MakeCellCenteredDerivativeProvider<CartoonSO2, NGHOST>(
              inverse_spacing, size, nx1, is, point.local_block, point.k,
              point.j, point.i);
          const auto diagnostic = ComputeZ4cCurvatureDiagnostics<NGHOST, false>(
              derivatives, adm.g_dd, adm.vK_dd, point.local_block, point.k,
              point.j, point.i);
          if (!Kokkos::isfinite(lapse) || lapse < 0.0 ||
              !Kokkos::isfinite(constraint_squared) || constraint_squared < 0.0 ||
              !diagnostic.valid || !Kokkos::isfinite(diagnostic.kretschmann)) {
            local_flags(kCartoonCentralMaxSources + s) = 1;
            return;
          }
          local_values(3 * s) = lapse;
          local_values(3 * s + 1) = constraint_squared;
          local_values(3 * s + 2) = diagnostic.kretschmann;
        });
  }

  auto host_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_values);
  auto host_flags =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_flags);
  Real values[3 * kCartoonCentralMaxSources] = {};
  int flags[2 * kCartoonCentralMaxSources] = {};
  for (int n = 0; n < 3 * kCartoonCentralMaxSources; ++n) {
    values[n] = host_values(n);
  }
  for (int n = 0; n < 2 * kCartoonCentralMaxSources; ++n) {
    flags[n] = host_flags(n);
  }
  int status_code = static_cast<int>(local_status);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 3 * kCartoonCentralMaxSources,
                MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, flags, 2 * kCartoonCentralMaxSources, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &status_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  sample.status = static_cast<CartoonCentralSample::Status>(status_code);
  if (sample.status != CartoonCentralSample::Status::valid) return sample;

  Real lapse[kCartoonCentralMaxSources] = {};
  Real constraint_squared[kCartoonCentralMaxSources] = {};
  Real kretschmann[kCartoonCentralMaxSources] = {};
  for (int s = 0; s < kCartoonCentralMaxSources; ++s) {
    if (!supports.point[s].expected) continue;
    if (flags[s] != 1) {
      sample.status = CartoonCentralSample::Status::invalid_owner;
      return sample;
    }
    if (flags[kCartoonCentralMaxSources + s] != 0) {
      sample.status = CartoonCentralSample::Status::nonfinite_support;
      return sample;
    }
    lapse[s] = values[3 * s];
    constraint_squared[s] = values[3 * s + 1];
    kretschmann[s] = values[3 * s + 2];
  }
  CartoonCentralSample reconstruction = ReconstructCartoonCentralSupportValues(
      supports, lapse, constraint_squared, kretschmann);
  sample.valid = reconstruction.valid;
  sample.lapse = reconstruction.lapse;
  sample.constraint_norm = reconstruction.constraint_norm;
  sample.abs_kretschmann = reconstruction.abs_kretschmann;
  sample.status = reconstruction.status;
  return sample;
}

//! Sample the synchronized physical origin directly on the native VC grid.
//!
//! Unlike the legacy CC diagnostic, no four-cell reconstruction is required:
//! rho=0,z=0 is an evolved vertex.  The globally selected leaf is deterministic
//! at same-level block interfaces, and shared-node exchange makes duplicate
//! copies bitwise identical before this accepted-state diagnostic.
template <int NGHOST>
inline CartoonCentralSample SampleCartoonCentralVertexDiagnostics(Mesh *mesh) {
  CartoonCentralSample sample;
  const CartoonMeridionalStencil center =
      LocateNativeCartoonMeridionalPoint(mesh, 0.0, 0.0);
  sample.gid = center.gid;
  sample.level = center.level;
  const Real tolerance = 128.0 * std::numeric_limits<Real>::epsilon();
  if (!center.valid || std::fabs(center.wi) > tolerance ||
      std::fabs(center.wj) > tolerance) {
    sample.status = center.valid
                        ? CartoonCentralSample::Status::invalid_common_lattice
                        : CartoonCentralSample::Status::missing_center_leaf;
    return sample;
  }

  Kokkos::View<Real *> local_values("Cartoon native vertex center values", 3);
  Kokkos::View<int *> local_flags("Cartoon native vertex center flags", 2);
  Kokkos::deep_copy(local_values, 0.0);
  Kokkos::deep_copy(local_flags, 0);
  if (center.owner_rank == global_variable::my_rank) {
    if (center.local_block < 0 ||
        center.local_block >= mesh->pmb_pack->nmb_thispack) {
      sample.status = CartoonCentralSample::Status::invalid_owner;
      return sample;
    }
    auto u0 = mesh->pmb_pack->pz4c->u0;
    auto constraints = mesh->pmb_pack->pz4c->u_con;
    auto metric = mesh->pmb_pack->pz4c->adm.g_dd;
    auto curvature = mesh->pmb_pack->pz4c->adm.vK_dd;
    auto size = mesh->pmb_pack->pmb->mb_size.d_view;
    const auto layout = mesh->pmb_pack->pz4c->layout;
    const int alpha = mesh->pmb_pack->pz4c->I_Z4C_ALPHA;
    Kokkos::parallel_for(
        "Cartoon native vertex center diagnostic",
        Kokkos::RangePolicy<DevExeSpace>(0, 1), KOKKOS_LAMBDA(const int) {
          const int m = center.local_block;
          const int k = center.k;
          const int j = center.j0;
          const int i = center.i0;
          const Real inverse_spacing[3] = {
              1.0 / size(m).dx1, 1.0 / size(m).dx2, 1.0 / size(m).dx3};
          auto derivatives =
              MakeVertexCenteredDerivativeProvider<CartoonSO2, NGHOST>(
                  inverse_spacing, size, layout.nx1, layout.is, m, k, j, i);
          const auto diagnostic = ComputeZ4cCurvatureDiagnostics<NGHOST, false>(
              derivatives, metric, curvature, m, k, j, i);
          const Real lapse = u0(m, alpha, k, j, i);
          const Real constraint_squared = constraints(m, 0, k, j, i);
          local_flags(0) = 1;
          if (!Kokkos::isfinite(lapse) || lapse < 0.0 ||
              !Kokkos::isfinite(constraint_squared) ||
              constraint_squared < 0.0 || !diagnostic.valid ||
              !Kokkos::isfinite(diagnostic.kretschmann)) {
            local_flags(1) = 1;
            return;
          }
          local_values(0) = lapse;
          local_values(1) = constraint_squared;
          local_values(2) = Kokkos::fabs(diagnostic.kretschmann);
        });
    Kokkos::fence();
  }
  auto host_values =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_values);
  auto host_flags =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), local_flags);
  Real values[3] = {host_values(0), host_values(1), host_values(2)};
  int flags[2] = {host_flags(0), host_flags(1)};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 3, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, flags, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (flags[0] != 1) {
    sample.status = flags[0] == 0
                        ? CartoonCentralSample::Status::missing_support
                        : CartoonCentralSample::Status::duplicate_support;
    return sample;
  }
  if (flags[1] != 0) {
    sample.status = CartoonCentralSample::Status::nonfinite_support;
    return sample;
  }
  sample.lapse = values[0];
  sample.constraint_norm = Z4cAggregateConstraintNorm(values[1]);
  sample.abs_kretschmann = values[2];
  sample.valid = std::isfinite(sample.lapse) &&
                 std::isfinite(sample.constraint_norm) &&
                 std::isfinite(sample.abs_kretschmann);
  sample.status = sample.valid ? CartoonCentralSample::Status::valid
                               : CartoonCentralSample::Status::nonfinite_support;
  return sample;
}

inline const char *CartoonCentralSampleStatusMessage(
    const CartoonCentralSample::Status status) {
  switch (status) {
    case CartoonCentralSample::Status::valid:
      return "";
    case CartoonCentralSample::Status::missing_center_leaf:
      return "axis-central diagnostic has no finest center leaf";
    case CartoonCentralSample::Status::invalid_common_lattice:
      return "axis-central diagnostic has invalid common-level geometry";
    case CartoonCentralSample::Status::unsupported_level_gap:
      return "axis-central diagnostic support exceeds a 2:1 level gap";
    case CartoonCentralSample::Status::missing_support:
      return "axis-central diagnostic is missing a physical support cell";
    case CartoonCentralSample::Status::duplicate_support:
      return "axis-central diagnostic has duplicate physical support cells";
    case CartoonCentralSample::Status::invalid_owner:
      return "axis-central diagnostic support ownership is not unique";
    case CartoonCentralSample::Status::insufficient_derivative_halo:
      return "axis-central diagnostic support lacks a stored derivative halo";
    case CartoonCentralSample::Status::nonfinite_support:
      return "axis-central diagnostic support is nonfinite or invalid";
  }
  return "axis-central diagnostic has an unknown failure status";
}

inline CartoonCentralSample DispatchCartoonCentralDiagnostics(Mesh *mesh) {
  const bool vertex = mesh->pmb_pack->pz4c->layout.centering ==
                      Z4cGridCentering::vertex;
  switch (mesh->pmb_pack->z4c_symmetry.stencil_width) {
    case 2:
      return vertex ? SampleCartoonCentralVertexDiagnostics<2>(mesh)
                    : SampleCartoonCentralDiagnostics<2>(mesh);
    case 3:
      return vertex ? SampleCartoonCentralVertexDiagnostics<3>(mesh)
                    : SampleCartoonCentralDiagnostics<3>(mesh);
    case 4:
      return vertex ? SampleCartoonCentralVertexDiagnostics<4>(mesh)
                    : SampleCartoonCentralDiagnostics<4>(mesh);
    default:
      return {};
  }
}

//! Initialize or advance the restart-authoritative central proper-time state.
//! Empty return means success; a nonempty string is a fail-closed diagnostic.
inline std::string UpdateCartoonCentralState(Mesh *mesh,
                                             const bool restart_initialization) {
  if (mesh->pmb_pack->z4c_symmetry.mode != Z4cSymmetryMode::cartoon_so2) return {};
  const CartoonCentralSample sample = DispatchCartoonCentralDiagnostics(mesh);
  if (!sample.valid) return CartoonCentralSampleStatusMessage(sample.status);

  Z4cCentralRestartState &state = mesh->pmb_pack->z4c_restart_state.central;
  const auto result = UpdateZ4cCentralRestartState(
      &state, sample.lapse, sample.constraint_norm, sample.abs_kretschmann,
      sample.gid, sample.level, mesh->ncycle, mesh->time,
      restart_initialization);
  return result.valid ? std::string{} : result.error;
}

}  // namespace z4c

#endif  // Z4C_CARTOON_MERIDIONAL_SAMPLER_HPP_
