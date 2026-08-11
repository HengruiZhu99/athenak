//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_meridional_sampler.hpp
//! \brief Shared geometry and scalar sampling on the signed-rho Cartoon plane.

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
//! Cartoon uses only the positive-rho half of the signed meridional plane.
KOKKOS_INLINE_FUNCTION
Real Z4cDiagnosticCellMeasure(const Z4cSymmetryMode mode, const Real signed_rho,
                              const Real dx1, const Real dx2, const Real dx3,
                              const Real spatial_determinant) {
  const Real proper_factor = Kokkos::sqrt(Kokkos::fabs(spatial_determinant));
  if (mode == Z4cSymmetryMode::cartoon_so2) {
    return signed_rho > 0.0
               ? kCartoonTwoPi * signed_rho * dx1 * dx2 * proper_factor
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
inline CartoonMeridionalStencil LocateCartoonMeridionalPoint(
    Mesh *mesh, const Real signed_rho, const Real z) {
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
    if (!meridional_detail::ContainsClosed(signed_rho, x1min, x1max) ||
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
  const Real dx1 = (selected_x1max - selected_x1min) / indices.nx1;
  const Real dx2 = (selected_x2max - selected_x2min) / indices.nx2;
  const Real offset_i = (signed_rho - selected_x1min) / dx1 - 0.5;
  const Real offset_j = (z - selected_x2min) / dx2 - 0.5;
  const int lower_i = static_cast<int>(std::floor(offset_i));
  const int lower_j = static_cast<int>(std::floor(offset_j));
  stencil.i0 = indices.is + lower_i;
  stencil.j0 = indices.js + lower_j;
  stencil.k = indices.ks;
  stencil.wi = offset_i - lower_i;
  stencil.wj = offset_j - lower_j;
  const int total_i = indices.nx1 + 2 * indices.ng;
  const int total_j = indices.nx2 + 2 * indices.ng;
  if (stencil.i0 < 0 || stencil.i0 + 1 >= total_i || stencil.j0 < 0 ||
      stencil.j0 + 1 >= total_j) {
    stencil.valid = false;
  }
  return stencil;
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
    missing_support,
    duplicate_support,
    mixed_level_support,
    invalid_owner,
    insufficient_derivative_halo,
    nonfinite_support
  } status = Status::missing_center_leaf;
};

struct CartoonCentralSupport {
  int matches = 0;
  bool covered_at_other_level = false;
  int gid = -1;
  int level = -1;
  int owner_rank = -1;
  int local_block = -1;
  int k = 0;
  int i = 0;
  int j = 0;
  Real rho = 0.0;
  Real z = 0.0;
};

struct CartoonCentralSupportSet {
  CartoonCentralSupport point[4];
  int gid = -1;
  int level = -1;
};

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
  for (int s = 0; s < 4; ++s) {
    const CartoonCentralSupport &point = supports.point[s];
    if (point.matches == 0) {
      return point.covered_at_other_level
                 ? CartoonCentralSample::Status::mixed_level_support
                 : CartoonCentralSample::Status::missing_support;
    }
    if (point.matches != 1) {
      return CartoonCentralSample::Status::duplicate_support;
    }
    if (point.level != supports.level) {
      return CartoonCentralSample::Status::mixed_level_support;
    }
    if (point.owner_rank < 0 || point.owner_rank >= nranks) {
      return CartoonCentralSample::Status::invalid_owner;
    }
    if (!CartoonCentralActiveCellHasStoredDerivativeHalo<NGHOST>(
            indices, point.i, point.j)) {
      return CartoonCentralSample::Status::insufficient_derivative_halo;
    }
    for (int previous = 0; previous < s; ++previous) {
      if (point.rho == supports.point[previous].rho &&
          point.z == supports.point[previous].z) {
        return CartoonCentralSample::Status::duplicate_support;
      }
    }
  }
  return CartoonCentralSample::Status::valid;
}

inline CartoonCentralSample ReconstructCartoonCentralFourPoint(
    const Real lapse[4], const Real constraint_squared[4],
    const Real kretschmann[4]) {
  CartoonCentralSample sample;
  Real lapse_sum = 0.0;
  Real constraint_sum = 0.0;
  Real kretschmann_sum = 0.0;
  for (int s = 0; s < 4; ++s) {
    if (!std::isfinite(lapse[s]) || lapse[s] < 0.0 ||
        !std::isfinite(constraint_squared[s]) || constraint_squared[s] < 0.0 ||
        !std::isfinite(kretschmann[s])) {
      sample.status = CartoonCentralSample::Status::nonfinite_support;
      return sample;
    }
    lapse_sum += lapse[s];
    constraint_sum += constraint_squared[s];
    kretschmann_sum += kretschmann[s];
  }
  sample.lapse = 0.25 * lapse_sum;
  sample.constraint_norm = Z4cAggregateConstraintNorm(0.25 * constraint_sum);
  sample.abs_kretschmann = std::fabs(0.25 * kretschmann_sum);
  sample.valid = std::isfinite(sample.lapse) &&
                 std::isfinite(sample.constraint_norm) &&
                 std::isfinite(sample.abs_kretschmann);
  sample.status = sample.valid ? CartoonCentralSample::Status::valid
                               : CartoonCentralSample::Status::nonfinite_support;
  return sample;
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
    return supports;
  }

  const int level_offset = center.level - mesh->root_level;
  const std::int64_t blocks_x1 =
      static_cast<std::int64_t>(mesh->nmb_rootx1) << level_offset;
  const std::int64_t blocks_x2 =
      static_cast<std::int64_t>(mesh->nmb_rootx2) << level_offset;
  const Real dx1 = (mesh->mesh_size.x1max - mesh->mesh_size.x1min) /
                   (static_cast<Real>(blocks_x1) * mesh->mb_indcs.nx1);
  const Real dx2 = (mesh->mesh_size.x2max - mesh->mesh_size.x2min) /
                   (static_cast<Real>(blocks_x2) * mesh->mb_indcs.nx2);
  if (!(dx1 > 0.0) || !(dx2 > 0.0) || !std::isfinite(dx1) ||
      !std::isfinite(dx2)) {
    return supports;
  }

  for (int s = 0; s < 4; ++s) {
    CartoonCentralSupport &point = supports.point[s];
    point.rho = (s & 1) == 0 ? -0.5 * dx1 : 0.5 * dx1;
    point.z = (s & 2) == 0 ? -0.5 * dx2 : 0.5 * dx2;
    for (int gid = 0; gid < mesh->nmb_total; ++gid) {
      Real x1min = 0.0;
      Real x1max = 0.0;
      Real x2min = 0.0;
      Real x2max = 0.0;
      const LogicalLocation &location = mesh->lloc_eachmb[gid];
      LogicalEdges(*mesh, location, &x1min, &x1max, &x2min, &x2max);
      if (!ContainsOpen(point.rho, x1min, x1max) ||
          !ContainsOpen(point.z, x2min, x2max)) {
        continue;
      }
      if (location.level != supports.level) {
        point.covered_at_other_level = true;
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
      point.level = location.level;
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
    for (int s = 0; s < 4; ++s) {
      const CartoonCentralSupport &point = supports.point[s];
      if (point.owner_rank == global_variable::my_rank &&
          (point.local_block < 0 ||
           point.local_block >= mesh->pmb_pack->nmb_thispack)) {
        local_status = CartoonCentralSample::Status::invalid_owner;
      }
    }
  }

  array_sum::GlobalSum local;
  if (local_status == CartoonCentralSample::Status::valid) {
    auto u0 = mesh->pmb_pack->pz4c->u0;
    auto constraints = mesh->pmb_pack->pz4c->u_con;
    auto adm = mesh->pmb_pack->padm->adm;
    auto size = mesh->pmb_pack->pmb->mb_size.d_view;
    const int alpha = mesh->pmb_pack->pz4c->I_Z4C_ALPHA;
    const int nx1 = mesh->mb_indcs.nx1;
    const int is = mesh->mb_indcs.is;
    Kokkos::parallel_reduce(
        "Cartoon axis-central physical support diagnostics",
        Kokkos::RangePolicy<DevExeSpace>(0, 4),
        KOKKOS_LAMBDA(const int s, array_sum::GlobalSum &values) {
          const CartoonCentralSupport point = supports.point[s];
          if (point.local_block < 0) return;
          values.the_array[12 + s] = 1.0;
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
            values.the_array[16 + s] = 1.0;
            return;
          }
          values.the_array[3 * s] = lapse;
          values.the_array[3 * s + 1] = constraint_squared;
          values.the_array[3 * s + 2] = diagnostic.kretschmann;
        },
        Kokkos::Sum<array_sum::GlobalSum>(local));
  }

  Real values[20] = {};
  for (int n = 0; n < 20; ++n) values[n] = local.the_array[n];
  int status_code = static_cast<int>(local_status);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 20, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &status_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  sample.status = static_cast<CartoonCentralSample::Status>(status_code);
  if (sample.status != CartoonCentralSample::Status::valid) return sample;

  Real lapse[4] = {};
  Real constraint_squared[4] = {};
  Real kretschmann[4] = {};
  for (int s = 0; s < 4; ++s) {
    if (values[12 + s] != 1.0) {
      sample.status = CartoonCentralSample::Status::invalid_owner;
      return sample;
    }
    if (values[16 + s] != 0.0) {
      sample.status = CartoonCentralSample::Status::nonfinite_support;
      return sample;
    }
    lapse[s] = values[3 * s];
    constraint_squared[s] = values[3 * s + 1];
    kretschmann[s] = values[3 * s + 2];
  }
  CartoonCentralSample reconstruction = ReconstructCartoonCentralFourPoint(
      lapse, constraint_squared, kretschmann);
  sample.valid = reconstruction.valid;
  sample.lapse = reconstruction.lapse;
  sample.constraint_norm = reconstruction.constraint_norm;
  sample.abs_kretschmann = reconstruction.abs_kretschmann;
  sample.status = reconstruction.status;
  return sample;
}

inline const char *CartoonCentralSampleStatusMessage(
    const CartoonCentralSample::Status status) {
  switch (status) {
    case CartoonCentralSample::Status::valid:
      return "";
    case CartoonCentralSample::Status::missing_center_leaf:
      return "axis-central diagnostic has no finest center leaf";
    case CartoonCentralSample::Status::missing_support:
      return "axis-central diagnostic is missing a physical support cell";
    case CartoonCentralSample::Status::duplicate_support:
      return "axis-central diagnostic has duplicate physical support cells";
    case CartoonCentralSample::Status::mixed_level_support:
      return "axis-central diagnostic support spans AMR levels";
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
  switch (mesh->pmb_pack->z4c_symmetry.stencil_width) {
    case 2:
      return SampleCartoonCentralDiagnostics<2>(mesh);
    case 3:
      return SampleCartoonCentralDiagnostics<3>(mesh);
    case 4:
      return SampleCartoonCentralDiagnostics<4>(mesh);
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
