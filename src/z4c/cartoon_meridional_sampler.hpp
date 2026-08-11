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
};

template <int NGHOST>
inline CartoonCentralSample SampleCartoonCentralDiagnostics(Mesh *mesh) {
  CartoonCentralSample sample;
  const CartoonMeridionalStencil stencil =
      LocateCartoonMeridionalPoint(mesh, 0.0, 0.0);
  sample.gid = stencil.gid;
  sample.level = stencil.level;
  if (!stencil.valid) return sample;

  array_sum::GlobalSum local;
  if (stencil.owner_rank == global_variable::my_rank) {
    auto u0 = mesh->pmb_pack->pz4c->u0;
    auto constraints = mesh->pmb_pack->pz4c->u_con;
    auto adm = mesh->pmb_pack->padm->adm;
    auto size = mesh->pmb_pack->pmb->mb_size.d_view;
    const int alpha = mesh->pmb_pack->pz4c->I_Z4C_ALPHA;
    const int nx1 = mesh->mb_indcs.nx1;
    const int is = mesh->mb_indcs.is;
    Kokkos::parallel_reduce(
        "Cartoon axis-central diagnostics", Kokkos::RangePolicy<DevExeSpace>(0, 1),
        KOKKOS_LAMBDA(const int, array_sum::GlobalSum &values) {
          values.the_array[0] =
              SampleCartoonMeridionalScalar(u0, alpha, stencil);
          const Real c = SampleCartoonMeridionalScalar(constraints, 0, stencil);
          values.the_array[1] = Z4cAggregateConstraintNorm(c);

          const Real inverse_spacing[3] = {
              1.0 / size(stencil.local_block).dx1,
              1.0 / size(stencil.local_block).dx2,
              1.0 / size(stencil.local_block).dx3};
          Real kretschmann = 0.0;
          for (int dj = 0; dj <= 1; ++dj) {
            for (int di = 0; di <= 1; ++di) {
              const int i = stencil.i0 + di;
              const int j = stencil.j0 + dj;
              auto derivatives =
                  MakeCellCenteredDerivativeProvider<CartoonSO2, NGHOST>(
                      inverse_spacing, size, nx1, is, stencil.local_block,
                      stencil.k, j, i);
              const auto diagnostic = ComputeZ4cCurvatureDiagnostics<NGHOST, false>(
                  derivatives, adm.g_dd, adm.vK_dd, stencil.local_block, stencil.k,
                  j, i);
              if (!diagnostic.valid) {
                kretschmann = std::numeric_limits<Real>::quiet_NaN();
              } else {
                const Real weight_i = di == 0 ? 1.0 - stencil.wi : stencil.wi;
                const Real weight_j = dj == 0 ? 1.0 - stencil.wj : stencil.wj;
                kretschmann += weight_i * weight_j * diagnostic.kretschmann;
              }
            }
          }
          values.the_array[2] = Kokkos::fabs(kretschmann);
        },
        Kokkos::Sum<array_sum::GlobalSum>(local));
  }

  Real values[3] = {local.the_array[0], local.the_array[1], local.the_array[2]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, values, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  sample.lapse = values[0];
  sample.constraint_norm = values[1];
  sample.abs_kretschmann = values[2];
  sample.valid = std::isfinite(sample.lapse) && sample.lapse >= 0.0 &&
                 std::isfinite(sample.constraint_norm) &&
                 sample.constraint_norm >= 0.0 &&
                 std::isfinite(sample.abs_kretschmann) && sample.gid >= 0 &&
                 sample.level >= 0;
  return sample;
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
  if (!sample.valid) return "nonfinite or uncovered axis-central diagnostic sample";

  Z4cCentralRestartState &state = mesh->pmb_pack->z4c_restart_state.central;
  const auto result = UpdateZ4cCentralRestartState(
      &state, sample.lapse, sample.constraint_norm, sample.abs_kretschmann,
      sample.gid, sample.level, mesh->ncycle, mesh->time,
      restart_initialization);
  return result.valid ? std::string{} : result.error;
}

}  // namespace z4c

#endif  // Z4C_CARTOON_MERIDIONAL_SAMPLER_HPP_
