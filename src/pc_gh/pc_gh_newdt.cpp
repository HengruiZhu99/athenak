//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_newdt.cpp
//! \brief characteristic CFL timestep for the PC-GH system

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"

namespace pc_gh {

TaskStatus PcGh::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) return TaskStatus::complete;
  ValidateState("pre-timestep state", false, false);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &pc = u;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  int const nkji = nx3*nx2*nx1;
  int const nmkji = pmy_pack->nmb_thispack*nkji;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  bool const use_z4c_mp = (opt.gauge == "z4c_mp"
                            || opt.gauge == "z4c_mp_hyperbolic");
  int first_bad_speed = 3*nmkji;
  Kokkos::parallel_reduce("PC-GH characteristic speed validation",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, 3*nmkji),
  KOKKOS_LAMBDA(int flat, int &bad) {
    int const direction = flat % 3;
    int const idx = flat/3;
    if ((direction == 1 && !multi_d) || (direction == 2 && !three_d)) return;
    int const m = idx/nkji;
    int const q = idx - m*nkji;
    int const k0 = q/(nx2*nx1);
    int const j0 = (q - k0*nx2*nx1)/nx1;
    int const i0 = q - k0*nx2*nx1 - j0*nx1;
    int const i = is + i0;
    int const j = js + j0;
    int const k = ks + k0;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_inv;
    Real const det_g = adm::SpatialDet(
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i));
    adm::SpatialInv(1.0/det_g,
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i),
        &g_inv(0, 0), &g_inv(0, 1), &g_inv(0, 2),
        &g_inv(1, 1), &g_inv(1, 2), &g_inv(2, 2));
    Real const w = pc.w(m, k, j, i);
    Real const rho = pc.rho(m, k, j, i);
    Real gauge_factor = rho*w*w;
    if (use_z4c_mp) {
      gauge_factor = std::fmax(gauge_factor, 2.0/std::sqrt(3.0));
      gauge_factor = std::fmax(gauge_factor, std::sqrt(2.0*rho*w*w*w));
    }
    Real const speed = std::fabs(pc.beta(m, direction, k, j, i))
                       + gauge_factor*std::sqrt(g_inv(direction, direction));
    if (!std::isfinite(speed) && flat < bad) bad = flat;
  }, Kokkos::Min<int>(first_bad_speed));
  int local_bad_speed = (first_bad_speed < 3*nmkji) ? 1 : 0;
  int global_bad_speed = local_bad_speed;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_bad_speed, &global_bad_speed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (global_bad_speed != 0) {
    if (local_bad_speed != 0) {
      std::cout << "### FATAL ERROR: PC-GH non-finite characteristic speed at t="
                << pmy_pack->pmesh->time << " on rank " << global_variable::my_rank
                << ", active-cell/direction flat index " << first_bad_speed
                << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
  Real local_dt = std::numeric_limits<float>::max();

  Kokkos::parallel_reduce("PC-GH characteristic dt",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &min_dt) {
    int const m = idx/nkji;
    int const q = idx - m*nkji;
    int const k0 = q/(nx2*nx1);
    int const j0 = (q - k0*nx2*nx1)/nx1;
    int const i0 = q - k0*nx2*nx1 - j0*nx1;
    int const i = is + i0;
    int const j = js + j0;
    int const k = ks + k0;
    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_inv;
    Real const det_g = adm::SpatialDet(
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i));
    adm::SpatialInv(1.0/det_g,
        pc.gtilde(m, 0, 0, k, j, i), pc.gtilde(m, 0, 1, k, j, i),
        pc.gtilde(m, 0, 2, k, j, i), pc.gtilde(m, 1, 1, k, j, i),
        pc.gtilde(m, 1, 2, k, j, i), pc.gtilde(m, 2, 2, k, j, i),
        &g_inv(0, 0), &g_inv(0, 1), &g_inv(0, 2),
        &g_inv(1, 1), &g_inv(1, 2), &g_inv(2, 2));
    Real const w = pc.w(m, k, j, i);
    Real const rho = pc.rho(m, k, j, i);
    Real const physical_factor = rho*w*w;
    Real gauge_factor = physical_factor;
    if (use_z4c_mp) {
      gauge_factor = std::fmax(gauge_factor, 2.0/std::sqrt(3.0));
      gauge_factor = std::fmax(gauge_factor, std::sqrt(2.0*rho*w*w*w));
    }
    Real const speed1 = std::fabs(pc.beta(m, 0, k, j, i))
                        + gauge_factor*std::sqrt(g_inv(0, 0));
    // A CFL estimate is intrinsically a grid-spacing/speed quotient.  The branch
    // avoids division by a zero characteristic speed; this quotient is not used by
    // the evolution RHS and no field floor is applied.
    if (speed1 > 0.0) min_dt = std::fmin(min_dt, size.d_view(m).dx1/speed1);
    if (multi_d) {
      Real const speed2 = std::fabs(pc.beta(m, 1, k, j, i))
                          + gauge_factor*std::sqrt(g_inv(1, 1));
      if (speed2 > 0.0) min_dt = std::fmin(min_dt, size.d_view(m).dx2/speed2);
    }
    if (three_d) {
      Real const speed3 = std::fabs(pc.beta(m, 2, k, j, i))
                          + gauge_factor*std::sqrt(g_inv(2, 2));
      if (speed3 > 0.0) min_dt = std::fmin(min_dt, size.d_view(m).dx3/speed3);
    }
  }, Kokkos::Min<Real>(local_dt));
  if (!(std::isfinite(local_dt) && local_dt > 0.0)) {
    std::cout << "### FATAL ERROR: PC-GH characteristic timestep is non-finite or "
              << "nonpositive at t=" << pmy_pack->pmesh->time << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The mesh applies the user CFL factor to dtnew. Keep lambda*dtnew <= 1
  // for the new explicit relaxation, independently of the spatial mesh speed.
  if (opt.reduction_system == "advective" && opt.reduction_rate > 0.0) {
    local_dt = std::min(local_dt, 1.0/opt.reduction_rate);
  }
  dtnew = local_dt;
  return TaskStatus::complete;
}

}  // namespace pc_gh
