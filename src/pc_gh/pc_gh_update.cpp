//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_update.cpp
//! \brief explicit Runge-Kutta register operations for PC-GH

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"

namespace pc_gh {

TaskStatus PcGh::CopyU(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nmb = pmy_pack->nmb_thispack;
  if (pdriver->integrator == "rk4") {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      Real const delta = pdriver->delta[stage - 1];
      par_for("PC-GH RK4 register accumulation", DevExeSpace(),
      0, nmb - 1, 0, npcgh - 1,
      indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        u1(m, n, k, j, i) += delta*u0(m, n, k, j, i);
      });
    }
  } else if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nmb = pmy_pack->nmb_thispack;
  Real const gam0 = pdriver->gam0[stage - 1];
  Real const gam1 = pdriver->gam1[stage - 1];
  Real const beta_dt = pdriver->beta[stage - 1]*pmy_pack->pmesh->dt;

  par_for("PC-GH RK update", DevExeSpace(),
  0, nmb - 1, 0, npcgh - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    u0(m, n, k, j, i) = gam0*u0(m, n, k, j, i)
                         + gam1*u1(m, n, k, j, i)
                         + beta_dt*u_rhs(m, n, k, j, i);
  });
  return TaskStatus::complete;
}

}  // namespace pc_gh
