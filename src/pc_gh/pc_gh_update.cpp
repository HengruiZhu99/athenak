//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_update.cpp
//! \brief explicit Runge-Kutta register operations for PC-GH

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "utils/compact_object_tracker.hpp"

namespace pc_gh {

TaskStatus PcGh::CopyU(Driver *pdriver, int stage) {
  reduction_monitor_stage = stage;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nmb = pmy_pack->nmb_thispack;
  int const nvars = EvolvedVariables();
  if (stage == 1) {
    for (auto &operation : transfer_reduction_change) operation.fill(0.0);
  }
  if (opt.reduction_follow_trackers) {
    for (std::size_t n=0; n<ptracker.size(); ++n) {
      for (int a=0; a<3; ++a) {
        if (stage == 1) reduction_tracker_register[n][a] = ptracker[n]->GetPos(a);
        else if (pdriver->integrator == "rk4") {
          reduction_tracker_register[n][a] += pdriver->delta[stage-1]*ptracker[n]->GetPos(a);
        }
      }
    }
  }
  if (pdriver->integrator == "rk4") {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      Real const delta = pdriver->delta[stage - 1];
      auto state = u0;
      auto accumulator = u1;
      par_for("PC-GH RK4 register accumulation", DevExeSpace(),
      0, nmb - 1, 0, nvars - 1,
      indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
        accumulator(m, n, k, j, i) += delta*state(m, n, k, j, i);
      });
    }
  } else if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

TaskStatus PcGh::ExpRKUpdate(Driver *pdriver, int stage) {
  BeginStateBudget(-10);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nmb = pmy_pack->nmb_thispack;
  int const nvars = EvolvedVariables();
  Real const gam0 = pdriver->gam0[stage - 1];
  Real const gam1 = pdriver->gam1[stage - 1];
  Real const beta_dt = pdriver->beta[stage - 1]*pmy_pack->pmesh->dt;

  // Capture Kokkos views by value; a captured host-side PcGh this pointer is
  // invalid in CUDA device code even when the referenced views hold device data.
  auto state = u0;
  auto accumulator = u1;
  auto source = u_rhs;
  par_for("PC-GH RK update", DevExeSpace(),
  0, nmb - 1, 0, nvars - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
    state(m, n, k, j, i) = gam0*state(m, n, k, j, i)
                         + gam1*accumulator(m, n, k, j, i)
                         + beta_dt*source(m, n, k, j, i);
  });
  Kokkos::fence();
  EndStateBudget(-10);
  ValidateState("post-RK update", false, false);
  if (opt.reduction_follow_trackers) {
    for (std::size_t n=0; n<ptracker.size(); ++n) {
      Real position[3];
      for (int a=0; a<3; ++a) {
        position[a] = gam0*ptracker[n]->GetPos(a)
            + gam1*reduction_tracker_register[n][a]+beta_dt*ptracker[n]->GetVelocity(a);
        if (!std::isfinite(position[a])) {
          std::cout << "### FATAL ERROR: nonfinite RK compact-object position" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      ptracker[n]->SetPos(position);
    }
  }
  return TaskStatus::complete;
}

}  // namespace pc_gh
