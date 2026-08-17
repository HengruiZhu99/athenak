//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh_tasks.cpp
//! \brief Driver tasks for the standalone regularized vacuum FO-GH system.

#include <algorithm>
#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "driver/driver.hpp"
#include "fo_gh/fo_gh.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "tasklist/numerical_relativity.hpp"

namespace fo_gh {

void FoGh::QueueTasks() {
  using namespace numrel; // NOLINT(build/namespaces)
  auto *pnr = pmy_pack->pnr;
  pnr->QueueTask(&FoGh::InitRecv, this, FoGh_Recv, "FoGh_Recv", Task_Start);
  pnr->QueueTask(&FoGh::CopyU, this, FoGh_CopyU, "FoGh_CopyU", Task_Run);
  switch (opt.fd_order) {
    case 2:
      pnr->QueueTask(&FoGh::CalcRHS<2>, this, FoGh_CalcRHS, "FoGh_CalcRHS",
                     Task_Run, {FoGh_CopyU});
      break;
    case 4:
      pnr->QueueTask(&FoGh::CalcRHS<3>, this, FoGh_CalcRHS, "FoGh_CalcRHS",
                     Task_Run, {FoGh_CopyU});
      break;
    case 6:
      pnr->QueueTask(&FoGh::CalcRHS<4>, this, FoGh_CalcRHS, "FoGh_CalcRHS",
                     Task_Run, {FoGh_CopyU});
      break;
  }
  pnr->QueueTask(&FoGh::ExpRKUpdate, this, FoGh_ExplRK, "FoGh_ExplRK", Task_Run,
                 {FoGh_CalcRHS});
  pnr->QueueTask(&FoGh::RestrictU, this, FoGh_RestU, "FoGh_RestU", Task_Run,
                 {FoGh_ExplRK});
  pnr->QueueTask(&FoGh::SendU, this, FoGh_SendU, "FoGh_SendU", Task_Run,
                 {FoGh_RestU});
  pnr->QueueTask(&FoGh::RecvU, this, FoGh_RecvU, "FoGh_RecvU", Task_Run,
                 {FoGh_SendU});
  pnr->QueueTask(&FoGh::Prolongate, this, FoGh_Prolong, "FoGh_Prolong", Task_Run,
                 {FoGh_RecvU});
  pnr->QueueTask(&FoGh::ApplyPhysicalBCs, this, FoGh_BCS, "FoGh_BCS", Task_Run,
                 {FoGh_Prolong});
  pnr->QueueTask(&FoGh::NewTimeStep, this, FoGh_Newdt, "FoGh_Newdt", Task_Run,
                 {FoGh_BCS});
  pnr->QueueTask(&FoGh::ClearSend, this, FoGh_ClearS, "FoGh_ClearS", Task_End);
  pnr->QueueTask(&FoGh::ClearRecv, this, FoGh_ClearR, "FoGh_ClearR", Task_End,
                 {FoGh_ClearS});
}

TaskStatus FoGh::InitRecv(Driver *pdriver, int stage) {
  return pbval_u->InitRecv(nfo_gh);
}

TaskStatus FoGh::ClearRecv(Driver *pdriver, int stage) {
  return pbval_u->ClearRecv();
}

TaskStatus FoGh::ClearSend(Driver *pdriver, int stage) {
  return pbval_u->ClearSend();
}

TaskStatus FoGh::CopyU(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (pdriver->integrator == "rk4") {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      const Real delta = pdriver->delta[stage - 1];
      const auto state = u0;
      const auto base = u1;
      par_for("fo_gh copy", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
      0, nfo_gh - 1, indcs.ks, indcs.ke, indcs.js, indcs.je,
      indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        base(m, n, k, j, i) += delta*state(m, n, k, j, i);
      });
    }
  } else if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

TaskStatus FoGh::ExpRKUpdate(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const Real gam0 = pdriver->gam0[stage - 1];
  const Real gam1 = pdriver->gam1[stage - 1];
  const Real beta_dt = pdriver->beta[stage - 1]*pmy_pack->pmesh->dt;
  const auto state = u0;
  const auto base = u1;
  const auto state_rhs = u_rhs;
  par_for("fo_gh RK update", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, nfo_gh - 1, indcs.ks, indcs.ke, indcs.js, indcs.je,
  indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    state(m, n, k, j, i) = gam0*state(m, n, k, j, i)
                              + gam1*base(m, n, k, j, i)
                              + beta_dt*state_rhs(m, n, k, j, i);
  });
  return TaskStatus::complete;
}

TaskStatus FoGh::RestrictU(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

TaskStatus FoGh::SendU(Driver *pdriver, int stage) {
  return pbval_u->PackAndSendCC(u0, coarse_u0);
}

TaskStatus FoGh::RecvU(Driver *pdriver, int stage) {
  return pbval_u->RecvAndUnpackCC(u0, coarse_u0);
}

TaskStatus FoGh::Prolongate(Driver *pdriver, int stage) {
  if (pmy_pack->pmesh->multilevel) {
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

TaskStatus FoGh::ApplyPhysicalBCs(Driver *pdriver, int stage) {
  // Periodic boundaries are completed by RecvU. Component-aware vacuum outer
  // boundary conditions are added separately; silently borrowing fluid BCs is unsafe.
  return TaskStatus::complete;
}

TaskStatus FoGh::NewTimeStep(Driver *pdriver, int stage) {
  if (stage != pdriver->nexp_stages) {
    return TaskStatus::complete;
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real minimum = std::numeric_limits<float>::max();
  Kokkos::parallel_reduce(
      "fo_gh dt", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_minimum) {
        const int m = idx/ncells;
        const Real dx = fmin(size.d_view(m).dx1,
                            fmin(size.d_view(m).dx2, size.d_view(m).dx3));
        local_minimum = fmin(local_minimum, dx);
      }, Kokkos::Min<Real>(minimum));
  dtnew = minimum;
  return TaskStatus::complete;
}

} // namespace fo_gh
