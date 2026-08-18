//========================================================================================
//! \file ref_gh_tasks.cpp
//! \brief Driver tasks for reference-frame first-order GH.
//========================================================================================
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/standard_gh_source.hpp"
#include "tasklist/numerical_relativity.hpp"

namespace ref_gh {

void RefGh::QueueTasks() {
  using namespace numrel;  // NOLINT(build/namespaces)
  auto *pnr = pmy_pack->pnr;
  pnr->QueueTask(&RefGh::InitRecv, this, RefGh_Recv, "RefGh_Recv", Task_Start);
  pnr->QueueTask(&RefGh::CopyU, this, RefGh_CopyU, "RefGh_CopyU", Task_Run);
  if (opt.fd_order == 2) {
    pnr->QueueTask(&RefGh::CalcRHS<2>, this, RefGh_CalcRHS, "RefGh_CalcRHS",
                   Task_Run, {RefGh_CopyU});
  } else if (opt.fd_order == 4) {
    pnr->QueueTask(&RefGh::CalcRHS<3>, this, RefGh_CalcRHS, "RefGh_CalcRHS",
                   Task_Run, {RefGh_CopyU});
  } else {
    pnr->QueueTask(&RefGh::CalcRHS<4>, this, RefGh_CalcRHS, "RefGh_CalcRHS",
                   Task_Run, {RefGh_CopyU});
  }
  pnr->QueueTask(&RefGh::ExpRKUpdate, this, RefGh_ExplRK, "RefGh_ExplRK", Task_Run,
                 {RefGh_CalcRHS});
  pnr->QueueTask(&RefGh::RestrictU, this, RefGh_RestU, "RefGh_RestU", Task_Run,
                 {RefGh_ExplRK});
  pnr->QueueTask(&RefGh::SendU, this, RefGh_SendU, "RefGh_SendU", Task_Run,
                 {RefGh_RestU});
  pnr->QueueTask(&RefGh::RecvU, this, RefGh_RecvU, "RefGh_RecvU", Task_Run,
                 {RefGh_SendU});
  pnr->QueueTask(&RefGh::Prolongate, this, RefGh_Prolong, "RefGh_Prolong", Task_Run,
                 {RefGh_RecvU});
  pnr->QueueTask(&RefGh::ApplyPhysicalBCs, this, RefGh_BCS, "RefGh_BCS", Task_Run,
                 {RefGh_Prolong});
  pnr->QueueTask(&RefGh::NewTimeStep, this, RefGh_Newdt, "RefGh_Newdt", Task_Run,
                 {RefGh_BCS});
  pnr->QueueTask(&RefGh::ClearSend, this, RefGh_ClearS, "RefGh_ClearS", Task_End);
  pnr->QueueTask(&RefGh::ClearRecv, this, RefGh_ClearR, "RefGh_ClearR", Task_End,
                 {RefGh_ClearS});
}

TaskStatus RefGh::InitRecv(Driver *, int) { return pbval_u->InitRecv(nref_gh); }
TaskStatus RefGh::ClearRecv(Driver *, int) { return pbval_u->ClearRecv(); }
TaskStatus RefGh::ClearSend(Driver *, int) { return pbval_u->ClearSend(); }

TaskStatus RefGh::CopyU(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  if (driver->integrator == "rk4") {
    if (stage == 1) {
      Kokkos::deep_copy(DevExeSpace(), u1, u0);
    } else {
      const Real delta = driver->delta[stage - 1];
      const auto state = u0;
      const auto base = u1;
      par_for("ref_gh rk4 base", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
      0, nref_gh - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
      KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
        base(m, n, k, j, i) += delta*state(m, n, k, j, i);
      });
    }
  } else if (stage == 1) {
    Kokkos::deep_copy(DevExeSpace(), u1, u0);
  }
  return TaskStatus::complete;
}

TaskStatus RefGh::ExpRKUpdate(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const Real gam0 = driver->gam0[stage - 1];
  const Real gam1 = driver->gam1[stage - 1];
  const Real beta_dt = driver->beta[stage - 1]*pmy_pack->pmesh->dt;
  const auto state = u0;
  const auto base = u1;
  const auto rhs = u_rhs;
  par_for("ref_gh RK update", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, nref_gh - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
    state(m, n, k, j, i) = gam0*state(m, n, k, j, i)
                               + gam1*base(m, n, k, j, i)
                               + beta_dt*rhs(m, n, k, j, i);
  });
  return TaskStatus::complete;
}

TaskStatus RefGh::RestrictU(Driver *, int) {
  if (pmy_pack->pmesh->multilevel) {
    pmy_pack->pmesh->pmr->RestrictCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}
TaskStatus RefGh::SendU(Driver *, int) { return pbval_u->PackAndSendCC(u0, coarse_u0); }
TaskStatus RefGh::RecvU(Driver *, int) { return pbval_u->RecvAndUnpackCC(u0, coarse_u0); }
TaskStatus RefGh::Prolongate(Driver *, int) {
  if (pmy_pack->pmesh->multilevel) pbval_u->ProlongateCC(u0, coarse_u0, true);
  return TaskStatus::complete;
}
TaskStatus RefGh::ApplyPhysicalBCs(Driver *, int) {
  // The first two required gates are periodic.  Non-periodic characteristic data are
  // added with the trumpet provider; failing closed here prevents unfilled ghost use.
  if (!pmy_pack->pmesh->strictly_periodic) {
    std::cout << "### FATAL ERROR: current ref_gh flat-reference prototype requires "
              << "periodic boundaries." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

TaskStatus RefGh::NewTimeStep(Driver *driver, int stage) {
  if (stage != driver->nexp_stages) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Kokkos::parallel_reduce(
      "ref_gh dt", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_minimum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        Real metric[4][4];  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          for (int b = a; b < 4; ++b) {
            metric[a][b] = metric[b][a] =
                state(m, PsiIndex(a, b), k, j, i);
          }
        }
        Real inverse[4][4], determinant = 0.0;  // NOLINT(runtime/arrays)
        if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
          local_minimum = 0.0;
          return;
        }
        Real spatial_inverse[3][3], spatial_det = 0.0; // NOLINT(runtime/arrays)
        if (!InvertSpatial3(metric, spatial_inverse, spatial_det)) {
          local_minimum = 0.0;
          return;
        }
        const Real alpha = 1.0/Kokkos::sqrt(-inverse[0][0]);
        for (int p = 0; p < 3; ++p) {
          const Real beta = alpha*alpha*inverse[0][p + 1];
          const Real speed = Kokkos::abs(beta)
                             + alpha*Kokkos::sqrt(spatial_inverse[p][p]);
          const Real dx = (p == 0) ? size.d_view(m).dx1
                          : ((p == 1) ? size.d_view(m).dx2 : size.d_view(m).dx3);
          const Real candidate = speed > 0.0 ? dx/speed : 0.0;
          if (candidate < local_minimum) local_minimum = candidate;
        }
      }, Kokkos::Min<Real>(dtnew));
  max_char_speed = 0.0;  // populated by the full conditioning diagnostic pass later
  if (!(dtnew > 0.0) || !std::isfinite(dtnew)) {
    std::cout << "### FATAL ERROR: ref_gh reached an invalid effective timestep."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.fail_closed_dt > 0.0 && dtnew < opt.fail_closed_dt) {
    std::cout << "### FATAL ERROR: ref_gh timestep crossed fail_closed_dt."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

}  // namespace ref_gh
