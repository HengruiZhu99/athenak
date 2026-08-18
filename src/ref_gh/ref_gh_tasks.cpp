//========================================================================================
//! \file ref_gh_tasks.cpp
//! \brief Driver tasks for reference-frame first-order GH.
//========================================================================================
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/standard_gh_source.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
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
  if (pmy_pack->pmesh->strictly_periodic) return TaskStatus::complete;
  if (opt.reference_kind != 1) {
    std::cout << "### FATAL ERROR: non-periodic ref_gh boundaries are currently "
              << "implemented only for the exact stationary trumpet state."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Exact analytic data for the stationary-reference gate.  In the regular frame the
  // complete incoming state is simply Psi=eta, Pi=Phi=0.  Filling the full state (rather
  // than only its incoming characteristic projection) is exact for this solution and
  // avoids finite-differencing or extrapolating singular coordinate-metric components.
  // Internal block faces have BoundaryFlag::block and are left untouched.
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int n1 = indcs.nx1 + 2*ng;
  const int n2 = indcs.nx2 + 2*ng;
  const int n3 = indcs.nx3 + 2*ng;
  const int is = indcs.is;
  const int ie = indcs.ie;
  const int js = indcs.js;
  const int je = indcs.je;
  const int ks = indcs.ks;
  const int ke = indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  const auto state = u0;
  const auto mb_bcs = pmy_pack->pmb->mb_bcs.d_view;

  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("ref_gh exact trumpet x1 boundaries", DevExeSpace(), 0, nmb - 1,
    0, nref_gh - 1, 0, n3 - 1, 0, n2 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
      Real value = 0.0;
      if (n == PsiIndex(0, 0)) value = -1.0;
      if (n == PsiIndex(1, 1) || n == PsiIndex(2, 2) || n == PsiIndex(3, 3)) {
        value = 1.0;
      }
      if (mb_bcs(m, BoundaryFace::inner_x1) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, j, is - g) = value;
      }
      if (mb_bcs(m, BoundaryFace::outer_x1) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, j, ie + g) = value;
      }
    });
  }
  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("ref_gh exact trumpet x2 boundaries", DevExeSpace(), 0, nmb - 1,
    0, nref_gh - 1, 0, n3 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int i) {
      Real value = 0.0;
      if (n == PsiIndex(0, 0)) value = -1.0;
      if (n == PsiIndex(1, 1) || n == PsiIndex(2, 2) || n == PsiIndex(3, 3)) {
        value = 1.0;
      }
      if (mb_bcs(m, BoundaryFace::inner_x2) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, js - g, i) = value;
      }
      if (mb_bcs(m, BoundaryFace::outer_x2) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, k, je + g, i) = value;
      }
    });
  }
  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
    par_for("ref_gh exact trumpet x3 boundaries", DevExeSpace(), 0, nmb - 1,
    0, nref_gh - 1, 0, n2 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
      Real value = 0.0;
      if (n == PsiIndex(0, 0)) value = -1.0;
      if (n == PsiIndex(1, 1) || n == PsiIndex(2, 2) || n == PsiIndex(3, 3)) {
        value = 1.0;
      }
      if (mb_bcs(m, BoundaryFace::inner_x3) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, ks - g, j, i) = value;
      }
      if (mb_bcs(m, BoundaryFace::outer_x3) != BoundaryFlag::block) {
        for (int g = 1; g <= ng; ++g) state(m, n, ke + g, j, i) = value;
      }
    });
  }
  return TaskStatus::complete;
}

TaskStatus RefGh::NewTimeStep(Driver *driver, int stage) {
  if (stage != driver->nexp_stages) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const auto table = reference_table;
  const int reference_kind = opt.reference_kind;
  const Real reference_mass = opt.reference_mass;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const Real time = pmy_pack->pmesh->time;
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
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        ReferenceGeometry reference;
        if (reference_kind == 0) {
          reference = MinkowskiReference()(time, x, y, z);
        } else {
          const TrumpetSchwarzschildReference provider{
              table, reference_mass, {center_x, center_y, center_z}};
          reference = provider(time, x, y, z);
        }
        Real psi[4][4], metric[4][4];  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          for (int b = a; b < 4; ++b) {
            psi[a][b] = psi[b][a] =
                state(m, PsiIndex(a, b), k, j, i);
          }
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            metric[a][b] = 0.0;
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                metric[a][b] += reference.coframe[A][a]
                                *reference.coframe[B][b]*psi[A][B];
              }
            }
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
