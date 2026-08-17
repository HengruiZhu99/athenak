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
#include "pgen/pgen.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "utils/finite_diff.hpp"

namespace fo_gh {

namespace {

template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real Extrapolate(const DvceArray5D<Real> &u, const int m, const int n,
                 const int k, const int j, const int i, const int dk,
                 const int dj, const int di, const int distance) {
  const Real f0 = u(m, n, k, j, i);
  const Real f1 = u(m, n, k + dk, j + dj, i + di);
  if constexpr (ORDER == 2) {
    return f0 + distance*(f0 - f1);
  } else {
    const Real f2 = u(m, n, k + 2*dk, j + 2*dj, i + 2*di);
    if constexpr (ORDER == 3) {
      return 0.5*(f0*(1 + distance)*(2 + distance)
                  + distance*(f2 + distance*f2 - 2*f1*(2 + distance)));
    } else {
      const Real f3 = u(m, n, k + 3*dk, j + 3*dj, i + 3*di);
      return (-3.0*f1*distance*(2 + distance)*(3 + distance)
              + f0*(1 + distance)*(2 + distance)*(3 + distance)
              + distance*(1 + distance)
                  *(-f3*(2 + distance) + 3*f2*(3 + distance)))/6.0;
    }
  }
}

KOKKOS_INLINE_FUNCTION
bool IsExtrapolationBoundary(const BoundaryFlag flag) {
  return flag == BoundaryFlag::outflow || flag == BoundaryFlag::diode
         || flag == BoundaryFlag::vacuum;
}

template <int ORDER>
void ApplyExtrapolation(MeshBlockPack *pmbp, DvceArray5D<Real> &state,
                        const int is, const int ie, const int js, const int je,
                        const int ks, const int ke, const int n1, const int n2,
                        const int n3) {
  const int ng = pmbp->pmesh->mb_indcs.ng;
  const int nmb = pmbp->nmb_thispack;
  const auto bcs = pmbp->pmb->mb_bcs;
  if (pmbp->pmesh->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::periodic) {
    par_for("fo_gh BC x1", DevExeSpace(), 0, nmb - 1, 0, FoGh::nfo_gh - 1,
    0, n3 - 1, 0, n2 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j) {
      if (IsExtrapolationBoundary(bcs.d_view(m, BoundaryFace::inner_x1))) {
        for (int q = 0; q < ng; ++q) {
          state(m, n, k, j, is - q - 1) =
              Extrapolate<ORDER>(state, m, n, k, j, is, 0, 0, 1, q + 1);
        }
      }
      if (IsExtrapolationBoundary(bcs.d_view(m, BoundaryFace::outer_x1))) {
        for (int q = 0; q < ng; ++q) {
          state(m, n, k, j, ie + q + 1) =
              Extrapolate<ORDER>(state, m, n, k, j, ie, 0, 0, -1, q + 1);
        }
      }
    });
  }
  if (pmbp->pmesh->mesh_bcs[BoundaryFace::inner_x2] != BoundaryFlag::periodic) {
    par_for("fo_gh BC x2", DevExeSpace(), 0, nmb - 1, 0, FoGh::nfo_gh - 1,
    0, n3 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int i) {
      if (IsExtrapolationBoundary(bcs.d_view(m, BoundaryFace::inner_x2))) {
        for (int q = 0; q < ng; ++q) {
          state(m, n, k, js - q - 1, i) =
              Extrapolate<ORDER>(state, m, n, k, js, i, 0, 1, 0, q + 1);
        }
      }
      if (IsExtrapolationBoundary(bcs.d_view(m, BoundaryFace::outer_x2))) {
        for (int q = 0; q < ng; ++q) {
          state(m, n, k, je + q + 1, i) =
              Extrapolate<ORDER>(state, m, n, k, je, i, 0, -1, 0, q + 1);
        }
      }
    });
  }
  if (pmbp->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
    par_for("fo_gh BC x3", DevExeSpace(), 0, nmb - 1, 0, FoGh::nfo_gh - 1,
    0, n2 - 1, 0, n1 - 1,
    KOKKOS_LAMBDA(const int m, const int n, const int j, const int i) {
      if (IsExtrapolationBoundary(bcs.d_view(m, BoundaryFace::inner_x3))) {
        for (int q = 0; q < ng; ++q) {
          state(m, n, ks - q - 1, j, i) =
              Extrapolate<ORDER>(state, m, n, ks, j, i, 1, 0, 0, q + 1);
        }
      }
      if (IsExtrapolationBoundary(bcs.d_view(m, BoundaryFace::outer_x3))) {
        for (int q = 0; q < ng; ++q) {
          state(m, n, ke + q + 1, j, i) =
              Extrapolate<ORDER>(state, m, n, ke, j, i, -1, 0, 0, q + 1);
        }
      }
    });
  }
}

} // namespace

template <int FDNG>
void RepairCompatibleGradients(FoGh *pfogh, MeshBlockPack *pmbp,
                               const DualArray1D<int> &repair) {
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int mbs = pmbp->gids;
  const auto vars = pfogh->u;
  par_for("fo_gh AMR gradient repair", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (repair.d_view(m + mbs) == 0) return;
    const Real idx[3] = {1.0/size.d_view(m).dx1,
                         1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    for (int p = 0; p < 3; ++p) {
      vars.X(m, p, k, j, i) = Dx<FDNG>(p, idx, vars.chi, m, k, j, i);
      vars.a(m, p, k, j, i) = Dx<FDNG>(p, idx, vars.alpha, m, k, j, i);
      for (int a = 0; a < 3; ++a) {
        vars.B(m, p, a, k, j, i) =
            Dx<FDNG>(p, idx, vars.beta, m, a, k, j, i);
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          vars.Q[p](m, a, b, k, j, i) =
              Dx<FDNG>(p, idx, vars.gtilde, m, a, b, k, j, i);
        }
      }
    }
  });
}

void FoGh::RepairGradients(const DualArray1D<int> &repair) {
  switch (opt.fd_order) {
    case 2: RepairCompatibleGradients<2>(this, pmy_pack, repair); break;
    case 4: RepairCompatibleGradients<3>(this, pmy_pack, repair); break;
    case 6: RepairCompatibleGradients<4>(this, pmy_pack, repair); break;
  }
}

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
    if (!(pmy_pack->pmesh->strictly_periodic)) {
      auto &indcs = pmy_pack->pmesh->mb_indcs;
      const int n1 = indcs.cnx1 + 2*indcs.ng;
      const int n2 = indcs.cnx2 + 2*indcs.ng;
      const int n3 = indcs.cnx3 + 2*indcs.ng;
      if (opt.extrap_order == 2) {
        ApplyExtrapolation<2>(pmy_pack, coarse_u0, indcs.cis, indcs.cie,
                              indcs.cjs, indcs.cje, indcs.cks, indcs.cke,
                              n1, n2, n3);
      } else if (opt.extrap_order == 3) {
        ApplyExtrapolation<3>(pmy_pack, coarse_u0, indcs.cis, indcs.cie,
                              indcs.cjs, indcs.cje, indcs.cks, indcs.cke,
                              n1, n2, n3);
      } else {
        ApplyExtrapolation<4>(pmy_pack, coarse_u0, indcs.cis, indcs.cie,
                              indcs.cjs, indcs.cje, indcs.cks, indcs.cke,
                              n1, n2, n3);
      }
    }
    pbval_u->ProlongateCC(u0, coarse_u0, true);
  }
  return TaskStatus::complete;
}

TaskStatus FoGh::ApplyPhysicalBCs(Driver *pdriver, int stage) {
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int n1 = indcs.nx1 + 2*indcs.ng;
    const int n2 = indcs.nx2 + 2*indcs.ng;
    const int n3 = indcs.nx3 + 2*indcs.ng;
    if (opt.extrap_order == 2) {
      ApplyExtrapolation<2>(pmy_pack, u0, indcs.is, indcs.ie, indcs.js,
                            indcs.je, indcs.ks, indcs.ke, n1, n2, n3);
    } else if (opt.extrap_order == 3) {
      ApplyExtrapolation<3>(pmy_pack, u0, indcs.is, indcs.ie, indcs.js,
                            indcs.je, indcs.ks, indcs.ke, n1, n2, n3);
    } else {
      ApplyExtrapolation<4>(pmy_pack, u0, indcs.is, indcs.ie, indcs.js,
                            indcs.je, indcs.ks, indcs.ke, n1, n2, n3);
    }
    if (pmy_pack->pmesh->pgen->user_bcs) {
      pmy_pack->pmesh->pgen->user_bcs_func(pmy_pack->pmesh);
    }
  }
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
