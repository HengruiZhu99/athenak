//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh_tasks.cpp
//! \brief Driver tasks for the standalone regularized vacuum FO-GH system.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "fo_gh/fo_gh.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "pgen/pgen.hpp"
#include "tasklist/numerical_relativity.hpp"
#include "utils/finite_diff.hpp"

namespace fo_gh {

namespace {

void SymmetricEigenvalues3(Real matrix[3][3], Real eigenvalues[3]) {
  for (int sweep = 0; sweep < 24; ++sweep) {
    int p = 0;
    int q = 1;
    Real largest = std::abs(matrix[0][1]);
    for (int i = 0; i < 3; ++i) {
      for (int j = i + 1; j < 3; ++j) {
        if (std::abs(matrix[i][j]) > largest) {
          largest = std::abs(matrix[i][j]);
          p = i;
          q = j;
        }
      }
    }
    if (largest < 1.0e-15) break;
    const Real angle = 0.5*std::atan2(2.0*matrix[p][q],
                                      matrix[q][q] - matrix[p][p]);
    const Real cosine = std::cos(angle);
    const Real sine = std::sin(angle);
    const Real app = matrix[p][p];
    const Real aqq = matrix[q][q];
    const Real apq = matrix[p][q];
    matrix[p][p] = cosine*cosine*app - 2.0*sine*cosine*apq
                   + sine*sine*aqq;
    matrix[q][q] = sine*sine*app + 2.0*sine*cosine*apq
                   + cosine*cosine*aqq;
    matrix[p][q] = matrix[q][p] = 0.0;
    for (int r = 0; r < 3; ++r) {
      if (r == p || r == q) continue;
      const Real arp = matrix[r][p];
      const Real arq = matrix[r][q];
      matrix[r][p] = matrix[p][r] = cosine*arp - sine*arq;
      matrix[r][q] = matrix[q][r] = sine*arp + cosine*arq;
    }
  }
  for (int i = 0; i < 3; ++i) eigenvalues[i] = matrix[i][i];
  std::sort(eigenvalues, eigenvalues + 3);
}

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
  const auto vars = u;
  const auto state = u0;
  using MinLoc = Kokkos::MinLoc<Real, int>;
  MinLoc::value_type minimum;
  Kokkos::parallel_reduce(
      "fo_gh dt", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, MinLoc::value_type &local_minimum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> metric;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
        for (int a = 0; a < 3; ++a) {
          for (int b = a; b < 3; ++b) {
            metric(a, b) = vars.gtilde(m, a, b, k, j, i);
          }
        }
        const Real determinant = Determinant3(metric);
        bool valid = Kokkos::isfinite(vars.alpha(m, k, j, i))
                     && Kokkos::isfinite(vars.chi(m, k, j, i))
                     && Kokkos::isfinite(determinant)
                     && vars.alpha(m, k, j, i) > 0.0
                     && vars.chi(m, k, j, i) > 0.0 && determinant > 0.0;
        for (int n = 0; n < nfo_gh; ++n) {
          valid = valid && Kokkos::isfinite(state(m, n, k, j, i));
        }
        if (!valid) {
          if (0.0 < local_minimum.val) {
            local_minimum.val = 0.0;
            local_minimum.loc = idx;
          }
          return;
        }
        Invert3(metric, inverse);
        const Real dx[3] = {size.d_view(m).dx1,
                            size.d_view(m).dx2,
                            size.d_view(m).dx3};
        for (int d = 0; d < 3; ++d) {
          const Real speed = Kokkos::abs(vars.beta(m, d, k, j, i))
              + vars.alpha(m, k, j, i)
                *std::sqrt(fmax(0.0, vars.chi(m, k, j, i)*inverse(d, d)));
          const Real candidate = speed > 0.0 && Kokkos::isfinite(speed)
                                 ? dx[d]/speed : 0.0;
          if (candidate < local_minimum.val) {
            local_minimum.val = candidate;
            local_minimum.loc = idx;
          }
        }
      }, MinLoc(minimum));
  dtnew = minimum.val;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "fo_gh max characteristic speed", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> metric;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
        for (int a = 0; a < 3; ++a) {
          for (int b = a; b < 3; ++b) {
            metric(a, b) = vars.gtilde(m, a, b, k, j, i);
          }
        }
        const Real determinant = Determinant3(metric);
        if (!Kokkos::isfinite(vars.alpha(m, k, j, i))
            || !Kokkos::isfinite(vars.chi(m, k, j, i))
            || !Kokkos::isfinite(determinant)
            || vars.alpha(m, k, j, i) <= 0.0
            || vars.chi(m, k, j, i) <= 0.0 || determinant <= 0.0) return;
        Invert3(metric, inverse);
        for (int d = 0; d < 3; ++d) {
          const Real speed = Kokkos::abs(vars.beta(m, d, k, j, i))
              + vars.alpha(m, k, j, i)
                *std::sqrt(fmax(0.0, vars.chi(m, k, j, i)*inverse(d, d)));
          local_maximum = fmax(local_maximum, speed);
        }
      }, Kokkos::Max<Real>(maximum));
  max_char_speed = maximum;
  const bool constraints_valid = std::isfinite(dtnew) && dtnew > 0.0;
  if (constraints_valid) UpdateDiagnostics();
  if (opt.fail_closed_dt > 0.0 &&
      (!std::isfinite(dtnew) || dtnew < opt.fail_closed_dt)) {
    using MaxLoc = Kokkos::MaxLoc<Real, int>;
    MaxLoc::value_type maximum_state;
    MaxLoc::value_type maximum_rhs;
    const int total_values = pmy_pack->nmb_thispack*nfo_gh*ncells;
    const auto state_rhs = u_rhs;
    Kokkos::parallel_reduce(
        "fo_gh failure maximum state",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, total_values),
        KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is;
          work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js;
          work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          work /= indcs.nx3;
          const int n = work % nfo_gh;
          const int m = work/nfo_gh;
          const Real value = Kokkos::abs(state(m, n, k, j, i));
          if (value > local_maximum.val || !Kokkos::isfinite(value)) {
            local_maximum.val = value;
            local_maximum.loc = idx;
          }
        }, MaxLoc(maximum_state));
    Kokkos::parallel_reduce(
        "fo_gh failure maximum rhs",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, total_values),
        KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is;
          work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js;
          work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          work /= indcs.nx3;
          const int n = work % nfo_gh;
          const int m = work/nfo_gh;
          const Real value = Kokkos::abs(state_rhs(m, n, k, j, i));
          if (value > local_maximum.val || !Kokkos::isfinite(value)) {
            local_maximum.val = value;
            local_maximum.loc = idx;
          }
        }, MaxLoc(maximum_rhs));
    WriteFailureTelemetry(minimum.loc, maximum_state.val, maximum_state.loc,
                          maximum_rhs.val, maximum_rhs.loc, constraints_valid);
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
    std::exit(EXIT_FAILURE);
#endif
  }
  return TaskStatus::complete;
}

void FoGh::WriteFailureTelemetry(const int flattened_cell,
                                 const Real maximum_state,
                                 const int maximum_state_location,
                                 const Real maximum_rhs,
                                 const int maximum_rhs_location,
                                 const bool constraints_valid) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int work = flattened_cell;
  const int i = work % indcs.nx1 + indcs.is;
  work /= indcs.nx1;
  const int j = work % indcs.nx2 + indcs.js;
  work /= indcs.nx2;
  const int k = work % indcs.nx3 + indcs.ks;
  const int m = work/indcs.nx3;
  DvceArray1D<Real> packed("fo_gh failure telemetry", 2*nfo_gh + ncon);
  const auto state = u0;
  const auto state_rhs = u_rhs;
  const auto constraints = u_con;
  Kokkos::parallel_for(
      "fo_gh pack failure telemetry", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int) {
        for (int n = 0; n < nfo_gh; ++n) {
          packed(n) = state(m, n, k, j, i);
          packed(nfo_gh + n) = state_rhs(m, n, k, j, i);
        }
        for (int n = 0; n < ncon; ++n) {
          packed(2*nfo_gh + n) = constraints(m, n, k, j, i);
        }
      });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), packed);
  auto &block = pmy_pack->pmb->mb_size.h_view(m);
  const Real x = CellCenterX(i - indcs.is, indcs.nx1, block.x1min, block.x1max);
  const Real y = CellCenterX(j - indcs.js, indcs.nx2, block.x2min, block.x2max);
  const Real z = CellCenterX(k - indcs.ks, indcs.nx3, block.x3min, block.x3max);

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde;
  gtilde(0, 0) = host(I_TGXX);
  gtilde(0, 1) = host(I_TGXY);
  gtilde(0, 2) = host(I_TGXZ);
  gtilde(1, 1) = host(I_TGYY);
  gtilde(1, 2) = host(I_TGYZ);
  gtilde(2, 2) = host(I_TGZZ);
  const Real determinant = Determinant3(gtilde);
  Real matrix[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) matrix[a][b] = gtilde(a, b);
  }
  Real eigenvalues[3];
  SymmetricEigenvalues3(matrix, eigenvalues);

  RegularPointState point;
  point.ZeroClear();
  point.gtilde = gtilde;
  point.chi = host(I_CHI);
  point.alpha = host(I_ALPHA);
  point.K = host(I_K);
  point.pi = host(I_PI);
  point.h_perp = host(I_H_PERP);
  for (int a = 0; a < 3; ++a) {
    point.beta(a) = host(I_BETAX + a);
    point.Lambda(a) = host(I_LAMBDAX + a);
    point.X(a) = host(I_XX + a);
    point.a(a) = host(I_AX + a);
    point.h(a) = host(I_HX + a);
  }
  Real f_perp = 0.0;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> f;
  GaugeTargets(point, opt.eta_beta, f_perp, f);

  auto DecodeMaximum = [&](const int location, int &rank_m, int &rank_n,
                           int &rank_k, int &rank_j, int &rank_i) {
    int index = location;
    rank_i = index % indcs.nx1 + indcs.is;
    index /= indcs.nx1;
    rank_j = index % indcs.nx2 + indcs.js;
    index /= indcs.nx2;
    rank_k = index % indcs.nx3 + indcs.ks;
    index /= indcs.nx3;
    rank_n = index % nfo_gh;
    rank_m = index/nfo_gh;
  };
  int sm = 0, sn = 0, sk = 0, sj = 0, si = 0;
  int rm = 0, rn = 0, rk = 0, rj = 0, ri = 0;
  DecodeMaximum(maximum_state_location, sm, sn, sk, sj, si);
  DecodeMaximum(maximum_rhs_location, rm, rn, rk, rj, ri);

  std::cerr << std::setprecision(17)
            << "FO_GH_FIRST_BAD_STATE rank=" << global_variable::my_rank
            << " cycle=" << pmy_pack->pmesh->ncycle
            << " time=" << pmy_pack->pmesh->time << " dtnew=" << dtnew
            << " max_char_speed=" << max_char_speed << '\n'
            << "cell local_block=" << m << " global_block=" << pmy_pack->gids + m
            << " i=" << i << " j=" << j << " k=" << k
            << " x=" << x << " y=" << y << " z=" << z << '\n'
            << "alpha=" << point.alpha << " A=" << point.alpha*point.alpha
            << " chi=" << point.chi << " det_gtilde=" << determinant
            << " eig_gtilde=" << eigenvalues[0] << ',' << eigenvalues[1]
            << ',' << eigenvalues[2] << '\n'
            << "h_minus_f=" << point.h_perp - f_perp << ','
            << point.h(0) - f(0) << ',' << point.h(1) - f(1) << ','
            << point.h(2) - f(2) << '\n'
            << "maximum_state=" << maximum_state << " component="
            << StateNames[sn] << " local_block=" << sm << " i=" << si
            << " j=" << sj << " k=" << sk << '\n'
            << "maximum_last_stage_rhs=" << maximum_rhs << " component="
            << StateNames[rn] << " local_block=" << rm << " i=" << ri
            << " j=" << rj << " k=" << rk << '\n';
  if (constraints_valid) {
    for (int n = 0; n < ncon; ++n) {
      std::cerr << ConstraintNames[n] << '=' << host(2*nfo_gh + n)
                << (n + 1 == ncon ? '\n' : ' ');
    }
  } else {
    std::cerr << "constraints=unavailable_invalid_metric_or_speed\n";
  }
  for (int n = 0; n < nfo_gh; ++n) {
    std::cerr << StateNames[n] << '=' << host(n)
              << " rhs=" << host(nfo_gh + n) << '\n';
  }
  std::cerr.flush();
}

void FoGh::FoGhToADM() {
  if (pmy_pack->padm == nullptr) return;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1 ? indcs.nx2 + 2*indcs.ng : 1);
  const int n3 = (indcs.nx3 > 1 ? indcs.nx3 + 2*indcs.ng : 1);
  const auto vars = u;
  const auto adm_vars = pmy_pack->padm->adm;
  par_for("fo_gh to ADM", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real psi4 = 1.0/vars.chi(m, k, j, i);
    adm_vars.psi4(m, k, j, i) = psi4;
    adm_vars.alpha(m, k, j, i) = vars.alpha(m, k, j, i);
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = vars.beta(m, a, k, j, i);
      for (int b = a; b < 3; ++b) {
        const Real gamma = psi4*vars.gtilde(m, a, b, k, j, i);
        adm_vars.g_dd(m, a, b, k, j, i) = gamma;
        adm_vars.vK_dd(m, a, b, k, j, i) =
            psi4*vars.Atilde(m, a, b, k, j, i)
            + vars.K(m, k, j, i)*gamma/3.0;
      }
    }
  });
}

void FoGh::UpdateDiagnostics() {
  FoGhToADM();
  switch (opt.fd_order) {
    case 2: CalcConstraints<2>(); break;
    case 4: CalcConstraints<3>(); break;
    case 6: CalcConstraints<4>(); break;
  }
}

} // namespace fo_gh
