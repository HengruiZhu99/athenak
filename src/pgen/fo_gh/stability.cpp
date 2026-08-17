//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stability.cpp
//! \brief Z4c-parity robust Minkowski stability test for vacuum FO-GH.

#include <cmath>
#include <cstdlib>
#include <iostream>

#include <Kokkos_Random.hpp>

#include "athena.hpp"
#include "fo_gh/fo_gh.hpp"
#include "fo_gh/fo_gh_state.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

KOKKOS_INLINE_FUNCTION
void StoreRegularState(const fo_gh::RegularPointState &u,
                       const DvceArray5D<Real> &state, const int m,
                       const int k, const int j, const int i) {
  state(m, fo_gh::I_CHI, k, j, i) = u.chi;
  state(m, fo_gh::I_ALPHA, k, j, i) = u.alpha;
  state(m, fo_gh::I_K, k, j, i) = u.K;
  state(m, fo_gh::I_PI, k, j, i) = u.pi;
  state(m, fo_gh::I_H_PERP, k, j, i) = u.h_perp;
  state(m, fo_gh::I_VARTHETA_PERP, k, j, i) = u.vartheta_perp;
  for (int a = 0; a < 3; ++a) {
    state(m, fo_gh::I_BETAX + a, k, j, i) = u.beta(a);
    state(m, fo_gh::I_LAMBDAX + a, k, j, i) = u.Lambda(a);
    state(m, fo_gh::I_XX + a, k, j, i) = u.X(a);
    state(m, fo_gh::I_AX + a, k, j, i) = u.a(a);
    state(m, fo_gh::I_HX + a, k, j, i) = u.h(a);
    state(m, fo_gh::I_VARTHETAX + a, k, j, i) = u.vartheta(a);
    for (int b = 0; b < 3; ++b) {
      state(m, fo_gh::I_BXX + 3*a + b, k, j, i) = u.B(a, b);
    }
    for (int n = 0; n < 6; ++n) {
      const int first[6] = {0, 0, 0, 1, 1, 2};
      const int second[6] = {0, 1, 2, 1, 2, 2};
      state(m, fo_gh::I_QXXX + 6*a + n, k, j, i) =
          u.Q(a, first[n], second[n]);
    }
  }
  for (int n = 0; n < 6; ++n) {
    const int first[6] = {0, 0, 0, 1, 1, 2};
    const int second[6] = {0, 1, 2, 1, 2, 2};
    state(m, fo_gh::I_TGXX + n, k, j, i) = u.gtilde(first[n], second[n]);
    state(m, fo_gh::I_TAXX + n, k, j, i) = u.Atilde(first[n], second[n]);
  }
}

void FoGhStabilityHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 2;
  pdata->label[0] = "LINF-Err";
  pdata->label[1] = "RMS-Err";
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const auto state = pmbp->pfogh->u0;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real sum_squared = 0.0;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "fo_gh stability RMS", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*6*ncells),
      KOKKOS_LAMBDA(const int idx, Real &sum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        work /= indcs.nx3;
        const int n = work % 6;
        const int m = work/6;
        const Real expected = (n == 0 || n == 3 || n == 5) ? 1.0 : 0.0;
        const Real error = state(m, fo_gh::I_TGXX + n, k, j, i) - expected;
        sum += error*error;
      }, sum_squared);
  Kokkos::parallel_reduce(
      "fo_gh stability Linf", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*6*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        work /= indcs.nx3;
        const int n = work % 6;
        const int m = work/6;
        const Real expected = (n == 0 || n == 3 || n == 5) ? 1.0 : 0.0;
        local_maximum = fmax(local_maximum,
            Kokkos::abs(state(m, fo_gh::I_TGXX + n, k, j, i) - expected));
      }, Kokkos::Max<Real>(maximum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &sum_squared, 1, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  const Real total_cells = static_cast<Real>(pm->nmb_total*ncells);
  pdata->hdata[0] = maximum;
  pdata->hdata[1] = std::sqrt(sum_squared/(6.0*total_cells));
}

} // namespace

void ProblemGenerator::FoGhStability(ParameterInput *pin, const bool restart) {
  user_hist_func = &FoGhStabilityHistory;
  if (restart) {
    return;
  }
  auto *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pfogh == nullptr) {
    std::cout << "FO-GH stability data require an <fo_gh> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;
  const Real rho = pin->GetOrAddReal("problem", "rho", 1.0);
  const Real amplitude = 1.0e-10/(rho*rho);
  DvceArray5D<Real> adm_data("fo_gh stability ADM", pmbp->nmb_thispack, 12,
                             ncells3, ncells2, ncells1);
  Kokkos::Random_XorShift64_Pool<> random_pool(pmbp->gids);
  const std::size_t scratch_size = ScrArray1D<Real>::shmem_size(12);
  par_for_outer("fo_gh stability random", DevExeSpace(), scratch_size, 0,
  0, pmbp->nmb_thispack - 1, 0, ncells3 - 1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k) {
    auto generator = random_pool.get_state();
    ScrArray1D<Real> random(member.team_scratch(0), 12);
    for (int n = 0; n < 12; ++n) {
      random(n) = amplitude*2.0*(generator.drand() - 0.5);
    }
    par_for_inner(member, 0, ncells1*ncells2 - 1, [&](const int index) {
      const int j = index/ncells1;
      const int i = index % ncells1;
      for (int n = 0; n < 12; ++n) {
        adm_data(m, n, k, j, i) = random(n);
      }
      adm_data(m, 0, k, j, i) += 1.0;
      adm_data(m, 3, k, j, i) += 1.0;
      adm_data(m, 5, k, j, i) += 1.0;
    });
    random_pool.free_state(generator);
  });
  const auto state = pmbp->pfogh->u0;
  const Real eta_beta = pmbp->pfogh->opt.eta_beta;
  par_for("fo_gh stability convert", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    fo_gh::AdmPointState adm;
    adm.ZeroClear();
    adm.alpha = 1.0;
    const int first[6] = {0, 0, 0, 1, 1, 2};
    const int second[6] = {0, 1, 2, 1, 2, 2};
    const Real inverse_spacing[3] = {1.0/size.d_view(m).dx1,
                                     1.0/size.d_view(m).dx2,
                                     1.0/size.d_view(m).dx3};
    for (int n = 0; n < 6; ++n) {
      adm.gamma(first[n], second[n]) = adm_data(m, n, k, j, i);
      adm.K(first[n], second[n]) = adm_data(m, 6 + n, k, j, i);
      for (int p = 0; p < 3; ++p) {
        const int di = (p == 0);
        const int dj = (p == 1);
        const int dk = (p == 2);
        adm.dgamma(p, first[n], second[n]) = 0.5*inverse_spacing[p]
            *(adm_data(m, n, k + dk, j + dj, i + di)
              - adm_data(m, n, k - dk, j - dj, i - di));
      }
    }
    fo_gh::RegularPointState regular;
    fo_gh::AdmToRegular(adm, eta_beta, regular);
    StoreRegularState(regular, state, m, k, j, i);
  });
  Kokkos::fence();
}
