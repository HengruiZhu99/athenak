//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file puncture.cpp
//! \brief Isotropic Schwarzschild wormhole data for regularized vacuum FO-GH.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "fo_gh/fo_gh.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

void CheckFoGhPuncture(ParameterInput *pin, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  pmbp->pfogh->FoGhToADM();
  switch (pmbp->pfogh->opt.fd_order) {
    case 2:
      (void)pmbp->pfogh->CalcRHS<2>(nullptr, 0);
      break;
    case 4:
      (void)pmbp->pfogh->CalcRHS<3>(nullptr, 0);
      break;
    case 6:
      (void)pmbp->pfogh->CalcRHS<4>(nullptr, 0);
      break;
  }
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const auto state = pmbp->pfogh->u0;
  const auto rhs = pmbp->pfogh->u_rhs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real minimum_radius = std::numeric_limits<float>::max();
  Real minimum_alpha = std::numeric_limits<float>::max();
  Real minimum_chi = std::numeric_limits<float>::max();
  Real maximum_state = 0.0;
  Real maximum_rhs = 0.0;
  Real maximum_near_rhs = 0.0;
  int nonfinite = 0;
  Kokkos::parallel_reduce(
      "fo_gh puncture diagnostics", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &min_radius, Real &min_alpha,
                    Real &min_chi, Real &max_state, Real &max_rhs,
                    Real &max_near_rhs, int &nan_count) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        const Real radius = std::sqrt(x*x + y*y + z*z);
        min_radius = fmin(min_radius, radius);
        min_alpha = fmin(min_alpha, state(m, fo_gh::I_ALPHA, k, j, i));
        min_chi = fmin(min_chi, state(m, fo_gh::I_CHI, k, j, i));
        for (int n = 0; n < fo_gh::nvar; ++n) {
          const Real state_value = state(m, n, k, j, i);
          const Real rhs_value = rhs(m, n, k, j, i);
          if (!Kokkos::isfinite(state_value) || !Kokkos::isfinite(rhs_value)) {
            ++nan_count;
          } else {
            max_state = fmax(max_state, Kokkos::abs(state_value));
            max_rhs = fmax(max_rhs, Kokkos::abs(rhs_value));
            if (radius < 3.0*size.d_view(m).dx1) {
              max_near_rhs = fmax(max_near_rhs, Kokkos::abs(rhs_value));
            }
          }
        }
      }, Kokkos::Min<Real>(minimum_radius), Kokkos::Min<Real>(minimum_alpha),
      Kokkos::Min<Real>(minimum_chi), Kokkos::Max<Real>(maximum_state),
      Kokkos::Max<Real>(maximum_rhs), Kokkos::Max<Real>(maximum_near_rhs),
      nonfinite);
  Real adm_adapter_error = 0.0;
  const auto vars = pmbp->pfogh->u;
  const auto adm_vars = pmbp->padm->adm;
  Kokkos::parallel_reduce(
      "fo_gh ADM adapter check", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &maximum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real psi4 = 1.0/vars.chi(m, k, j, i);
        maximum = fmax(maximum,
                       Kokkos::abs(adm_vars.psi4(m, k, j, i) - psi4));
        maximum = fmax(maximum, Kokkos::abs(adm_vars.alpha(m, k, j, i)
                                             - vars.alpha(m, k, j, i)));
        for (int a = 0; a < 3; ++a) {
          maximum = fmax(maximum, Kokkos::abs(adm_vars.beta_u(m, a, k, j, i)
                                               - vars.beta(m, a, k, j, i)));
          for (int b = a; b < 3; ++b) {
            const Real gamma = psi4*vars.gtilde(m, a, b, k, j, i);
            const Real extrinsic = psi4*vars.Atilde(m, a, b, k, j, i)
                                   + vars.K(m, k, j, i)*gamma/3.0;
            maximum = fmax(maximum, Kokkos::abs(
                adm_vars.g_dd(m, a, b, k, j, i) - gamma));
            maximum = fmax(maximum, Kokkos::abs(
                adm_vars.vK_dd(m, a, b, k, j, i) - extrinsic));
          }
        }
      }, Kokkos::Max<Real>(adm_adapter_error));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &minimum_radius, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minimum_alpha, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minimum_chi, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_state, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_rhs, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &maximum_near_rhs, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nonfinite, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &adm_adapter_error, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (nonfinite != 0 || minimum_radius <= 0.0 || adm_adapter_error > 1.0e-13) {
    std::cout << "FO-GH puncture boundedness failed: nonfinite=" << nonfinite
              << ", minimum radius=" << minimum_radius
              << ", ADM adapter error=" << adm_adapter_error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename")
                                 + "-puncture.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) {
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file, "# nx dx rmin alpha_min chi_min state_max rhs_max ");
    std::fprintf(file, "near_rhs_max nonfinite adm_adapter_error\n");
    std::fprintf(file, "%d %.17e %.17e %.17e %.17e %.17e %.17e %.17e %d %.17e\n",
                 pm->mesh_indcs.nx1,
                 (pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1,
                 minimum_radius, minimum_alpha, minimum_chi, maximum_state,
                 maximum_rhs, maximum_near_rhs, nonfinite, adm_adapter_error);
    std::fclose(file);
    std::cout << "FO-GH puncture boundedness passed: rmin=" << minimum_radius
              << ", state max=" << maximum_state
              << ", near RHS max=" << maximum_near_rhs
              << ", ADM adapter error=" << adm_adapter_error << std::endl;
  }
}

} // namespace

void ProblemGenerator::FoGhPuncture(ParameterInput *pin, const bool restart) {
  pgen_final_func = &CheckFoGhPuncture;
  if (restart) {
    return;
  }
  auto *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pfogh == nullptr) {
    std::cout << "FO-GH puncture data require an <fo_gh> block." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real mass = pin->GetOrAddReal("problem", "mass", 1.0);
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const auto state = pmbp->pfogh->u0;
  par_for("fo_gh puncture data", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const Real radius = std::sqrt(x*x + y*y + z*z);
    const Real psi = 1.0 + mass/(2.0*radius);
    const Real alpha = 1.0/(psi*psi);
    const Real chi = alpha*alpha;
    for (int n = 0; n < fo_gh::nvar; ++n) {
      state(m, n, k, j, i) = 0.0;
    }
    state(m, fo_gh::I_TGXX, k, j, i) = 1.0;
    state(m, fo_gh::I_TGYY, k, j, i) = 1.0;
    state(m, fo_gh::I_TGZZ, k, j, i) = 1.0;
    state(m, fo_gh::I_CHI, k, j, i) = chi;
    state(m, fo_gh::I_ALPHA, k, j, i) = alpha;
    const Real coordinates[3] = {x, y, z};
    for (int p = 0; p < 3; ++p) {
      state(m, fo_gh::I_AX + p, k, j, i) =
          mass*coordinates[p]/(psi*psi*psi*radius*radius*radius);
      state(m, fo_gh::I_XX + p, k, j, i) =
          2.0*mass*coordinates[p]
          /(psi*psi*psi*psi*psi*radius*radius*radius);
    }
  });
}
