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
#include "z4c/fastflow.hpp"
#include "pgen/pgen.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

KOKKOS_INLINE_FUNCTION
Real ConstraintMagnitude(const DvceArray5D<Real> &constraints, const int group,
                         const int m, const int k, const int j, const int i) {
  if (group == 0) return Kokkos::abs(constraints(m, fo_gh::FoGh::I_CON_H, k, j, i));
  Real value2 = 0.0;
  if (group == 1) {
    for (int n = fo_gh::FoGh::I_CON_MX; n <= fo_gh::FoGh::I_CON_MZ; ++n) {
      value2 += SQR(constraints(m, n, k, j, i));
    }
  } else if (group == 2) {
    for (int n = fo_gh::FoGh::I_CON_GH_PERP; n <= fo_gh::FoGh::I_CON_GHZ; ++n) {
      value2 += SQR(constraints(m, n, k, j, i));
    }
  } else {
    for (int n = fo_gh::FoGh::I_CON_RQ; n <= fo_gh::FoGh::I_CON_RB; ++n) {
      value2 += SQR(constraints(m, n, k, j, i));
    }
  }
  return std::sqrt(value2);
}

KOKKOS_INLINE_FUNCTION
Real StandardGhMaximum(const fo_gh::FoGh::Variables &vars, const int group,
                       const int m, const int k, const int j, const int i) {
  fo_gh::RegularPointState point;
  fo_gh::StandardGhPointState gh;
  fo_gh::LoadPoint(vars, m, k, j, i, point);
  if (group == 3) {
    Real maximum = fmax(Kokkos::abs(point.h_perp),
                        Kokkos::abs(point.vartheta_perp));
    for (int a = 0; a < 3; ++a) {
      maximum = fmax(maximum, Kokkos::abs(point.h(a)));
      maximum = fmax(maximum, Kokkos::abs(point.vartheta(a)));
    }
    return maximum;
  }
  fo_gh::RegularToStandardGh(point, gh);
  Real maximum = 0.0;
  if (group == 0) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) maximum = fmax(maximum, Kokkos::abs(gh.g(a, b)));
    }
  } else if (group == 1) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) maximum = fmax(maximum, Kokkos::abs(gh.Pi(a, b)));
    }
  } else {
    for (int p = 0; p < 3; ++p) {
      for (int a = 0; a < 4; ++a) {
        for (int b = a; b < 4; ++b) {
          maximum = fmax(maximum, Kokkos::abs(gh.Phi(p, a, b)));
        }
      }
    }
  }
  return maximum;
}

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
  const auto constraints = pmbp->pfogh->u_con;
  const auto vars = pmbp->pfogh->u;
  const auto adm_vars = pmbp->padm->adm;
  const Real near_radius = pin->GetOrAddReal("problem", "near_radius", 1.0);
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

  // Global and fixed-physical-radius near-puncture L1/L2 constraint integrals.
  // Layout is four groups times (L1,L2sq), volume, then the same near the puncture.
  array_sum::GlobalSum constraint_sums;
  Kokkos::parallel_reduce(
      "fo_gh puncture constraint sums", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
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
        const Real detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i), adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i), adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i), adm_vars.g_dd(m, 2, 2, k, j, i));
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2
                            *size.d_view(m).dx3*std::sqrt(Kokkos::abs(detg));
        for (int group = 0; group < 4; ++group) {
          const Real value = ConstraintMagnitude(constraints, group, m, k, j, i);
          sum.the_array[2*group] += volume*value;
          sum.the_array[2*group + 1] += volume*value*value;
          if (radius < near_radius) {
            sum.the_array[9 + 2*group] += volume*value;
            sum.the_array[10 + 2*group] += volume*value*value;
          }
        }
        sum.the_array[8] += volume;
        if (radius < near_radius) sum.the_array[17] += volume;
      }, Kokkos::Sum<array_sum::GlobalSum>(constraint_sums));

  Real constraint_linf[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int group = 0; group < 4; ++group) {
    Kokkos::parallel_reduce(
        "fo_gh puncture constraint Linf", Kokkos::RangePolicy<>(DevExeSpace(),
        0, pmbp->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is;
          work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js;
          work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          maximum = fmax(maximum,
                         ConstraintMagnitude(constraints, group, m, k, j, i));
        }, Kokkos::Max<Real>(constraint_linf[group]));
    Kokkos::parallel_reduce(
        "fo_gh puncture near constraint Linf", Kokkos::RangePolicy<>(DevExeSpace(),
        0, pmbp->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
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
          if (std::sqrt(x*x + y*y + z*z) < near_radius) {
            maximum = fmax(maximum,
                           ConstraintMagnitude(constraints, group, m, k, j, i));
          }
        }, Kokkos::Max<Real>(constraint_linf[4 + group]));
  }

  Real gh_extrema[4] = {0.0, 0.0, 0.0, 0.0};
  for (int group = 0; group < 4; ++group) {
    Kokkos::parallel_reduce(
        "fo_gh puncture field extrema", Kokkos::RangePolicy<>(DevExeSpace(),
        0, pmbp->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is;
          work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js;
          work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          maximum = fmax(maximum, StandardGhMaximum(vars, group, m, k, j, i));
        }, Kokkos::Max<Real>(gh_extrema[group]));
  }
  const Real domain_min[3] = {pm->mesh_size.x1min, pm->mesh_size.x2min,
                              pm->mesh_size.x3min};
  const Real domain_max[3] = {pm->mesh_size.x1max, pm->mesh_size.x2max,
                              pm->mesh_size.x3max};
  Real adm_mass = 0.0;
  Kokkos::parallel_reduce(
      "fo_gh puncture ADM mass", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmbp->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &mass_sum) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real block_min[3] = {size.d_view(m).x1min, size.d_view(m).x2min,
                                   size.d_view(m).x3min};
        const Real block_max[3] = {size.d_view(m).x1max, size.d_view(m).x2max,
                                   size.d_view(m).x3max};
        const int cell[3] = {i, j, k};
        const int start[3] = {indcs.is, indcs.js, indcs.ks};
        const int end[3] = {indcs.ie, indcs.je, indcs.ke};
        const Real area[3] = {size.d_view(m).dx2*size.d_view(m).dx3,
                              size.d_view(m).dx1*size.d_view(m).dx3,
                              size.d_view(m).dx1*size.d_view(m).dx2};
        const Real chi = vars.chi(m, k, j, i);
        for (int p = 0; p < 3; ++p) {
          for (int side = 0; side < 2; ++side) {
            const bool boundary = (side == 0)
                ? (cell[p] == start[p] && block_min[p] == domain_min[p])
                : (cell[p] == end[p] && block_max[p] == domain_max[p]);
            if (!boundary) continue;
            Real integrand = 0.0;
            for (int q = 0; q < 3; ++q) {
              const Real d_q_g_pq = vars.Q[q](m, p, q, k, j, i)/chi
                  - vars.X(m, q, k, j, i)*vars.gtilde(m, p, q, k, j, i)/(chi*chi);
              const Real d_p_g_qq = vars.Q[p](m, q, q, k, j, i)/chi
                  - vars.X(m, p, k, j, i)*vars.gtilde(m, q, q, k, j, i)/(chi*chi);
              integrand += d_q_g_pq - d_p_g_qq;
            }
            mass_sum += (side == 0 ? -1.0 : 1.0)*area[p]*integrand/(16.0*M_PI);
          }
        }
      }, adm_mass);
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
  MPI_Allreduce(MPI_IN_PLACE, constraint_sums.the_array, NREDUCTION_VARIABLES,
                MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, constraint_linf, 8, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, gh_extrema, 4, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &adm_mass, 1, MPI_ATHENA_REAL, MPI_SUM,
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
    const std::string checkpoint_filename = pin->GetString("job", "basename")
                                            + "-checkpoint.dat";
    FILE *checkpoint = std::fopen(checkpoint_filename.c_str(), "w");
    if (checkpoint == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(checkpoint, "# time cycle near_radius finite alpha_min chi_min ");
    std::fprintf(checkpoint, "metric_max Pi_max Phi_max gauge_driver_max ");
    std::fprintf(checkpoint, "max_char_speed dt_candidate effective_cfl_next ");
    std::fprintf(checkpoint, "adm_mass adm_mass_drift horizon_status ");
    std::fprintf(checkpoint, "horizon_area horizon_mass ");
    const char *group_names[4] = {"H", "M", "GH", "R"};
    for (int group = 0; group < 4; ++group) {
      std::fprintf(checkpoint, "%s_L1 %s_L2 %s_Linf ", group_names[group],
                   group_names[group], group_names[group]);
    }
    for (int group = 0; group < 4; ++group) {
      std::fprintf(checkpoint, "%s_near_L1 %s_near_L2 %s_near_Linf ",
                   group_names[group], group_names[group], group_names[group]);
    }
    std::fprintf(checkpoint, "global_volume near_volume\n");
    const Real global_volume = constraint_sums.the_array[8];
    const Real near_volume = constraint_sums.the_array[17];
    const Real effective_cfl = (pmbp->pfogh->dtnew > 0.0
                                ? pm->dt/pmbp->pfogh->dtnew : 0.0);
    std::fprintf(checkpoint, "%.17e %d %.17e %d %.17e %.17e ",
                 pm->time, pm->ncycle, near_radius, (nonfinite == 0),
                 minimum_alpha, minimum_chi);
    const Real expected_mass = pin->GetReal("problem", "mass");
    std::fprintf(checkpoint, "%.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e ",
                 gh_extrema[0], gh_extrema[1], gh_extrema[2], gh_extrema[3],
                 pmbp->pfogh->max_char_speed, pmbp->pfogh->dtnew, effective_cfl,
                 adm_mass, adm_mass - expected_mass);
    int horizon_status = -1;
    Real horizon_area = 0.0;
    Real horizon_mass = 0.0;
    if (!(pmbp->pfogh->pfastflow.empty())) {
      horizon_status = pmbp->pfogh->pfastflow[0]->ah_found;
      if (horizon_status == 1) {
        horizon_area = pmbp->pfogh->pfastflow[0]->GetArea();
        horizon_mass = pmbp->pfogh->pfastflow[0]->GetMass();
      }
    }
    std::fprintf(checkpoint, "%d %.17e %.17e ", horizon_status,
                 horizon_area, horizon_mass);
    for (int group = 0; group < 4; ++group) {
      const Real l1 = constraint_sums.the_array[2*group]/global_volume;
      const Real l2 = std::sqrt(constraint_sums.the_array[2*group + 1]/global_volume);
      std::fprintf(checkpoint, "%.17e %.17e %.17e ", l1, l2,
                   constraint_linf[group]);
    }
    for (int group = 0; group < 4; ++group) {
      const Real l1 = (near_volume > 0.0
                       ? constraint_sums.the_array[9 + 2*group]/near_volume : 0.0);
      const Real l2 = (near_volume > 0.0
                       ? std::sqrt(constraint_sums.the_array[10 + 2*group]/near_volume)
                       : 0.0);
      std::fprintf(checkpoint, "%.17e %.17e %.17e ", l1, l2,
                   constraint_linf[4 + group]);
    }
    std::fprintf(checkpoint, "%.17e %.17e\n", global_volume, near_volume);
    std::fclose(checkpoint);
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
