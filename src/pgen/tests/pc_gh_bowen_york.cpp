//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_bowen_york.cpp
//! \brief exact time-symmetric isotropic Bowen-York/Schwarzschild puncture audit

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {

KOKKOS_INLINE_FUNCTION
void ExactTimeSymmetricBowenYork(Real x, Real y, Real z, Real mass,
                                 Real *state, Real *rhs) {
  for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) state[v] = rhs[v] = 0.0;
  Real const radius = std::sqrt(x*x + y*y + z*z);
  Real const puncture_scale = 0.5*mass;
  Real const denominator = radius + puncture_scale;
  Real const chi = std::pow(radius/denominator, 4);
  Real const sqrt_chi = std::sqrt(chi);
  Real const dchi_dr = 4.0*puncture_scale*radius*radius*radius
                       /std::pow(denominator, 5);
  Real const normal[3] = {x/radius, y/radius, z/radius};
  Real const gradient[3] = {dchi_dr*normal[0], dchi_dr*normal[1],
                            dchi_dr*normal[2]};
  state[pc_gh::PcGh::I_CHI] = chi;
  state[pc_gh::PcGh::I_GTXX] = 1.0;
  state[pc_gh::PcGh::I_GTYY] = 1.0;
  state[pc_gh::PcGh::I_GTZZ] = 1.0;
  state[pc_gh::PcGh::I_A] = chi;
  for (int d = 0; d < 3; ++d) {
    state[pc_gh::PcGh::I_X1 + d] = gradient[d];
    state[pc_gh::PcGh::I_Y1 + d] = gradient[d];
  }

  Real const gradient_sq = dchi_dr*dchi_dr;
  rhs[pc_gh::PcGh::I_K] = -gradient_sq/(8.0*sqrt_chi);
  rhs[pc_gh::PcGh::I_PI] = -rhs[pc_gh::PcGh::I_K];
  Real const coefficient = gradient_sq/(2.0*sqrt_chi);
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real const delta = (a == b) ? 1.0 : 0.0;
      rhs[pc_gh::PcGh::I_ATXX + pc_gh::PcGh::SymmetricIndex(a, b)] =
          -coefficient*(normal[a]*normal[b] - delta/3.0);
    }
  }
}

void CalculateDiagnostics(MeshBlockPack *pmbp) {
  switch (pmbp->ppcgh->opt.fd_stencil) {
    case 2:
      (void)pmbp->ppcgh->CalcRHS<2>(nullptr, 0);
      (void)pmbp->ppcgh->CalcConstraints<2>(nullptr, 0);
      break;
    case 3:
      (void)pmbp->ppcgh->CalcRHS<3>(nullptr, 0);
      (void)pmbp->ppcgh->CalcConstraints<3>(nullptr, 0);
      break;
    case 4:
      (void)pmbp->ppcgh->CalcRHS<4>(nullptr, 0);
      (void)pmbp->ppcgh->CalcConstraints<4>(nullptr, 0);
      break;
    default:
      std::abort();
  }
  Kokkos::fence();
}

void CheckPcGhBowenYork(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  CalculateDiagnostics(pmbp);
  auto &pcgh = *pmbp->ppcgh;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto state = pcgh.u0;
  auto state_rhs = pcgh.u_rhs;
  auto constraints = pcgh.u_con;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const nkji = nx3*nx2*nx1;
  int const nji = nx2*nx1;
  int const nmkji = pmbp->nmb_thispack*nkji;
  Real const mass = pin->GetReal("problem", "mass");
  Real const center[3] = {pin->GetReal("problem", "center_x"),
                          pin->GetReal("problem", "center_y"),
                          pin->GetReal("problem", "center_z")};
  Real const audit_r_min = pin->GetReal("problem", "audit_r_min");
  Real const audit_r_max = pin->GetReal("problem", "audit_r_max");

  Real maxima[6] = {};
  Real square_sums[6] = {};
  int sample_counts[6] = {};
  for (int family = 0; family < 6; ++family) {
    Kokkos::parallel_reduce("PC-GH Bowen-York max error",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &maximum) {
      int const m = idx/nkji;
      int const k0 = (idx - m*nkji)/nji;
      int const j0 = (idx - m*nkji - k0*nji)/nx1;
      int const i0 = idx - m*nkji - k0*nji - j0*nx1;
      int const k = indcs.ks + k0;
      int const j = indcs.js + j0;
      int const i = indcs.is + i0;
      Real const x = CellCenterX(i0, nx1, size.d_view(m).x1min,
                                 size.d_view(m).x1max) - center[0];
      Real const y = CellCenterX(j0, nx2, size.d_view(m).x2min,
                                 size.d_view(m).x2max) - center[1];
      Real const z = CellCenterX(k0, nx3, size.d_view(m).x3min,
                                 size.d_view(m).x3max) - center[2];
      Real const radius = std::sqrt(x*x + y*y + z*z);
      if (radius < audit_r_min || radius > audit_r_max) return;
      Real exact_state[pc_gh::PcGh::npcgh];
      Real exact_rhs[pc_gh::PcGh::npcgh];
      ExactTimeSymmetricBowenYork(x, y, z, mass, exact_state, exact_rhs);
      if (family < 4) {
        int const first = (family % 2 == 0) ? 0 : pc_gh::PcGh::I_X1;
        int const last = (family % 2 == 0) ? pc_gh::PcGh::I_X1
                                           : pc_gh::PcGh::npcgh;
        for (int v = first; v < last; ++v) {
          Real const value = (family < 2) ? state(m, v, k, j, i) - exact_state[v]
                                          : state_rhs(m, v, k, j, i) - exact_rhs[v];
          maximum = std::fmax(maximum, std::fabs(value));
        }
      } else {
        int const first = (family == 4) ? pc_gh::PcGh::I_CON_CPERP
                                         : pc_gh::PcGh::I_CON_RED_X;
        int const last = (family == 4) ? pc_gh::PcGh::I_CON_RED_X
                                        : pc_gh::PcGh::I_CON_RMINUS;
        for (int v = first; v < last; ++v) {
          maximum = std::fmax(maximum, std::fabs(constraints(m, v, k, j, i)));
        }
      }
    }, Kokkos::Max<Real>(maxima[family]));

    Kokkos::parallel_reduce("PC-GH Bowen-York RMS error",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &sum, int &count) {
      int const m = idx/nkji;
      int const k0 = (idx - m*nkji)/nji;
      int const j0 = (idx - m*nkji - k0*nji)/nx1;
      int const i0 = idx - m*nkji - k0*nji - j0*nx1;
      int const k = indcs.ks + k0;
      int const j = indcs.js + j0;
      int const i = indcs.is + i0;
      Real const x = CellCenterX(i0, nx1, size.d_view(m).x1min,
                                 size.d_view(m).x1max) - center[0];
      Real const y = CellCenterX(j0, nx2, size.d_view(m).x2min,
                                 size.d_view(m).x2max) - center[1];
      Real const z = CellCenterX(k0, nx3, size.d_view(m).x3min,
                                 size.d_view(m).x3max) - center[2];
      Real const radius = std::sqrt(x*x + y*y + z*z);
      if (radius < audit_r_min || radius > audit_r_max) return;
      Real exact_state[pc_gh::PcGh::npcgh];
      Real exact_rhs[pc_gh::PcGh::npcgh];
      ExactTimeSymmetricBowenYork(x, y, z, mass, exact_state, exact_rhs);
      if (family < 4) {
        int const first = (family % 2 == 0) ? 0 : pc_gh::PcGh::I_X1;
        int const last = (family % 2 == 0) ? pc_gh::PcGh::I_X1
                                           : pc_gh::PcGh::npcgh;
        for (int v = first; v < last; ++v) {
          Real const value = (family < 2) ? state(m, v, k, j, i) - exact_state[v]
                                          : state_rhs(m, v, k, j, i) - exact_rhs[v];
          sum += value*value;
          ++count;
        }
      } else {
        int const first = (family == 4) ? pc_gh::PcGh::I_CON_CPERP
                                         : pc_gh::PcGh::I_CON_RED_X;
        int const last = (family == 4) ? pc_gh::PcGh::I_CON_RED_X
                                        : pc_gh::PcGh::I_CON_RMINUS;
        for (int v = first; v < last; ++v) {
          Real const value = constraints(m, v, k, j, i);
          sum += value*value;
          ++count;
        }
      }
    }, Kokkos::Sum<Real>(square_sums[family]), Kokkos::Sum<int>(sample_counts[family]));
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, maxima, 6, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, square_sums, 6, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sample_counts, 6, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  Real rms[6];
  for (int family = 0; family < 6; ++family) {
    if (sample_counts[family] <= 0 || !std::isfinite(maxima[family])) {
      std::cout << "PC-GH Bowen-York audit shell is empty or nonfinite" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rms[family] = std::sqrt(square_sums[family]/sample_counts[family]);
  }
  if (global_variable::my_rank != 0) return;

  FILE *file = std::fopen("pc_gh_bowen_york-residuals.dat", "a+");
  if (file == nullptr) std::exit(EXIT_FAILURE);
  std::fseek(file, 0, SEEK_END);
  if (std::ftell(file) == 0) {
    std::fprintf(file, "# N max_state_primary max_state_gradient max_rhs_primary "
                       "max_rhs_gradient max_GH_physical max_reduction_curl_algebraic "
                       "rms_state_primary rms_state_gradient rms_rhs_primary "
                       "rms_rhs_gradient rms_GH_physical rms_reduction_curl_algebraic\n");
  }
  std::fprintf(file, "%d", pm->mesh_indcs.nx1);
  for (Real value : maxima) std::fprintf(file, " %.17e", static_cast<double>(value));
  for (Real value : rms) std::fprintf(file, " %.17e", static_cast<double>(value));
  std::fprintf(file, "\n");
  std::fclose(file);
  std::cout << "PC-GH Bowen-York pointwise N=" << pm->mesh_indcs.nx1
            << " RMS state/RHS/GH/reduction=(" << rms[0] << ',' << rms[1] << ','
            << rms[2] << ',' << rms[3] << ',' << rms[4] << ',' << rms[5] << ')'
            << std::endl;

  if (global_variable::nranks != 1) return;
  auto state_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), state);
  auto rhs_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), state_rhs);
  auto con_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), constraints);
  size.template sync<HostMemSpace>();
  std::vector<Real> state_max(pc_gh::PcGh::npcgh, 0.0);
  std::vector<Real> rhs_max(pc_gh::PcGh::npcgh, 0.0);
  std::vector<Real> con_max(pc_gh::PcGh::ncon, 0.0);
  std::vector<Real> state_xyz(3*pc_gh::PcGh::npcgh, 0.0);
  std::vector<Real> rhs_xyz(3*pc_gh::PcGh::npcgh, 0.0);
  std::vector<Real> con_xyz(3*pc_gh::PcGh::ncon, 0.0);
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      Real const z = CellCenterX(k - indcs.ks, nx3, size.h_view(m).x3min,
                                 size.h_view(m).x3max) - center[2];
      for (int j = indcs.js; j <= indcs.je; ++j) {
        Real const y = CellCenterX(j - indcs.js, nx2, size.h_view(m).x2min,
                                   size.h_view(m).x2max) - center[1];
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          Real const x = CellCenterX(i - indcs.is, nx1, size.h_view(m).x1min,
                                     size.h_view(m).x1max) - center[0];
          Real const radius = std::sqrt(x*x + y*y + z*z);
          if (radius < audit_r_min || radius > audit_r_max) continue;
          for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
            Real value = std::fabs(state_host(m, v, k, j, i));
            if (value > state_max[v]) {
              state_max[v] = value;
              state_xyz[3*v] = x; state_xyz[3*v + 1] = y; state_xyz[3*v + 2] = z;
            }
            value = std::fabs(rhs_host(m, v, k, j, i));
            if (value > rhs_max[v]) {
              rhs_max[v] = value;
              rhs_xyz[3*v] = x; rhs_xyz[3*v + 1] = y; rhs_xyz[3*v + 2] = z;
            }
          }
          for (int v = 0; v < pc_gh::PcGh::ncon; ++v) {
            Real const value = std::fabs(con_host(m, v, k, j, i));
            if (value > con_max[v]) {
              con_max[v] = value;
              con_xyz[3*v] = x; con_xyz[3*v + 1] = y; con_xyz[3*v + 2] = z;
            }
          }
        }
      }
    }
  }
  std::string const maxima_name = "pc_gh_bowen_york-maxima-N"
      + std::to_string(pm->mesh_indcs.nx1) + ".dat";
  FILE *max_file = std::fopen(maxima_name.c_str(), "w");
  if (max_file == nullptr) std::exit(EXIT_FAILURE);
  std::fprintf(max_file, "# kind name max_abs x y z\n");
  for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
    std::fprintf(max_file, "state %s %.17e %.17e %.17e %.17e\n",
        pc_gh::PcGh::PcGhNames[v], static_cast<double>(state_max[v]),
        static_cast<double>(state_xyz[3*v]), static_cast<double>(state_xyz[3*v + 1]),
        static_cast<double>(state_xyz[3*v + 2]));
    std::fprintf(max_file, "rhs %s %.17e %.17e %.17e %.17e\n",
        pc_gh::PcGh::PcGhNames[v], static_cast<double>(rhs_max[v]),
        static_cast<double>(rhs_xyz[3*v]), static_cast<double>(rhs_xyz[3*v + 1]),
        static_cast<double>(rhs_xyz[3*v + 2]));
  }
  for (int v = 0; v < pc_gh::PcGh::ncon; ++v) {
    std::fprintf(max_file, "con %s %.17e %.17e %.17e %.17e\n",
        pc_gh::PcGh::ConstraintNames[v], static_cast<double>(con_max[v]),
        static_cast<double>(con_xyz[3*v]), static_cast<double>(con_xyz[3*v + 1]),
        static_cast<double>(con_xyz[3*v + 2]));
  }
  std::fclose(max_file);
}

void CheckPcGhOnePuncture(ParameterInput *, Mesh *pm) {
  if (global_variable::nranks != 1) {
    if (global_variable::my_rank == 0) {
      std::cout << "PC-GH one-puncture final diagnostics currently require Serial"
                << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
  MeshBlockPack *pmbp = pm->pmb_pack;
  CalculateDiagnostics(pmbp);
  auto state = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->ppcgh->u0);
  auto con = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->ppcgh->u_con);
  auto &indcs = pm->mb_indcs;
  Real min_a = std::numeric_limits<Real>::max();
  Real min_chi = std::numeric_limits<Real>::max();
  Real min_spd = std::numeric_limits<Real>::max();
  Real max_state = 0.0;
  Real max_group[4] = {};
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int k = indcs.ks; k <= indcs.ke; ++k) {
      for (int j = indcs.js; j <= indcs.je; ++j) {
        for (int i = indcs.is; i <= indcs.ie; ++i) {
          Real const gxx = state(m, pc_gh::PcGh::I_GTXX, k, j, i);
          Real const gxy = state(m, pc_gh::PcGh::I_GTXY, k, j, i);
          Real const gxz = state(m, pc_gh::PcGh::I_GTXZ, k, j, i);
          Real const gyy = state(m, pc_gh::PcGh::I_GTYY, k, j, i);
          Real const gyz = state(m, pc_gh::PcGh::I_GTYZ, k, j, i);
          Real const gzz = state(m, pc_gh::PcGh::I_GTZZ, k, j, i);
          Real const minor2 = gxx*gyy - gxy*gxy;
          Real const det = gxx*(gyy*gzz - gyz*gyz)
                         - gxy*(gxy*gzz - gxz*gyz)
                         + gxz*(gxy*gyz - gxz*gyy);
          min_a = std::fmin(min_a, state(m, pc_gh::PcGh::I_A, k, j, i));
          min_chi = std::fmin(min_chi, state(m, pc_gh::PcGh::I_CHI, k, j, i));
          min_spd = std::fmin(min_spd, std::fmin(gxx, std::fmin(minor2, det)));
          for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
            max_state = std::fmax(max_state, std::fabs(state(m, v, k, j, i)));
          }
          for (int v = 0; v < pc_gh::PcGh::I_CON_RMINUS; ++v) {
            int group = 3;
            if (v < pc_gh::PcGh::I_CON_H) group = 0;
            else if (v < pc_gh::PcGh::I_CON_RED_X) group = 1;
            else if (v < pc_gh::PcGh::I_CON_DETG) group = 2;
            max_group[group] = std::fmax(
                max_group[group], std::fabs(con(m, v, k, j, i)));
          }
        }
      }
    }
  }
  if (!(std::isfinite(max_state) && std::isfinite(max_group[0])
        && std::isfinite(max_group[1]) && std::isfinite(max_group[2])
        && std::isfinite(max_group[3]) && min_a > 0.0 && min_chi > 0.0
        && min_spd > 0.0)) {
    std::cout << "PC-GH one-puncture run lost finiteness, positivity, or SPD"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank != 0) return;
  FILE *file = std::fopen("pc_gh_one_puncture-final.dat", "a+");
  if (file == nullptr) std::exit(EXIT_FAILURE);
  std::fseek(file, 0, SEEK_END);
  if (std::ftell(file) == 0) {
    std::fprintf(file, "# nx1 time cycles max_state max_GH max_ADM "
                       "max_reduction_curl max_algebraic min_A min_chi min_SPD\n");
  }
  std::fprintf(file, "%d %.17e %d %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
      pm->mesh_indcs.nx1, static_cast<double>(pm->time), pm->ncycle,
      static_cast<double>(max_state), static_cast<double>(max_group[0]),
      static_cast<double>(max_group[1]), static_cast<double>(max_group[2]),
      static_cast<double>(max_group[3]), static_cast<double>(min_a),
      static_cast<double>(min_chi), static_cast<double>(min_spd));
  std::fclose(file);
  std::cout << "PC-GH one puncture: t=" << pm->time << " max(GH,ADM,red,alg)=("
            << max_group[0] << ',' << max_group[1] << ',' << max_group[2] << ','
            << max_group[3] << ") min(A,chi,SPD)=(" << min_a << ',' << min_chi
            << ',' << min_spd << ')' << std::endl;
}

}  // namespace

void ProblemGenerator::PcGhBowenYork(ParameterInput *pin, const bool restart) {
  pgen_final_func = CheckPcGhBowenYork;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppcgh == nullptr || pmbp->padm == nullptr || !pmy_mesh_->three_d) {
    std::cout << "pc_gh_bowen_york requires three-dimensional PC-GH and ADM storage"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real const mass = pin->GetOrAddReal("problem", "mass", 1.0);
  Real const center[3] = {pin->GetOrAddReal("problem", "center_x", 0.0),
                          pin->GetOrAddReal("problem", "center_y", 0.0),
                          pin->GetOrAddReal("problem", "center_z", 0.0)};
  pin->GetOrAddReal("problem", "audit_r_min", 0.5);
  pin->GetOrAddReal("problem", "audit_r_max", 4.0);
  if (!(mass > 0.0)) {
    std::cout << "PC-GH Bowen-York mass must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real const xmin[3] = {pmy_mesh_->mesh_size.x1min, pmy_mesh_->mesh_size.x2min,
                        pmy_mesh_->mesh_size.x3min};
  Real const xmax[3] = {pmy_mesh_->mesh_size.x1max, pmy_mesh_->mesh_size.x2max,
                        pmy_mesh_->mesh_size.x3max};
  int const nx[3] = {pmy_mesh_->mesh_indcs.nx1, pmy_mesh_->mesh_indcs.nx2,
                     pmy_mesh_->mesh_indcs.nx3};
  for (int d = 0; d < 3; ++d) {
    Real const location = (center[d] - xmin[d])*nx[d]/(xmax[d] - xmin[d]);
    if (std::fabs(location - std::nearbyint(location)) > 1.0e-12) {
      std::cout << "PC-GH Bowen-York center must lie on cell faces, not a cell center"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &adm_vars = pmbp->padm->adm;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = indcs.js - indcs.ng;
  int const jeg = indcs.je + indcs.ng;
  int const ksg = indcs.ks - indcs.ng;
  int const keg = indcs.ke + indcs.ng;
  par_for("PC-GH isotropic Bowen-York ADM data", DevExeSpace(),
  0, pmbp->nmb_thispack - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max) - center[0];
    Real const y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max) - center[1];
    Real const z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max) - center[2];
    Real const radius = std::sqrt(x*x + y*y + z*z);
    Real const psi = 1.0 + 0.5*mass/radius;
    Real const psi4 = std::pow(psi, 4);
    adm_vars.alpha(m, k, j, i) = 1.0/(psi*psi);
    adm_vars.psi4(m, k, j, i) = psi4;
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = 0.0;
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = (a == b) ? psi4 : 0.0;
        adm_vars.vK_dd(m, a, b, k, j, i) = 0.0;
      }
    }
  });
  switch (pmbp->ppcgh->opt.fd_stencil) {
    case 2: pmbp->ppcgh->ADMToPcGh<2>(pmbp); break;
    case 3: pmbp->ppcgh->ADMToPcGh<3>(pmbp); break;
    case 4: pmbp->ppcgh->ADMToPcGh<4>(pmbp); break;
    default: std::abort();
  }
  pmbp->ppcgh->PcGhToADM(pmbp);
}

void ProblemGenerator::PcGhOnePuncture(ParameterInput *pin, const bool restart) {
  PcGhBowenYork(pin, restart);
  pgen_final_func = CheckPcGhOnePuncture;
}
