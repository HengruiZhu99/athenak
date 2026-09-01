//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_trumpet_a0.cpp
//! \brief stationary Schwarzschild 1+log trumpet target for prescribed Gauge A0

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {

void CheckPcGhFrozenOperator(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &pcgh = *pmbp->ppcgh;
  auto &indcs = pm->mb_indcs;
  if (global_variable::nranks != 1 || pmbp->nmb_thispack != 1
      || indcs.nx1 < 5 || indcs.nx2 < 5 || indcs.nx3 < 5
      || indcs.nx1%2 == 0 || indcs.nx2%2 == 0 || indcs.nx3%2 == 0) {
    std::cout << "PC-GH frozen operator requires one rank, one MeshBlock, and odd "
              << "three-dimensional extents of at least five cells" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &size = pmbp->pmb->mb_size;
  size.template sync<HostMemSpace>();
  int const ic = indcs.is + indcs.nx1/2;
  int const jc = indcs.js + indcs.nx2/2;
  int const kc = indcs.ks + indcs.nx3/2;
  Real const x0 = CellCenterX(ic - indcs.is, indcs.nx1,
                              size.h_view(0).x1min, size.h_view(0).x1max);
  Real const y0 = CellCenterX(jc - indcs.js, indcs.nx2,
                              size.h_view(0).x2min, size.h_view(0).x2max);
  Real const z0 = CellCenterX(kc - indcs.ks, indcs.nx3,
                              size.h_view(0).x3min, size.h_view(0).x3max);
  Real const wave_kx = pin->GetOrAddReal("problem", "frozen_kx", 1.0);
  Real const wave_ky = pin->GetOrAddReal("problem", "frozen_ky", 0.0);
  Real const wave_kz = pin->GetOrAddReal("problem", "frozen_kz", 0.0);
  Real const relative_epsilon =
      pin->GetOrAddReal("problem", "frozen_epsilon", 1.0e-7);
  if (!(relative_epsilon > 0.0)) {
    std::cout << "PC-GH frozen-operator epsilon must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto calculate_rhs = [&]() {
    switch (pcgh.opt.fd_stencil) {
      case 2: (void)pcgh.CalcRHS<2>(nullptr, 0); break;
      case 3: (void)pcgh.CalcRHS<3>(nullptr, 0); break;
      case 4: (void)pcgh.CalcRHS<4>(nullptr, 0); break;
      default: std::abort();
    }
    auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pcgh.u_rhs);
    std::vector<Real> result(pc_gh::PcGh::npcgh);
    for (int row = 0; row < pc_gh::PcGh::npcgh; ++row) {
      result[row] = host(0, row, kc, jc, ic);
    }
    return result;
  };

  Kokkos::deep_copy(pcgh.u1, pcgh.u0);
  auto background_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pcgh.u1);
  std::vector<Real> const background_rhs = calculate_rhs();
  int constexpr nstate = pc_gh::PcGh::npcgh;
  std::vector<Real> lower_order(nstate*nstate, 0.0);
  std::vector<Real> fd_response(nstate*nstate, 0.0);
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = indcs.js - indcs.ng;
  int const jeg = indcs.je + indcs.ng;
  int const ksg = indcs.ks - indcs.ng;
  int const keg = indcs.ke + indcs.ng;
  auto state = pcgh.u0;

  for (int column = 0; column < nstate; ++column) {
    Real const scale = std::max(1.0, std::fabs(background_host(0, column, kc, jc, ic)));
    Real const epsilon = relative_epsilon*scale;
    for (int mode = 0; mode < 2; ++mode) {
      std::vector<Real> side[2];
      for (int side_index = 0; side_index < 2; ++side_index) {
        Real const sign = (side_index == 0) ? -1.0 : 1.0;
        Kokkos::deep_copy(pcgh.u0, pcgh.u1);
        par_for("PC-GH frozen operator perturbation", DevExeSpace(),
        0, 0, ksg, keg, jsg, jeg, isg, ieg,
        KOKKOS_LAMBDA(int m, int k, int j, int i) {
          Real waveform = 1.0;
          if (mode == 1) {
            Real const x = CellCenterX(i - indcs.is, indcs.nx1,
                size.d_view(m).x1min, size.d_view(m).x1max);
            Real const y = CellCenterX(j - indcs.js, indcs.nx2,
                size.d_view(m).x2min, size.d_view(m).x2max);
            Real const z = CellCenterX(k - indcs.ks, indcs.nx3,
                size.d_view(m).x3min, size.d_view(m).x3max);
            waveform = std::sin(wave_kx*(x - x0) + wave_ky*(y - y0)
                                + wave_kz*(z - z0));
          }
          state(m, column, k, j, i) += sign*epsilon*waveform;
        });
        Kokkos::fence();
        side[side_index] = calculate_rhs();
      }
      for (int row = 0; row < nstate; ++row) {
        Real const derivative = (side[1][row] - side[0][row])/(2.0*epsilon);
        if (!std::isfinite(derivative)) {
          std::cout << "PC-GH frozen operator contains a nonfinite entry" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        ((mode == 0) ? lower_order : fd_response)[row*nstate + column] = derivative;
      }
    }
  }
  Kokkos::deep_copy(pcgh.u0, pcgh.u1);

  FILE *file = std::fopen("pc_gh_frozen_operator.dat", "w");
  if (file == nullptr) std::exit(EXIT_FAILURE);
  std::fprintf(file, "# PC-GH frozen operator at x=(%.17e,%.17e,%.17e)\n",
      static_cast<double>(x0), static_cast<double>(y0), static_cast<double>(z0));
  std::fprintf(file, "# k=(%.17e,%.17e,%.17e) epsilon=%.17e order=%d\n",
      static_cast<double>(wave_kx), static_cast<double>(wave_ky),
      static_cast<double>(wave_kz), static_cast<double>(relative_epsilon),
      pcgh.opt.spatial_order);
  std::fprintf(file, "# kind row column value\n");
  for (int row = 0; row < nstate; ++row) {
    std::fprintf(file, "S %d -1 %.17e\n", row,
                 static_cast<double>(background_host(0, row, kc, jc, ic)));
    std::fprintf(file, "R %d -1 %.17e\n", row,
                 static_cast<double>(background_rhs[row]));
    for (int column = 0; column < nstate; ++column) {
      std::fprintf(file, "B %d %d %.17e\n", row, column,
                   static_cast<double>(lower_order[row*nstate + column]));
      std::fprintf(file, "D %d %d %.17e\n", row, column,
                   static_cast<double>(fd_response[row*nstate + column]));
    }
  }
  std::fclose(file);
  std::cout << "PASS: wrote 55x55 PC-GH lower-order and FD-response matrices at ("
            << x0 << ',' << y0 << ',' << z0 << ")" << std::endl;
}

void CheckPcGhTrumpetA0(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &pcgh = *pmbp->ppcgh;
  switch (pcgh.opt.fd_stencil) {
    case 2:
      (void)pcgh.CalcRHS<2>(nullptr, 0);
      (void)pcgh.CalcConstraints<2>(nullptr, 0);
      break;
    case 3:
      (void)pcgh.CalcRHS<3>(nullptr, 0);
      (void)pcgh.CalcConstraints<3>(nullptr, 0);
      break;
    case 4:
      (void)pcgh.CalcRHS<4>(nullptr, 0);
      (void)pcgh.CalcConstraints<4>(nullptr, 0);
      break;
    default:
      std::abort();
  }
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto state_rhs = pcgh.u_rhs;
  auto constraints = pcgh.u_con;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const nkji = nx3*nx2*nx1;
  int const nji = nx2*nx1;
  int const nmkji = pmbp->nmb_thispack*nkji;
  Real const audit_r_min = pin->GetOrAddReal("problem", "audit_r_min", 0.1);
  Real const audit_r_max = pin->GetOrAddReal("problem", "audit_r_max", 8.0);
  Real const center_x = pcgh.opt.gauge_center[0];
  Real const center_y = pcgh.opt.gauge_center[1];
  Real const center_z = pcgh.opt.gauge_center[2];

  Real maxima[4] = {0.0, 0.0, 0.0, 0.0};
  Real rms[4] = {0.0, 0.0, 0.0, 0.0};
  Real square_sums[4] = {0.0, 0.0, 0.0, 0.0};
  int sample_counts[4] = {0, 0, 0, 0};
  for (int family = 0; family < 4; ++family) {
    Kokkos::parallel_reduce("PC-GH trumpet residual family",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &maximum) {
      int const m = idx/nkji;
      int const k0 = (idx - m*nkji)/nji;
      int const j0 = (idx - m*nkji - k0*nji)/nx1;
      int const i0 = idx - m*nkji - k0*nji - j0*nx1;
      int const k = indcs.ks + k0;
      int const j = indcs.js + j0;
      int const i = indcs.is + i0;
      Real const x = CellCenterX(i - indcs.is, nx1, size.d_view(m).x1min,
                                 size.d_view(m).x1max) - center_x;
      Real const y = CellCenterX(j - indcs.js, nx2, size.d_view(m).x2min,
                                 size.d_view(m).x2max) - center_y;
      Real const z = CellCenterX(k - indcs.ks, nx3, size.d_view(m).x3min,
                                 size.d_view(m).x3max) - center_z;
      Real const radius = std::sqrt(x*x + y*y + z*z);
      if (radius < audit_r_min || radius > audit_r_max) return;
      if (family < 2) {
        int const first = (family == 0) ? 0 : pc_gh::PcGh::I_X1;
        int const last = (family == 0) ? pc_gh::PcGh::I_X1 : pc_gh::PcGh::npcgh;
        for (int v = first; v < last; ++v) {
          maximum = std::fmax(maximum, std::fabs(state_rhs(m, v, k, j, i)));
        }
      } else {
        int const first = (family == 2) ? pc_gh::PcGh::I_CON_CPERP
                                         : pc_gh::PcGh::I_CON_RED_X;
        int const last = (family == 2) ? pc_gh::PcGh::I_CON_RED_X
                                        : pc_gh::PcGh::I_CON_RMINUS;
        for (int v = first; v < last; ++v) {
          maximum = std::fmax(maximum, std::fabs(constraints(m, v, k, j, i)));
        }
      }
    }, Kokkos::Max<Real>(maxima[family]));
    Kokkos::parallel_reduce("PC-GH trumpet residual RMS",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &sum, int &count) {
      int const m = idx/nkji;
      int const k0 = (idx - m*nkji)/nji;
      int const j0 = (idx - m*nkji - k0*nji)/nx1;
      int const i0 = idx - m*nkji - k0*nji - j0*nx1;
      int const k = indcs.ks + k0;
      int const j = indcs.js + j0;
      int const i = indcs.is + i0;
      Real const x = CellCenterX(i - indcs.is, nx1, size.d_view(m).x1min,
                                 size.d_view(m).x1max) - center_x;
      Real const y = CellCenterX(j - indcs.js, nx2, size.d_view(m).x2min,
                                 size.d_view(m).x2max) - center_y;
      Real const z = CellCenterX(k - indcs.ks, nx3, size.d_view(m).x3min,
                                 size.d_view(m).x3max) - center_z;
      Real const radius = std::sqrt(x*x + y*y + z*z);
      if (radius < audit_r_min || radius > audit_r_max) return;
      if (family < 2) {
        int const first = (family == 0) ? 0 : pc_gh::PcGh::I_X1;
        int const last = (family == 0) ? pc_gh::PcGh::I_X1 : pc_gh::PcGh::npcgh;
        for (int v = first; v < last; ++v) {
          sum += SQR(state_rhs(m, v, k, j, i));
          ++count;
        }
      } else {
        int const first = (family == 2) ? pc_gh::PcGh::I_CON_CPERP
                                         : pc_gh::PcGh::I_CON_RED_X;
        int const last = (family == 2) ? pc_gh::PcGh::I_CON_RED_X
                                        : pc_gh::PcGh::I_CON_RMINUS;
        for (int v = first; v < last; ++v) {
          sum += SQR(constraints(m, v, k, j, i));
          ++count;
        }
      }
    }, Kokkos::Sum<Real>(square_sums[family]),
       Kokkos::Sum<int>(sample_counts[family]));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, maxima, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, square_sums, 4, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sample_counts, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int family = 0; family < 4; ++family) {
    if (sample_counts[family] <= 0) {
      std::cout << "PC-GH Gauge A0 audit shell contains no samples" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rms[family] = std::sqrt(square_sums[family]
                            /static_cast<Real>(sample_counts[family]));
  }
  for (Real value : maxima) {
    if (!std::isfinite(value)) {
      std::cout << "PC-GH Gauge A0 pointwise residual is nonfinite" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (global_variable::my_rank == 0) {
    FILE *file = std::fopen("pc_gh_trumpet_a0-residuals.dat", "a");
    if (file == nullptr) {
      std::cout << "Unable to open pc_gh_trumpet_a0-residuals.dat" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file,
        "%d %.17e %.17e %.17e %.17e %.17e %.17e %.17e %.17e\n",
        pm->mesh_indcs.nx1,
        static_cast<double>(maxima[0]), static_cast<double>(maxima[1]),
        static_cast<double>(maxima[2]), static_cast<double>(maxima[3]),
        static_cast<double>(rms[0]), static_cast<double>(rms[1]),
        static_cast<double>(rms[2]), static_cast<double>(rms[3]));
    std::fclose(file);
    std::cout << "PC-GH Gauge A0 pointwise: primary=" << maxima[0]
              << " gradient=" << maxima[1] << " GH=" << maxima[2]
              << " reduction/curl/algebraic=" << maxima[3]
              << " RMS=(" << rms[0] << ',' << rms[1] << ',' << rms[2]
              << ',' << rms[3] << ')' << std::endl;
    if (global_variable::nranks == 1) {
      auto rhs_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), state_rhs);
      auto con_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), constraints);
      std::vector<Real> rhs_max(pc_gh::PcGh::npcgh, 0.0);
      std::vector<Real> con_max(pc_gh::PcGh::ncon, 0.0);
      std::vector<Real> rhs_x(pc_gh::PcGh::npcgh, 0.0);
      std::vector<Real> rhs_y(pc_gh::PcGh::npcgh, 0.0);
      std::vector<Real> rhs_z(pc_gh::PcGh::npcgh, 0.0);
      std::vector<Real> con_x(pc_gh::PcGh::ncon, 0.0);
      std::vector<Real> con_y(pc_gh::PcGh::ncon, 0.0);
      std::vector<Real> con_z(pc_gh::PcGh::ncon, 0.0);
      size.template sync<HostMemSpace>();
      for (int m = 0; m < pmbp->nmb_thispack; ++m) {
        for (int k = indcs.ks; k <= indcs.ke; ++k) {
          Real const z = CellCenterX(k - indcs.ks, nx3, size.h_view(m).x3min,
                                     size.h_view(m).x3max) - center_z;
          for (int j = indcs.js; j <= indcs.je; ++j) {
            Real const y = CellCenterX(j - indcs.js, nx2, size.h_view(m).x2min,
                                       size.h_view(m).x2max) - center_y;
            for (int i = indcs.is; i <= indcs.ie; ++i) {
              Real const x = CellCenterX(i - indcs.is, nx1, size.h_view(m).x1min,
                                         size.h_view(m).x1max) - center_x;
              Real const radius = std::sqrt(x*x + y*y + z*z);
              if (radius < audit_r_min || radius > audit_r_max) continue;
              for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
                Real const value = std::fabs(rhs_host(m, v, k, j, i));
                if (value > rhs_max[v]) {
                  rhs_max[v] = value;
                  rhs_x[v] = x; rhs_y[v] = y; rhs_z[v] = z;
                }
              }
              for (int v = 0; v < pc_gh::PcGh::ncon; ++v) {
                Real const value = std::fabs(con_host(m, v, k, j, i));
                if (value > con_max[v]) {
                  con_max[v] = value;
                  con_x[v] = x; con_y[v] = y; con_z[v] = z;
                }
              }
            }
          }
        }
      }
      FILE *max_file = std::fopen("pc_gh_trumpet_a0-maxima.dat", "w");
      if (max_file == nullptr) std::exit(EXIT_FAILURE);
      std::fprintf(max_file, "# kind name max_abs x y z\n");
      for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
        std::fprintf(max_file, "rhs %s %.17e %.17e %.17e %.17e\n",
            pc_gh::PcGh::PcGhNames[v], static_cast<double>(rhs_max[v]),
            static_cast<double>(rhs_x[v]), static_cast<double>(rhs_y[v]),
            static_cast<double>(rhs_z[v]));
      }
      for (int v = 0; v < pc_gh::PcGh::ncon; ++v) {
        std::fprintf(max_file, "con %s %.17e %.17e %.17e %.17e\n",
            pc_gh::PcGh::ConstraintNames[v], static_cast<double>(con_max[v]),
            static_cast<double>(con_x[v]), static_cast<double>(con_y[v]),
            static_cast<double>(con_z[v]));
      }
      std::fclose(max_file);
    }
  }
}

}  // namespace

void ProblemGenerator::PcGhTrumpetA0(ParameterInput *pin, const bool restart) {
  pgen_final_func = pin->GetOrAddBoolean("problem", "frozen_operator", false)
      ? CheckPcGhFrozenOperator : CheckPcGhTrumpetA0;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppcgh == nullptr || pmbp->padm == nullptr
      || pmbp->ppcgh->opt.gauge != "a0") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "pc_gh_trumpet_a0 requires <pc_gh>/gauge=a0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &pcgh = *pmbp->ppcgh;
  auto &state = pcgh.u0;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto table = pcgh.gauge_a0_table;
  int const npoints = pcgh.gauge_a0_npoints;
  Real const log_r_min = pcgh.gauge_a0_log_r_min;
  Real const inv_dlog_r = pcgh.gauge_a0_inv_dlog_r;
  Real const mass = pcgh.opt.gauge_mass;
  Real const center_x = pcgh.opt.gauge_center[0];
  Real const center_y = pcgh.opt.gauge_center[1];
  Real const center_z = pcgh.opt.gauge_center[2];
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = indcs.js - indcs.ng;
  int const jeg = indcs.je + indcs.ng;
  int const ksg = indcs.ks - indcs.ng;
  int const keg = indcs.ke + indcs.ng;

  par_for("PC-GH Gauge A0 trumpet target", DevExeSpace(),
  0, pmbp->nmb_thispack - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const coord[3] = {
        CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                    size.d_view(m).x1max) - center_x,
        CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                    size.d_view(m).x2max) - center_y,
        CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                    size.d_view(m).x3max) - center_z};
    Real const radius = std::sqrt(coord[0]*coord[0] + coord[1]*coord[1]
                                  + coord[2]*coord[2]);
    Real const log_radius = std::log(radius/mass);
    pc_gh::PcGh::GaugeA0Point const target = pc_gh::PcGh::EvaluateGaugeA0(
        table, npoints, log_r_min, inv_dlog_r, log_radius);
    for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) state(m, v, k, j, i) = 0.0;
    state(m, pc_gh::PcGh::I_A, k, j, i) = target.A;
    state(m, pc_gh::PcGh::I_CHI, k, j, i) = target.chi;
    state(m, pc_gh::PcGh::I_K, k, j, i) = target.K/mass;
    state(m, pc_gh::PcGh::I_PI, k, j, i) = -target.K/mass;
    state(m, pc_gh::PcGh::I_GTXX, k, j, i) = 1.0;
    state(m, pc_gh::PcGh::I_GTYY, k, j, i) = 1.0;
    state(m, pc_gh::PcGh::I_GTZZ, k, j, i) = 1.0;
    for (int q = 0; q < 3; ++q) {
      Real const normal_q = coord[q]/radius;
      state(m, pc_gh::PcGh::I_BETAX + q, k, j, i) = target.beta_r*normal_q;
      state(m, pc_gh::PcGh::I_X1 + q, k, j, i) = target.dx_chi*normal_q/radius;
      state(m, pc_gh::PcGh::I_Y1 + q, k, j, i) = target.dx_A*normal_q/radius;
      for (int p = 0; p < 3; ++p) {
        Real const normal_p = coord[p]/radius;
        Real const delta = (p == q) ? 1.0 : 0.0;
        state(m, pc_gh::PcGh::BIndex(p, q), k, j, i) =
            (target.b_tangential*delta
             + std::sqrt(target.A)*target.at_radial*normal_p*normal_q)/mass;
      }
      for (int p = q; p < 3; ++p) {
        Real const normal_p = coord[p]/radius;
        Real const delta = (p == q) ? 1.0 : 0.0;
        state(m, pc_gh::PcGh::I_ATXX + pc_gh::PcGh::SymmetricIndex(q, p),
              k, j, i) = target.at_radial*(normal_q*normal_p - delta/3.0)/mass;
      }
    }
  });
  pcgh.PcGhToADM(pmbp);
}
