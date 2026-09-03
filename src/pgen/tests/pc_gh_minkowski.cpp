//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_minkowski.cpp
//! \brief exact Minkowski initial data for PC-GH plumbing and algebra tests

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

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
Real RobustNoise(std::uint64_t seed, int variable, int i, int j, int k) {
  std::uint64_t value = seed;
  value ^= static_cast<std::uint64_t>(variable + 1)*0x9e3779b97f4a7c15ULL;
  value ^= static_cast<std::uint64_t>(i + 1)*0xbf58476d1ce4e5b9ULL;
  value ^= static_cast<std::uint64_t>(j + 1)*0x94d049bb133111ebULL;
  value ^= static_cast<std::uint64_t>(k + 1)*0xd6e8feb86659fd93ULL;
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30))*0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27))*0x94d049bb133111ebULL;
  value ^= value >> 31;
  return 2.0*static_cast<Real>(value >> 11)
      *static_cast<Real>(1.0/9007199254740992.0) - 1.0;
}

void CalculatePcGhDiagnostics(MeshBlockPack *pmbp) {
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

void CheckPcGhMinkowski(ParameterInput *, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &state = pmbp->ppcgh->u0;
  auto &state_rhs = pmbp->ppcgh->u_rhs;
  auto &constraints = pmbp->ppcgh->u_con;
  auto &adm_vars = pmbp->padm->adm;
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
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  int const nmkji = pmbp->nmb_thispack*nx3*nx2*nx1;
  int const nkji = nx3*nx2*nx1;
  int const nji = nx2*nx1;
  Real max_error = 0.0;
  Kokkos::parallel_reduce("check exact PC-GH Minkowski state",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, Real &thread_max) {
    int const m = idx/nkji;
    int const k0 = (idx - m*nkji)/nji;
    int const j0 = (idx - m*nkji - k0*nji)/nx1;
    int const i0 = idx - m*nkji - k0*nji - j0*nx1;
    int const k = k0 + ks;
    int const j = j0 + js;
    int const i = i0 + is;
    for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
      Real expected = 0.0;
      if (v == pc_gh::PcGh::I_W || v == pc_gh::PcGh::I_RHO
          || v == pc_gh::PcGh::I_GTXX || v == pc_gh::PcGh::I_GTYY
          || v == pc_gh::PcGh::I_GTZZ) {
        expected = 1.0;
      }
      thread_max = fmax(thread_max, fabs(state(m, v, k, j, i) - expected));
      thread_max = fmax(thread_max, fabs(state_rhs(m, v, k, j, i)));
    }
    for (int v = 0; v < pc_gh::PcGh::ncon; ++v) {
      if (v == pc_gh::PcGh::I_CON_PHYSICAL_VALID) continue;
      Real const expected = (v == pc_gh::PcGh::I_CON_MINOR1
                             || v == pc_gh::PcGh::I_CON_MINOR2
                             || v == pc_gh::PcGh::I_CON_MINEIG) ? 1.0 : 0.0;
      thread_max = fmax(thread_max,
          fabs(constraints(m, v, k, j, i) - expected));
    }
    thread_max = fmax(thread_max, fabs(adm_vars.alpha(m, k, j, i) - 1.0));
    thread_max = fmax(thread_max, fabs(adm_vars.psi4(m, k, j, i) - 1.0));
    for (int a = 0; a < 3; ++a) {
      thread_max = fmax(thread_max, fabs(adm_vars.beta_u(m, a, k, j, i)));
      for (int b = a; b < 3; ++b) {
        Real const expected = (a == b) ? 1.0 : 0.0;
        thread_max = fmax(thread_max,
            fabs(adm_vars.g_dd(m, a, b, k, j, i) - expected));
        thread_max = fmax(thread_max, fabs(adm_vars.vK_dd(m, a, b, k, j, i)));
      }
    }
  }, Kokkos::Max<Real>(max_error));
  if (max_error != 0.0) {
    std::cout << "PC-GH Minkowski state, ADM, RHS, or diagnostic residual = "
              << max_error
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "PASS: exact PC-GH Minkowski state, ADM round trip, RHS, and diagnostics"
              << std::endl;
  }
}

void CheckPcGhRobustMinkowski(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  CalculatePcGhDiagnostics(pmbp);
  auto state = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->ppcgh->u0);
  auto con = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->ppcgh->u_con);
  auto &indcs = pm->mb_indcs;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  long double state_sum2 = 0.0L;
  long double con_sum2 = 0.0L;
  Real state_max = 0.0;
  Real con_max = 0.0;
  Real min_a = std::numeric_limits<Real>::max();
  Real min_chi = std::numeric_limits<Real>::max();
  Real min_spd = std::numeric_limits<Real>::max();
  std::uint64_t count = 0;
  for (int m = 0; m < pmbp->nmb_thispack; ++m) {
    for (int k = ks; k < ks + indcs.nx3; ++k) {
      for (int j = js; j < js + indcs.nx2; ++j) {
        for (int i = is; i < is + indcs.nx1; ++i) {
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
          Real const w = state(m, pc_gh::PcGh::I_W, k, j, i);
          Real const rho = state(m, pc_gh::PcGh::I_RHO, k, j, i);
          min_a = std::fmin(min_a, rho*w);
          min_chi = std::fmin(min_chi, w*w);
          min_spd = std::fmin(min_spd, std::fmin(gxx, std::fmin(minor2, det)));
          for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) {
            Real expected = 0.0;
            if (v == pc_gh::PcGh::I_W || v == pc_gh::PcGh::I_RHO
                || v == pc_gh::PcGh::I_GTXX || v == pc_gh::PcGh::I_GTYY
                || v == pc_gh::PcGh::I_GTZZ) expected = 1.0;
            Real const error = state(m, v, k, j, i) - expected;
            state_sum2 += static_cast<long double>(error)*error;
            state_max = std::fmax(state_max, std::fabs(error));
          }
          for (int v = 0; v < pc_gh::PcGh::I_CON_MINOR1; ++v) {
            Real const value = con(m, v, k, j, i);
            con_sum2 += static_cast<long double>(value)*value;
            con_max = std::fmax(con_max, std::fabs(value));
          }
          ++count;
        }
      }
    }
  }
  double const state_rms = std::sqrt(
      static_cast<double>(state_sum2/(count*pc_gh::PcGh::npcgh)));
  double const con_rms = std::sqrt(
      static_cast<double>(con_sum2/(count*pc_gh::PcGh::I_CON_MINOR1)));
  if (!(std::isfinite(state_rms) && std::isfinite(con_rms)
        && min_a > 0.0 && min_chi > 0.0 && min_spd > 0.0)) {
    std::cout << "PC-GH robust Minkowski lost finiteness, positivity, or SPD" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    FILE *file = std::fopen("pc_gh_robust_minkowski-final.dat", "a+");
    if (file == nullptr) {
      std::cout << "Unable to open pc_gh_robust_minkowski-final.dat" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fseek(file, 0, SEEK_END);
    if (std::ftell(file) == 0) {
      std::fprintf(file, "# nx1 nx2 nx3 time cycles amplitude seed state_rms "
                         "state_linf constraint_rms constraint_linf min_A min_chi min_SPD\n");
    }
    std::fprintf(file, "%d %d %d %.17e %d %.17e %d %.17e %.17e %.17e %.17e "
                       "%.17e %.17e %.17e\n",
                 pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3,
                 static_cast<double>(pm->time), pm->ncycle,
                 static_cast<double>(pin->GetReal("problem", "amp")),
                 pin->GetInteger("problem", "seed"), state_rms,
                 static_cast<double>(state_max), con_rms,
                 static_cast<double>(con_max), static_cast<double>(min_a),
                 static_cast<double>(min_chi), static_cast<double>(min_spd));
    std::fclose(file);
    std::cout << "PC-GH robust Minkowski: nx1=" << pm->mesh_indcs.nx1
              << " state_rms=" << state_rms << " constraint_rms=" << con_rms
              << " min_A=" << min_a << " min_chi=" << min_chi
              << " min_SPD=" << min_spd << std::endl;
  }
}

}  // namespace

void ProblemGenerator::PcGhMinkowski(ParameterInput *pin, const bool restart) {
  pgen_final_func = CheckPcGhMinkowski;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppcgh == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "pc_gh_minkowski requires a <pc_gh> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmbp->pmesh->mb_indcs;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  int const nmb = pmbp->nmb_thispack;
  auto &adm_vars = pmbp->padm->adm;

  par_for("PC-GH Minkowski ADM data", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    adm_vars.alpha(m, k, j, i) = 1.0;
    adm_vars.psi4(m, k, j, i) = 1.0;
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = 0.0;
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = (a == b) ? 1.0 : 0.0;
        adm_vars.vK_dd(m, a, b, k, j, i) = 0.0;
      }
    }
  });

  switch (pmbp->ppcgh->opt.fd_stencil) {
    case 2:
      pmbp->ppcgh->ADMToPcGh<2>(pmbp);
      break;
    case 3:
      pmbp->ppcgh->ADMToPcGh<3>(pmbp);
      break;
    case 4:
      pmbp->ppcgh->ADMToPcGh<4>(pmbp);
      break;
    default:
      std::abort();
  }
  pmbp->ppcgh->PcGhToADM(pmbp);
}

void ProblemGenerator::PcGhRobustMinkowski(ParameterInput *pin, const bool restart) {
  pgen_final_func = CheckPcGhRobustMinkowski;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppcgh == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "pc_gh_robust_minkowski requires a <pc_gh> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->multilevel) {
    std::cout << "pc_gh_robust_minkowski is a uniform-grid gate" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto state = pmbp->ppcgh->u0;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  int const nmb = pmbp->nmb_thispack;
  int const nx[3] = {pmy_mesh_->mesh_indcs.nx1, pmy_mesh_->mesh_indcs.nx2,
                     pmy_mesh_->mesh_indcs.nx3};
  Real const xmin[3] = {pmy_mesh_->mesh_size.x1min, pmy_mesh_->mesh_size.x2min,
                        pmy_mesh_->mesh_size.x3min};
  Real const dx[3] = {
      (pmy_mesh_->mesh_size.x1max - xmin[0])/nx[0],
      (pmy_mesh_->mesh_size.x2max - xmin[1])/nx[1],
      (pmy_mesh_->mesh_size.x3max - xmin[2])/nx[2]};
  Real const amplitude = pin->GetOrAddReal("problem", "amp", 1.0e-10);
  int const signed_seed = pin->GetOrAddInteger("problem", "seed", 20260901);
  std::uint64_t const seed = static_cast<std::uint64_t>(signed_seed);
  if (!(amplitude > 0.0 && amplitude < 1.0e-2)) {
    std::cout << "<problem>/amp must lie in (0,1e-2) for robust Minkowski" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  par_for("PC-GH algebraic random Minkowski perturbation", DevExeSpace(),
  0, nmb - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x[3] = {
        CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                    size.d_view(m).x1max),
        CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                    size.d_view(m).x2max),
        CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                    size.d_view(m).x3max)};
    int gi[3];
    for (int d = 0; d < 3; ++d) {
      int raw = static_cast<int>(std::floor((x[d] - xmin[d])/dx[d]));
      raw %= nx[d];
      gi[d] = (raw < 0) ? raw + nx[d] : raw;
    }
    auto noise = [&](int v) {
      return RobustNoise(seed, v, gi[0], gi[1], gi[2]);
    };
    for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) state(m, v, k, j, i) = 0.0;
    state(m, pc_gh::PcGh::I_W, k, j, i) = std::exp(amplitude*noise(0));
    Real const e0 = amplitude*noise(1);
    Real const e1 = amplitude*noise(2);
    Real const gdiag[3] = {std::exp(e0), std::exp(e1), std::exp(-e0 - e1)};
    state(m, pc_gh::PcGh::I_GTXX, k, j, i) = gdiag[0];
    state(m, pc_gh::PcGh::I_GTYY, k, j, i) = gdiag[1];
    state(m, pc_gh::PcGh::I_GTZZ, k, j, i) = gdiag[2];
    Real const a0 = amplitude*noise(3);
    Real const a1 = amplitude*noise(4);
    state(m, pc_gh::PcGh::I_ATXX, k, j, i) = gdiag[0]*a0;
    state(m, pc_gh::PcGh::I_ATYY, k, j, i) = gdiag[1]*a1;
    state(m, pc_gh::PcGh::I_ATZZ, k, j, i) = gdiag[2]*(-a0 - a1);
    state(m, pc_gh::PcGh::I_K, k, j, i) = amplitude*noise(5);
    for (int d = 0; d < 3; ++d) {
      state(m, pc_gh::PcGh::I_ZX + d, k, j, i) = amplitude*noise(6 + d);
      state(m, pc_gh::PcGh::I_BETAX + d, k, j, i) = amplitude*noise(10 + d);
      state(m, pc_gh::PcGh::I_P1 + d, k, j, i) = amplitude*noise(13 + d);
      state(m, pc_gh::PcGh::I_L1 + d, k, j, i) = amplitude*noise(17 + d);
      Real const q0 = amplitude*noise(20 + 2*d);
      Real const q1 = amplitude*noise(21 + 2*d);
      state(m, pc_gh::PcGh::QIndex(d, 0, 0), k, j, i) = gdiag[0]*q0;
      state(m, pc_gh::PcGh::QIndex(d, 1, 1), k, j, i) = gdiag[1]*q1;
      state(m, pc_gh::PcGh::QIndex(d, 2, 2), k, j, i) = gdiag[2]*(-q0 - q1);
      for (int a = 0; a < 3; ++a) {
        state(m, pc_gh::PcGh::BIndex(d, a), k, j, i) =
            amplitude*noise(26 + 3*d + a);
      }
    }
    state(m, pc_gh::PcGh::I_CPERP, k, j, i) = amplitude*noise(9);
    state(m, pc_gh::PcGh::I_RHO, k, j, i) = std::exp(amplitude*noise(16));
  });
  pmbp->ppcgh->PcGhToADM(pmbp);
  CalculatePcGhDiagnostics(pmbp);
}
