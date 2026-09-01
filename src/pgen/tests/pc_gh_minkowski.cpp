//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_minkowski.cpp
//! \brief exact Minkowski initial data for PC-GH plumbing and algebra tests

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {

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
      if (v == pc_gh::PcGh::I_CHI || v == pc_gh::PcGh::I_A
          || v == pc_gh::PcGh::I_GTXX || v == pc_gh::PcGh::I_GTYY
          || v == pc_gh::PcGh::I_GTZZ) {
        expected = 1.0;
      }
      thread_max = fmax(thread_max, fabs(state(m, v, k, j, i) - expected));
      thread_max = fmax(thread_max, fabs(state_rhs(m, v, k, j, i)));
    }
    for (int v = 0; v < pc_gh::PcGh::ncon; ++v) {
      Real const expected =
          (v == pc_gh::PcGh::I_CON_RMINUS || v == pc_gh::PcGh::I_CON_RPLUS)
              ? 1.0 : 0.0;
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
