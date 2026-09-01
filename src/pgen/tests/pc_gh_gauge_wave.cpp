//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_gauge_wave.cpp
//! \brief exact periodic nonlinear harmonic gauge wave for PC-GH

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

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
void SetGaugeWaveState(DvceArray5D<Real> state, int m, int k, int j, int i,
                       Real x, Real time, Real amplitude, Real wave_number) {
  Real const phase = wave_number*(x - time);
  Real const h = 1.0 - amplitude*std::sin(phase);
  Real const hx = -amplitude*wave_number*std::cos(phase);
  Real const ht = -hx;
  Real const sqrt_h = std::sqrt(h);
  Real const chi = std::pow(h, -1.0/3.0);
  Real const k_xx = -ht/(2.0*sqrt_h);
  Real const trace_k = k_xx/h;

  for (int v = 0; v < pc_gh::PcGh::npcgh; ++v) state(m, v, k, j, i) = 0.0;
  state(m, pc_gh::PcGh::I_CHI, k, j, i) = chi;
  state(m, pc_gh::PcGh::I_GTXX, k, j, i) = std::pow(h, 2.0/3.0);
  state(m, pc_gh::PcGh::I_GTYY, k, j, i) = chi;
  state(m, pc_gh::PcGh::I_GTZZ, k, j, i) = chi;
  state(m, pc_gh::PcGh::I_K, k, j, i) = trace_k;
  state(m, pc_gh::PcGh::I_ATXX, k, j, i) = 2.0*chi*k_xx/3.0;
  state(m, pc_gh::PcGh::I_ATYY, k, j, i) = -chi*trace_k/3.0;
  state(m, pc_gh::PcGh::I_ATZZ, k, j, i) = -chi*trace_k/3.0;
  state(m, pc_gh::PcGh::I_LAMX, k, j, i) =
      2.0*std::pow(h, -5.0/3.0)*hx/3.0;
  state(m, pc_gh::PcGh::I_PI, k, j, i) = -trace_k;
  state(m, pc_gh::PcGh::I_A, k, j, i) = h;
  state(m, pc_gh::PcGh::I_X1, k, j, i) =
      -std::pow(h, -4.0/3.0)*hx/3.0;
  state(m, pc_gh::PcGh::I_Q1XX, k, j, i) =
      2.0*std::pow(h, -1.0/3.0)*hx/3.0;
  state(m, pc_gh::PcGh::I_Q1YY, k, j, i) =
      -std::pow(h, -4.0/3.0)*hx/3.0;
  state(m, pc_gh::PcGh::I_Q1ZZ, k, j, i) =
      -std::pow(h, -4.0/3.0)*hx/3.0;
  state(m, pc_gh::PcGh::I_Y1, k, j, i) = hx;
}

void CheckPcGhGaugeWave(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &state = pmbp->ppcgh->u0;
  auto &expected = pmbp->ppcgh->u1;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  int const nkji = nx3*nx2*nx1;
  int const nji = nx2*nx1;
  int const nmkji = pmbp->nmb_thispack*nkji;
  Real const amplitude = pin->GetReal("problem", "amp");
  Real const length = pm->mesh_size.x1max - pm->mesh_size.x1min;
  Real const wave_number = 2.0*M_PI/length;
  Real const time = pm->time;

  par_for("exact PC-GH gauge wave", DevExeSpace(),
  0, pmbp->nmb_thispack - 1, indcs.ks, indcs.ke, indcs.js, indcs.je,
  indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    SetGaugeWaveState(expected, m, k, j, i, x, time, amplitude, wave_number);
  });

  Real l1_sum = 0.0;
  Real linfty = 0.0;
  Kokkos::parallel_reduce("PC-GH gauge-wave error",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji*pc_gh::PcGh::npcgh),
  KOKKOS_LAMBDA(int idx, Real &sum, Real &maximum) {
    int const v = idx % pc_gh::PcGh::npcgh;
    int const cell = idx/pc_gh::PcGh::npcgh;
    int const m = cell/nkji;
    int const q = cell - m*nkji;
    int const k0 = q/nji;
    int const j0 = (q - k0*nji)/nx1;
    int const i0 = q - k0*nji - j0*nx1;
    int const k = ks + k0;
    int const j = js + j0;
    int const i = is + i0;
    Real const error = std::fabs(state(m, v, k, j, i) - expected(m, v, k, j, i));
    sum += error;
    maximum = std::fmax(maximum, error);
  }, Kokkos::Sum<Real>(l1_sum), Kokkos::Max<Real>(linfty));
  Real const l1 = l1_sum/(static_cast<Real>(nmkji)*pc_gh::PcGh::npcgh);

  if (!std::isfinite(l1) || !std::isfinite(linfty)) {
    std::cout << "PC-GH gauge-wave error is nonfinite" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    FILE *file = std::fopen("pc_gh_gauge_wave-errors.dat", "a");
    if (file == nullptr) {
      std::cout << "Unable to open pc_gh_gauge_wave-errors.dat" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file, "%d %.17e %.17e\n", pm->mesh_indcs.nx1,
                 static_cast<double>(l1), static_cast<double>(linfty));
    std::fclose(file);
    std::cout << "PC-GH gauge wave: nx1=" << pm->mesh_indcs.nx1
              << " L1=" << l1 << " Linf=" << linfty << std::endl;
  }
}

}  // namespace

void ProblemGenerator::PcGhGaugeWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = CheckPcGhGaugeWave;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->ppcgh == nullptr || pmbp->padm == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "pc_gh_gauge_wave requires a <pc_gh> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &adm_vars = pmbp->padm->adm;
  int const isg = indcs.is - indcs.ng;
  int const ieg = indcs.ie + indcs.ng;
  int const jsg = pmbp->pmesh->multi_d ? indcs.js - indcs.ng : indcs.js;
  int const jeg = pmbp->pmesh->multi_d ? indcs.je + indcs.ng : indcs.je;
  int const ksg = pmbp->pmesh->three_d ? indcs.ks - indcs.ng : indcs.ks;
  int const keg = pmbp->pmesh->three_d ? indcs.ke + indcs.ng : indcs.ke;
  int const is = indcs.is;
  int const nx1 = indcs.nx1;
  Real const amplitude = pin->GetOrAddReal("problem", "amp", 0.01);
  Real const length = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  Real const wave_number = 2.0*M_PI/length;

  par_for("PC-GH gauge-wave ADM data", DevExeSpace(),
  0, pmbp->nmb_thispack - 1, ksg, keg, jsg, jeg, isg, ieg,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x = CellCenterX(i - is, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max);
    Real const phase = wave_number*x;
    Real const h = 1.0 - amplitude*std::sin(phase);
    Real const ht = amplitude*wave_number*std::cos(phase);
    Real const sqrt_h = std::sqrt(h);
    adm_vars.alpha(m, k, j, i) = sqrt_h;
    adm_vars.psi4(m, k, j, i) = std::pow(h, 1.0/3.0);
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = 0.0;
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = (a == b) ? 1.0 : 0.0;
        adm_vars.vK_dd(m, a, b, k, j, i) = 0.0;
      }
    }
    adm_vars.g_dd(m, 0, 0, k, j, i) = h;
    adm_vars.vK_dd(m, 0, 0, k, j, i) = -ht/(2.0*sqrt_h);
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
