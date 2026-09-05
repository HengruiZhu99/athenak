// Measure the actual production principal matrix using affine local point jets.
// No evolution, projection, or fitted finite-wavelength dispersion is involved.
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"
#include "pulse.hpp"

namespace {
using PC = pc_gh::PcGh;
using State = Kokkos::Array<Real, PC::npcgh>;

void Fill(Mesh *pm, State seed, int variable, Real amplitude,
          Kokkos::Array<Real, 3> direction) {
  auto state = pm->pmb_pack->ppcgh->u0;
  auto size = pm->pmb_pack->pmb->mb_size;
  auto ind = pm->mb_indcs;
  int const ni = state.extent_int(4), nj = state.extent_int(3), nk = state.extent_int(2);
  par_for("regular extension affine probe", DevExeSpace(),
  0, pm->pmb_pack->nmb_thispack - 1, 0, nk - 1, 0, nj - 1, 0, ni - 1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const offset[3] = {
        (i-ind.is-ind.nx1/2)*size.d_view(m).dx1,
        (j-ind.js-ind.nx2/2)*size.d_view(m).dx2,
        (k-ind.ks-ind.nx3/2)*size.d_view(m).dx3};
    Real jet = 0.0;
    for (int d = 0; d < 3; ++d) jet += direction[d]*offset[d];
    for (int n = 0; n < PC::npcgh; ++n) {
      state(m, n, k, j, i) = seed[n] + (n == variable ? amplitude*jet : 0.0);
    }
  });
  Kokkos::fence();
}

State Evaluate(Mesh *pm, State seed, int variable, Real amplitude,
               Kokkos::Array<Real, 3> direction) {
  Fill(pm, seed, variable, amplitude, direction);
  auto *pc = pm->pmb_pack->ppcgh;
  (void)pc->CalcRHS<2>(nullptr, 0);
  auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u_rhs);
  auto ind = pm->mb_indcs;
  State result{};
  for (int n = 0; n < PC::npcgh; ++n) {
    result[n] = host(0, n, ind.ks+ind.nx3/2, ind.js+ind.nx2/2, ind.is+ind.nx1/2);
  }
  return result;
}

void Audit(ParameterInput *, Mesh *pm) {
  auto *pc = pm->pmb_pack->ppcgh;
  std::ofstream matrix("principal.csv"), states("states.csv");
  matrix << "case,row,col,value\n" << std::setprecision(17);
  states << "case,var,value\n" << std::setprecision(17);
  std::ofstream description("cases.csv");
  description << "case,w,rho,nx,ny,nz,rate,curved\n" << std::setprecision(17);
  for (int c = 0; c < 12; ++c) {
    State seed{};
    seed[PC::I_W] = c == 0 ? 1.0 : 0.5;
    seed[PC::I_RHO] = c == 0 ? 1.0 : 1.5;
    seed[PC::I_GTXX] = seed[PC::I_GTYY] = seed[PC::I_GTZZ] = 1.0;
    seed[PC::I_BETAX] = 0.25;
    seed[PC::I_BETAY] = -0.375;
    seed[PC::I_BETAZ] = 0.125;
    bool const curved = c >= 4 && c < 8;
    if (curved) {
      seed[PC::I_GTXX] = 1.0625;
      seed[PC::I_GTXY] = 0.25;
    }
    if (c >= 8) {
      seed[PC::I_K] = 0.025;
      seed[PC::I_CPERP] = -0.03;
      Real const at[6] = {0.007, 0.014, 0.021, -0.014, 0.007, 0.007};
      for (int n = 0; n < 6; ++n) seed[PC::I_ATXX+n] = at[n];
      for (int d = 0; d < 3; ++d) {
        seed[PC::I_ZX+d] = 0.003*(d-1);
        seed[PC::I_P1+d] = 0.004*(d+1);
        seed[PC::I_L1+d] = 0.005*(d-1);
        Real const q[6] = {0.011*(d+1), 0.022, -0.011,
                          -0.011*(d+2), 0.033, 0.011};
        for (int n = 0; n < 6; ++n) seed[PC::I_Q1XX+6*d+n] = q[n];
        for (int a = 0; a < 3; ++a) seed[PC::BIndex(d, a)] = 0.002*(d+a-2);
      }
    }
    Kokkos::Array<Real, 3> direction{};
    if (c % 4 < 3) direction[c % 4] = 1.0;
    else direction = {0.25, -0.5, 0.75};
    description << c << ',' << seed[PC::I_W] << ',' << seed[PC::I_RHO];
    for (int d = 0; d < 3; ++d) description << ',' << direction[d];
    description << ',' << pc->opt.reduction_rate << ',' << curved << '\n';
    for (int n = 0; n < PC::npcgh; ++n) states << c << ',' << n << ',' << seed[n] << '\n';
    for (int col = 0; col < PC::npcgh; ++col) {
      Real const epsilon = 0.125;
      auto plus = Evaluate(pm, seed, col, epsilon, direction);
      auto minus = Evaluate(pm, seed, col, -epsilon, direction);
      for (int row = 0; row < PC::npcgh; ++row) {
        Real const measured = (plus[row]-minus[row])/(2.0*epsilon);
        matrix << c << ',' << row << ',' << col << ',' << measured << '\n';
      }
    }
  }
  std::cout << "PASS: recorded twelve full 55-field affine-jet principal matrices\n";
  // Linearize the source at an exact shifted Minkowski equilibrium. Disabling
  // eta here keeps its constant shift stationary; kappa remains independently set.
  Real const saved_eta = pc->opt.shift_eta;
  pc->opt.shift_eta = 0.0;
  State flat{};
  flat[PC::I_W] = flat[PC::I_RHO] = 1.0;
  flat[PC::I_GTXX] = flat[PC::I_GTYY] = flat[PC::I_GTZZ] = 1.0;
  flat[PC::I_BETAX] = 0.25;
  flat[PC::I_BETAY] = -0.375;
  flat[PC::I_BETAZ] = 0.125;
  std::ofstream source("source-jacobian.csv");
  source << "row,col,value\n" << std::setprecision(17);
  for (int col = 0; col < PC::npcgh; ++col) {
    State plus = flat, minus = flat;
    plus[col] += 0.125;
    minus[col] -= 0.125;
    auto forward = Evaluate(pm, plus, -1, 0.0, {0.0, 0.0, 0.0});
    auto backward = Evaluate(pm, minus, -1, 0.0, {0.0, 0.0, 0.0});
    for (int row = 0; row < PC::npcgh; ++row) {
      source << row << ',' << col << ',' << (forward[row]-backward[row])/0.25 << '\n';
    }
  }
  pc->opt.shift_eta = saved_eta;
  std::cout << "PASS: recorded independent homogeneous source Jacobian\n";

  // An independent constraint-satisfying polynomial jet. w, rho and beta are
  // affine, alpha=rho*w is quadratic, and L=2*d(alpha). At/Q are zero; K, C and
  // Z need not satisfy the Einstein constraints. The new extension must leave
  // every RHS row unchanged even for this off-GH-constraint state.
  auto state = pc->u0;
  auto size = pm->pmb_pack->pmb->mb_size;
  auto ind = pm->mb_indcs;
  int const ni = state.extent_int(4), nj = state.extent_int(3), nk = state.extent_int(2);
  par_for("regular extension reduction-manifold oracle", DevExeSpace(),
  0, 0, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x[3] = {(i-ind.is-ind.nx1/2)*size.d_view(m).dx1,
                       (j-ind.js-ind.nx2/2)*size.d_view(m).dx2,
                       (k-ind.ks-ind.nx3/2)*size.d_view(m).dx3};
    Real const dw[3] = {0.03125, -0.0625, 0.015625};
    Real const drho[3] = {-0.125, 0.0625, 0.03125};
    Real w = 0.5, rho = 1.5;
    for (int d = 0; d < 3; ++d) { w += dw[d]*x[d]; rho += drho[d]*x[d]; }
    for (int n = 0; n < PC::npcgh; ++n) state(m, n, k, j, i) = 0.0;
    state(m, PC::I_W, k, j, i) = w;
    state(m, PC::I_RHO, k, j, i) = rho;
    state(m, PC::I_GTXX, k, j, i) = 1.0;
    state(m, PC::I_GTYY, k, j, i) = 1.0;
    state(m, PC::I_GTZZ, k, j, i) = 1.0;
    state(m, PC::I_K, k, j, i) = 0.0625;
    state(m, PC::I_CPERP, k, j, i) = -0.03125;
    for (int a = 0; a < 3; ++a) {
      state(m, PC::I_ZX+a, k, j, i) = (a-1)*0.015625;
      state(m, PC::I_P1+a, k, j, i) = dw[a];
      state(m, PC::I_L1+a, k, j, i) = 2.0*(rho*dw[a]+w*drho[a]);
      Real beta = (a+1)*0.125;
      for (int d = 0; d < 3; ++d) {
        Real const b = (d+a-1)*0.03125;
        beta += b*x[d];
        state(m, PC::BIndex(d, a), k, j, i) = b;
      }
      state(m, PC::I_BETAX+a, k, j, i) = beta;
    }
  });
  std::string const saved_system = pc->opt.reduction_system;
  pc->opt.reduction_system = "legacy";
  (void)pc->CalcRHS<2>(nullptr, 0);
  auto legacy = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u_rhs);
  pc->opt.reduction_system = "advective";
  (void)pc->CalcRHS<2>(nullptr, 0);
  auto candidate = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u_rhs);
  pc->opt.reduction_system = saved_system;
  Real max_difference = 0.0;
  for (int n = 0; n < PC::npcgh; ++n) {
    max_difference = std::fmax(max_difference, std::fabs(
        candidate(0, n, ind.ks+ind.nx3/2, ind.js+ind.nx2/2, ind.is+ind.nx1/2)
        -legacy(0, n, ind.ks+ind.nx3/2, ind.js+ind.nx2/2, ind.is+ind.nx1/2)));
  }
  if (max_difference > 1.0e-13) {
    std::cerr << "FAIL: reduction-manifold RHS changed by " << max_difference << '\n';
    std::exit(EXIT_FAILURE);
  }
  std::cout << "PASS: all 55 production RHS rows unchanged on independent reduction "
               "manifold jet; maximum difference=" << max_difference << '\n';
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, bool restart) {
  std::cout << "Research qualification execution backend: " << DevExeSpace::name() << '\n';
  if (pin->GetOrAddBoolean("problem", "require_cuda", false)
      && std::string(DevExeSpace::name()) != "Cuda") {
    std::cerr << "This qualification input requires the CUDA backend\n";
    std::exit(EXIT_FAILURE);
  }
  std::string const name = pin->GetString("problem", "pgen_name");
  if (name == "regular_extension_pulse") {
    pgen_final_func = regular_pulse::Final;
    return regular_pulse::Initialize(pin, pmy_mesh_, restart);
  }
  if (name == "pc_gh_robust_minkowski") return PcGhRobustMinkowski(pin, restart);
  if (name == "pc_gh_minkowski") return PcGhMinkowski(pin, restart);
  if (name == "pc_gh_gauge_wave") return PcGhGaugeWave(pin, restart);
  if (name == "pc_gh_one_puncture") return PcGhOnePuncture(pin, restart);
  auto *pm = pmy_mesh_;
  if (restart || pm->pmb_pack->ppcgh == nullptr || pm->multilevel
      || global_variable::nranks != 1 || pm->pmb_pack->nmb_thispack != 1
      || pm->pmb_pack->ppcgh->opt.fd_stencil != 2
      || pin->GetInteger("time", "nlim") != 0
      || pm->pmb_pack->ppcgh->opt.dissipation != 0.0
      || pm->pmb_pack->ppcgh->opt.project_reduction_constraints) {
    std::cerr << "The affine oracle requires one uniform block, one rank, order 2, "
                 "zero steps, no KO, and no reduction projection.\n";
    std::exit(EXIT_FAILURE);
  }
  pgen_final_func = Audit;
  State seed{};
  seed[PC::I_W] = seed[PC::I_RHO] = 1.0;
  seed[PC::I_GTXX] = seed[PC::I_GTYY] = seed[PC::I_GTZZ] = 1.0;
  Fill(pm, seed, -1, 0.0, {0.0, 0.0, 0.0});
}
