// Independent, zero-step audit of the actual PC-GH configuration RHS.
// Build with -DPROBLEM=../../analysis/pc_gh_gamma2/production_oracle.
// This is a diagnostic pgen; it does not change any production equation.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pgen/pgen.hpp"

namespace {
using PC = pc_gh::PcGh;

void Fill(Mesh *pm, Real w, Real shift, Real lapse_gradient, Real rho_gradient) {
  auto state = pm->pmb_pack->ppcgh->u0;
  auto size = pm->pmb_pack->pmb->mb_size;
  auto ind = pm->mb_indcs;
  int const nj = state.extent_int(3);
  int const nk = state.extent_int(2);
  int const ni = state.extent_int(4);
  par_for("FO-GH map audit state", DevExeSpace(),
  0, pm->pmb_pack->nmb_thispack - 1, 0, nk - 1, 0, nj - 1, 0, ni - 1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    for (int n = 0; n < PC::npcgh; ++n) state(m, n, k, j, i) = 0.0;
    Real const x = CellCenterX(i - ind.is, ind.nx1,
        size.d_view(m).x1min, size.d_view(m).x1max);
    state(m, PC::I_W, k, j, i) = w;
    state(m, PC::I_RHO, k, j, i) = 1.0 + rho_gradient*x;
    state(m, PC::I_GTXX, k, j, i) = 1.0;
    state(m, PC::I_GTYY, k, j, i) = 1.0;
    state(m, PC::I_GTZZ, k, j, i) = 1.0;
    state(m, PC::I_BETAX, k, j, i) = shift;
    state(m, PC::I_L1, k, j, i) = lapse_gradient;
  });
  Kokkos::fence();
}

void Audit(ParameterInput *, Mesh *pm) {
  auto *pc = pm->pmb_pack->ppcgh;
  std::ofstream out("configuration-map.csv");
  out << "case,w,shift,Lx,drho,production_rho_rhs,fo_gh_rho_rhs,residual\n"
      << std::setprecision(17);
  Real const shift = 0.25;
  Real const amplitude = 0.125;
  for (int c = 0; c < 6; ++c) {
    Real const w = c < 4 ? std::pow(0.5, c) : 1.0;
    Real const gradient = c >= 4 ? amplitude : 0.0;
    Real const lx = c == 4 ? 0.0 : (c == 5 ? 2.0*w*gradient : amplitude);
    Fill(pm, w, shift, lx, gradient);
    (void)pc->CalcRHS<2>(nullptr, 0);
    (void)pc->CalcConstraints<2>(nullptr, 0);
    auto rhs = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u_rhs);
    auto ind = pm->mb_indcs;
    Real const expected_production = shift*gradient;
    // Equation (35), gamma1=-1: alpha_t=beta^i L_i/2 and w_t=0
    // for this state; rho_t=(alpha_t-rho*w_t)/w.
    Real const expected_fo = shift*lx/(2.0*w);
    for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
      for (int k = ind.ks; k <= ind.ke; ++k) {
        for (int j = ind.js; j <= ind.je; ++j) {
          for (int i = ind.is; i <= ind.ie; ++i) {
            Real const measured = rhs(m, PC::I_RHO, k, j, i);
            if (!std::isfinite(measured)
                || std::abs(measured - expected_production) > 1.e-13) {
              std::cerr << "FAIL: unexpected production rho row\n";
              std::exit(EXIT_FAILURE);
            }
          }
        }
      }
    }
    Real const measured = rhs(0, PC::I_RHO, ind.ks, ind.js, ind.is);
    out << c << ',' << w << ',' << shift << ',' << lx << ',' << gradient
        << ',' << measured << ',' << expected_fo << ',' << measured - expected_fo
        << '\n';
    if ((c < 5 && measured == expected_fo)
        || (c == 5 && std::abs(measured - expected_fo) > 1.e-13)) {
      std::cerr << "FAIL: off-constraint discrepancy/control not reproduced\n";
      std::exit(EXIT_FAILURE);
    }
  }
  std::cout << "PASS: five production/FO-GH off-constraint counterexamples; "
               "one constraint-satisfying control\n";
}
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, bool restart) {
  auto *pm = pmy_mesh_;
  if (restart || pm->pmb_pack->ppcgh == nullptr || pm->multilevel
      || global_variable::nranks != 1 || pm->pmb_pack->ppcgh->opt.fd_stencil != 2
      || pm->pmb_pack->ppcgh->opt.gauge != "harmonic"
      || pin->GetInteger("time", "nlim") != 0
      || pm->pmb_pack->ppcgh->opt.dissipation != 0.0
      || pm->pmb_pack->ppcgh->opt.project_reduction_constraints) {
    std::cerr << "This oracle requires a serial, uniform, zero-step harmonic "
                 "PC-GH run with order 2, no KO, and no reduction projection.\n";
    std::exit(EXIT_FAILURE);
  }
  pgen_final_func = Audit;
  Fill(pm, 1.0, 0.0, 0.0, 0.0);
}
