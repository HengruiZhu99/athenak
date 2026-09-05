// Independent compact reduction pulses and component dumps for CUDA qualification.
// Included only by the research problem generator, not by production builds.
#ifndef PC_GH_REGULAR_EXTENSION_PULSE_HPP_
#define PC_GH_REGULAR_EXTENSION_PULSE_HPP_
#include "utils/finite_diff.hpp"

namespace regular_pulse {
using PC = pc_gh::PcGh;

KOKKOS_INLINE_FUNCTION
Real Bump(Real radius_squared, Real width) {
  Real const s2 = radius_squared/(width*width);
  return s2 < 1.0 ? std::exp(1.0-1.0/(1.0-s2)) : 0.0;
}

void Dump(ParameterInput *pin, Mesh *pm, std::string phase) {
  auto *pc = pm->pmb_pack->ppcgh;
  switch (pc->opt.fd_stencil) {
    case 2: (void)pc->CalcConstraints<2>(nullptr, 0); break;
    case 3: (void)pc->CalcConstraints<3>(nullptr, 0); break;
    case 4: (void)pc->CalcConstraints<4>(nullptr, 0); break;
  }
  auto state = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u0);
  auto con = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pc->u_con);
  auto ind = pm->mb_indcs;
  std::ofstream file("pulse-"+phase+"-rank"+std::to_string(global_variable::my_rank)+".csv");
  if (!file) { std::cerr << "Unable to write pulse component dump\n"; std::exit(1); }
  file << "time,block,level,x,y,z,volume";
  for (int v = 0; v < PC::npcgh; ++v) file << ",u" << v;
  for (int v = 0; v < 33; ++v) file << ",E" << v;
  for (int v = 0; v < 8; ++v) file << ",norm" << v;
  file << '\n' << std::setprecision(17);
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    auto size = pm->pmb_pack->pmb->mb_size.h_view(m);
    Real const inverse[3] = {1.0/size.dx1, 1.0/size.dx2, 1.0/size.dx3};
    Real const volume = size.dx1*(pm->multi_d ? size.dx2 : 1.0)
                                *(pm->three_d ? size.dx3 : 1.0);
    for (int k = ind.ks; k <= ind.ke; ++k) {
      for (int j = ind.js; j <= ind.je; ++j) {
        for (int i = ind.is; i <= ind.ie; ++i) {
          auto derivative = [&](int d, int v) -> Real {
            if ((d == 1 && !pm->multi_d) || (d == 2 && !pm->three_d)) return 0.0;
            switch (pc->opt.fd_stencil) {
              case 2: return Dx<2>(d, inverse, state, m, v, k, j, i);
              case 3: return Dx<3>(d, inverse, state, m, v, k, j, i);
              case 4: return Dx<4>(d, inverse, state, m, v, k, j, i);
            }
            std::abort();
          };
          Real error[33] = {};
          for (int d = 0; d < 3; ++d) {
            error[d] = state(m, PC::I_P1+d, k, j, i)-derivative(d, PC::I_W);
            for (int t = 0; t < 6; ++t) {
              error[3+6*d+t] = state(m, PC::I_Q1XX+6*d+t, k, j, i)
                              -derivative(d, PC::I_GTXX+t);
            }
            // True ell=L-2*d(alpha), evaluated with the continuum product rule
            // on the two configuration factors, as in the production source.
            error[21+d] = state(m, PC::I_L1+d, k, j, i)-2.0*(
                state(m, PC::I_RHO, k, j, i)*derivative(d, PC::I_W)
                +state(m, PC::I_W, k, j, i)*derivative(d, PC::I_RHO));
            for (int a = 0; a < 3; ++a) {
              error[24+3*d+a] = state(m, PC::BIndex(d, a), k, j, i)
                               -derivative(d, PC::I_BETAX+a);
            }
          }
          file << pm->time << ',' << pm->pmb_pack->pmb->mb_gid.h_view(m) << ','
               << pm->pmb_pack->pmb->mb_lev.h_view(m)-pm->root_level << ','
               << CellCenterX(i-ind.is, ind.nx1, size.x1min, size.x1max) << ','
               << CellCenterX(j-ind.js, ind.nx2, size.x2min, size.x2max) << ','
               << CellCenterX(k-ind.ks, ind.nx3, size.x3min, size.x3max) << ',' << volume;
          for (int v = 0; v < PC::npcgh; ++v) file << ',' << state(m, v, k, j, i);
          for (Real value : error) file << ',' << value;
          for (int v = 0; v < 8; ++v) file << ',' << con(m, PC::I_CON_RED_W+v, k, j, i);
          file << '\n';
        }
      }
    }
  }
}

void Final(ParameterInput *pin, Mesh *pm) { Dump(pin, pm, "final"); }

void Initialize(ParameterInput *pin, Mesh *pm, bool restart) {
  if (restart) return;
  auto *pc = pm->pmb_pack->ppcgh;
  if (pc == nullptr || pc->opt.gauge != "z4c_mp_hyperbolic"
      || pc->opt.shift_eta != 0.0 || pc->opt.project_reduction_constraints) {
    std::cerr << "Compact pulses require switched moving gauge, eta=0, no projection\n";
    std::exit(EXIT_FAILURE);
  }
  std::string const family = pin->GetString("problem", "pulse_family");
  int const which = family == "p" ? 0 : family == "Q" ? 1
                  : family == "L" ? 2 : family == "B" ? 3 : -1;
  int const direction = pin->GetOrAddInteger("problem", "pulse_direction", 1);
  Real const amplitude = pin->GetOrAddReal("problem", "pulse_amplitude", 1.0e-8);
  Real const width = pin->GetOrAddReal("problem", "pulse_width", 0.75);
  Real const center = pin->GetOrAddReal("problem", "pulse_center_x", 0.0);
  Real const shift = pin->GetOrAddReal("problem", "pulse_shift", 0.5);
  bool const radial = pin->GetOrAddBoolean("problem", "pulse_radial", false);
  if (which < 0 || direction < 0 || direction > 2 || !(width > 0.0)
      || !(amplitude > 0.0 && amplitude < 1.0e-2)) {
    std::cerr << "Invalid compact pulse family/direction/width/amplitude\n";
    std::exit(EXIT_FAILURE);
  }
  auto state = pc->u0;
  auto size = pm->pmb_pack->pmb->mb_size;
  auto ind = pm->mb_indcs;
  int const ni = state.extent_int(4), nj = state.extent_int(3), nk = state.extent_int(2);
  bool const multi = pm->multi_d, three = pm->three_d;
  par_for("regular extension independent compact pulse", DevExeSpace(),
  0, pm->pmb_pack->nmb_thispack-1, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real const x = CellCenterX(i-ind.is, ind.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max)-center;
    Real const y = CellCenterX(j-ind.js, ind.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max);
    Real const z = CellCenterX(k-ind.ks, ind.nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max);
    Real const radius2 = x*x+(radial && multi ? y*y : 0.0)
                            +(radial && three ? z*z : 0.0);
    Real const pulse = amplitude*Bump(radius2, width);
    for (int v = 0; v < PC::npcgh; ++v) state(m, v, k, j, i) = 0.0;
    state(m, PC::I_W, k, j, i) = state(m, PC::I_RHO, k, j, i) = 1.0;
    state(m, PC::I_GTXX, k, j, i) = state(m, PC::I_GTYY, k, j, i)
                                = state(m, PC::I_GTZZ, k, j, i) = 1.0;
    state(m, PC::I_BETAX, k, j, i) = shift;
    if (which == 0) state(m, PC::I_P1+direction, k, j, i) = pulse;
    if (which == 1) {
      state(m, PC::QIndex(direction, 0, 0), k, j, i) = pulse;
      state(m, PC::QIndex(direction, 2, 2), k, j, i) = -pulse;
    }
    if (which == 2) state(m, PC::I_L1+direction, k, j, i) = pulse;
    if (which == 3) state(m, PC::BIndex(direction, 2), k, j, i) = pulse;
  });
  // CalcConstraints also reads the RHS for diagnostic norms.
  Kokkos::deep_copy(pc->u_rhs, 0.0);
  pc->PcGhToADM(pm->pmb_pack);
  Dump(pin, pm, "initial");
}
}  // namespace regular_pulse
#endif
