//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_calcrhs.cpp
//! \brief denominator-free puncture-conformal generalized-harmonic vacuum RHS

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "utils/finite_diff.hpp"

namespace pc_gh {

template <int FD_STENCIL>
TaskStatus PcGh::CalcRHS(Driver *, int) {
  ValidateState("pre-RHS state", false, false);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int const nmb = pmy_pack->nmb_thispack;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  Real const kappa = opt.kappa;
  bool const use_gauge_a0 = (opt.gauge == "a0");
  bool const use_z4c_mp = (opt.gauge == "z4c_mp"
                            || opt.gauge == "z4c_mp_hyperbolic");
  bool const use_hyperbolic_switch = (opt.gauge == "z4c_mp_hyperbolic");
  Real const shift_eta = opt.shift_eta;
  Real const shift_switch_z0 = opt.shift_switch_z0;
  Real const shift_switch_z1 = opt.shift_switch_z1;
  auto gauge_a0_table_ = gauge_a0_table;
  int const gauge_a0_npoints_ = gauge_a0_npoints;
  Real const gauge_a0_log_r_min_ = gauge_a0_log_r_min;
  Real const gauge_a0_inv_dlog_r_ = gauge_a0_inv_dlog_r;
  Real const gauge_mass = opt.gauge_mass;
  Real const gauge_center_x = opt.gauge_center[0];
  Real const gauge_center_y = opt.gauge_center[1];
  Real const gauge_center_z = opt.gauge_center[2];
  auto &pc = u;
  auto &pc_rhs = rhs;
  auto &state = u0;
  auto &state_rhs = u_rhs;

  par_for("regular PC-GH first-order RHS", DevExeSpace(),
  0, nmb - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1,
                   1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    Real g[3][3];
    Real gu[3][3];
    Real at[3][3];
    Real at_uu[3][3];
    Real at_ud[3][3];
    Real aa[3][3];
    Real q[3][3][3];
    Real b[3][3];
    Real gamma_u[3][3][3];
    Real gamma_d[3][3][3];
    Real gamma_contracted[3] = {0.0, 0.0, 0.0};
    Real z[3];
    Real z_d[3] = {0.0, 0.0, 0.0};
    Real p_vec[3];
    Real l_vec[3];
    Real beta[3];
    Real lambda[3];

    Real d_rho[3] = {0.0, 0.0, 0.0};
    Real d_k[3] = {0.0, 0.0, 0.0};
    Real d_cperp[3] = {0.0, 0.0, 0.0};
    Real d_z[3][3] = {};
    Real d_lambda[3][3] = {};
    Real d_at[3][3][3] = {};
    Real d_p[3][3] = {};
    Real d_l[3][3] = {};
    Real d_q[3][3][3][3] = {};
    Real d_b[3][3][3] = {};

    Real const w = pc.w(m, k, j, i);
    Real const rho = pc.rho(m, k, j, i);
    Real const w2 = w*w;
    Real const w3 = w2*w;
    Real const alpha = rho*w;
    Real const trace_k = pc.K(m, k, j, i);
    Real const c_perp = pc.Cperp(m, k, j, i);
    Real const pi = c_perp - trace_k;
    Real const zeta = rho*w3;
    Real shift_switch = 0.0;
    Real d_shift_switch_dz = 0.0;
    if (use_hyperbolic_switch) {
      if (zeta >= shift_switch_z1) {
        shift_switch = 1.0;
      } else if (zeta > shift_switch_z0) {
        Real const t = (zeta - shift_switch_z0)
                       /(shift_switch_z1 - shift_switch_z0);
        shift_switch = t*t*(3.0 - 2.0*t);
        d_shift_switch_dz = 6.0*t*(1.0 - t)
                            /(shift_switch_z1 - shift_switch_z0);
      }
    }

    Real h_perp = 0.0;
    Real d_h_perp[3] = {0.0, 0.0, 0.0};
    Real h_source[3] = {0.0, 0.0, 0.0};
    Real d_h_source[3][3] = {};
    if (use_gauge_a0) {
      Real const coord[3] = {
          CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                      size.d_view(m).x1max) - gauge_center_x,
          CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                      size.d_view(m).x2max) - gauge_center_y,
          CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                      size.d_view(m).x3max) - gauge_center_z};
      Real const radius = std::sqrt(coord[0]*coord[0] + coord[1]*coord[1]
                                    + coord[2]*coord[2]);
      Real const log_radius = std::log(radius/gauge_mass);
      GaugeA0Point const target = EvaluateGaugeA0(gauge_a0_table_,
          gauge_a0_npoints_, gauge_a0_log_r_min_, gauge_a0_inv_dlog_r_, log_radius);
      h_perp = target.h_perp/gauge_mass;
      for (int a = 0; a < 3; ++a) {
        Real const normal_a = coord[a]/radius;
        h_source[a] = target.h_radial*normal_a/gauge_mass;
        d_h_perp[a] = target.dx_h_perp*normal_a/(gauge_mass*radius);
        for (int ell = 0; ell < 3; ++ell) {
          Real const normal_ell = coord[ell]/radius;
          Real const tangent = ((a == ell) ? 1.0 : 0.0) - normal_a*normal_ell;
          d_h_source[ell][a] =
              (target.dx_h_radial*normal_a*normal_ell
               + target.h_radial*tangent)/(gauge_mass*radius);
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      beta[a] = pc.beta(m, a, k, j, i);
      z[a] = pc.Z(m, a, k, j, i);
      p_vec[a] = pc.p(m, a, k, j, i);
      l_vec[a] = pc.L(m, a, k, j, i);
      for (int c = 0; c < 3; ++c) {
        b[a][c] = state(m, BIndex(a, c), k, j, i);
      }
      for (int c = a; c < 3; ++c) {
        g[a][c] = g[c][a] = pc.gtilde(m, a, c, k, j, i);
        at[a][c] = at[c][a] = pc.Atilde(m, a, c, k, j, i);
        for (int d = 0; d < 3; ++d) {
          q[d][a][c] = q[d][c][a] = state(m, QIndex(d, a, c), k, j, i);
        }
      }
    }

    AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
    Real const det_g = adm::SpatialDet(g[0][0], g[0][1], g[0][2],
                                       g[1][1], g[1][2], g[2][2]);
    adm::SpatialInv(1.0/det_g, g[0][0], g[0][1], g[0][2],
                    g[1][1], g[1][2], g[2][2],
                    &inverse(0, 0), &inverse(0, 1), &inverse(0, 2),
                    &inverse(1, 1), &inverse(1, 2), &inverse(2, 2));
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) gu[a][c] = inverse(a, c);
    }

    for (int d = 0; d < 3; ++d) {
      bool const active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
      if (!active) continue;
      d_rho[d] = Dx<FD_STENCIL>(d, idx, pc.rho, m, k, j, i);
      d_k[d] = Dx<FD_STENCIL>(d, idx, pc.K, m, k, j, i);
      d_cperp[d] = Dx<FD_STENCIL>(d, idx, pc.Cperp, m, k, j, i);
      for (int a = 0; a < 3; ++a) {
        d_z[d][a] = Dx<FD_STENCIL>(d, idx, pc.Z, m, a, k, j, i);
        d_p[d][a] = Dx<FD_STENCIL>(d, idx, pc.p, m, a, k, j, i);
        d_l[d][a] = Dx<FD_STENCIL>(d, idx, pc.L, m, a, k, j, i);
        for (int c = 0; c < 3; ++c) {
          d_b[d][a][c] = Dx<FD_STENCIL>(
              d, idx, state, m, BIndex(a, c), k, j, i);
        }
        for (int c = a; c < 3; ++c) {
          Real const dat = Dx<FD_STENCIL>(d, idx, pc.Atilde, m, a, c, k, j, i);
          d_at[d][a][c] = d_at[d][c][a] = dat;
          for (int e = 0; e < 3; ++e) {
            Real const dq = Dx<FD_STENCIL>(
                d, idx, state, m, QIndex(e, a, c), k, j, i);
            d_q[d][e][a][c] = d_q[d][e][c][a] = dq;
          }
        }
      }
    }

    Real trace_b = 0.0;
    Real at_sq = 0.0;
    Real p_sq = 0.0;
    Real p_dot_l = 0.0;
    for (int a = 0; a < 3; ++a) {
      trace_b += b[a][a];
      for (int c = 0; c < 3; ++c) {
        p_sq += gu[a][c]*p_vec[a]*p_vec[c];
        p_dot_l += gu[a][c]*p_vec[a]*l_vec[c];
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        at_uu[a][c] = 0.0;
        at_ud[a][c] = 0.0;
        aa[a][c] = 0.0;
        for (int r = 0; r < 3; ++r) {
          at_ud[a][c] += gu[a][r]*at[r][c];
          for (int s = 0; s < 3; ++s) {
            at_uu[a][c] += gu[a][r]*gu[c][s]*at[r][s];
            aa[a][c] += at[a][r]*gu[r][s]*at[s][c];
          }
        }
        at_sq += at[a][c]*at_uu[a][c];
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        for (int e = 0; e < 3; ++e) {
          gamma_d[a][c][e] = 0.5*(q[c][a][e] + q[e][a][c] - q[a][c][e]);
        }
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        for (int e = 0; e < 3; ++e) {
          gamma_u[a][c][e] = 0.0;
          for (int r = 0; r < 3; ++r) {
            gamma_u[a][c][e] += gu[a][r]*gamma_d[r][c][e];
          }
        }
      }
      for (int c = 0; c < 3; ++c) {
        for (int e = 0; e < 3; ++e) {
          gamma_contracted[a] += gu[c][e]*gamma_u[a][c][e];
        }
      }
      lambda[a] = gamma_contracted[a] - z[a];
      for (int c = 0; c < 3; ++c) z_d[a] += g[a][c]*z[c];
    }

    // Differentiate GammaTilde^i(Q) explicitly.  This supplies both the Brown-Ricci
    // derivative and the complete STANDARD B equation without evolving LambdaTilde^i.
    for (int ell = 0; ell < 3; ++ell) {
      for (int a = 0; a < 3; ++a) {
        Real d_gamma_contracted = 0.0;
        for (int c = 0; c < 3; ++c) {
          for (int e = 0; e < 3; ++e) {
            Real d_gu_ce = 0.0;
            for (int r = 0; r < 3; ++r) {
              for (int s = 0; s < 3; ++s) {
                d_gu_ce -= gu[c][r]*gu[e][s]*q[ell][r][s];
              }
            }
            Real d_gamma_u = 0.0;
            for (int r = 0; r < 3; ++r) {
              Real d_gu_ar = 0.0;
              for (int s = 0; s < 3; ++s) {
                for (int t = 0; t < 3; ++t) {
                  d_gu_ar -= gu[a][s]*gu[r][t]*q[ell][s][t];
                }
              }
              Real const d_gamma_d = 0.5*(
                  d_q[ell][c][r][e] + d_q[ell][e][r][c]
                  - d_q[ell][r][c][e]);
              d_gamma_u += d_gu_ar*gamma_d[r][c][e] + gu[a][r]*d_gamma_d;
            }
            d_gamma_contracted += d_gu_ce*gamma_u[a][c][e]
                                  + gu[c][e]*d_gamma_u;
          }
        }
        d_lambda[ell][a] = d_gamma_contracted - d_z[ell][a];
      }
    }

    Real cov_p[3][3];
    Real cov_l[3][3];
    Real ricci[3][3];
    Real s_tensor[3][3];
    Real t_tensor[3][3];
    Real div_p = 0.0;
    Real trace_e = 0.0;
    Real ricci_scalar = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        cov_p[a][c] = d_p[a][c];
        cov_l[a][c] = d_l[a][c];
        for (int r = 0; r < 3; ++r) {
          cov_p[a][c] -= gamma_u[r][a][c]*p_vec[r];
          cov_l[a][c] -= gamma_u[r][a][c]*l_vec[r];
        }
        ricci[a][c] = 0.0;
        for (int r = 0; r < 3; ++r) {
          for (int s = 0; s < 3; ++s) {
            ricci[a][c] -= 0.5*gu[r][s]*d_q[r][s][a][c];
            Real nonlinear = 0.0;
            for (int t = 0; t < 3; ++t) {
              Real const gamma_sym = 0.5*(gamma_d[a][c][t]
                                           + gamma_d[c][a][t]);
              nonlinear += gamma_u[t][r][s]*gamma_sym
                  + gamma_u[t][r][a]*gamma_d[c][t][s]
                  + gamma_u[t][r][c]*gamma_d[a][t][s]
                  + gamma_u[t][a][r]*gamma_d[t][c][s];
            }
            ricci[a][c] += gu[r][s]*nonlinear;
          }
          ricci[a][c] += 0.5*(g[r][a]*d_lambda[c][r]
                              + g[r][c]*d_lambda[a][r]);
        }
        s_tensor[a][c] = rho*w3*ricci[a][c]
            + rho*w2*cov_p[a][c] - 0.5*w2*cov_l[a][c]
            - 0.5*w*(l_vec[a]*p_vec[c] + l_vec[c]*p_vec[a]);
        t_tensor[a][c] = -w*(z_d[a]*p_vec[c] + z_d[c]*p_vec[a]);
        for (int r = 0; r < 3; ++r) {
          t_tensor[a][c] -= 0.5*w2*z[r]*q[r][a][c];
        }
        div_p += gu[a][c]*cov_p[a][c];
        trace_e += 0.5*w2*gu[a][c]*cov_l[a][c];
        ricci_scalar += gu[a][c]*ricci[a][c];
      }
    }

    Real trace_s = 0.0;
    Real trace_t = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        trace_s += gu[a][c]*s_tensor[a][c];
        trace_t += gu[a][c]*t_tensor[a][c];
      }
    }
    Real const hamiltonian = 2.0*trace_k*trace_k/3.0 - at_sq
        + w2*ricci_scalar + 4.0*w*div_p - 6.0*p_sq;

    Real momentum[3] = {0.0, 0.0, 0.0};
    Real alpha_momentum[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 3; ++a) {
      Real divergence = 0.0;
      for (int d = 0; d < 3; ++d) {
        for (int r = 0; r < 3; ++r) {
          Real d_gu = 0.0;
          for (int s = 0; s < 3; ++s) {
            for (int t = 0; t < 3; ++t) {
              d_gu -= gu[d][s]*gu[r][t]*q[d][s][t];
            }
          }
          divergence += d_gu*at[r][a] + gu[d][r]*d_at[d][r][a];
          divergence += gamma_u[d][d][r]*at_ud[r][a];
          divergence -= gamma_u[r][d][a]*at_ud[d][r];
        }
      }
      momentum[a] = divergence - 2.0*d_k[a]/3.0;
      alpha_momentum[a] = rho*w*momentum[a];
      for (int d = 0; d < 3; ++d) {
        alpha_momentum[a] -= 3.0*rho*at_ud[d][a]*p_vec[d];
      }
    }

    Real adv_w = 0.0;
    Real adv_rho = 0.0;
    Real adv_k = 0.0;
    Real adv_cperp = 0.0;
    Real p_dot_z = 0.0;
    Real lapse_gradient_dot_z = 0.0;
    for (int d = 0; d < 3; ++d) {
      adv_w += beta[d]*p_vec[d];
      adv_rho += beta[d]*d_rho[d];
      adv_k += beta[d]*d_k[d];
      adv_cperp += beta[d]*d_cperp[d];
      p_dot_z += p_vec[d]*z[d];
      lapse_gradient_dot_z += (rho*p_vec[d] + 0.5*l_vec[d])*z[d];
    }
    pc_rhs.w(m, k, j, i) = adv_w + w*(alpha*trace_k - trace_b)/3.0;
    Real const lapse_driver = alpha*pi - h_perp;
    pc_rhs.rho(m, k, j, i) = use_z4c_mp
        ? adv_rho + rho*(-2.0*trace_k - (alpha*trace_k - trace_b)/3.0)
        : adv_rho + rho*(lapse_driver - (alpha*trace_k - trace_b)/3.0);
    pc_rhs.K(m, k, j, i) = adv_k
        + alpha*at_sq + alpha*trace_k*trace_k/3.0
        - trace_e + 0.5*w*p_dot_l
        + alpha*(hamiltonian - trace_k*c_perp)
        + alpha*w*p_dot_z - 1.5*kappa*alpha*c_perp;
    pc_rhs.Cperp(m, k, j, i) = adv_cperp
        + alpha*(hamiltonian - trace_k*c_perp)
        + w2*lapse_gradient_dot_z - 2.0*kappa*alpha*c_perp;

    for (int a = 0; a < 3; ++a) {
      Real adv_beta = 0.0;
      Real adv_z = 0.0;
      Real metric_source = 0.0;
      for (int d = 0; d < 3; ++d) {
        adv_beta += beta[d]*b[d][a];
        adv_z += beta[d]*d_z[d][a];
        metric_source += zeta*gu[a][d]*(rho*p_vec[d] - 0.5*l_vec[d]);
      }
      Real const shift_source = use_z4c_mp
          ? lambda[a] - shift_eta*beta[a] + shift_switch*metric_source
          : h_source[a] + rho*rho*w2*w2*lambda[a] + metric_source;
      pc_rhs.beta(m, a, k, j, i) = adv_beta + shift_source;

      Real z_source = 0.0;
      for (int c = 0; c < 3; ++c) {
        z_source -= 2.0*gu[a][c]*alpha_momentum[c];
        z_source -= rho*w*gu[a][c]*d_cperp[c];
        z_source += 0.5*c_perp*gu[a][c]*l_vec[c];
        z_source -= z[c]*b[c][a];
      }
      z_source += 2.0*z[a]*trace_b/3.0
          - (2.0*rho*w*trace_k/3.0 + kappa*rho*w)*z[a];
      pc_rhs.Z(m, a, k, j, i) = adv_z + z_source;
    }

    for (int a = 0; a < 3; ++a) {
      for (int c = a; c < 3; ++c) {
        Real adv_g = 0.0;
        Real adv_at = 0.0;
        for (int d = 0; d < 3; ++d) {
          adv_g += beta[d]*q[d][a][c];
          adv_at += beta[d]*d_at[d][a][c];
        }
        Real shift_g = 0.0;
        Real shift_at = 0.0;
        for (int d = 0; d < 3; ++d) {
          shift_g += g[d][a]*b[c][d] + g[d][c]*b[a][d];
          shift_at += at[d][a]*b[c][d] + at[d][c]*b[a][d];
        }
        pc_rhs.gtilde(m, a, c, k, j, i) = adv_g - 2.0*alpha*at[a][c]
            + shift_g - 2.0*g[a][c]*trace_b/3.0;

        Real const s_tf = s_tensor[a][c] - g[a][c]*trace_s/3.0;
        Real const t_tf = t_tensor[a][c] - g[a][c]*trace_t/3.0;
        pc_rhs.Atilde(m, a, c, k, j, i) = adv_at + s_tf + shift_at
            - 2.0*at[a][c]*trace_b/3.0 - 2.0*alpha*aa[a][c]
            + alpha*trace_k*at[a][c] - alpha*c_perp*at[a][c]
            + alpha*t_tf;
      }
    }

    Real d_f_w[3] = {0.0, 0.0, 0.0};
    Real d_f_beta[3][3] = {};
    Real d_f_g[3][3][3] = {};
    for (int ell = 0; ell < 3; ++ell) {
      Real d_trace_b = 0.0;
      for (int r = 0; r < 3; ++r) d_trace_b += d_b[ell][r][r];
      d_f_w[ell] = (p_vec[ell]*(alpha*trace_k - trace_b)
          + w*(0.5*l_vec[ell]*trace_k + alpha*d_k[ell] - d_trace_b))/3.0;

      Real const d_zeta = w2*(0.5*l_vec[ell] + 2.0*rho*p_vec[ell]);
      Real const d_coefficient = rho*w3*(l_vec[ell] + 2.0*rho*p_vec[ell]);
      for (int a = 0; a < 3; ++a) {
        Real metric_source = 0.0;
        Real d_metric_source = 0.0;
        for (int c = 0; c < 3; ++c) {
          Real d_gu_ac = 0.0;
          for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
              d_gu_ac -= gu[a][r]*gu[c][s]*q[ell][r][s];
            }
          }
          Real const v_c = rho*p_vec[c] - 0.5*l_vec[c];
          Real const regular_dv =
              rho*w2*(0.5*l_vec[ell] - rho*p_vec[ell])*p_vec[c]
              + rho*rho*w3*d_p[ell][c] - 0.5*rho*w3*d_l[ell][c];
          metric_source += zeta*gu[a][c]*v_c;
          d_metric_source += d_zeta*gu[a][c]*v_c
              + zeta*d_gu_ac*v_c + gu[a][c]*regular_dv;
        }
        d_f_beta[ell][a] = use_z4c_mp
            ? d_lambda[ell][a] - shift_eta*b[ell][a]
                + shift_switch*d_metric_source
                + d_shift_switch_dz*d_zeta*metric_source
            : d_h_source[ell][a] + d_coefficient*lambda[a]
                + rho*rho*w2*w2*d_lambda[ell][a] + d_metric_source;
      }

      for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) {
          d_f_g[ell][a][c] = -l_vec[ell]*at[a][c]
                              - 2.0*alpha*d_at[ell][a][c];
          for (int r = 0; r < 3; ++r) {
            d_f_g[ell][a][c] += q[ell][r][a]*b[c][r]
                + g[r][a]*d_b[ell][c][r]
                + q[ell][r][c]*b[a][r]
                + g[r][c]*d_b[ell][a][r];
          }
          d_f_g[ell][a][c] -= 2.0*(q[ell][a][c]*trace_b
                                    + g[a][c]*d_trace_b)/3.0;
        }
      }
    }

    for (int ell = 0; ell < 3; ++ell) {
      Real rhs_p = d_f_w[ell];
      Real rhs_l = 0.0;
      if (use_z4c_mp) {
        rhs_l = -2.0*trace_k*l_vec[ell] - 4.0*alpha*d_k[ell];
      } else {
        rhs_l = l_vec[ell]*(2.0*alpha*pi - h_perp)
            + 2.0*alpha*alpha*(d_cperp[ell] - d_k[ell])
            - 2.0*alpha*d_h_perp[ell];
      }
      for (int d = 0; d < 3; ++d) {
        rhs_p += beta[d]*d_p[d][ell] + b[ell][d]*p_vec[d];
        rhs_l += beta[d]*d_l[d][ell] + b[ell][d]*l_vec[d];
      }
      pc_rhs.p(m, ell, k, j, i) = rhs_p;
      pc_rhs.L(m, ell, k, j, i) = rhs_l;

      for (int a = 0; a < 3; ++a) {
        Real rhs_b = d_f_beta[ell][a];
        for (int d = 0; d < 3; ++d) {
          rhs_b += beta[d]*d_b[d][ell][a] + b[ell][d]*b[d][a];
        }
        state_rhs(m, BIndex(ell, a), k, j, i) = rhs_b;
        for (int c = a; c < 3; ++c) {
          Real rhs_q = d_f_g[ell][a][c];
          for (int d = 0; d < 3; ++d) {
            rhs_q += beta[d]*d_q[d][ell][a][c] + b[ell][d]*q[d][a][c];
          }
          state_rhs(m, QIndex(ell, a, c), k, j, i) = rhs_q;
        }
      }
    }
  });

  Real const dissipation = opt.dissipation;
  if (dissipation != 0.0) {
    par_for("PC-GH Kreiss-Oliger dissipation", DevExeSpace(),
    0, nmb - 1, 0, npcgh - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(int m, int n, int k, int j, int i) {
      Real idx[3] = {1.0/size.d_view(m).dx1,
                     1.0/size.d_view(m).dx2,
                     1.0/size.d_view(m).dx3};
      state_rhs(m, n, k, j, i) += dissipation*Diss<FD_STENCIL>(
          0, idx, state, m, n, k, j, i);
      if (multi_d) {
        state_rhs(m, n, k, j, i) += dissipation*Diss<FD_STENCIL>(
            1, idx, state, m, n, k, j, i);
      }
      if (three_d) {
        state_rhs(m, n, k, j, i) += dissipation*Diss<FD_STENCIL>(
            2, idx, state, m, n, k, j, i);
      }
    });
  }
  Kokkos::fence();
  ValidateState("post-RHS state and RHS", true, false);
  return TaskStatus::complete;
}

template TaskStatus PcGh::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcRHS<4>(Driver *pdriver, int stage);

}  // namespace pc_gh
