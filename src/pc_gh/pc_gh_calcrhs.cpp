//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_calcrhs.cpp
//! \brief puncture-conformal first-order generalized-harmonic vacuum RHS

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
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int const nmb = pmy_pack->nmb_thispack;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  Real const kappa = opt.kappa;
  bool const use_gauge_a0 = (opt.gauge == "a0");
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

  par_for("PC-GH first-order RHS", DevExeSpace(),
  0, nmb - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1,
                   1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    Real g[3][3];
    Real gu[3][3];
    Real at[3][3];
    Real at_uu[3][3];
    Real aa[3][3];
    Real q[3][3][3];
    Real b[3][3];
    Real gamma_u[3][3][3];
    Real gamma_d[3][3][3];
    Real gamma_contracted[3] = {0.0, 0.0, 0.0};
    Real z[3];
    Real z_d[3] = {0.0, 0.0, 0.0};
    Real x[3];
    Real y[3];
    Real w[3];
    Real l[3];
    Real beta[3];
    Real lambda[3];

    Real d_k[3] = {0.0, 0.0, 0.0};
    Real d_pi[3] = {0.0, 0.0, 0.0};
    Real d_lambda[3][3] = {};
    Real d_at[3][3][3] = {};
    Real d_x[3][3] = {};
    Real d_y[3][3] = {};
    Real d_q[3][3][3][3] = {};
    Real d_b[3][3][3] = {};

    Real const chi = pc.chi(m, k, j, i);
    Real const lapse_sq = pc.A(m, k, j, i);
    Real const alpha = std::sqrt(lapse_sq);
    Real const sqrt_chi = std::sqrt(chi);
    Real const r_minus = chi/alpha;
    Real const r_plus = alpha/sqrt_chi;
    Real const trace_k = pc.K(m, k, j, i);
    Real const pi = pc.pi(m, k, j, i);
    Real const c_perp = pi + trace_k;
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
      Real dx_h_perp;
      Real h_radial;
      Real dx_h_radial;
      InterpolateGaugeA0(gauge_a0_table_, gauge_a0_npoints_,
          gauge_a0_log_r_min_, gauge_a0_inv_dlog_r_, I_A0_H_PERP,
          log_radius, h_perp, dx_h_perp);
      InterpolateGaugeA0(gauge_a0_table_, gauge_a0_npoints_,
          gauge_a0_log_r_min_, gauge_a0_inv_dlog_r_, I_A0_H_RADIAL,
          log_radius, h_radial, dx_h_radial);
      h_perp /= gauge_mass;
      for (int a = 0; a < 3; ++a) {
        Real const normal_a = coord[a]/radius;
        h_source[a] = h_radial*normal_a/gauge_mass;
        d_h_perp[a] = dx_h_perp*normal_a/(gauge_mass*radius);
        for (int ell = 0; ell < 3; ++ell) {
          Real const normal_ell = coord[ell]/radius;
          Real const tangent = ((a == ell) ? 1.0 : 0.0) - normal_a*normal_ell;
          d_h_source[ell][a] =
              (dx_h_radial*normal_a*normal_ell + h_radial*tangent)
              /(gauge_mass*radius);
        }
      }
    }

    for (int a = 0; a < 3; ++a) {
      beta[a] = pc.beta(m, a, k, j, i);
      lambda[a] = pc.Lambda(m, a, k, j, i);
      x[a] = pc.X(m, a, k, j, i);
      y[a] = pc.Y(m, a, k, j, i);
      w[a] = x[a]/sqrt_chi;
      l[a] = y[a]/alpha;
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
      d_k[d] = Dx<FD_STENCIL>(d, idx, pc.K, m, k, j, i);
      d_pi[d] = Dx<FD_STENCIL>(d, idx, pc.pi, m, k, j, i);
      for (int a = 0; a < 3; ++a) {
        d_lambda[d][a] = Dx<FD_STENCIL>(d, idx, pc.Lambda, m, a, k, j, i);
        d_x[d][a] = Dx<FD_STENCIL>(d, idx, pc.X, m, a, k, j, i);
        d_y[d][a] = Dx<FD_STENCIL>(d, idx, pc.Y, m, a, k, j, i);
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
    Real w_sq = 0.0;
    Real x_dot_l = 0.0;
    for (int a = 0; a < 3; ++a) {
      trace_b += b[a][a];
      for (int c = 0; c < 3; ++c) {
        w_sq += gu[a][c]*w[a]*w[c];
        x_dot_l += gu[a][c]*x[a]*l[c];
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        at_uu[a][c] = 0.0;
        aa[a][c] = 0.0;
        for (int p = 0; p < 3; ++p) {
          for (int r = 0; r < 3; ++r) {
            at_uu[a][c] += gu[a][p]*gu[c][r]*at[p][r];
            aa[a][c] += at[a][p]*gu[p][r]*at[r][c];
          }
        }
        at_sq += at[a][c]*at_uu[a][c];
      }
    }

    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        for (int e = 0; e < 3; ++e) {
          gamma_d[a][c][e] = 0.5*(q[c][a][e] + q[e][a][c] - q[a][c][e]);
          gamma_u[a][c][e] = 0.0;
          for (int p = 0; p < 3; ++p) {
            gamma_u[a][c][e] += gu[a][p]*0.5*(
                q[c][p][e] + q[e][p][c] - q[p][c][e]);
          }
        }
      }
      for (int c = 0; c < 3; ++c) {
        for (int e = 0; e < 3; ++e) {
          gamma_contracted[a] += gu[c][e]*gamma_u[a][c][e];
        }
      }
      z[a] = gamma_contracted[a] - lambda[a];
      for (int c = 0; c < 3; ++c) z_d[a] += g[a][c]*z[c];
    }

    Real cal_x[3][3];
    Real cal_y[3][3];
    Real cal_a[3][3];
    Real ricci[3][3];
    Real s_tensor[3][3];
    Real z_tensor[3][3];
    Real trace_cal_a = 0.0;
    Real trace_cal_x = 0.0;
    Real ricci_scalar = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        cal_x[a][c] = d_x[a][c];
        cal_y[a][c] = d_y[a][c];
        for (int p = 0; p < 3; ++p) {
          cal_x[a][c] -= gamma_u[p][a][c]*x[p];
          cal_y[a][c] -= gamma_u[p][a][c]*y[p];
        }
        cal_a[a][c] = 0.5*r_minus*(cal_y[a][c] - 0.5*l[a]*l[c]);

        ricci[a][c] = 0.0;
        for (int p = 0; p < 3; ++p) {
          for (int r = 0; r < 3; ++r) {
            ricci[a][c] -= 0.5*gu[p][r]*d_q[p][r][a][c];
            Real nonlinear = 0.0;
            for (int t = 0; t < 3; ++t) {
              Real const gamma_sym = 0.5*(gamma_d[a][c][t]
                                           + gamma_d[c][a][t]);
              nonlinear += gamma_u[t][p][r]*gamma_sym
                  + gamma_u[t][p][a]*gamma_d[c][t][r]
                  + gamma_u[t][p][c]*gamma_d[a][t][r]
                  + gamma_u[t][a][p]*gamma_d[t][c][r];
            }
            ricci[a][c] += gu[p][r]*nonlinear;
          }
          ricci[a][c] += 0.5*(g[p][a]*d_lambda[c][p]
                              + g[p][c]*d_lambda[a][p]);
        }
        s_tensor[a][c] = alpha*chi*ricci[a][c]
            + 0.5*alpha*cal_x[a][c] - 0.25*alpha*w[a]*w[c]
            - cal_a[a][c] - 0.25*(l[a]*x[c] + l[c]*x[a]);
        z_tensor[a][c] = -0.5*(z_d[a]*x[c] + z_d[c]*x[a]);
        for (int p = 0; p < 3; ++p) {
          z_tensor[a][c] -= 0.5*chi*z[p]*q[p][a][c];
        }
        trace_cal_a += gu[a][c]*cal_a[a][c];
        trace_cal_x += gu[a][c]*cal_x[a][c];
        ricci_scalar += gu[a][c]*ricci[a][c];
      }
    }

    Real trace_s = 0.0;
    Real trace_z_tensor = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int c = 0; c < 3; ++c) {
        trace_s += gu[a][c]*s_tensor[a][c];
        trace_z_tensor += gu[a][c]*z_tensor[a][c];
      }
    }
    Real const hamiltonian = 2.0*trace_k*trace_k/3.0 - at_sq
        + chi*ricci_scalar + 2.0*trace_cal_x - 2.5*w_sq;

    Real adv_chi = 0.0;
    Real adv_a = 0.0;
    Real adv_k = 0.0;
    Real adv_pi = 0.0;
    for (int d = 0; d < 3; ++d) {
      adv_chi += beta[d]*x[d];
      adv_a += beta[d]*y[d];
      adv_k += beta[d]*d_k[d];
      adv_pi += beta[d]*d_pi[d];
    }
    pc_rhs.chi(m, k, j, i) = adv_chi
        + 2.0*chi*(alpha*trace_k - trace_b)/3.0;
    pc_rhs.A(m, k, j, i) = adv_a
        + 2.0*lapse_sq*(alpha*pi - h_perp);
    pc_rhs.K(m, k, j, i) = adv_k
        + alpha*at_sq + alpha*trace_k*trace_k/3.0
        - trace_cal_a + 0.25*x_dot_l
        + alpha*(hamiltonian - trace_k*c_perp)
        + 0.5*alpha*(x[0]*z[0] + x[1]*z[1] + x[2]*z[2])
        - 1.5*kappa*alpha*c_perp;
    pc_rhs.pi(m, k, j, i) = adv_pi
        - alpha*at_sq - alpha*trace_k*trace_k/3.0
        + trace_cal_a - 0.25*x_dot_l
        + 0.5*chi*(z[0]*l[0] + z[1]*l[1] + z[2]*l[2])
        - 0.5*kappa*alpha*c_perp;

    for (int a = 0; a < 3; ++a) {
      Real adv_beta = 0.0;
      Real adv_lambda = 0.0;
      for (int d = 0; d < 3; ++d) {
        adv_beta += beta[d]*b[d][a];
        adv_lambda += beta[d]*d_lambda[d][a];
      }
      Real shift_source = h_source[a] + lapse_sq*chi*lambda[a];
      for (int c = 0; c < 3; ++c) {
        shift_source += 0.5*gu[a][c]*(lapse_sq*x[c] - chi*y[c]);
      }
      pc_rhs.beta(m, a, k, j, i) = adv_beta + shift_source;

      Real lambda_source = 0.0;
      for (int c = 0; c < 3; ++c) {
        Real d_trace_b = 0.0;
        for (int p = 0; p < 3; ++p) {
          lambda_source += gu[c][p]*d_b[c][p][a];
          d_trace_b += d_b[c][p][p];
        }
        lambda_source += gu[a][c]*d_trace_b/3.0;
        lambda_source -= lambda[c]*b[c][a];
        lambda_source -= at_uu[a][c]*l[c];
        lambda_source -= 3.0*r_plus*at_uu[a][c]*w[c];
        lambda_source -= 4.0*alpha*gu[a][c]*d_k[c]/3.0;
        lambda_source += alpha*gu[a][c]*(d_pi[c] + d_k[c]);
        lambda_source -= 0.5*chi*c_perp*gu[a][c]*l[c];
        for (int p = 0; p < 3; ++p) {
          lambda_source += 2.0*alpha*at_uu[c][p]*gamma_u[a][c][p];
        }
      }
      lambda_source += 2.0*lambda[a]*trace_b/3.0
          + (2.0*alpha*trace_k/3.0 + kappa*alpha)*z[a];
      pc_rhs.Lambda(m, a, k, j, i) = adv_lambda + lambda_source;
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
        Real const z_tf = z_tensor[a][c] - g[a][c]*trace_z_tensor/3.0;
        pc_rhs.Atilde(m, a, c, k, j, i) = adv_at + s_tf + shift_at
            - 2.0*at[a][c]*trace_b/3.0 - 2.0*alpha*aa[a][c]
            + alpha*trace_k*at[a][c] - alpha*c_perp*at[a][c]
            + alpha*z_tf;
      }
    }

    Real d_f_chi[3] = {0.0, 0.0, 0.0};
    Real d_f_a[3] = {0.0, 0.0, 0.0};
    Real d_f_beta[3][3] = {};
    Real d_f_g[3][3][3] = {};
    for (int ell = 0; ell < 3; ++ell) {
      Real d_trace_b = 0.0;
      for (int p = 0; p < 3; ++p) d_trace_b += d_b[ell][p][p];
      d_f_chi[ell] = 2.0*(x[ell]*(alpha*trace_k - trace_b)
          + chi*(0.5*l[ell]*trace_k + alpha*d_k[ell] - d_trace_b))/3.0;
      d_f_a[ell] = 2.0*y[ell]*(alpha*pi - h_perp)
          + 2.0*lapse_sq*(0.5*l[ell]*pi + alpha*d_pi[ell] - d_h_perp[ell]);

      for (int a = 0; a < 3; ++a) {
        d_f_beta[ell][a] = d_h_source[ell][a]
                           + (y[ell]*chi + lapse_sq*x[ell])*lambda[a]
                           + lapse_sq*chi*d_lambda[ell][a];
        for (int c = 0; c < 3; ++c) {
          Real const v_c = lapse_sq*x[c] - chi*y[c];
          Real inverse_derivative = 0.0;
          for (int p = 0; p < 3; ++p) {
            for (int r = 0; r < 3; ++r) {
              inverse_derivative -= gu[a][p]*gu[c][r]*q[ell][p][r];
            }
          }
          d_f_beta[ell][a] += 0.5*inverse_derivative*v_c
              + 0.5*gu[a][c]*(y[ell]*x[c] + lapse_sq*d_x[ell][c]
                               - x[ell]*y[c] - chi*d_y[ell][c]);
        }
      }

      for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) {
          d_f_g[ell][a][c] = -l[ell]*at[a][c] - 2.0*alpha*d_at[ell][a][c];
          for (int p = 0; p < 3; ++p) {
            d_f_g[ell][a][c] += q[ell][p][a]*b[c][p]
                + g[p][a]*d_b[ell][c][p]
                + q[ell][p][c]*b[a][p]
                + g[p][c]*d_b[ell][a][p];
          }
          d_f_g[ell][a][c] -= 2.0*(q[ell][a][c]*trace_b
                                    + g[a][c]*d_trace_b)/3.0;
        }
      }
    }

    for (int ell = 0; ell < 3; ++ell) {
      Real rhs_x = d_f_chi[ell];
      Real rhs_y = d_f_a[ell];
      for (int d = 0; d < 3; ++d) {
        rhs_x += beta[d]*d_x[d][ell] + b[ell][d]*x[d];
        rhs_y += beta[d]*d_y[d][ell] + b[ell][d]*y[d];
      }
      pc_rhs.X(m, ell, k, j, i) = rhs_x;
      pc_rhs.Y(m, ell, k, j, i) = rhs_y;

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
  return TaskStatus::complete;
}

template TaskStatus PcGh::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcRHS<4>(Driver *pdriver, int stage);

}  // namespace pc_gh
