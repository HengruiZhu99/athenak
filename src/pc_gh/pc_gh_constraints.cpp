//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_constraints.cpp
//! \brief independent PC-GH constraint, reduction, curl, and regularity diagnostics

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "utils/finite_diff.hpp"

namespace pc_gh {

template <int FD_STENCIL>
TaskStatus PcGh::CalcConstraints(Driver *pdriver, int stage) {
  if (pdriver != nullptr && stage != pdriver->nexp_stages) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int const nmb = pmy_pack->nmb_thispack;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  auto &pc = u;
  auto &state = u0;
  auto &state_rhs = u_rhs;
  auto &con = u_con;

  par_for("PC-GH constraint diagnostics", DevExeSpace(),
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
    Real q[3][3][3];
    Real gamma_u[3][3][3];
    Real gamma_d[3][3][3];
    Real gamma_contracted[3] = {0.0, 0.0, 0.0};
    Real x[3];
    Real y[3];
    Real w[3];
    Real l[3];
    Real z[3];
    Real d_k[3] = {0.0, 0.0, 0.0};
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
    Real const trace_k = pc.K(m, k, j, i);
    Real const c_perp = pc.pi(m, k, j, i) + trace_k;
    for (int a = 0; a < 3; ++a) {
      x[a] = pc.X(m, a, k, j, i);
      y[a] = pc.Y(m, a, k, j, i);
      w[a] = x[a]/sqrt_chi;
      l[a] = y[a]/alpha;
      for (int b = a; b < 3; ++b) {
        g[a][b] = g[b][a] = pc.gtilde(m, a, b, k, j, i);
        at[a][b] = at[b][a] = pc.Atilde(m, a, b, k, j, i);
        for (int d = 0; d < 3; ++d) {
          q[d][a][b] = q[d][b][a] = state(m, QIndex(d, a, b), k, j, i);
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
      for (int b = 0; b < 3; ++b) gu[a][b] = inverse(a, b);
    }

    Real red_x2 = 0.0;
    Real red_q2 = 0.0;
    Real red_y2 = 0.0;
    Real red_b2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      bool const active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
      if (active) {
        Real const dx_chi = Dx<FD_STENCIL>(d, idx, pc.chi, m, k, j, i);
        Real const dx_a = Dx<FD_STENCIL>(d, idx, pc.A, m, k, j, i);
        red_x2 += (x[d] - dx_chi)*(x[d] - dx_chi);
        red_y2 += (y[d] - dx_a)*(y[d] - dx_a);
        d_k[d] = Dx<FD_STENCIL>(d, idx, pc.K, m, k, j, i);
      } else {
        red_x2 += x[d]*x[d];
        red_y2 += y[d]*y[d];
      }
      for (int a = 0; a < 3; ++a) {
        if (active) {
          d_lambda[d][a] = Dx<FD_STENCIL>(d, idx, pc.Lambda, m, a, k, j, i);
          d_x[d][a] = Dx<FD_STENCIL>(d, idx, pc.X, m, a, k, j, i);
          d_y[d][a] = Dx<FD_STENCIL>(d, idx, pc.Y, m, a, k, j, i);
        }
        for (int b = 0; b < 3; ++b) {
          Real const stored_b = state(m, BIndex(a, b), k, j, i);
          Real dx_beta = 0.0;
          if (active && a == d) {
            dx_beta = Dx<FD_STENCIL>(d, idx, pc.beta, m, b, k, j, i);
          }
          if (a == d) red_b2 += (stored_b - dx_beta)*(stored_b - dx_beta);
          if (active) {
            d_b[d][a][b] = Dx<FD_STENCIL>(
                d, idx, state, m, BIndex(a, b), k, j, i);
          }
        }
        for (int b = a; b < 3; ++b) {
          if (active) {
            Real const dat = Dx<FD_STENCIL>(d, idx, pc.Atilde, m, a, b, k, j, i);
            d_at[d][a][b] = d_at[d][b][a] = dat;
          }
          for (int e = 0; e < 3; ++e) {
            if (active) {
              Real const dq = Dx<FD_STENCIL>(
                  d, idx, state, m, QIndex(e, a, b), k, j, i);
              d_q[d][e][a][b] = d_q[d][e][b][a] = dq;
            }
          }
          if (active) {
            Real const dg = Dx<FD_STENCIL>(d, idx, pc.gtilde, m, a, b, k, j, i);
            Real const rq = q[d][a][b] - dg;
            red_q2 += rq*rq;
          } else {
            red_q2 += q[d][a][b]*q[d][a][b];
          }
        }
      }
    }

    Real at_sq = 0.0;
    Real trace_at = 0.0;
    Real trace_q2 = 0.0;
    Real w2 = 0.0;
    Real l2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        at_uu[a][b] = 0.0;
        at_ud[a][b] = 0.0;
        for (int p = 0; p < 3; ++p) {
          at_ud[a][b] += gu[a][p]*at[p][b];
          for (int r = 0; r < 3; ++r) {
            at_uu[a][b] += gu[a][p]*gu[b][r]*at[p][r];
          }
        }
        at_sq += at[a][b]*at_uu[a][b];
        trace_at += gu[a][b]*at[a][b];
        w2 += gu[a][b]*w[a]*w[b];
        l2 += gu[a][b]*l[a]*l[b];
      }
    }
    for (int d = 0; d < 3; ++d) {
      Real trace_q = 0.0;
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) trace_q += gu[a][b]*q[d][a][b];
      }
      trace_q2 += trace_q*trace_q;
    }

    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          gamma_d[a][b][c] = 0.5*(q[b][a][c] + q[c][a][b] - q[a][b][c]);
          gamma_u[a][b][c] = 0.0;
          for (int p = 0; p < 3; ++p) {
            gamma_u[a][b][c] += gu[a][p]*0.5*(
                q[b][p][c] + q[c][p][b] - q[p][b][c]);
          }
        }
      }
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          gamma_contracted[a] += gu[b][c]*gamma_u[a][b][c];
        }
      }
      z[a] = gamma_contracted[a] - pc.Lambda(m, a, k, j, i);
    }

    Real cal_x[3][3];
    Real ricci[3][3];
    Real trace_cal_x = 0.0;
    Real ricci_scalar = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        cal_x[a][b] = d_x[a][b];
        for (int p = 0; p < 3; ++p) cal_x[a][b] -= gamma_u[p][a][b]*x[p];
        ricci[a][b] = 0.0;
        for (int p = 0; p < 3; ++p) {
          for (int r = 0; r < 3; ++r) {
            ricci[a][b] -= 0.5*gu[p][r]*d_q[p][r][a][b];
            Real nonlinear = 0.0;
            for (int t = 0; t < 3; ++t) {
              nonlinear += gamma_u[t][p][r]*0.5*(gamma_d[a][b][t]
                                                  + gamma_d[b][a][t])
                  + gamma_u[t][p][a]*gamma_d[b][t][r]
                  + gamma_u[t][p][b]*gamma_d[a][t][r]
                  + gamma_u[t][a][p]*gamma_d[t][b][r];
            }
            ricci[a][b] += gu[p][r]*nonlinear;
          }
          ricci[a][b] += 0.5*(g[p][a]*d_lambda[b][p]
                              + g[p][b]*d_lambda[a][p]);
        }
        trace_cal_x += gu[a][b]*cal_x[a][b];
        ricci_scalar += gu[a][b]*ricci[a][b];
      }
    }
    Real const hamiltonian = 2.0*trace_k*trace_k/3.0 - at_sq
        + chi*ricci_scalar + 2.0*trace_cal_x - 2.5*w2;

    Real momentum[3] = {0.0, 0.0, 0.0};
    for (int a = 0; a < 3; ++a) {
      Real divergence = 0.0;
      for (int d = 0; d < 3; ++d) {
        for (int p = 0; p < 3; ++p) {
          Real d_gu = 0.0;
          for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
              d_gu -= gu[d][r]*gu[p][s]*q[d][r][s];
            }
          }
          divergence += d_gu*at[p][a] + gu[d][p]*d_at[d][p][a];
          divergence += gamma_u[d][d][p]*at_ud[p][a];
          divergence -= gamma_u[p][d][a]*at_ud[d][p];
        }
      }
      momentum[a] = sqrt_chi*(divergence - 2.0*d_k[a]/3.0);
      for (int d = 0; d < 3; ++d) momentum[a] -= 1.5*at_ud[d][a]*w[d];
    }

    Real curl_x2 = 0.0;
    Real curl_y2 = 0.0;
    Real curl_q2 = 0.0;
    Real curl_b2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a + 1; b < 3; ++b) {
        Real const cx = d_x[a][b] - d_x[b][a];
        Real const cy = d_y[a][b] - d_y[b][a];
        curl_x2 += cx*cx;
        curl_y2 += cy*cy;
        for (int p = 0; p < 3; ++p) {
          Real const cb = d_b[a][b][p] - d_b[b][a][p];
          curl_b2 += cb*cb;
          for (int r = p; r < 3; ++r) {
            Real const cq = d_q[a][b][p][r] - d_q[b][a][p][r];
            curl_q2 += cq*cq;
          }
        }
      }
    }

    Real rhs_primary2 = 0.0;
    Real rhs_gradient2 = 0.0;
    for (int v = 0; v < npcgh; ++v) {
      Real const value = state_rhs(m, v, k, j, i);
      if (v < I_X1) rhs_primary2 += value*value;
      else rhs_gradient2 += value*value;
    }
    con(m, I_CON_CPERP, k, j, i) = c_perp;
    con(m, I_CON_ZX, k, j, i) = z[0];
    con(m, I_CON_ZY, k, j, i) = z[1];
    con(m, I_CON_ZZ, k, j, i) = z[2];
    con(m, I_CON_H, k, j, i) = hamiltonian;
    con(m, I_CON_MX, k, j, i) = momentum[0];
    con(m, I_CON_MY, k, j, i) = momentum[1];
    con(m, I_CON_MZ, k, j, i) = momentum[2];
    con(m, I_CON_RED_X, k, j, i) = std::sqrt(red_x2);
    con(m, I_CON_RED_Q, k, j, i) = std::sqrt(red_q2);
    con(m, I_CON_RED_Y, k, j, i) = std::sqrt(red_y2);
    con(m, I_CON_RED_B, k, j, i) = std::sqrt(red_b2);
    con(m, I_CON_CURL_X, k, j, i) = std::sqrt(curl_x2);
    con(m, I_CON_CURL_Q, k, j, i) = std::sqrt(curl_q2);
    con(m, I_CON_CURL_Y, k, j, i) = std::sqrt(curl_y2);
    con(m, I_CON_CURL_B, k, j, i) = std::sqrt(curl_b2);
    con(m, I_CON_DETG, k, j, i) = det_g - 1.0;
    con(m, I_CON_TRA, k, j, i) = trace_at;
    con(m, I_CON_TRQ, k, j, i) = std::sqrt(trace_q2);
    con(m, I_CON_RMINUS, k, j, i) = chi/alpha;
    con(m, I_CON_RPLUS, k, j, i) = alpha/sqrt_chi;
    con(m, I_CON_W, k, j, i) = std::sqrt(w2);
    con(m, I_CON_L, k, j, i) = std::sqrt(l2);
    con(m, I_CON_RHS_PRIMARY, k, j, i) = std::sqrt(rhs_primary2);
    con(m, I_CON_RHS_GRADIENT, k, j, i) = std::sqrt(rhs_gradient2);
  });
  return TaskStatus::complete;
}

template TaskStatus PcGh::CalcConstraints<2>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcConstraints<3>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcConstraints<4>(Driver *pdriver, int stage);

}  // namespace pc_gh
