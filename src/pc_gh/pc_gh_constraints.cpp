//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh_constraints.cpp
//! \brief regular PC-GH constraint, curl, reduction, and boundedness diagnostics

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "pc_gh/pc_gh.hpp"
#include "utils/finite_diff.hpp"

namespace pc_gh {
namespace {

KOKKOS_INLINE_FUNCTION
Real MinimumSymmetricEigenvalue(Real a00, Real a01, Real a02,
                                Real a11, Real a12, Real a22) {
  Real const mean = (a00 + a11 + a22)/3.0;
  Real const d00 = a00 - mean;
  Real const d11 = a11 - mean;
  Real const d22 = a22 - mean;
  Real const scale2 = (d00*d00 + d11*d11 + d22*d22
                       + 2.0*(a01*a01 + a02*a02 + a12*a12))/6.0;
  if (scale2 == 0.0) return mean;
  Real const scale = std::sqrt(scale2);
  Real const b00 = d00/scale;
  Real const b01 = a01/scale;
  Real const b02 = a02/scale;
  Real const b11 = d11/scale;
  Real const b12 = a12/scale;
  Real const b22 = d22/scale;
  Real const half_det = 0.5*adm::SpatialDet(b00, b01, b02, b11, b12, b22);
  Real const bounded = std::fmax(-1.0, std::fmin(1.0, half_det));
  Real const angle = std::acos(bounded)/3.0;
  Real const largest = mean + 2.0*scale*std::cos(angle);
  Real const middle = mean + 2.0*scale*std::cos(
      angle + 4.1887902047863909846168578443727);
  return 3.0*mean - largest - middle;
}

}  // namespace

template <int FD_STENCIL>
TaskStatus PcGh::CalcConstraints(Driver *pdriver, int stage) {
  if (pdriver != nullptr && stage != pdriver->nexp_stages) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int const nmb = pmy_pack->nmb_thispack;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  Real const inner_radius = opt.physical_output_inner_radius;
  Real const center_x = opt.gauge_center[0];
  Real const center_y = opt.gauge_center[1];
  Real const center_z = opt.gauge_center[2];
  auto &pc = u;
  auto &state = u0;
  auto &state_rhs = u_rhs;
  auto &con = u_con;

  par_for("regular PC-GH constraint diagnostics", DevExeSpace(),
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
    Real p_vec[3];
    Real l_vec[3];
    Real z[3];
    Real d_k[3] = {0.0, 0.0, 0.0};
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
    Real const trace_k = pc.K(m, k, j, i);
    Real const c_perp = pc.Cperp(m, k, j, i);
    for (int a = 0; a < 3; ++a) {
      p_vec[a] = pc.p(m, a, k, j, i);
      l_vec[a] = pc.L(m, a, k, j, i);
      z[a] = pc.Z(m, a, k, j, i);
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

    Real red_w2 = 0.0;
    Real red_q2 = 0.0;
    Real red_alpha2 = 0.0;
    Real red_b2 = 0.0;
    for (int d = 0; d < 3; ++d) {
      bool const active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
      if (active) {
        Real const dw = Dx<FD_STENCIL>(d, idx, pc.w, m, k, j, i);
        Real const drho = Dx<FD_STENCIL>(d, idx, pc.rho, m, k, j, i);
        Real const rw = p_vec[d] - dw;
        Real const ralpha = l_vec[d] - 2.0*(w*drho + rho*p_vec[d]);
        red_w2 += rw*rw;
        red_alpha2 += ralpha*ralpha;
        d_k[d] = Dx<FD_STENCIL>(d, idx, pc.K, m, k, j, i);
      } else {
        red_w2 += p_vec[d]*p_vec[d];
        red_alpha2 += l_vec[d]*l_vec[d];
      }
      for (int a = 0; a < 3; ++a) {
        if (active) {
          d_z[d][a] = Dx<FD_STENCIL>(d, idx, pc.Z, m, a, k, j, i);
          d_p[d][a] = Dx<FD_STENCIL>(d, idx, pc.p, m, a, k, j, i);
          d_l[d][a] = Dx<FD_STENCIL>(d, idx, pc.L, m, a, k, j, i);
        }
        for (int b = 0; b < 3; ++b) {
          Real const stored_b = state(m, BIndex(a, b), k, j, i);
          Real dbeta = 0.0;
          if (active && a == d) {
            dbeta = Dx<FD_STENCIL>(d, idx, pc.beta, m, b, k, j, i);
          }
          if (a == d) red_b2 += (stored_b - dbeta)*(stored_b - dbeta);
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
    Real p2 = 0.0;
    Real l2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        at_uu[a][b] = 0.0;
        at_ud[a][b] = 0.0;
        for (int r = 0; r < 3; ++r) {
          at_ud[a][b] += gu[a][r]*at[r][b];
          for (int s = 0; s < 3; ++s) {
            at_uu[a][b] += gu[a][r]*gu[b][s]*at[r][s];
          }
        }
        at_sq += at[a][b]*at_uu[a][b];
        trace_at += gu[a][b]*at[a][b];
        p2 += gu[a][b]*p_vec[a]*p_vec[b];
        l2 += gu[a][b]*l_vec[a]*l_vec[b];
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
        }
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          gamma_u[a][b][c] = 0.0;
          for (int r = 0; r < 3; ++r) {
            gamma_u[a][b][c] += gu[a][r]*gamma_d[r][b][c];
          }
        }
      }
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          gamma_contracted[a] += gu[b][c]*gamma_u[a][b][c];
        }
      }
    }
    for (int ell = 0; ell < 3; ++ell) {
      for (int a = 0; a < 3; ++a) {
        Real d_gamma_contracted = 0.0;
        for (int b = 0; b < 3; ++b) {
          for (int c = 0; c < 3; ++c) {
            Real d_gu_bc = 0.0;
            for (int r = 0; r < 3; ++r) {
              for (int s = 0; s < 3; ++s) {
                d_gu_bc -= gu[b][r]*gu[c][s]*q[ell][r][s];
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
                  d_q[ell][b][r][c] + d_q[ell][c][r][b]
                  - d_q[ell][r][b][c]);
              d_gamma_u += d_gu_ar*gamma_d[r][b][c] + gu[a][r]*d_gamma_d;
            }
            d_gamma_contracted += d_gu_bc*gamma_u[a][b][c]
                                  + gu[b][c]*d_gamma_u;
          }
        }
        d_lambda[ell][a] = d_gamma_contracted - d_z[ell][a];
      }
    }

    Real div_p = 0.0;
    Real ricci_scalar = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        Real cov_p = d_p[a][b];
        for (int r = 0; r < 3; ++r) cov_p -= gamma_u[r][a][b]*p_vec[r];
        div_p += gu[a][b]*cov_p;
        Real ricci = 0.0;
        for (int r = 0; r < 3; ++r) {
          for (int s = 0; s < 3; ++s) {
            ricci -= 0.5*gu[r][s]*d_q[r][s][a][b];
            Real nonlinear = 0.0;
            for (int t = 0; t < 3; ++t) {
              nonlinear += gamma_u[t][r][s]*0.5*(gamma_d[a][b][t]
                                                  + gamma_d[b][a][t])
                  + gamma_u[t][r][a]*gamma_d[b][t][s]
                  + gamma_u[t][r][b]*gamma_d[a][t][s]
                  + gamma_u[t][a][r]*gamma_d[t][b][s];
            }
            ricci += gu[r][s]*nonlinear;
          }
          ricci += 0.5*(g[r][a]*d_lambda[b][r]
                        + g[r][b]*d_lambda[a][r]);
        }
        ricci_scalar += gu[a][b]*ricci;
      }
    }
    Real const hamiltonian = 2.0*trace_k*trace_k/3.0 - at_sq
        + w2*ricci_scalar + 4.0*w*div_p - 6.0*p2;

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
      alpha_momentum[a] = rho*w*(divergence - 2.0*d_k[a]/3.0);
      for (int d = 0; d < 3; ++d) {
        alpha_momentum[a] -= 3.0*rho*at_ud[d][a]*p_vec[d];
      }
    }

    Real curl_p2 = 0.0;
    Real curl_l2 = 0.0;
    Real curl_q2 = 0.0;
    Real curl_b2 = 0.0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a + 1; b < 3; ++b) {
        Real const cp = d_p[a][b] - d_p[b][a];
        Real const cl = d_l[a][b] - d_l[b][a];
        curl_p2 += cp*cp;
        curl_l2 += cl*cl;
        for (int r = 0; r < 3; ++r) {
          Real const cb = d_b[a][b][r] - d_b[b][a][r];
          curl_b2 += cb*cb;
          for (int s = r; s < 3; ++s) {
            Real const cq = d_q[a][b][r][s] - d_q[b][a][r][s];
            curl_q2 += cq*cq;
          }
        }
      }
    }

    Real rhs_primary2 = 0.0;
    Real rhs_gradient2 = 0.0;
    for (int v = 0; v < npcgh; ++v) {
      Real const value = state_rhs(m, v, k, j, i);
      if (v < I_P1) rhs_primary2 += value*value;
      else rhs_gradient2 += value*value;
    }
    Real const minor1 = std::fmin(g[0][0], std::fmin(g[1][1], g[2][2]));
    Real const minor2 = std::fmin(
        g[0][0]*g[1][1] - g[0][1]*g[0][1],
        std::fmin(g[0][0]*g[2][2] - g[0][2]*g[0][2],
                  g[1][1]*g[2][2] - g[1][2]*g[1][2]));
    Real const min_eigenvalue = MinimumSymmetricEigenvalue(
        g[0][0], g[0][1], g[0][2], g[1][1], g[1][2], g[2][2]);
    Real const x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max) - center_x;
    Real const y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max) - center_y;
    Real const zz = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min,
                                size.d_view(m).x3max) - center_z;

    con(m, I_CON_CPERP, k, j, i) = c_perp;
    con(m, I_CON_ZX, k, j, i) = z[0];
    con(m, I_CON_ZY, k, j, i) = z[1];
    con(m, I_CON_ZZ, k, j, i) = z[2];
    con(m, I_CON_H, k, j, i) = hamiltonian;
    con(m, I_CON_MX, k, j, i) = alpha_momentum[0];
    con(m, I_CON_MY, k, j, i) = alpha_momentum[1];
    con(m, I_CON_MZ, k, j, i) = alpha_momentum[2];
    con(m, I_CON_RED_W, k, j, i) = std::sqrt(red_w2);
    con(m, I_CON_RED_Q, k, j, i) = std::sqrt(red_q2);
    con(m, I_CON_RED_ALPHA, k, j, i) = std::sqrt(red_alpha2);
    con(m, I_CON_RED_B, k, j, i) = std::sqrt(red_b2);
    con(m, I_CON_CURL_P, k, j, i) = std::sqrt(curl_p2);
    con(m, I_CON_CURL_Q, k, j, i) = std::sqrt(curl_q2);
    con(m, I_CON_CURL_L, k, j, i) = std::sqrt(curl_l2);
    con(m, I_CON_CURL_B, k, j, i) = std::sqrt(curl_b2);
    con(m, I_CON_DETG, k, j, i) = det_g - 1.0;
    con(m, I_CON_TRA, k, j, i) = trace_at;
    con(m, I_CON_TRQ, k, j, i) = std::sqrt(trace_q2);
    con(m, I_CON_MINOR1, k, j, i) = minor1;
    con(m, I_CON_MINOR2, k, j, i) = minor2;
    con(m, I_CON_MINEIG, k, j, i) = min_eigenvalue;
    con(m, I_CON_PHYSICAL_VALID, k, j, i) =
        (x*x + y*y + zz*zz >= inner_radius*inner_radius) ? 1.0 : 0.0;
    con(m, I_CON_P, k, j, i) = std::sqrt(p2);
    con(m, I_CON_L, k, j, i) = std::sqrt(l2);
    con(m, I_CON_RHS_PRIMARY, k, j, i) = std::sqrt(rhs_primary2);
    con(m, I_CON_RHS_GRADIENT, k, j, i) = std::sqrt(rhs_gradient2);
  });
  Kokkos::fence();
  ValidateState("constraint diagnostics", false, true);
  if (pdriver != nullptr && opt.boundedness_output
      && pmy_pack->pmesh->ncycle % opt.boundedness_dcycle == 0) {
    WriteBoundednessDiagnostics();
  }
  return TaskStatus::complete;
}

template <int FD_STENCIL>
void PcGh::MeasureReductionTransfer(bool save_before, int operation) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  int const nmb = pmy_pack->nmb_thispack;
  bool const multi_d = pmy_pack->pmesh->multi_d;
  bool const three_d = pmy_pack->pmesh->three_d;
  auto pc = u;
  auto state = u0;
  auto before = transfer_reduction_before;
  auto after = transfer_reduction_after;
  auto destination = save_before ? before : after;

  par_for("PC-GH transfer reduction measurement", DevExeSpace(),
  0, nmb - 1, indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real idx[3] = {1.0/size.d_view(m).dx1,
                   1.0/size.d_view(m).dx2,
                   1.0/size.d_view(m).dx3};
    Real norm2[4] = {0.0, 0.0, 0.0, 0.0};
    Real const w = pc.w(m, k, j, i);
    Real const rho = pc.rho(m, k, j, i);
    for (int d = 0; d < 3; ++d) {
      bool const active = (d == 0) || (d == 1 && multi_d) || (d == 2 && three_d);
      Real const p_d = pc.p(m, d, k, j, i);
      Real const l_d = pc.L(m, d, k, j, i);
      Real const dw = active ? Dx<FD_STENCIL>(d, idx, pc.w, m, k, j, i) : 0.0;
      Real const drho = active
          ? Dx<FD_STENCIL>(d, idx, pc.rho, m, k, j, i) : 0.0;
      Real const red_w = p_d - dw;
      Real const red_alpha = l_d - 2.0*(w*drho + rho*p_d);
      norm2[0] += red_w*red_w;
      norm2[2] += red_alpha*red_alpha;
      for (int a = 0; a < 3; ++a) {
        Real const dbeta = active
            ? Dx<FD_STENCIL>(d, idx, pc.beta, m, a, k, j, i) : 0.0;
        Real const red_b = state(m, BIndex(d, a), k, j, i) - dbeta;
        norm2[3] += red_b*red_b;
        for (int b = a; b < 3; ++b) {
          Real const dg = active
              ? Dx<FD_STENCIL>(d, idx, pc.gtilde, m, a, b, k, j, i) : 0.0;
          Real const red_q = state(m, QIndex(d, a, b), k, j, i) - dg;
          norm2[1] += red_q*red_q;
        }
      }
    }
    for (int n = 0; n < 4; ++n) destination(m, n, k, j, i) = std::sqrt(norm2[n]);
  });
  Kokkos::fence();
  if (save_before) return;

  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const nkji = nx1*nx2*nx3;
  int const ncells = nmb*nkji;
  Real changes[4] = {0.0, 0.0, 0.0, 0.0};
  for (int n = 0; n < 4; ++n) {
    Real local_change = 0.0;
    Kokkos::parallel_reduce("PC-GH transfer reduction change",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(int idx, Real &maximum) {
      int const m = idx/nkji;
      int const q = idx - m*nkji;
      int const k = indcs.ks + q/(nx1*nx2);
      int const j = indcs.js + (q % (nx1*nx2))/nx1;
      int const i = indcs.is + q % nx1;
      maximum = std::fmax(maximum,
          std::fabs(after(m, n, k, j, i) - before(m, n, k, j, i)));
    }, Kokkos::Max<Real>(local_change));
    changes[n] = local_change;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, changes, 4, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  for (int n = 0; n < 4; ++n) {
    transfer_reduction_change[operation][n] = std::fmax(
        transfer_reduction_change[operation][n], changes[n]);
  }
}

void PcGh::BeginReductionTransfer(int operation) {
  switch (opt.fd_stencil) {
    case 2: MeasureReductionTransfer<2>(true, operation); break;
    case 3: MeasureReductionTransfer<3>(true, operation); break;
    case 4: MeasureReductionTransfer<4>(true, operation); break;
    default: std::abort();
  }
}

void PcGh::EndReductionTransfer(int operation) {
  switch (opt.fd_stencil) {
    case 2: MeasureReductionTransfer<2>(false, operation); break;
    case 3: MeasureReductionTransfer<3>(false, operation); break;
    case 4: MeasureReductionTransfer<4>(false, operation); break;
    default: std::abort();
  }
}

void PcGh::WriteBoundednessDiagnostics() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const nkji = nx1*nx2*nx3;
  int const ncells = pmy_pack->nmb_thispack*nkji;
  auto state = u0;
  auto con = u_con;

  constexpr int nfield_max = 12;
  constexpr int nfield_min = 7;
  Real field_max[nfield_max] = {};
  Real field_min[nfield_min];
  for (Real &value : field_min) value = std::numeric_limits<Real>::max();

  for (int quantity = 0; quantity < nfield_max; ++quantity) {
    Real local_max = 0.0;
    Kokkos::parallel_reduce("PC-GH boundedness maxima",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(int idx, Real &maximum) {
      int const m = idx/nkji;
      int const q = idx - m*nkji;
      int const k = indcs.ks + q/(nx1*nx2);
      int const j = indcs.js + (q % (nx1*nx2))/nx1;
      int const i = indcs.is + q % nx1;
      Real value = 0.0;
      if (quantity == 0) value = state(m, I_W, k, j, i);
      if (quantity == 1) value = state(m, I_RHO, k, j, i);
      if (quantity == 2) value = state(m, I_RHO, k, j, i)*state(m, I_W, k, j, i);
      if (quantity == 3 || quantity == 4 || quantity == 6) {
        int const first = (quantity == 3) ? I_P1 : ((quantity == 4) ? I_L1 : I_ZX);
        for (int n = 0; n < 3; ++n) value += state(m, first + n, k, j, i)
                                                    *state(m, first + n, k, j, i);
        value = std::sqrt(value);
      }
      if (quantity == 5) value = std::fabs(state(m, I_CPERP, k, j, i));
      if (quantity == 7) value = std::fabs(state(m, I_K, k, j, i));
      if (quantity == 8 || quantity == 9 || quantity == 10 || quantity == 11) {
        int first = I_ATXX;
        int count = 6;
        if (quantity == 9) { first = I_BETAX; count = 3; }
        if (quantity == 10) { first = I_Q1XX; count = 18; }
        if (quantity == 11) { first = I_B11; count = 9; }
        for (int n = 0; n < count; ++n) value += state(m, first + n, k, j, i)
                                                        *state(m, first + n, k, j, i);
        value = std::sqrt(value);
      }
      maximum = std::fmax(maximum, value);
    }, Kokkos::Max<Real>(local_max));
    field_max[quantity] = local_max;
  }

  for (int quantity = 0; quantity < nfield_min; ++quantity) {
    Real local_min = std::numeric_limits<Real>::max();
    Kokkos::parallel_reduce("PC-GH boundedness minima",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(int idx, Real &minimum) {
      int const m = idx/nkji;
      int const q = idx - m*nkji;
      int const k = indcs.ks + q/(nx1*nx2);
      int const j = indcs.js + (q % (nx1*nx2))/nx1;
      int const i = indcs.is + q % nx1;
      Real value = 0.0;
      if (quantity == 0) value = state(m, I_W, k, j, i);
      if (quantity == 1) value = state(m, I_RHO, k, j, i);
      if (quantity == 2) value = state(m, I_RHO, k, j, i)*state(m, I_W, k, j, i);
      if (quantity >= 3) {
        int const c = (quantity == 3) ? I_CON_DETG
            : ((quantity == 4) ? I_CON_MINOR1
            : ((quantity == 5) ? I_CON_MINOR2 : I_CON_MINEIG));
        value = con(m, c, k, j, i);
        if (quantity == 3) value += 1.0;
      }
      minimum = std::fmin(minimum, value);
    }, Kokkos::Min<Real>(local_min));
    field_min[quantity] = local_min;
  }

  Real constraint_max[ncon] = {};
  for (int quantity = 0; quantity < ncon; ++quantity) {
    Real local_max = 0.0;
    Kokkos::parallel_reduce("PC-GH diagnostic maxima",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(int idx, Real &maximum) {
      int const m = idx/nkji;
      int const q = idx - m*nkji;
      int const k = indcs.ks + q/(nx1*nx2);
      int const j = indcs.js + (q % (nx1*nx2))/nx1;
      int const i = indcs.is + q % nx1;
      maximum = std::fmax(maximum, std::fabs(con(m, quantity, k, j, i)));
    }, Kokkos::Max<Real>(local_max));
    constraint_max[quantity] = local_max;
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, field_max, nfield_max, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, field_min, nfield_min, MPI_ATHENA_REAL,
                MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, constraint_max, ncon, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
#endif
  if (!opt.boundedness_output || global_variable::my_rank != 0) return;
  std::ifstream existing(opt.boundedness_file, std::ios::binary | std::ios::ate);
  bool const write_header = !existing.good() || existing.tellg() == 0;
  existing.close();
  std::ofstream output(opt.boundedness_file, std::ios::app);
  if (!output.is_open()) {
    std::cout << "### FATAL ERROR: unable to open PC-GH boundedness file '"
              << opt.boundedness_file << "'" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (write_header) {
    output << "# time cycle min_w max_w min_rho max_rho min_alpha max_alpha"
           << " max_p max_L max_Cperp max_Z max_K max_Atilde max_beta max_Q max_B"
           << " min_detg max_abs_detg_minus_1 min_minor1 min_minor2 min_eigenvalue";
    for (int n = 0; n < ncon; ++n) output << ' ' << ConstraintNames[n] << "_max";
    char const * const operations[3] = {"restrict", "prolong", "project"};
    char const * const reductions[4] = {"Rw", "RQ", "Ralpha", "RB"};
    for (int op = 0; op < 3; ++op) {
      for (int red = 0; red < 4; ++red) {
        output << " d" << reductions[red] << '_' << operations[op] << "_max";
      }
    }
    output << '\n';
  }
  output << std::setprecision(17) << pmy_pack->pmesh->time << ' '
         << pmy_pack->pmesh->ncycle << ' '
         << field_min[0] << ' ' << field_max[0] << ' '
         << field_min[1] << ' ' << field_max[1] << ' '
         << field_min[2] << ' ' << field_max[2];
  for (int n = 3; n < nfield_max; ++n) output << ' ' << field_max[n];
  output << ' ' << field_min[3] << ' ' << constraint_max[I_CON_DETG]
         << ' ' << field_min[4] << ' ' << field_min[5] << ' ' << field_min[6];
  for (int n = 0; n < ncon; ++n) output << ' ' << constraint_max[n];
  for (auto const &operation : transfer_reduction_change) {
    for (Real const value : operation) output << ' ' << value;
  }
  output << '\n';
}

template TaskStatus PcGh::CalcConstraints<2>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcConstraints<3>(Driver *pdriver, int stage);
template TaskStatus PcGh::CalcConstraints<4>(Driver *pdriver, int stage);

}  // namespace pc_gh
