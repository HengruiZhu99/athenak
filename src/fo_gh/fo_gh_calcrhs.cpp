//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh_calcrhs.cpp
//! \brief Two-pass compatible-gradient RHS for regularized vacuum FO-GH.

#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "fo_gh/fo_gh.hpp"
#include "fo_gh/fo_gh_rhs.hpp"
#include "mesh/mesh.hpp"
#include "utils/finite_diff.hpp"

namespace fo_gh {

KOKKOS_INLINE_FUNCTION
void LoadPoint(const FoGh::Variables &v, const int m, const int k,
               const int j, const int i, RegularPointState &u) {
  u.chi = v.chi(m, k, j, i);
  u.alpha = v.alpha(m, k, j, i);
  u.K = v.K(m, k, j, i);
  u.pi = v.pi(m, k, j, i);
  u.h_perp = v.h_perp(m, k, j, i);
  u.vartheta_perp = v.vartheta_perp(m, k, j, i);
  for (int a = 0; a < 3; ++a) {
    u.beta(a) = v.beta(m, a, k, j, i);
    u.Lambda(a) = v.Lambda(m, a, k, j, i);
    u.X(a) = v.X(m, a, k, j, i);
    u.a(a) = v.a(m, a, k, j, i);
    u.h(a) = v.h(m, a, k, j, i);
    u.vartheta(a) = v.vartheta(m, a, k, j, i);
    for (int b = 0; b < 3; ++b) {
      u.B(a, b) = v.B(m, a, b, k, j, i);
    }
    for (int b = 0; b < 3; ++b) {
      for (int c = b; c < 3; ++c) {
        u.Q(a, b, c) = v.Q[a](m, b, c, k, j, i);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      u.gtilde(a, b) = v.gtilde(m, a, b, k, j, i);
      u.Atilde(a, b) = v.Atilde(m, a, b, k, j, i);
    }
  }
}

template <int FDNG>
KOKKOS_INLINE_FUNCTION
void LoadDerivatives(const FoGh::Variables &v, const Real idx[3],
                     const int m, const int k, const int j, const int i,
                     EvolutionDerivatives &d) {
  d.ZeroClear();
  for (int p = 0; p < 3; ++p) {
    d.geometry.dK(p) = Dx<FDNG>(p, idx, v.K, m, k, j, i);
    d.dpi(p) = Dx<FDNG>(p, idx, v.pi, m, k, j, i);
    d.dh_perp(p) = Dx<FDNG>(p, idx, v.h_perp, m, k, j, i);
    for (int a = 0; a < 3; ++a) {
      d.geometry.dX(p, a) = Dx<FDNG>(p, idx, v.X, m, a, k, j, i);
      d.geometry.da(p, a) = Dx<FDNG>(p, idx, v.a, m, a, k, j, i);
      d.geometry.dLambda(p, a) = Dx<FDNG>(p, idx, v.Lambda, m, a, k, j, i);
      d.dh(p, a) = Dx<FDNG>(p, idx, v.h, m, a, k, j, i);
      for (int b = 0; b < 3; ++b) {
        d.dB(p, a, b) = Dx<FDNG>(p, idx, v.B, m, a, b, k, j, i);
      }
      for (int b = 0; b < 3; ++b) {
        for (int c = b; c < 3; ++c) {
          d.geometry.dQ[p](a, b, c) =
              Dx<FDNG>(p, idx, v.Q[a], m, b, c, k, j, i);
        }
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        d.geometry.dA(p, a, b) =
            Dx<FDNG>(p, idx, v.Atilde, m, a, b, k, j, i);
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void StorePrimary(const PrimaryRhs &r, const FoGh::Variables &rhs,
                  const int m, const int k, const int j, const int i) {
  rhs.chi(m, k, j, i) = r.chi;
  rhs.alpha(m, k, j, i) = r.alpha;
  rhs.K(m, k, j, i) = r.K;
  rhs.pi(m, k, j, i) = r.pi;
  rhs.h_perp(m, k, j, i) = r.h_perp;
  rhs.vartheta_perp(m, k, j, i) = r.vartheta_perp;
  for (int a = 0; a < 3; ++a) {
    rhs.beta(m, a, k, j, i) = r.beta(a);
    rhs.Lambda(m, a, k, j, i) = r.Lambda(a);
    rhs.h(m, a, k, j, i) = r.h(a);
    rhs.vartheta(m, a, k, j, i) = r.vartheta(a);
    for (int b = a; b < 3; ++b) {
      rhs.gtilde(m, a, b, k, j, i) = r.gtilde(a, b);
      rhs.Atilde(m, a, b, k, j, i) = r.Atilde(a, b);
    }
  }
}

template <int FDNG>
TaskStatus FoGh::CalcRHS(Driver *pdriver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int radius = FDNG - 1;
  const int nmb = pmy_pack->nmb_thispack;
  const auto vars = u;
  const auto rhs_vars = rhs;
  const Real kappa = opt.kappa;
  const Real mu_H = opt.mu_H;
  const Real eta_H = opt.eta_H;
  const Real eta_beta = opt.eta_beta;

  par_for("fo_gh primary rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks - radius, indcs.ke + radius,
  indcs.js - radius, indcs.je + radius,
  indcs.is - radius, indcs.ie + radius,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[3] = {1.0/size.d_view(m).dx1,
                         1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    RegularPointState point;
    EvolutionDerivatives derivatives;
    PrimaryRhs point_rhs;
    LoadPoint(vars, m, k, j, i, point);
    LoadDerivatives<FDNG>(vars, idx, m, k, j, i, derivatives);
    ComputePrimaryRhs(point, derivatives, kappa, mu_H, eta_H, eta_beta,
                      point_rhs);
    StorePrimary(point_rhs, rhs_vars, m, k, j, i);
  });

  par_for("fo_gh compatible rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[3] = {1.0/size.d_view(m).dx1,
                         1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    for (int p = 0; p < 3; ++p) {
      rhs_vars.X(m, p, k, j, i) =
          Dx<FDNG>(p, idx, rhs_vars.chi, m, k, j, i);
      rhs_vars.a(m, p, k, j, i) =
          Dx<FDNG>(p, idx, rhs_vars.alpha, m, k, j, i);
      for (int a = 0; a < 3; ++a) {
        rhs_vars.B(m, p, a, k, j, i) =
            Dx<FDNG>(p, idx, rhs_vars.beta, m, a, k, j, i);
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          rhs_vars.Q[p](m, a, b, k, j, i) =
              Dx<FDNG>(p, idx, rhs_vars.gtilde, m, a, b, k, j, i);
        }
      }
    }
  });

  if (opt.diss > 0.0) {
    const Real sign = (FDNG % 2 == 0) ? -1.0 : 1.0;
    const Real diss = opt.diss*std::pow(2.0, -2.0*FDNG)*sign;
    const auto state = u0;
    const auto state_rhs = u_rhs;
    par_for("fo_gh dissipation", DevExeSpace(), 0, nmb - 1, 0, nfo_gh - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      const Real idx[3] = {1.0/size.d_view(m).dx1,
                           1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};
      for (int p = 0; p < 3; ++p) {
        state_rhs(m, n, k, j, i) +=
            diss*Diss<FDNG>(p, idx, state, m, n, k, j, i);
      }
    });
  }
  return TaskStatus::complete;
}

template TaskStatus FoGh::CalcRHS<2>(Driver *, int);
template TaskStatus FoGh::CalcRHS<3>(Driver *, int);
template TaskStatus FoGh::CalcRHS<4>(Driver *, int);

template <int FDNG>
void FoGh::CalcConstraints() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto vars = u;
  const auto constraints = u_con;
  Kokkos::deep_copy(constraints, 0.0);
  par_for("fo_gh constraints", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[3] = {1.0/size.d_view(m).dx1,
                         1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    RegularPointState point;
    EvolutionDerivatives derivatives;
    GeometryPoint geometry;
    LoadPoint(vars, m, k, j, i, point);
    LoadDerivatives<FDNG>(vars, idx, m, k, j, i, derivatives);
    ComputeGeometry(point, derivatives.geometry, geometry);
    constraints(m, I_CON_H, k, j, i) = geometry.hamiltonian;
    constraints(m, I_CON_GH_PERP, k, j, i) = point.pi + point.K;
    Real rq2 = 0.0;
    Real rx2 = 0.0;
    Real ra2 = 0.0;
    Real rb2 = 0.0;
    for (int p = 0; p < 3; ++p) {
      constraints(m, I_CON_MX + p, k, j, i) = geometry.momentum(p);
      constraints(m, I_CON_GHX + p, k, j, i) = geometry.c_up(p);
      const Real rx = point.X(p) - Dx<FDNG>(p, idx, vars.chi, m, k, j, i);
      const Real ra = point.a(p) - Dx<FDNG>(p, idx, vars.alpha, m, k, j, i);
      rx2 += rx*rx;
      ra2 += ra*ra;
      for (int a = 0; a < 3; ++a) {
        const Real rb = point.B(p, a)
                        - Dx<FDNG>(p, idx, vars.beta, m, a, k, j, i);
        rb2 += rb*rb;
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = a; b < 3; ++b) {
          const Real rq = point.Q(p, a, b)
                          - Dx<FDNG>(p, idx, vars.gtilde, m, a, b, k, j, i);
          rq2 += rq*rq;
        }
      }
    }
    constraints(m, I_CON_RQ, k, j, i) = std::sqrt(rq2);
    constraints(m, I_CON_RX, k, j, i) = std::sqrt(rx2);
    constraints(m, I_CON_RA, k, j, i) = std::sqrt(ra2);
    constraints(m, I_CON_RB, k, j, i) = std::sqrt(rb2);
  });
}

template void FoGh::CalcConstraints<2>();
template void FoGh::CalcConstraints<3>();
template void FoGh::CalcConstraints<4>();

} // namespace fo_gh
