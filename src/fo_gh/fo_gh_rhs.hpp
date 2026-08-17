#ifndef FO_GH_FO_GH_RHS_HPP_
#define FO_GH_FO_GH_RHS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh_rhs.hpp
//! \brief Pointwise primary-field RHS for regularized first-order GH.

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "fo_gh/fo_gh_geometry.hpp"
#include "fo_gh/fo_gh_state.hpp"

namespace fo_gh {

struct EvolutionDerivatives {
  GeometryDerivatives geometry;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dpi;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 3> dB;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dh_perp;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dh;

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    geometry.ZeroClear();
    dpi.ZeroClear();
    dB.ZeroClear();
    dh_perp.ZeroClear();
    dh.ZeroClear();
  }
};

//! Coordinate-time RHSs for the 30 primary fields. Q, X, a, and B are
//! produced in a separate compatible-gradient pass by differentiating the
//! gtilde, chi, alpha, and beta RHSs.
struct PrimaryRhs {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde;
  Real chi;
  Real alpha;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> beta;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Atilde;
  Real K;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Lambda;
  Real pi;
  Real h_perp;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> h;
  Real vartheta_perp;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> vartheta;

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    gtilde.ZeroClear();
    chi = 0.0;
    alpha = 0.0;
    beta.ZeroClear();
    Atilde.ZeroClear();
    K = 0.0;
    Lambda.ZeroClear();
    pi = 0.0;
    h_perp = 0.0;
    h.ZeroClear();
    vartheta_perp = 0.0;
    vartheta.ZeroClear();
  }
};

KOKKOS_INLINE_FUNCTION
Real DivergenceC(const RegularPointState &u, const GeometryDerivatives &d,
                 const GeometryPoint &geo) {
  Real divergence = 0.0;
  for (int p = 0; p < 3; ++p) {
    Real derivative_contracted = 0.0;
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        Real derivative_inverse = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            derivative_inverse -= geo.inverse(j, a)*geo.inverse(k, b)*u.Q(p, a, b);
          }
        }
        derivative_contracted += derivative_inverse*geo.Gamma(p, j, k);

        Real derivative_gamma = 0.0;
        for (int l = 0; l < 3; ++l) {
          Real derivative_inverse_pl = 0.0;
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              derivative_inverse_pl -= geo.inverse(p, a)*geo.inverse(l, b)
                                       *u.Q(p, a, b);
            }
          }
          derivative_gamma += 0.5*derivative_inverse_pl
              *(u.Q(j, l, k) + u.Q(k, l, j) - u.Q(l, j, k));
          derivative_gamma += 0.5*geo.inverse(p, l)
              *(d.dQ[p](j, l, k) + d.dQ[p](k, l, j) - d.dQ[p](l, j, k));
        }
        derivative_contracted += geo.inverse(j, k)*derivative_gamma;
      }
    }
    divergence += derivative_contracted - d.dLambda(p, p);
    for (int k = 0; k < 3; ++k) {
      divergence += geo.Gamma(p, p, k)*geo.c_up(k);
    }
  }
  return divergence;
}

KOKKOS_INLINE_FUNCTION
Real Trace(const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &tensor,
           const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &inverse) {
  Real trace = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      trace += inverse(i, j)*tensor(i, j);
    }
  }
  return trace;
}

KOKKOS_INLINE_FUNCTION
void MakeTraceFree(AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &tensor,
                   const RegularPointState &u, const GeometryPoint &geo) {
  const Real trace = Trace(tensor, geo.inverse);
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      tensor(i, j) -= (trace/3.0)*u.gtilde(i, j);
    }
  }
}

KOKKOS_INLINE_FUNCTION
void ComputePrimaryRhs(const RegularPointState &u, const EvolutionDerivatives &d,
                       const Real kappa, const Real mu_H, const Real eta_H,
                       const Real eta_beta, PrimaryRhs &rhs) {
  rhs.ZeroClear();
  GeometryPoint geo;
  ComputeGeometry(u, d.geometry, geo);

  Real div_beta = 0.0;
  for (int k = 0; k < 3; ++k) {
    div_beta += u.B(k, k);
  }
  const Real C_perp = u.pi + u.K;

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> A_up;
  A_up.ZeroClear();
  Real A_squared = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          A_up(i, j) += geo.inverse(i, k)*geo.inverse(j, l)*u.Atilde(k, l);
        }
      }
      A_squared += u.Atilde(i, j)*A_up(i, j);
    }
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> d0_gtilde;
  d0_gtilde.ZeroClear();
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      d0_gtilde(i, j) = -2.0*u.alpha*u.Atilde(i, j)
                        - (2.0/3.0)*u.gtilde(i, j)*div_beta;
      for (int k = 0; k < 3; ++k) {
        d0_gtilde(i, j) += u.gtilde(i, k)*u.B(j, k)
                           + u.gtilde(j, k)*u.B(i, k);
        rhs.gtilde(i, j) += u.beta(k)*u.Q(k, i, j);
      }
      rhs.gtilde(i, j) += d0_gtilde(i, j);
    }
  }
  const Real d0_chi = (2.0/3.0)*u.chi*(u.alpha*u.K - div_beta);
  rhs.chi = d0_chi;
  rhs.alpha = u.alpha*u.alpha*u.pi - u.alpha*u.h_perp;
  for (int k = 0; k < 3; ++k) {
    rhs.chi += u.beta(k)*u.X(k);
    rhs.alpha += u.beta(k)*u.a(k);
  }

  for (int i = 0; i < 3; ++i) {
    Real d0_beta = u.h(i) + u.alpha*u.alpha*u.chi*u.Lambda(i);
    for (int j = 0; j < 3; ++j) {
      d0_beta += geo.inverse(i, j)
                 *(0.5*u.alpha*u.alpha*u.X(j) - u.alpha*u.chi*u.a(j));
      rhs.beta(i) += u.beta(j)*u.B(j, i);
    }
    rhs.beta(i) += d0_beta;
  }

  Real laplacian_alpha = 0.0;
  Real X_dot_a = 0.0;
  Real c_dot_X = 0.0;
  for (int i = 0; i < 3; ++i) {
    c_dot_X += geo.c_up(i)*u.X(i);
    for (int j = 0; j < 3; ++j) {
      laplacian_alpha += geo.inverse(i, j)*geo.dd_alpha(i, j);
      X_dot_a += geo.inverse(i, j)*u.X(i)*u.a(j);
    }
  }
  const Real div_c = DivergenceC(u, d.geometry, geo);
  Real d0_K = u.alpha*A_squared + (u.alpha/3.0)*u.K*u.K
              - u.chi*laplacian_alpha + 0.5*X_dot_a
              + u.alpha*(geo.hamiltonian - u.K*C_perp - u.chi*div_c
                         + 0.5*c_dot_X)
              - 1.5*u.alpha*kappa*C_perp;
  rhs.K = d0_K;
  for (int k = 0; k < 3; ++k) {
    rhs.K += u.beta(k)*d.geometry.dK(k);
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> curvature;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> c_terms;
  curvature.ZeroClear();
  c_terms.ZeroClear();
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      curvature(i, j) = u.alpha*u.chi*geo.Ricci(i, j)
                        + 0.5*u.alpha*geo.dd_chi(i, j)
                        - (u.alpha/(4.0*u.chi))*u.X(i)*u.X(j)
                        - u.chi*geo.dd_alpha(i, j)
                        - 0.5*(u.a(i)*u.X(j) + u.a(j)*u.X(i));
      c_terms(i, j) = -0.5*(geo.c_down(i)*u.X(j) + geo.c_down(j)*u.X(i));
      for (int k = 0; k < 3; ++k) {
        c_terms(i, j) -= u.chi*geo.c_down(k)*geo.Gamma(k, i, j);
      }
    }
  }
  MakeTraceFree(curvature, u, geo);
  MakeTraceFree(c_terms, u, geo);

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      Real d0_A = curvature(i, j) + u.alpha*c_terms(i, j)
                  - (2.0/3.0)*u.Atilde(i, j)*div_beta
                  + u.alpha*u.K*u.Atilde(i, j)
                  - u.alpha*C_perp*u.Atilde(i, j);
      for (int k = 0; k < 3; ++k) {
        d0_A += u.Atilde(i, k)*u.B(j, k) + u.Atilde(j, k)*u.B(i, k);
        for (int l = 0; l < 3; ++l) {
          d0_A -= 2.0*u.alpha*u.Atilde(i, k)*geo.inverse(k, l)*u.Atilde(l, j);
        }
        rhs.Atilde(i, j) += u.beta(k)*d.geometry.dA(k, i, j);
      }
      rhs.Atilde(i, j) += d0_A;
    }
  }

  for (int i = 0; i < 3; ++i) {
    Real d0_Lambda = (2.0/3.0)*u.Lambda(i)*div_beta
                     + ((2.0/3.0)*u.alpha*u.K + kappa*u.alpha)*geo.c_up(i);
    for (int k = 0; k < 3; ++k) {
      d0_Lambda -= u.Lambda(k)*u.B(k, i);
      rhs.Lambda(i) += u.beta(k)*d.geometry.dLambda(k, i);
      for (int l = 0; l < 3; ++l) {
        d0_Lambda += geo.inverse(k, l)*d.dB(k, l, i);
      }
    }
    for (int j = 0; j < 3; ++j) {
      Real d_div_beta = 0.0;
      for (int k = 0; k < 3; ++k) {
        d_div_beta += d.dB(j, k, k);
      }
      d0_Lambda += (1.0/3.0)*geo.inverse(i, j)*d_div_beta
                   - (4.0/3.0)*u.alpha*geo.inverse(i, j)*d.geometry.dK(j)
                   + u.alpha*geo.inverse(i, j)
                       *(d.dpi(j) + d.geometry.dK(j));
    }
    for (int k = 0; k < 3; ++k) {
      Real A_mixed = 0.0;
      for (int l = 0; l < 3; ++l) {
        A_mixed += geo.inverse(i, l)*u.Atilde(l, k);
      }
      d0_Lambda -= 2.0*A_mixed*u.a(k)
                   + (3.0*u.alpha/u.chi)*A_mixed*u.X(k);
    }
    for (int k = 0; k < 3; ++k) {
      for (int l = 0; l < 3; ++l) {
        d0_Lambda += 2.0*u.alpha*A_up(k, l)*geo.Gamma(i, k, l);
      }
    }
    rhs.Lambda(i) += d0_Lambda;
  }

  Real d0_pi = -u.alpha*A_squared - (u.alpha/3.0)*u.K*u.K
               + u.chi*laplacian_alpha - 0.5*X_dot_a
               - 0.5*kappa*u.alpha*C_perp;
  for (int i = 0; i < 3; ++i) {
    d0_pi += u.chi*geo.c_up(i)*u.a(i);
    rhs.pi += u.beta(i)*d.dpi(i);
  }
  rhs.pi += d0_pi;

  Real f_perp = 0.0;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> f;
  GaugeTargets(u, eta_beta, f_perp, f);
  rhs.h_perp = -mu_H*(u.h_perp - f_perp) + u.vartheta_perp;
  rhs.vartheta_perp = -eta_H*u.vartheta_perp;
  for (int k = 0; k < 3; ++k) {
    rhs.h_perp += u.beta(k)*d.dh_perp(k);
    rhs.vartheta_perp -= eta_H*u.beta(k)*d.dh_perp(k);
  }
  for (int i = 0; i < 3; ++i) {
    rhs.h(i) = -mu_H*(u.h(i) - f(i)) + u.vartheta(i);
    rhs.vartheta(i) = -eta_H*u.vartheta(i);
    for (int k = 0; k < 3; ++k) {
      rhs.h(i) += u.beta(k)*d.dh(k, i);
      rhs.vartheta(i) -= eta_H*u.beta(k)*d.dh(k, i);
    }
  }
}

} // namespace fo_gh

#endif // FO_GH_FO_GH_RHS_HPP_
