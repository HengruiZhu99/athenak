#ifndef FO_GH_FO_GH_STATE_HPP_
#define FO_GH_FO_GH_STATE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh_state.hpp
//! \brief State ordering and pointwise algebra for regularized first-order GH.

#include <cmath>

#include "athena.hpp"
#include "athena_tensor.hpp"

namespace fo_gh {

// The production state contains only the everywhere-regular variables. Physical
// g_ab, Pi_ab, and Phi_iab are reconstructed diagnostics and are not listed here.
enum StateIndex {
  I_TGXX = 0, I_TGXY, I_TGXZ, I_TGYY, I_TGYZ, I_TGZZ,
  I_CHI,
  I_ALPHA,
  I_BETAX, I_BETAY, I_BETAZ,
  I_TAXX, I_TAXY, I_TAXZ, I_TAYY, I_TAYZ, I_TAZZ,
  I_K,
  I_LAMBDAX, I_LAMBDAY, I_LAMBDAZ,
  I_PI,
  I_QXXX, I_QXXY, I_QXXZ, I_QXYY, I_QXYZ, I_QXZZ,
  I_QYXX, I_QYXY, I_QYXZ, I_QYYY, I_QYYZ, I_QYZZ,
  I_QZXX, I_QZXY, I_QZXZ, I_QZYY, I_QZYZ, I_QZZZ,
  I_XX, I_XY, I_XZ,
  I_AX, I_AY, I_AZ,
  I_BXX, I_BXY, I_BXZ,
  I_BYX, I_BYY, I_BYZ,
  I_BZX, I_BZY, I_BZZ,
  I_H_PERP,
  I_HX, I_HY, I_HZ,
  I_VARTHETA_PERP,
  I_VARTHETAX, I_VARTHETAY, I_VARTHETAZ,
  nvar
};

static_assert(nvar == 63, "The regularized FO-GH production state must have 63 fields.");

struct RegularPointState {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde;
  Real chi;
  Real alpha;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> beta;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Atilde;
  Real K;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Lambda;
  Real pi;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Q;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> X;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> a;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> B;
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
    Q.ZeroClear();
    X.ZeroClear();
    a.ZeroClear();
    B.ZeroClear();
    h_perp = 0.0;
    h.ZeroClear();
    vartheta_perp = 0.0;
    vartheta.ZeroClear();
  }
};

struct AdmPointState {
  Real alpha;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> beta;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> K;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dgamma;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta;

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    alpha = 0.0;
    beta.ZeroClear();
    gamma.ZeroClear();
    K.ZeroClear();
    dgamma.ZeroClear();
    dalpha.ZeroClear();
    dbeta.ZeroClear();
  }
};

struct StandardGhPointState {
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> g;
  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> Pi;
  MixedTensor<Real, 3, 4> Phi;
};

KOKKOS_INLINE_FUNCTION
Real Determinant3(const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &g) {
  return g(0, 0)*(g(1, 1)*g(2, 2) - g(1, 2)*g(1, 2))
       - g(0, 1)*(g(0, 1)*g(2, 2) - g(0, 2)*g(1, 2))
       + g(0, 2)*(g(0, 1)*g(1, 2) - g(0, 2)*g(1, 1));
}

KOKKOS_INLINE_FUNCTION
Real Invert3(const AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &g,
             AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> &inverse) {
  const Real det = Determinant3(g);
  const Real invdet = 1.0/det;
  inverse(0, 0) = (g(1, 1)*g(2, 2) - g(1, 2)*g(1, 2))*invdet;
  inverse(0, 1) = (g(0, 2)*g(1, 2) - g(0, 1)*g(2, 2))*invdet;
  inverse(0, 2) = (g(0, 1)*g(1, 2) - g(0, 2)*g(1, 1))*invdet;
  inverse(1, 1) = (g(0, 0)*g(2, 2) - g(0, 2)*g(0, 2))*invdet;
  inverse(1, 2) = (g(0, 1)*g(0, 2) - g(0, 0)*g(1, 2))*invdet;
  inverse(2, 2) = (g(0, 0)*g(1, 1) - g(0, 1)*g(0, 1))*invdet;
  return det;
}

KOKKOS_INLINE_FUNCTION
void GaugeTargets(const RegularPointState &u, const Real eta_beta,
                  Real &f_perp,
                  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> &f) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde_inv;
  Invert3(u.gtilde, gtilde_inv);
  f_perp = u.alpha*u.pi + 2.0*u.K;
  for (int i = 0; i < 3; ++i) {
    Real grad_term = 0.0;
    for (int j = 0; j < 3; ++j) {
      grad_term += gtilde_inv(i, j)*(-0.5*u.alpha*u.alpha*u.X(j)
                                    + u.alpha*u.chi*u.a(j));
    }
    f(i) = (0.75 - u.alpha*u.alpha*u.chi)*u.Lambda(i)
           + grad_term - eta_beta*u.beta(i);
  }
}

// Convert ADM data and its first spatial derivatives to a constraint-satisfying regular
// state. The GH constraints are initialized with pi=-K and Lambda equal to the contracted
// conformal Christoffel symbol. Driver fields start on the moving-puncture target.
KOKKOS_INLINE_FUNCTION
void AdmToRegular(const AdmPointState &adm, const Real eta_beta,
                  RegularPointState &u) {
  u.ZeroClear();
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_inv;
  const Real det_gamma = Invert3(adm.gamma, gamma_inv);
  u.chi = std::pow(det_gamma, -1.0/3.0);
  u.alpha = adm.alpha;

  u.K = 0.0;
  for (int i = 0; i < 3; ++i) {
    u.beta(i) = adm.beta(i);
    u.a(i) = adm.dalpha(i);
    for (int j = 0; j < 3; ++j) {
      u.K += gamma_inv(i, j)*adm.K(i, j);
      u.B(i, j) = adm.dbeta(i, j);
    }
  }
  u.pi = -u.K;

  for (int k = 0; k < 3; ++k) {
    Real gamma_trace = 0.0;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        gamma_trace += gamma_inv(i, j)*adm.dgamma(k, i, j);
      }
    }
    u.X(k) = -(u.chi/3.0)*gamma_trace;
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      u.gtilde(i, j) = u.chi*adm.gamma(i, j);
      u.Atilde(i, j) = u.chi*(adm.K(i, j) - adm.gamma(i, j)*u.K/3.0);
      for (int k = 0; k < 3; ++k) {
        u.Q(k, i, j) = u.X(k)*adm.gamma(i, j) + u.chi*adm.dgamma(k, i, j);
      }
    }
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde_inv;
  Invert3(u.gtilde, gtilde_inv);
  for (int i = 0; i < 3; ++i) {
    u.Lambda(i) = 0.0;
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          const Real christoffel = 0.5*gtilde_inv(i, l)
              *(u.Q(j, l, k) + u.Q(k, l, j) - u.Q(l, j, k));
          u.Lambda(i) += gtilde_inv(j, k)*christoffel;
        }
      }
    }
  }

  GaugeTargets(u, eta_beta, u.h_perp, u.h);
}

KOKKOS_INLINE_FUNCTION
void RegularToAdm(const RegularPointState &u, AdmPointState &adm) {
  adm.alpha = u.alpha;
  for (int i = 0; i < 3; ++i) {
    adm.beta(i) = u.beta(i);
    for (int j = i; j < 3; ++j) {
      adm.gamma(i, j) = u.gtilde(i, j)/u.chi;
      adm.K(i, j) = (u.Atilde(i, j) + u.gtilde(i, j)*u.K/3.0)/u.chi;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void RegularToStandardGh(const RegularPointState &u, StandardGhPointState &gh) {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Kij;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> G;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> d0gamma;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> d0beta;

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      gamma(i, j) = u.gtilde(i, j)/u.chi;
      Kij(i, j) = (u.Atilde(i, j) + u.gtilde(i, j)*u.K/3.0)/u.chi;
    }
  }
  // Populate the complete symmetric metric before contracting it below.  Computing
  // d0gamma in the reconstruction loop above would read not-yet-initialized
  // off-diagonal/diagonal entries for general non-diagonal data.
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      d0gamma(i, j) = -2.0*u.alpha*Kij(i, j);
      for (int k = 0; k < 3; ++k) {
        d0gamma(i, j) += gamma(i, k)*u.B(j, k) + gamma(j, k)*u.B(i, k);
      }
    }
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde_inv;
  Invert3(u.gtilde, gtilde_inv);
  for (int i = 0; i < 3; ++i) {
    Real gradient = 0.0;
    for (int j = 0; j < 3; ++j) {
      gradient += gtilde_inv(i, j)
          *(0.5*u.alpha*u.alpha*u.X(j) - u.alpha*u.chi*u.a(j));
    }
    d0beta(i) = u.h(i) + u.alpha*u.alpha*u.chi*u.Lambda(i) + gradient;
  }
  const Real d0alpha = u.alpha*u.alpha*u.pi - u.alpha*u.h_perp;

  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        G(k, i, j) = u.Q(k, i, j)/u.chi
                     - u.X(k)*u.gtilde(i, j)/(u.chi*u.chi);
      }
    }
  }

  gh.g.ZeroClear();
  gh.Pi.ZeroClear();
  gh.Phi.ZeroClear();
  Real beta_lower[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      beta_lower[i] += gamma(i, j)*u.beta(j);
    }
    gh.g(0, i + 1) = beta_lower[i];
    for (int j = i; j < 3; ++j) {
      gh.g(i + 1, j + 1) = gamma(i, j);
    }
  }
  gh.g(0, 0) = -u.alpha*u.alpha;
  for (int i = 0; i < 3; ++i) {
    gh.g(0, 0) += beta_lower[i]*u.beta(i);
  }

  AthenaPointTensor<Real, TensorSymm::SYM2, 4, 2> d0g;
  d0g.ZeroClear();
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      d0g(i + 1, j + 1) = d0gamma(i, j);
    }
    for (int j = 0; j < 3; ++j) {
      d0g(0, i + 1) += d0gamma(i, j)*u.beta(j) + gamma(i, j)*d0beta(j);
    }
  }
  d0g(0, 0) = -2.0*u.alpha*d0alpha;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      d0g(0, 0) += d0gamma(i, j)*u.beta(i)*u.beta(j)
                   + 2.0*gamma(i, j)*u.beta(i)*d0beta(j);
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      gh.Pi(a, b) = -d0g(a, b)/u.alpha;
    }
  }

  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        gh.Phi(k, i + 1, j + 1) = G(k, i, j);
      }
      for (int j = 0; j < 3; ++j) {
        gh.Phi(k, 0, i + 1) += G(k, i, j)*u.beta(j)
                               + gamma(i, j)*u.B(k, j);
      }
    }
    gh.Phi(k, 0, 0) = -2.0*u.alpha*u.a(k);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        gh.Phi(k, 0, 0) += G(k, i, j)*u.beta(i)*u.beta(j)
                            + 2.0*gamma(i, j)*u.beta(i)*u.B(k, j);
      }
    }
  }
}

} // namespace fo_gh

#endif // FO_GH_FO_GH_STATE_HPP_
