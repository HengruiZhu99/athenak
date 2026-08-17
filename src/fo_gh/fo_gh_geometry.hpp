#ifndef FO_GH_FO_GH_GEOMETRY_HPP_
#define FO_GH_FO_GH_GEOMETRY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh_geometry.hpp
//! \brief Pointwise conformal geometry and vacuum constraints for regularized FO-GH.

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "fo_gh/fo_gh_state.hpp"

namespace fo_gh {

//! First spatial derivatives required by conformal geometry and constraints.
//! dQ[k](l,i,j) means partial_k Q_lij; all other rank-2 derivatives use the
//! convention derivative(first index, field index).
struct GeometryDerivatives {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dQ[3]; // NOLINT(runtime/arrays)
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dX;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> da;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dA;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dK;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dLambda;

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int k = 0; k < 3; ++k) {
      dQ[k].ZeroClear();
    }
    dX.ZeroClear();
    da.ZeroClear();
    dA.ZeroClear();
    dK.ZeroClear();
    dLambda.ZeroClear();
  }
};

struct GeometryPoint {
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 3> Gamma;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> contracted_Gamma;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> c_up;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> c_down;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ricci;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dd_chi;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dd_alpha;
  Real Ricci_scalar;
  Real hamiltonian;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> momentum;
};

KOKKOS_INLINE_FUNCTION
Real LoweredGamma(const RegularPointState &u, const GeometryPoint &geo,
                  const int i, const int j, const int k) {
  Real value = 0.0;
  for (int l = 0; l < 3; ++l) {
    value += u.gtilde(i, l)*geo.Gamma(l, j, k);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
void ComputeGeometry(const RegularPointState &u, const GeometryDerivatives &d,
                     GeometryPoint &geo) {
  Invert3(u.gtilde, geo.inverse);
  geo.Gamma.ZeroClear();
  geo.contracted_Gamma.ZeroClear();
  geo.c_up.ZeroClear();
  geo.c_down.ZeroClear();
  geo.Ricci.ZeroClear();
  geo.dd_chi.ZeroClear();
  geo.dd_alpha.ZeroClear();
  geo.momentum.ZeroClear();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          geo.Gamma(i, j, k) += 0.5*geo.inverse(i, l)
              *(u.Q(j, l, k) + u.Q(k, l, j) - u.Q(l, j, k));
        }
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        geo.contracted_Gamma(i) += geo.inverse(j, k)*geo.Gamma(i, j, k);
      }
    }
    geo.c_up(i) = -u.Lambda(i) + geo.contracted_Gamma(i);
    for (int j = 0; j < 3; ++j) {
      geo.c_down(i) += u.gtilde(i, j)*geo.c_up(j);
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      geo.dd_chi(i, j) = d.dX(i, j);
      geo.dd_alpha(i, j) = d.da(i, j);
      for (int k = 0; k < 3; ++k) {
        geo.dd_chi(i, j) -= geo.Gamma(k, i, j)*u.X(k);
        geo.dd_alpha(i, j) -= geo.Gamma(k, i, j)*u.a(k);
      }
    }
  }

  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          geo.Ricci(i, j) -= 0.5*geo.inverse(k, l)*d.dQ[k](l, i, j);
          for (int m = 0; m < 3; ++m) {
            const Real quadratic =
                0.5*geo.Gamma(m, k, l)
                    *(LoweredGamma(u, geo, i, j, m)
                      + LoweredGamma(u, geo, j, i, m))
                + geo.Gamma(m, k, i)*LoweredGamma(u, geo, j, m, l)
                + geo.Gamma(m, k, j)*LoweredGamma(u, geo, i, m, l)
                + geo.Gamma(m, i, k)*LoweredGamma(u, geo, m, j, l);
            geo.Ricci(i, j) += geo.inverse(k, l)*quadratic;
          }
        }
        geo.Ricci(i, j) += 0.5*(u.gtilde(k, i)*d.dLambda(j, k)
                                + u.gtilde(k, j)*d.dLambda(i, k));
      }
    }
  }

  geo.Ricci_scalar = 0.0;
  Real A_squared = 0.0;
  Real laplacian_chi = 0.0;
  Real X_squared = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      geo.Ricci_scalar += geo.inverse(i, j)*geo.Ricci(i, j);
      laplacian_chi += geo.inverse(i, j)*geo.dd_chi(i, j);
      X_squared += geo.inverse(i, j)*u.X(i)*u.X(j);
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          A_squared += geo.inverse(i, k)*geo.inverse(j, l)
                       *u.Atilde(i, j)*u.Atilde(k, l);
        }
      }
    }
  }
  geo.hamiltonian = (2.0/3.0)*u.K*u.K - A_squared
                    + u.chi*geo.Ricci_scalar + 2.0*laplacian_chi
                    - (5.0/(2.0*u.chi))*X_squared;

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        Real covariant_dA = d.dA(j, k, i);
        for (int m = 0; m < 3; ++m) {
          covariant_dA -= geo.Gamma(m, j, k)*u.Atilde(m, i)
                          + geo.Gamma(m, j, i)*u.Atilde(k, m);
        }
        geo.momentum(i) += geo.inverse(j, k)*covariant_dA;
      }
    }
    geo.momentum(i) -= (2.0/3.0)*d.dK(i);
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        geo.momentum(i) -= (3.0/(2.0*u.chi))*geo.inverse(j, k)
                           *u.Atilde(k, i)*u.X(j);
      }
    }
  }
}

} // namespace fo_gh

#endif // FO_GH_FO_GH_GEOMETRY_HPP_
