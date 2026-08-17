//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file geometry_unit.cpp
//! \brief CPU/GPU conformal-geometry tests for regularized first-order GH.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "fo_gh/fo_gh_rhs.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::FoGhGeometryUnit()
//! \brief Test conformal Ricci and vacuum constraints on the configured device.

void ProblemGenerator::FoGhGeometryUnit(ParameterInput *pin, const bool restart) {
  (void)pin;
  (void)restart;
  int errors = 0;
  Kokkos::parallel_reduce(
      "FO-GH geometry unit", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int, int &local_errors) {
        constexpr Real tol = 3.0e-13;
        fo_gh::RegularPointState u;
        fo_gh::GeometryDerivatives d;
        fo_gh::GeometryPoint geo;

        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 1.0;
        u.chi = 1.0;
        for (int i = 0; i < 3; ++i) {
          u.gtilde(i, i) = 1.0;
        }
        fo_gh::ComputeGeometry(u, d, geo);
        if (Kokkos::abs(geo.hamiltonian) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          if (Kokkos::abs(geo.c_up(i)) > tol ||
              Kokkos::abs(geo.momentum(i)) > tol) {
            ++local_errors;
          }
          for (int j = 0; j < 3; ++j) {
            if (Kokkos::abs(geo.Ricci(i, j)) > tol) {
              ++local_errors;
            }
          }
        }

        // gtilde_ij = exp(q*x^2) delta_ij at x=0. Here f=q*x^2/2 in
        // gtilde=exp(2f)delta, so R_xx=-2q and R_yy=R_zz=-q.
        constexpr Real q = 0.17;
        d.ZeroClear();
        for (int i = 0; i < 3; ++i) {
          d.dQ[0](0, i, i) = 2.0*q;
        }
        d.dLambda(0, 0) = -q;
        fo_gh::ComputeGeometry(u, d, geo);
        if (Kokkos::abs(geo.Ricci(0, 0) + 2.0*q) > tol ||
            Kokkos::abs(geo.Ricci(1, 1) + q) > tol ||
            Kokkos::abs(geo.Ricci(2, 2) + q) > tol ||
            Kokkos::abs(geo.hamiltonian + 4.0*q) > tol) {
          ++local_errors;
        }

        // gtilde_ij = exp(2*s*x) delta_ij at x=0 exercises the quadratic
        // Christoffel terms: R_xx=0 and R_yy=R_zz=-s^2.
        constexpr Real s = 0.11;
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 1.0;
        u.chi = 1.0;
        for (int i = 0; i < 3; ++i) {
          u.gtilde(i, i) = 1.0;
          u.Q(0, i, i) = 2.0*s;
          d.dQ[0](0, i, i) = 4.0*s*s;
        }
        u.Lambda(0) = -s;
        d.dLambda(0, 0) = 2.0*s*s;
        fo_gh::ComputeGeometry(u, d, geo);
        if (Kokkos::abs(geo.c_up(0)) > tol ||
            Kokkos::abs(geo.Ricci(0, 0)) > tol ||
            Kokkos::abs(geo.Ricci(1, 1) + s*s) > tol ||
            Kokkos::abs(geo.Ricci(2, 2) + s*s) > tol) {
          ++local_errors;
        }

        // Direct momentum-constraint check in flat conformal geometry.
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 1.0;
        u.chi = 1.0;
        for (int i = 0; i < 3; ++i) {
          u.gtilde(i, i) = 1.0;
        }
        u.Atilde(0, 0) = 0.2;
        u.X(0) = 0.3;
        d.dK(0) = 0.6;
        fo_gh::ComputeGeometry(u, d, geo);
        if (Kokkos::abs(geo.momentum(0) + 0.49) > tol ||
            Kokkos::abs(geo.momentum(1)) > tol ||
            Kokkos::abs(geo.momentum(2)) > tol) {
          ++local_errors;
        }

        // Independent non-diagonal metric-jet oracle.  Build Christoffels and
        // Ricci directly from
        //   R_ij = d_k Gamma^k_ij - d_j Gamma^k_ik
        //          + Gamma^k_ij Gamma^l_kl - Gamma^k_il Gamma^l_jk,
        // rather than reusing the conformal-Ricci rearrangement implemented by
        // ComputeGeometry.  The second metric derivatives are symmetric in
        // their two derivative indices, as required for a compatible smooth
        // local jet.
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 0.91;
        u.chi = 0.83;
        u.K = -0.14;
        u.gtilde(0, 0) = 1.40;
        u.gtilde(0, 1) = 0.12;
        u.gtilde(0, 2) = -0.07;
        u.gtilde(1, 1) = 1.10;
        u.gtilde(1, 2) = 0.05;
        u.gtilde(2, 2) = 0.90;
        for (int p = 0; p < 3; ++p) {
          u.X(p) = 0.021*(p + 1) - 0.013;
          d.dK(p) = -0.017*(p + 1) + 0.006;
          for (int i = 0; i < 3; ++i) {
            d.dX(p, i) = 0.009*(p + 1)*(i + 1) - 0.004*(p + i + 1);
            for (int j = i; j < 3; ++j) {
              u.Q(p, i, j) = 0.011*(p + 1)*(i + j + 2)
                              - 0.007*(i + 1)*(j + 1);
              d.dA(p, i, j) = 0.008*(p + 1)*(i + j + 1)
                               - 0.003*(i + 1)*(j + 1);
              for (int qd = 0; qd < 3; ++qd) {
                d.dQ[p](qd, i, j) =
                    0.002*(p + qd + 2)*(i + j + 2)
                    - 0.001*(p + 1)*(qd + 1)*(i + 1)*(j + 1);
              }
            }
          }
        }
        u.Atilde(0, 0) = 0.07;
        u.Atilde(0, 1) = -0.03;
        u.Atilde(0, 2) = 0.02;
        u.Atilde(1, 1) = -0.05;
        u.Atilde(1, 2) = 0.04;
        u.Atilde(2, 2) = 0.01;

        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
        fo_gh::Invert3(u.gtilde, inverse);
        Real gamma_ref[3][3][3];
        Real dgamma_ref[3][3][3][3];
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            for (int c = 0; c < 3; ++c) {
              gamma_ref[a][b][c] = 0.0;
              for (int p = 0; p < 3; ++p) {
                dgamma_ref[p][a][b][c] = 0.0;
              }
              for (int l = 0; l < 3; ++l) {
                const Real metric_sum = u.Q(b, l, c) + u.Q(c, l, b)
                                        - u.Q(l, b, c);
                gamma_ref[a][b][c] += 0.5*inverse(a, l)*metric_sum;
                for (int p = 0; p < 3; ++p) {
                  Real d_inverse = 0.0;
                  for (int r = 0; r < 3; ++r) {
                    for (int s2 = 0; s2 < 3; ++s2) {
                      d_inverse -= inverse(a, r)*inverse(l, s2)*u.Q(p, r, s2);
                    }
                  }
                  const Real d_metric_sum = d.dQ[p](b, l, c)
                                            + d.dQ[p](c, l, b)
                                            - d.dQ[p](l, b, c);
                  dgamma_ref[p][a][b][c] +=
                      0.5*(d_inverse*metric_sum + inverse(a, l)*d_metric_sum);
                }
              }
            }
          }
        }

        // Set Lambda and dLambda to the independently differentiated
        // contracted Christoffel symbol, making c^i and D_i c^i exactly zero.
        for (int a = 0; a < 3; ++a) {
          u.Lambda(a) = 0.0;
          for (int p = 0; p < 3; ++p) {
            d.dLambda(p, a) = 0.0;
          }
          for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
              u.Lambda(a) += inverse(j, k)*gamma_ref[a][j][k];
              for (int p = 0; p < 3; ++p) {
                Real d_inverse = 0.0;
                for (int r = 0; r < 3; ++r) {
                  for (int s2 = 0; s2 < 3; ++s2) {
                    d_inverse -= inverse(j, r)*inverse(k, s2)*u.Q(p, r, s2);
                  }
                }
                d.dLambda(p, a) += d_inverse*gamma_ref[a][j][k]
                                          + inverse(j, k)*dgamma_ref[p][a][j][k];
              }
            }
          }
        }

        fo_gh::ComputeGeometry(u, d, geo);
        Real ricci_ref[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            ricci_ref[i][j] = 0.0;
            for (int k = 0; k < 3; ++k) {
              ricci_ref[i][j] += dgamma_ref[k][k][i][j]
                                 - dgamma_ref[j][k][i][k];
              for (int l = 0; l < 3; ++l) {
                ricci_ref[i][j] += gamma_ref[k][i][j]*gamma_ref[l][k][l]
                                   - gamma_ref[k][i][l]*gamma_ref[l][j][k];
              }
            }
            if (Kokkos::abs(geo.Ricci(i, j) - ricci_ref[i][j]) > 8.0e-13) {
              ++local_errors;
            }
          }
          if (Kokkos::abs(geo.c_up(i)) > 8.0e-13) {
            ++local_errors;
          }
        }
        if (Kokkos::abs(fo_gh::DivergenceC(u, d, geo)) > 2.0e-12) {
          ++local_errors;
        }

        Real ricci_scalar_ref = 0.0;
        Real laplacian_chi_ref = 0.0;
        Real x_squared_ref = 0.0;
        Real a_squared_ref = 0.0;
        Real momentum_ref[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            ricci_scalar_ref += inverse(i, j)*ricci_ref[i][j];
            Real dd_chi_ref = d.dX(i, j);
            for (int k = 0; k < 3; ++k) {
              dd_chi_ref -= gamma_ref[k][i][j]*u.X(k);
            }
            laplacian_chi_ref += inverse(i, j)*dd_chi_ref;
            x_squared_ref += inverse(i, j)*u.X(i)*u.X(j);
            for (int k = 0; k < 3; ++k) {
              for (int l = 0; l < 3; ++l) {
                a_squared_ref += inverse(i, k)*inverse(j, l)
                                 *u.Atilde(i, j)*u.Atilde(k, l);
              }
              Real covariant_dA = d.dA(j, k, i);
              for (int m = 0; m < 3; ++m) {
                covariant_dA -= gamma_ref[m][j][k]*u.Atilde(m, i)
                                + gamma_ref[m][j][i]*u.Atilde(k, m);
              }
              momentum_ref[i] += inverse(j, k)*covariant_dA;
            }
          }
          momentum_ref[i] -= (2.0/3.0)*d.dK(i);
          for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
              momentum_ref[i] -= (3.0/(2.0*u.chi))*inverse(j, k)
                                 *u.Atilde(k, i)*u.X(j);
            }
          }
          if (Kokkos::abs(geo.momentum(i) - momentum_ref[i]) > 8.0e-13) {
            ++local_errors;
          }
        }
        const Real hamiltonian_ref = (2.0/3.0)*u.K*u.K - a_squared_ref
                                     + u.chi*ricci_scalar_ref
                                     + 2.0*laplacian_chi_ref
                                     - (5.0/(2.0*u.chi))*x_squared_ref;
        if (Kokkos::abs(geo.hamiltonian - hamiltonian_ref) > 8.0e-13) {
          ++local_errors;
        }
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH geometry unit test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH geometry unit test passed." << std::endl;
}
