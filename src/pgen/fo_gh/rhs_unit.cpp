//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rhs_unit.cpp
//! \brief CPU/GPU pointwise RHS tests for regularized first-order GH.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "fo_gh/fo_gh_rhs.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::FoGhRhsUnit()
//! \brief Test exact Minkowski stationarity and moving-puncture target identities.

void ProblemGenerator::FoGhRhsUnit(ParameterInput *pin, const bool restart) {
  (void)pin;
  (void)restart;
  int errors = 0;
  Kokkos::parallel_reduce(
      "FO-GH RHS unit", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int, int &local_errors) {
        constexpr Real tol = 3.0e-13;
        fo_gh::RegularPointState u;
        fo_gh::EvolutionDerivatives d;
        fo_gh::PrimaryRhs rhs;
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 1.0;
        u.chi = 1.0;
        for (int i = 0; i < 3; ++i) {
          u.gtilde(i, i) = 1.0;
        }
        fo_gh::ComputePrimaryRhs(u, d, 1.0, 1.0, 1.0, 2.0, rhs);
        if (Kokkos::abs(rhs.chi) > tol || Kokkos::abs(rhs.alpha) > tol ||
            Kokkos::abs(rhs.K) > tol || Kokkos::abs(rhs.pi) > tol ||
            Kokkos::abs(rhs.h_perp) > tol ||
            Kokkos::abs(rhs.vartheta_perp) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          if (Kokkos::abs(rhs.beta(i)) > tol ||
              Kokkos::abs(rhs.Lambda(i)) > tol ||
              Kokkos::abs(rhs.h(i)) > tol ||
              Kokkos::abs(rhs.vartheta(i)) > tol) {
            ++local_errors;
          }
          for (int j = 0; j < 3; ++j) {
            if (Kokkos::abs(rhs.gtilde(i, j)) > tol ||
                Kokkos::abs(rhs.Atilde(i, j)) > tol) {
              ++local_errors;
            }
          }
        }

        // The divergence of c^i must vanish when Lambda and its derivative match
        // the contracted Christoffel symbol for a nonconstant conformal metric.
        constexpr Real s = 0.11;
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 1.0;
        u.chi = 1.0;
        for (int i = 0; i < 3; ++i) {
          u.gtilde(i, i) = 1.0;
          u.Q(0, i, i) = 2.0*s;
          d.geometry.dQ[0](0, i, i) = 4.0*s*s;
        }
        u.Lambda(0) = -s;
        d.geometry.dLambda(0, 0) = 2.0*s*s;
        fo_gh::GeometryPoint geo;
        fo_gh::ComputeGeometry(u, d.geometry, geo);
        if (Kokkos::abs(fo_gh::DivergenceC(u, d.geometry, geo)) > tol) {
          ++local_errors;
        }

        // With h=f and zero shift advection, the weight-zero equations must reduce
        // exactly to 1+log lapse and the Gamma-driver target.
        u.ZeroClear();
        d.ZeroClear();
        u.chi = 0.74;
        for (int i = 0; i < 3; ++i) {
          u.gtilde(i, i) = 1.0;
        }
        u.K = 0.23;
        u.pi = -0.07;
        u.alpha = 0.81;
        u.Lambda(0) = 0.13;
        u.Lambda(1) = -0.04;
        u.X(0) = 0.03;
        u.a(0) = -0.02;
        Real f_perp = 0.0;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> f;
        fo_gh::GaugeTargets(u, 2.0, f_perp, f);
        u.h_perp = f_perp;
        for (int i = 0; i < 3; ++i) {
          u.h(i) = f(i);
        }
        fo_gh::ComputePrimaryRhs(u, d, 1.0, 1.0, 1.0, 2.0, rhs);
        if (Kokkos::abs(rhs.alpha + 2.0*u.alpha*u.K) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          const Real expected = 0.75*u.Lambda(i) - 2.0*u.beta(i);
          if (Kokkos::abs(rhs.beta(i) - expected) > tol ||
              Kokkos::abs(rhs.h(i)) > tol ||
              Kokkos::abs(rhs.vartheta(i)) > tol) {
            ++local_errors;
          }
        }

        // The Lambda equation contracts the covectors a_k and X_k with the
        // twice-raised Atilde^{ik}.  A diagonal/identity metric cannot detect
        // an accidental mixed-index Atilde^i_k here, so use a non-diagonal
        // conformal metric and isolate these two terms.
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 0.82;
        u.chi = 0.71;
        u.gtilde(0, 0) = 1.35;
        u.gtilde(0, 1) = 0.16;
        u.gtilde(0, 2) = -0.08;
        u.gtilde(1, 1) = 1.12;
        u.gtilde(1, 2) = 0.06;
        u.gtilde(2, 2) = 0.93;
        u.Atilde(0, 0) = 0.09;
        u.Atilde(0, 1) = -0.04;
        u.Atilde(0, 2) = 0.03;
        u.Atilde(1, 1) = -0.06;
        u.Atilde(1, 2) = 0.05;
        u.Atilde(2, 2) = 0.02;
        u.a(0) = 0.07;
        u.a(1) = -0.03;
        u.a(2) = 0.04;
        u.X(0) = -0.02;
        u.X(1) = 0.06;
        u.X(2) = 0.01;
        fo_gh::ComputePrimaryRhs(u, d, 1.0, 1.0, 1.0, 2.0, rhs);
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
        fo_gh::Invert3(u.gtilde, inverse);
        Real A_squared = 0.0;
        Real X_squared = 0.0;
        Real X_dot_a = 0.0;
        for (int i = 0; i < 3; ++i) {
          Real expected = 0.0;
          for (int k = 0; k < 3; ++k) {
            Real A_up = 0.0;
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                A_up += inverse(i, a)*inverse(k, b)*u.Atilde(a, b);
              }
            }
            A_squared += u.Atilde(i, k)*A_up;
            expected -= 2.0*A_up*u.a(k)
                        + (3.0*u.alpha/u.chi)*A_up*u.X(k);
            X_squared += inverse(i, k)*u.X(i)*u.X(k);
            X_dot_a += inverse(i, k)*u.X(i)*u.a(k);
          }
          if (Kokkos::abs(rhs.Lambda(i) - expected) > tol) {
            ++local_errors;
          }
        }
        const Real hamiltonian = -A_squared
                                 - (5.0/(2.0*u.chi))*X_squared;
        const Real expected_K = u.alpha*A_squared + 0.5*X_dot_a
                                + u.alpha*hamiltonian;
        const Real expected_pi = -u.alpha*A_squared - 0.5*X_dot_a;
        if (Kokkos::abs(rhs.K - expected_K) > tol ||
            Kokkos::abs(rhs.pi - expected_pi) > tol) {
          ++local_errors;
        }

        // Full non-diagonal Atilde/Lambda oracle.  Assemble the fixed equations
        // directly from independently indexed 3x3 work arrays, including both
        // trace-free projections, shift second derivatives, the Lambda vector
        // Lie-index term, and an explicit advection payload.
        u.ZeroClear();
        d.ZeroClear();
        u.alpha = 0.79;
        u.chi = 0.68;
        u.K = 0.13;
        u.pi = -0.08;
        u.gtilde(0, 0) = 1.28;
        u.gtilde(0, 1) = -0.11;
        u.gtilde(0, 2) = 0.09;
        u.gtilde(1, 1) = 1.17;
        u.gtilde(1, 2) = 0.07;
        u.gtilde(2, 2) = 0.88;
        u.Atilde(0, 0) = 0.06;
        u.Atilde(0, 1) = 0.025;
        u.Atilde(0, 2) = -0.035;
        u.Atilde(1, 1) = -0.045;
        u.Atilde(1, 2) = 0.055;
        u.Atilde(2, 2) = 0.015;
        for (int p = 0; p < 3; ++p) {
          u.beta(p) = 0.04*(p + 1) - 0.07;
          u.Lambda(p) = -0.03*(p + 1) + 0.05;
          u.X(p) = 0.018*(p + 1) - 0.027;
          u.a(p) = -0.014*(p + 1) + 0.031;
          d.geometry.dK(p) = 0.012*(p + 1) - 0.019;
          d.dpi(p) = -0.016*(p + 1) + 0.021;
          for (int a = 0; a < 3; ++a) {
            u.B(p, a) = 0.013*(p + 1)*(a + 1) - 0.022*(p + a + 1);
            d.geometry.dX(p, a) = 0.007*(p + a + 2)
                                  - 0.003*(p + 1)*(a + 1);
            d.geometry.da(p, a) = -0.006*(p + a + 2)
                                  + 0.004*(p + 1)*(a + 1);
            d.geometry.dLambda(p, a) = 0.009*(p + 1)*(a + 1)
                                       - 0.005*(p + a + 1);
            for (int b = 0; b < 3; ++b) {
              d.dB(p, a, b) = -0.004*(p + 1)*(a + 2)*(b + 1)
                               + 0.003*(p + a + b + 2);
            }
            for (int b = a; b < 3; ++b) {
              u.Q(p, a, b) = 0.01*(p + 1)*(a + b + 2)
                              - 0.006*(a + 1)*(b + 1);
              d.geometry.dA(p, a, b) = 0.005*(p + 1)*(a + b + 1);
              for (int qd = 0; qd < 3; ++qd) {
                d.geometry.dQ[p](qd, a, b) =
                    0.0015*(p + qd + 2)*(a + b + 2)
                    - 0.0007*(p + 1)*(qd + 1)*(a + 1)*(b + 1);
              }
            }
          }
        }
        fo_gh::EvolutionAdvection advection;
        advection.ZeroClear();
        advection.K = 0.017;
        advection.pi = -0.012;
        advection.h_perp = 0.008;
        for (int i = 0; i < 3; ++i) {
          advection.Lambda(i) = 0.011*(i + 1) - 0.019;
          advection.h(i) = -0.007*(i + 1) + 0.013;
          for (int j = i; j < 3; ++j) {
            advection.Atilde(i, j) = 0.006*(i + j + 1) - 0.009;
          }
        }
        constexpr Real kappa = 0.73;
        fo_gh::ComputePrimaryRhs(u, d, advection, kappa, 1.1, 0.9, 1.7, rhs);
        fo_gh::GeometryPoint full_geo;
        fo_gh::ComputeGeometry(u, d.geometry, full_geo);

        Real Aup_ref[3][3];
        Real curvature_ref[3][3];
        Real c_ref[3][3];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            Aup_ref[i][j] = 0.0;
            curvature_ref[i][j] =
                u.alpha*u.chi*full_geo.Ricci(i, j)
                + 0.5*u.alpha*full_geo.dd_chi(i, j)
                - (u.alpha/(4.0*u.chi))*u.X(i)*u.X(j)
                - u.chi*full_geo.dd_alpha(i, j)
                - 0.5*(u.a(i)*u.X(j) + u.a(j)*u.X(i));
            c_ref[i][j] = -0.5*(full_geo.c_down(i)*u.X(j)
                                 + full_geo.c_down(j)*u.X(i));
            for (int k = 0; k < 3; ++k) {
              c_ref[i][j] -= u.chi*full_geo.c_down(k)*full_geo.Gamma(k, i, j);
              for (int l = 0; l < 3; ++l) {
                Aup_ref[i][j] += full_geo.inverse(i, k)*full_geo.inverse(j, l)
                                     *u.Atilde(k, l);
              }
            }
          }
        }
        Real curvature_trace = 0.0;
        Real c_trace = 0.0;
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            curvature_trace += full_geo.inverse(i, j)*curvature_ref[i][j];
            c_trace += full_geo.inverse(i, j)*c_ref[i][j];
          }
        }
        Real div_beta = 0.0;
        for (int k = 0; k < 3; ++k) div_beta += u.B(k, k);
        const Real C_perp = u.pi + u.K;
        for (int i = 0; i < 3; ++i) {
          for (int j = i; j < 3; ++j) {
            Real expected = advection.Atilde(i, j)
                + curvature_ref[i][j] - (curvature_trace/3.0)*u.gtilde(i, j)
                + u.alpha*(c_ref[i][j] - (c_trace/3.0)*u.gtilde(i, j))
                - (2.0/3.0)*u.Atilde(i, j)*div_beta
                + u.alpha*(u.K - C_perp)*u.Atilde(i, j);
            for (int k = 0; k < 3; ++k) {
              expected += u.Atilde(i, k)*u.B(j, k)
                          + u.Atilde(j, k)*u.B(i, k);
              for (int l = 0; l < 3; ++l) {
                expected -= 2.0*u.alpha*u.Atilde(i, k)
                            *full_geo.inverse(k, l)*u.Atilde(l, j);
              }
            }
            if (Kokkos::abs(rhs.Atilde(i, j) - expected) > 2.0e-12) {
              ++local_errors;
            }
          }

          Real expected_lambda = advection.Lambda(i)
              + (2.0/3.0)*u.Lambda(i)*div_beta
              + ((2.0/3.0)*u.alpha*u.K + kappa*u.alpha)*full_geo.c_up(i);
          for (int k = 0; k < 3; ++k) {
            expected_lambda -= u.Lambda(k)*u.B(k, i)
                               + 2.0*Aup_ref[i][k]*u.a(k)
                               + (3.0*u.alpha/u.chi)*Aup_ref[i][k]*u.X(k);
            for (int l = 0; l < 3; ++l) {
              expected_lambda += full_geo.inverse(k, l)*d.dB(k, l, i)
                                 + 2.0*u.alpha*Aup_ref[k][l]
                                       *full_geo.Gamma(i, k, l);
            }
          }
          for (int j = 0; j < 3; ++j) {
            Real d_div_beta = 0.0;
            for (int k = 0; k < 3; ++k) d_div_beta += d.dB(j, k, k);
            expected_lambda += (1.0/3.0)*full_geo.inverse(i, j)*d_div_beta
                               - (4.0/3.0)*u.alpha*full_geo.inverse(i, j)
                                     *d.geometry.dK(j)
                               + u.alpha*full_geo.inverse(i, j)
                                     *(d.dpi(j) + d.geometry.dK(j));
          }
          if (Kokkos::abs(rhs.Lambda(i) - expected_lambda) > 2.0e-12) {
            ++local_errors;
          }
        }
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH RHS unit test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH RHS unit test passed." << std::endl;
}
