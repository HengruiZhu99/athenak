//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file algebra_unit.cpp
//! \brief CPU/GPU pointwise algebra tests for regularized first-order GH.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "fo_gh/fo_gh_state.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::FoGhAlgebraUnit()
//! \brief Test metric inversion and ADM/regular/standard-GH maps on the device.

void ProblemGenerator::FoGhAlgebraUnit(ParameterInput *pin, const bool restart) {
  (void)pin;
  (void)restart;
  int errors = 0;
  Kokkos::parallel_reduce(
      "FO-GH algebra unit", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int, int &local_errors) {
        constexpr Real tol = 2.0e-13;

        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> matrix;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
        matrix(0, 0) = 2.0;
        matrix(0, 1) = 0.2;
        matrix(0, 2) = 0.1;
        matrix(1, 1) = 1.5;
        matrix(1, 2) = 0.3;
        matrix(2, 2) = 1.2;
        fo_gh::Invert3(matrix, inverse);
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            Real product = 0.0;
            for (int k = 0; k < 3; ++k) {
              product += matrix(i, k)*inverse(k, j);
            }
            const Real expected = (i == j ? 1.0 : 0.0);
            if (Kokkos::abs(product - expected) > tol) {
              ++local_errors;
            }
          }
        }

        fo_gh::AdmPointState adm;
        adm.ZeroClear();
        adm.alpha = 1.0;
        for (int i = 0; i < 3; ++i) {
          adm.gamma(i, i) = 1.0;
        }
        fo_gh::RegularPointState regular;
        fo_gh::AdmToRegular(adm, 2.0, regular);
        fo_gh::StandardGhPointState gh;
        fo_gh::RegularToStandardGh(regular, gh);
        if (Kokkos::abs(regular.chi - 1.0) > tol ||
            Kokkos::abs(gh.g(0, 0) + 1.0) > tol) {
          ++local_errors;
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            const Real expected = (a == b ? (a == 0 ? -1.0 : 1.0) : 0.0);
            if (Kokkos::abs(gh.g(a, b) - expected) > tol ||
                Kokkos::abs(gh.Pi(a, b)) > tol) {
              ++local_errors;
            }
          }
        }
        for (int k = 0; k < 3; ++k) {
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              if (Kokkos::abs(gh.Phi(k, a, b)) > tol) {
                ++local_errors;
              }
            }
          }
        }

        adm.ZeroClear();
        adm.alpha = 0.8;
        adm.dalpha(0) = 0.02;
        for (int i = 0; i < 3; ++i) {
          adm.gamma(i, i) = 4.0;
          adm.K(i, i) = 0.4;
          adm.dgamma(0, i, i) = 0.4;
        }
        fo_gh::AdmToRegular(adm, 2.0, regular);
        if (Kokkos::abs(regular.chi - 0.25) > tol ||
            Kokkos::abs(regular.K - 0.3) > tol ||
            Kokkos::abs(regular.pi + 0.3) > tol ||
            Kokkos::abs(regular.X(0) + 0.025) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          if (Kokkos::abs(regular.gtilde(i, i) - 1.0) > tol ||
              Kokkos::abs(regular.Atilde(i, i)) > tol ||
              Kokkos::abs(regular.Q(0, i, i)) > tol) {
            ++local_errors;
          }
        }

        fo_gh::RegularToStandardGh(regular, gh);
        if (Kokkos::abs(gh.Phi(0, 0, 0) + 0.032) > tol ||
            Kokkos::abs(gh.Pi(0, 0) + 0.96) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          if (Kokkos::abs(gh.g(i + 1, i + 1) - 4.0) > tol ||
              Kokkos::abs(gh.Pi(i + 1, i + 1) - 0.8) > tol) {
            ++local_errors;
          }
        }

        Real f_perp = 0.0;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> f;
        fo_gh::GaugeTargets(regular, 2.0, f_perp, f);
        if (Kokkos::abs(f_perp - regular.h_perp) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          if (Kokkos::abs(f(i) - regular.h(i)) > tol) {
            ++local_errors;
          }
        }

        // Non-diagonal ADM-jet oracle for the complete regular and standard-GH
        // maps.  In particular, exercise all mixed spacetime Phi components
        // and the shift-dependent Pi reconstruction, which diagonal static
        // data cannot constrain.
        adm.ZeroClear();
        adm.alpha = 0.84;
        adm.gamma(0, 0) = 1.43;
        adm.gamma(0, 1) = 0.18;
        adm.gamma(0, 2) = -0.09;
        adm.gamma(1, 1) = 1.21;
        adm.gamma(1, 2) = 0.08;
        adm.gamma(2, 2) = 0.96;
        adm.K(0, 0) = 0.07;
        adm.K(0, 1) = -0.025;
        adm.K(0, 2) = 0.035;
        adm.K(1, 1) = -0.045;
        adm.K(1, 2) = 0.055;
        adm.K(2, 2) = 0.015;
        for (int p = 0; p < 3; ++p) {
          adm.beta(p) = 0.06*(p + 1) - 0.10;
          adm.dalpha(p) = -0.017*(p + 1) + 0.029;
          for (int a = 0; a < 3; ++a) {
            adm.dbeta(p, a) = 0.012*(p + 1)*(a + 1)
                               - 0.019*(p + a + 1);
            for (int b = a; b < 3; ++b) {
              adm.dgamma(p, a, b) = 0.014*(p + 1)*(a + b + 2)
                                     - 0.008*(a + 1)*(b + 1);
            }
          }
        }
        constexpr Real eta_beta = 1.6;
        fo_gh::AdmToRegular(adm, eta_beta, regular);

        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gamma_inverse;
        const Real det_gamma = fo_gh::Invert3(adm.gamma, gamma_inverse);
        const Real chi_ref = std::pow(det_gamma, -1.0/3.0);
        Real K_trace_ref = 0.0;
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            K_trace_ref += gamma_inverse(i, j)*adm.K(i, j);
          }
        }
        if (Kokkos::abs(regular.chi - chi_ref) > tol ||
            Kokkos::abs(regular.K - K_trace_ref) > tol ||
            Kokkos::abs(regular.pi + K_trace_ref) > tol) {
          ++local_errors;
        }
        for (int p = 0; p < 3; ++p) {
          Real metric_trace = 0.0;
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              metric_trace += gamma_inverse(i, j)*adm.dgamma(p, i, j);
            }
          }
          const Real X_ref = -(chi_ref/3.0)*metric_trace;
          if (Kokkos::abs(regular.X(p) - X_ref) > tol ||
              Kokkos::abs(regular.a(p) - adm.dalpha(p)) > tol) {
            ++local_errors;
          }
          for (int a = 0; a < 3; ++a) {
            if (Kokkos::abs(regular.B(p, a) - adm.dbeta(p, a)) > tol) {
              ++local_errors;
            }
            for (int b = a; b < 3; ++b) {
              const Real gtilde_ref = chi_ref*adm.gamma(a, b);
              const Real Atilde_ref = chi_ref
                  *(adm.K(a, b) - adm.gamma(a, b)*K_trace_ref/3.0);
              const Real Q_ref = X_ref*adm.gamma(a, b)
                                 + chi_ref*adm.dgamma(p, a, b);
              if (Kokkos::abs(regular.gtilde(a, b) - gtilde_ref) > tol ||
                  Kokkos::abs(regular.Atilde(a, b) - Atilde_ref) > tol ||
                  Kokkos::abs(regular.Q(p, a, b) - Q_ref) > tol) {
                ++local_errors;
              }
            }
          }
        }

        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> gtilde_inverse;
        fo_gh::Invert3(regular.gtilde, gtilde_inverse);
        for (int a = 0; a < 3; ++a) {
          Real lambda_ref = 0.0;
          for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
              Real gamma_tilde = 0.0;
              for (int l = 0; l < 3; ++l) {
                gamma_tilde += 0.5*gtilde_inverse(a, l)
                    *(regular.Q(j, l, k) + regular.Q(k, l, j)
                      - regular.Q(l, j, k));
              }
              lambda_ref += gtilde_inverse(j, k)*gamma_tilde;
            }
          }
          if (Kokkos::abs(regular.Lambda(a) - lambda_ref) > tol) {
            ++local_errors;
          }
        }

        fo_gh::RegularToStandardGh(regular, gh);
        Real beta_lower_ref[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            beta_lower_ref[i] += adm.gamma(i, j)*adm.beta(j);
          }
        }
        Real g00_ref = -adm.alpha*adm.alpha;
        for (int i = 0; i < 3; ++i) g00_ref += beta_lower_ref[i]*adm.beta(i);
        if (Kokkos::abs(gh.g(0, 0) - g00_ref) > tol) {
          ++local_errors;
        }
        for (int i = 0; i < 3; ++i) {
          if (Kokkos::abs(gh.g(0, i + 1) - beta_lower_ref[i]) > tol) {
            ++local_errors;
          }
          for (int j = i; j < 3; ++j) {
            if (Kokkos::abs(gh.g(i + 1, j + 1) - adm.gamma(i, j)) > tol) {
              ++local_errors;
            }
          }
        }

        for (int p = 0; p < 3; ++p) {
          Real phi00_ref = -2.0*adm.alpha*adm.dalpha(p);
          for (int i = 0; i < 3; ++i) {
            Real phi0i_ref = 0.0;
            for (int j = 0; j < 3; ++j) {
              phi0i_ref += adm.dgamma(p, i, j)*adm.beta(j)
                           + adm.gamma(i, j)*adm.dbeta(p, j);
              phi00_ref += adm.dgamma(p, i, j)*adm.beta(i)*adm.beta(j)
                           + 2.0*adm.gamma(i, j)*adm.beta(i)*adm.dbeta(p, j);
            }
            if (Kokkos::abs(gh.Phi(p, 0, i + 1) - phi0i_ref) > tol) {
              ++local_errors;
            }
            for (int j = i; j < 3; ++j) {
              if (Kokkos::abs(gh.Phi(p, i + 1, j + 1)
                              - adm.dgamma(p, i, j)) > tol) {
                ++local_errors;
              }
            }
          }
          if (Kokkos::abs(gh.Phi(p, 0, 0) - phi00_ref) > tol) {
            ++local_errors;
          }
        }

        Real d0gamma_ref[3][3];
        Real d0beta_ref[3];
        for (int i = 0; i < 3; ++i) {
          d0beta_ref[i] = 0.75*regular.Lambda(i) - eta_beta*regular.beta(i);
          for (int j = 0; j < 3; ++j) {
            d0gamma_ref[i][j] = -2.0*adm.alpha*adm.K(i, j);
            for (int k = 0; k < 3; ++k) {
              d0gamma_ref[i][j] += adm.gamma(i, k)*adm.dbeta(j, k)
                                   + adm.gamma(j, k)*adm.dbeta(i, k);
            }
          }
        }
        const Real d0alpha_ref = -2.0*adm.alpha*K_trace_ref;
        Real d0g00_ref = -2.0*adm.alpha*d0alpha_ref;
        for (int i = 0; i < 3; ++i) {
          Real d0g0i_ref = 0.0;
          for (int j = 0; j < 3; ++j) {
            d0g0i_ref += d0gamma_ref[i][j]*adm.beta(j)
                         + adm.gamma(i, j)*d0beta_ref[j];
            d0g00_ref += d0gamma_ref[i][j]*adm.beta(i)*adm.beta(j)
                         + 2.0*adm.gamma(i, j)*adm.beta(i)*d0beta_ref[j];
          }
          if (Kokkos::abs(gh.Pi(0, i + 1) + d0g0i_ref/adm.alpha) > tol) {
            ++local_errors;
          }
          for (int j = i; j < 3; ++j) {
            if (Kokkos::abs(gh.Pi(i + 1, j + 1)
                            + d0gamma_ref[i][j]/adm.alpha) > tol) {
              ++local_errors;
            }
          }
        }
        if (Kokkos::abs(gh.Pi(0, 0) + d0g00_ref/adm.alpha) > tol) {
          ++local_errors;
        }
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH algebra unit test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH algebra unit test passed." << std::endl;
}
