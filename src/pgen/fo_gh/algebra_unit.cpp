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
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH algebra unit test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH algebra unit test passed." << std::endl;
}
