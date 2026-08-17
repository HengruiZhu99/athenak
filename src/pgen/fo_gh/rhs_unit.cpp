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
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH RHS unit test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH RHS unit test passed." << std::endl;
}
