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
#include "fo_gh/fo_gh_geometry.hpp"
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
      }, errors);

  if (errors != 0) {
    std::cout << "FO-GH geometry unit test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH geometry unit test passed." << std::endl;
}
