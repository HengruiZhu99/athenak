//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file compatible_unit.cpp
//! \brief Device test for compatible FO-GH reduction-constraint evolution.

#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "fo_gh/fo_gh.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "utils/finite_diff.hpp"

void ProblemGenerator::FoGhCompatibleUnit(ParameterInput *pin, const bool restart) {
  (void)pin;
  (void)restart;
  auto *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const auto state = pmbp->pfogh->u0;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = indcs.nx3 + 2*indcs.ng;
  par_for("fo_gh compatible data", DevExeSpace(), 0, pmbp->nmb_thispack - 1,
  0, ncells3 - 1, 0, ncells2 - 1, 0, ncells1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    for (int n = 0; n < fo_gh::nvar; ++n) {
      state(m, n, k, j, i) = 0.0;
    }
    state(m, fo_gh::I_TGXX, k, j, i) = 1.0 + 0.01*std::sin(2.0*M_PI*x);
    state(m, fo_gh::I_TGYY, k, j, i) = 1.0 + 0.01*std::sin(2.0*M_PI*y);
    state(m, fo_gh::I_TGZZ, k, j, i) = 1.0 + 0.01*std::sin(2.0*M_PI*z);
    state(m, fo_gh::I_CHI, k, j, i) =
        1.0 + 0.005*std::cos(2.0*M_PI*(x + y));
    state(m, fo_gh::I_ALPHA, k, j, i) =
        1.0 + 0.004*std::sin(2.0*M_PI*(y + z));
    state(m, fo_gh::I_BETAX, k, j, i) = 0.003*std::sin(2.0*M_PI*z);
    state(m, fo_gh::I_BETAY, k, j, i) = 0.002*std::cos(2.0*M_PI*x);
    state(m, fo_gh::I_BETAZ, k, j, i) = 0.001*std::sin(2.0*M_PI*y);
  });
  const auto vars = pmbp->pfogh->u;
  par_for("fo_gh compatible gradients", DevExeSpace(),
  0, pmbp->nmb_thispack - 1, indcs.ks - 1, indcs.ke + 1,
  indcs.js - 1, indcs.je + 1, indcs.is - 1, indcs.ie + 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[3] = {1.0/size.d_view(m).dx1,
                         1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    for (int p = 0; p < 3; ++p) {
      vars.X(m, p, k, j, i) = Dx<2>(p, idx, vars.chi, m, k, j, i);
      vars.a(m, p, k, j, i) = Dx<2>(p, idx, vars.alpha, m, k, j, i);
      for (int a = 0; a < 3; ++a) {
        vars.B(m, p, a, k, j, i) =
            Dx<2>(p, idx, vars.beta, m, a, k, j, i);
        for (int b = a; b < 3; ++b) {
          vars.Q[p](m, a, b, k, j, i) =
              Dx<2>(p, idx, vars.gtilde, m, a, b, k, j, i);
        }
      }
    }
  });

  int errors = 0;
  Kokkos::parallel_reduce(
      "fo_gh compatible initial constraints", Kokkos::MDRangePolicy<
      Kokkos::Rank<4>, DevExeSpace>({0, indcs.ks, indcs.js, indcs.is},
      {pmbp->nmb_thispack, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    int &local_errors) {
        constexpr Real tolerance = 2.0e-13;
        const Real idx[3] = {1.0/size.d_view(m).dx1,
                             1.0/size.d_view(m).dx2,
                             1.0/size.d_view(m).dx3};
        for (int p = 0; p < 3; ++p) {
          if (Kokkos::abs(vars.X(m, p, k, j, i)
                          - Dx<2>(p, idx, vars.chi, m, k, j, i)) > tolerance ||
              Kokkos::abs(vars.a(m, p, k, j, i)
                          - Dx<2>(p, idx, vars.alpha, m, k, j, i)) > tolerance) {
            ++local_errors;
          }
          for (int a = 0; a < 3; ++a) {
            if (Kokkos::abs(vars.B(m, p, a, k, j, i)
                            - Dx<2>(p, idx, vars.beta, m, a, k, j, i))
                > tolerance) {
              ++local_errors;
            }
            for (int b = a; b < 3; ++b) {
              if (Kokkos::abs(vars.Q[p](m, a, b, k, j, i)
                              - Dx<2>(p, idx, vars.gtilde, m, a, b, k, j, i))
                  > tolerance) {
                ++local_errors;
              }
            }
          }
          for (int q = p + 1; q < 3; ++q) {
            const Real curl_x = Dx<2>(p, idx, vars.X, m, q, k, j, i)
                                - Dx<2>(q, idx, vars.X, m, p, k, j, i);
            if (Kokkos::abs(curl_x) > tolerance) {
              ++local_errors;
            }
          }
        }
      }, errors);
  (void)pmbp->pfogh->CalcRHS<2>(nullptr, 0);
  const auto rhs = pmbp->pfogh->rhs;
  Kokkos::parallel_reduce(
      "fo_gh compatible rhs constraints", Kokkos::MDRangePolicy<
      Kokkos::Rank<4>, DevExeSpace>({0, indcs.ks, indcs.js, indcs.is},
      {pmbp->nmb_thispack, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    int &local_errors) {
        constexpr Real tolerance = 2.0e-13;
        const Real idx[3] = {1.0/size.d_view(m).dx1,
                             1.0/size.d_view(m).dx2,
                             1.0/size.d_view(m).dx3};
        for (int p = 0; p < 3; ++p) {
          if (Kokkos::abs(rhs.X(m, p, k, j, i)
                          - Dx<2>(p, idx, rhs.chi, m, k, j, i)) > tolerance ||
              Kokkos::abs(rhs.a(m, p, k, j, i)
                          - Dx<2>(p, idx, rhs.alpha, m, k, j, i)) > tolerance) {
            ++local_errors;
          }
          for (int a = 0; a < 3; ++a) {
            if (Kokkos::abs(rhs.B(m, p, a, k, j, i)
                            - Dx<2>(p, idx, rhs.beta, m, a, k, j, i))
                > tolerance) {
              ++local_errors;
            }
            for (int b = a; b < 3; ++b) {
              if (Kokkos::abs(rhs.Q[p](m, a, b, k, j, i)
                              - Dx<2>(p, idx, rhs.gtilde, m, a, b, k, j, i))
                  > tolerance) {
                ++local_errors;
              }
            }
          }
        }
      }, errors);
  if (errors != 0) {
    std::cout << "FO-GH compatible-gradient test failed with " << errors
              << " errors." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH compatible-gradient test passed." << std::endl;
}
