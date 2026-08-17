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
    state(m, fo_gh::I_H_PERP, k, j, i) =
        0.007*std::sin(2.0*M_PI*(x + 2.0*y));
    state(m, fo_gh::I_HX, k, j, i) = 0.006*std::cos(2.0*M_PI*(y + z));
    state(m, fo_gh::I_HY, k, j, i) = 0.005*std::sin(2.0*M_PI*(z + x));
    state(m, fo_gh::I_HZ, k, j, i) = 0.004*std::cos(2.0*M_PI*(x + y));
    state(m, fo_gh::I_VARTHETA_PERP, k, j, i) =
        0.001*std::cos(2.0*M_PI*z);
    state(m, fo_gh::I_VARTHETAX, k, j, i) = 0.001*std::sin(2.0*M_PI*x);
    state(m, fo_gh::I_VARTHETAY, k, j, i) = 0.001*std::cos(2.0*M_PI*y);
    state(m, fo_gh::I_VARTHETAZ, k, j, i) = 0.001*std::sin(2.0*M_PI*z);
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

  int initial_errors = 0;
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
            const Real curl_a = Dx<2>(p, idx, vars.a, m, q, k, j, i)
                                - Dx<2>(q, idx, vars.a, m, p, k, j, i);
            if (Kokkos::abs(curl_x) > tolerance ||
                Kokkos::abs(curl_a) > tolerance) {
              ++local_errors;
            }
            for (int a = 0; a < 3; ++a) {
              const Real curl_b = Dx<2>(p, idx, vars.B, m, q, a, k, j, i)
                                  - Dx<2>(q, idx, vars.B, m, p, a, k, j, i);
              if (Kokkos::abs(curl_b) > tolerance) ++local_errors;
              for (int b = a; b < 3; ++b) {
                const Real curl_q = Dx<2>(p, idx, vars.Q[q], m, a, b, k, j, i)
                                    - Dx<2>(q, idx, vars.Q[p], m, a, b, k, j, i);
                if (Kokkos::abs(curl_q) > tolerance) ++local_errors;
              }
            }
          }
        }
      }, initial_errors);
  (void)pmbp->pfogh->CalcRHS<2>(nullptr, 0);
  const auto rhs = pmbp->pfogh->rhs;
  int rhs_errors = 0;
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
        // CalcRHS produces Q/X/a/B RHS values on physical cells.  Their curls
        // therefore have a valid centered stencil only one cell inside that
        // region; direct compatible-gradient identities remain checked below
        // on every physical cell.
        const bool curl_stencil_in_bounds =
            (k > indcs.ks && k < indcs.ke && j > indcs.js && j < indcs.je &&
             i > indcs.is && i < indcs.ie);
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
          for (int q = p + 1; q < 3 && curl_stencil_in_bounds; ++q) {
            const Real curl_x = Dx<2>(p, idx, rhs.X, m, q, k, j, i)
                                - Dx<2>(q, idx, rhs.X, m, p, k, j, i);
            const Real curl_a = Dx<2>(p, idx, rhs.a, m, q, k, j, i)
                                - Dx<2>(q, idx, rhs.a, m, p, k, j, i);
            if (Kokkos::abs(curl_x) > tolerance ||
                Kokkos::abs(curl_a) > tolerance) {
              ++local_errors;
            }
            for (int a = 0; a < 3; ++a) {
              const Real curl_b = Dx<2>(p, idx, rhs.B, m, q, a, k, j, i)
                                  - Dx<2>(q, idx, rhs.B, m, p, a, k, j, i);
              if (Kokkos::abs(curl_b) > tolerance) ++local_errors;
              for (int b = a; b < 3; ++b) {
                const Real curl_q = Dx<2>(p, idx, rhs.Q[q], m, a, b, k, j, i)
                                    - Dx<2>(q, idx, rhs.Q[p], m, a, b, k, j, i);
                if (Kokkos::abs(curl_q) > tolerance) ++local_errors;
              }
            }
          }
        }
      }, rhs_errors);

  // The gauge-driver equations expose both scalar and vector Lx paths without
  // requiring any continuum-source reconstruction.  Verify the production RHS
  // uses robust advection and that this dataset actually distinguishes it from
  // the centered beta.Dh expression used by pointwise algebra tests.
  int advection_errors = 0;
  Kokkos::parallel_reduce(
      "fo_gh robust advection", Kokkos::MDRangePolicy<
      Kokkos::Rank<4>, DevExeSpace>({0, indcs.ks, indcs.js, indcs.is},
      {pmbp->nmb_thispack, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    int &local_errors) {
        constexpr Real tolerance = 2.0e-13;
        const Real idx[3] = {1.0/size.d_view(m).dx1,
                             1.0/size.d_view(m).dx2,
                             1.0/size.d_view(m).dx3};
        fo_gh::RegularPointState point;
        fo_gh::LoadPoint(vars, m, k, j, i, point);
        Real f_perp = 0.0;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> f;
        fo_gh::GaugeTargets(point, 2.0, f_perp, f);
        Real robust_perp = 0.0;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> robust_h;
        robust_h.ZeroClear();
        for (int p = 0; p < 3; ++p) {
          robust_perp += Lx<2>(p, idx, vars.beta, vars.h_perp, m, p, k, j, i);
          for (int a = 0; a < 3; ++a) {
            robust_h(a) += Lx<2>(p, idx, vars.beta, vars.h,
                                 m, p, a, k, j, i);
          }
        }
        const Real expected_h_perp = robust_perp - (point.h_perp - f_perp)
                                     + point.vartheta_perp;
        const Real expected_vartheta_perp =
            -(robust_perp + point.vartheta_perp);
        if (Kokkos::abs(rhs.h_perp(m, k, j, i) - expected_h_perp) > tolerance ||
            Kokkos::abs(rhs.vartheta_perp(m, k, j, i)
                        - expected_vartheta_perp) > tolerance) {
          ++local_errors;
        }
        for (int a = 0; a < 3; ++a) {
          const Real expected_h = robust_h(a) - (point.h(a) - f(a))
                                  + point.vartheta(a);
          const Real expected_vartheta = -(robust_h(a) + point.vartheta(a));
          if (Kokkos::abs(rhs.h(m, a, k, j, i) - expected_h) > tolerance ||
              Kokkos::abs(rhs.vartheta(m, a, k, j, i)
                          - expected_vartheta) > tolerance) {
            ++local_errors;
          }
        }
      }, advection_errors);
  Real max_advection_difference = 0.0;
  Kokkos::parallel_reduce(
      "fo_gh robust advection distinction", Kokkos::MDRangePolicy<
      Kokkos::Rank<4>, DevExeSpace>({0, indcs.ks, indcs.js, indcs.is},
      {pmbp->nmb_thispack, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    Real &local_max) {
        const Real idx[3] = {1.0/size.d_view(m).dx1,
                             1.0/size.d_view(m).dx2,
                             1.0/size.d_view(m).dx3};
        for (int p = 0; p < 3; ++p) {
          const Real difference = Kokkos::abs(
              Lx<2>(p, idx, vars.beta, vars.h_perp, m, p, k, j, i)
              - vars.beta(m, p, k, j, i)
                *Dx<2>(p, idx, vars.h_perp, m, k, j, i));
          if (difference > local_max) local_max = difference;
          for (int a = 0; a < 3; ++a) {
            const Real vector_difference = Kokkos::abs(
                Lx<2>(p, idx, vars.beta, vars.h, m, p, a, k, j, i)
                - vars.beta(m, p, k, j, i)
                  *Dx<2>(p, idx, vars.h, m, a, k, j, i));
            if (vector_difference > local_max) local_max = vector_difference;
          }
        }
      }, Kokkos::Max<Real>(max_advection_difference));
  if (max_advection_difference <= 1.0e-10) {
    ++advection_errors;
  }

  // A flat, trace-free Atilde profile isolates the tensor Lx overload.  With
  // Axx=f, Ayy=-f, all other non-gauge fields flat, and constant beta, the
  // fixed equations reduce to R_Axx=Lx(Axx)-2f^2,
  // R_Ayy=Lx(Ayy)-2f^2, R_pi=-2f^2, and R_K=0.
  par_for("fo_gh tensor advection data", DevExeSpace(),
  0, pmbp->nmb_thispack - 1, 0, ncells3 - 1, 0, ncells2 - 1, 0, ncells1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real f = 0.01*std::sin(2.0*M_PI*(x + y));
    for (int n = 0; n < fo_gh::nvar; ++n) {
      state(m, n, k, j, i) = 0.0;
    }
    state(m, fo_gh::I_TGXX, k, j, i) = 1.0;
    state(m, fo_gh::I_TGYY, k, j, i) = 1.0;
    state(m, fo_gh::I_TGZZ, k, j, i) = 1.0;
    state(m, fo_gh::I_CHI, k, j, i) = 1.0;
    state(m, fo_gh::I_ALPHA, k, j, i) = 1.0;
    state(m, fo_gh::I_BETAX, k, j, i) = 0.2;
    state(m, fo_gh::I_BETAY, k, j, i) = -0.1;
    state(m, fo_gh::I_BETAZ, k, j, i) = 0.05;
    state(m, fo_gh::I_TAXX, k, j, i) = f;
    state(m, fo_gh::I_TAYY, k, j, i) = -f;
  });
  (void)pmbp->pfogh->CalcRHS<2>(nullptr, 0);
  int tensor_advection_errors = 0;
  Kokkos::parallel_reduce(
      "fo_gh tensor robust advection", Kokkos::MDRangePolicy<
      Kokkos::Rank<4>, DevExeSpace>({0, indcs.ks, indcs.js, indcs.is},
      {pmbp->nmb_thispack, indcs.ke + 1, indcs.je + 1, indcs.ie + 1}),
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i,
                    int &local_errors) {
        constexpr Real tolerance = 2.0e-13;
        const Real idx[3] = {1.0/size.d_view(m).dx1,
                             1.0/size.d_view(m).dx2,
                             1.0/size.d_view(m).dx3};
        const Real f = vars.Atilde(m, 0, 0, k, j, i);
        Real adv_xx = 0.0;
        Real adv_yy = 0.0;
        for (int p = 0; p < 3; ++p) {
          adv_xx += Lx<2>(p, idx, vars.beta, vars.Atilde,
                          m, p, 0, 0, k, j, i);
          adv_yy += Lx<2>(p, idx, vars.beta, vars.Atilde,
                          m, p, 1, 1, k, j, i);
        }
        if (Kokkos::abs(rhs.Atilde(m, 0, 0, k, j, i)
                        - (adv_xx - 2.0*f*f)) > tolerance ||
            Kokkos::abs(rhs.Atilde(m, 1, 1, k, j, i)
                        - (adv_yy - 2.0*f*f)) > tolerance ||
            Kokkos::abs(rhs.Atilde(m, 2, 2, k, j, i)) > tolerance ||
            Kokkos::abs(rhs.pi(m, k, j, i) + 2.0*f*f) > tolerance ||
            Kokkos::abs(rhs.K(m, k, j, i)) > tolerance) {
          ++local_errors;
        }
      }, tensor_advection_errors);

  const int errors = initial_errors + rhs_errors + advection_errors
                     + tensor_advection_errors;
  if (errors != 0) {
    std::cout << "FO-GH compatible/advection test failed with " << errors
              << " errors (initial=" << initial_errors << ", rhs=" << rhs_errors
              << ", advection=" << advection_errors << ", tensor="
              << tensor_advection_errors << ")." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "FO-GH compatible-gradient and robust-advection test passed."
            << std::endl;
}
