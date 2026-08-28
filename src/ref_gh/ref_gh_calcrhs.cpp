//========================================================================================
//! \file ref_gh_calcrhs.cpp
//! \brief Flat-reference nonlinear GH RHS and compatible Phi update.
//========================================================================================
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "ref_gh/analytic_radial_q_production.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/gamma2_damping.hpp"
#include "ref_gh/gauge_driver.hpp"
#include "ref_gh/phi_ordering.hpp"
#include "ref_gh/physical_gauge_target.hpp"
#include "ref_gh/reference_gauge_baseline.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"
#include "utils/finite_diff.hpp"

namespace ref_gh {

template <int FDNG>
TaskStatus RefGh::CalcRHS(Driver *driver, int stage) {
  return opt.reference_backend == 1
      ? CalcRHSImpl<FDNG, true>(driver, stage)
      : CalcRHSImpl<FDNG, false>(driver, stage);
}

template <int FDNG, bool Analytic>
TaskStatus RefGh::CalcRHSImpl(Driver *driver, int stage) {
  // The queued UpdateReference task normally prepares this cache.  Keep the
  // guard here for initialization/unit-test callers that invoke CalcRHS
  // directly outside the stage task list.
  FillReferenceCache(StageTime(driver, stage),
                     opt.source_kind != 0 || opt.gauge_reference_subtraction);
  const std::uint64_t expected_reference_generation =
      opt.reference_q_controlled ? q_controller_generation
                                 : controller_generation;
  if ((opt.reference_controlled || opt.reference_q_controlled)
      && reference_cache_generation != expected_reference_generation) {
    std::cout << "### FATAL ERROR: Ref-GH RHS observed a stale reference cache: "
              << "cache_generation=" << reference_cache_generation
              << " state_generation=" << expected_reference_generation
              << " stage=" << stage << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int radius = FDNG - 1;
  const int nmb = pmy_pack->nmb_thispack;
  const auto state = u0;
  const auto state_rhs = u_rhs;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto analytic_static = reference_static;
  const auto analytic_stage = reference_stage;
  constexpr int reference_backend = Analytic ? 1 : 0;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const int source_kind = opt.source_kind;
  const int phi_ordering = opt.phi_ordering;
  const Real gamma0 = opt.gamma0;
  const Real gamma2 = opt.gamma2;
  const bool gauge_driver_enabled = opt.gauge_driver_enabled;
  const bool gauge_reference_subtraction = opt.gauge_reference_subtraction;
  const bool reference_time_dependent = opt.reference_time_dependent;
  const Real gauge_mu = opt.gauge_mu;
  const Real gauge_eta = opt.gauge_eta;
  const Real shift_nu = opt.shift_nu;
  const Real shift_eta = opt.shift_eta;
  Kokkos::deep_copy(state_rhs, 0.0);
  DebugFence("ref_gh CalcRHS zero");

  // Psi_t is required on a stencil halo by the compatible Phi update.  Keep its
  // point-local working set separate from the lower-order source and principal Pi
  // update so large reference-frame temporaries do not all remain live together.
  par_for("ref_gh psi rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks - radius, indcs.ke + radius,
  indcs.js - radius, indcs.je + radius,
  indcs.is - radius, indcs.ie + radius,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const auto reference = MakeTypedProductionReferencePoint<Analytic>(
        reference_cache, reference_extra, analytic_static, analytic_stage,
        m, k, j, i, x, y, z, center_x, center_y, center_z);
    Real psi[4][4], metric[4][4], inverse[4][4], pi[4][4]; // NOLINT
    Real phi[3][4][4]; // NOLINT
    LoadSymmetric(state, kPsiOffset, m, k, j, i, psi);
    LoadSymmetric(state, kPiOffset, m, k, j, i, pi);
    for (int p = 0; p < 3; ++p) {
      for (int a = 0; a < 4; ++a) {
        for (int b = a; b < 4; ++b) {
          phi[p][a][b] = phi[p][b][a] =
              state(m, PhiIndex(p, a, b), k, j, i);
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        metric[a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            metric[a][b] += ReferenceCoframe(reference, A, a)
                            *ReferenceCoframe(reference, B, b)*psi[A][B];
          }
        }
      }
    }
    Real determinant = 0.0;
    if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
      for (int n = 0; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
      return;
    }
    const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
    Real shift[3];  // NOLINT(runtime/arrays)
    for (int p = 0; p < 3; ++p) {
      shift[p] = lapse*lapse*inverse[0][p + 1];
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        Real psi_rhs = -lapse*pi[a][b];
        for (int p = 0; p < 3; ++p) {
          Real coordinate_d_psi = 0.0;
          for (int I = 0; I < 3; ++I) {
            coordinate_d_psi +=
                ReferenceSpatialCoframe(reference, I, p)*phi[I][a][b];
          }
          psi_rhs += shift[p]*coordinate_d_psi;
        }
        state_rhs(m, PsiIndex(a, b), k, j, i) = psi_rhs;
      }
    }
  });
  DebugFence("ref_gh CalcRHS psi");

  // The generic oracle retains its independently qualified split kernel.  The
  // analytic production backend evaluates this block in the main active-cell
  // kernel below so physical point geometry is reconstructed only once.
  if (reference_backend == 0 && gauge_driver_enabled) {
    par_for("ref_gh improved gauge driver", DevExeSpace(), 0, nmb - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const auto reference = MakeTypedProductionReferencePoint<Analytic>(
          reference_cache, reference_extra, analytic_static, analytic_stage,
          m, k, j, i, x, y, z, center_x, center_y, center_z);
      Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4]; // NOLINT
      Real d_psi[4][4][4], d_metric[4][4][4]; // NOLINT
      CoordinateGhGeometry geometry;
      Real determinant = 0.0;
      if (!LoadProductionPointGeometry(state, reference, m, k, j, i, psi, pi,
                                       phi, d_psi, metric, d_metric, geometry,
                                       determinant)) {
        for (int n = kHhatOffset; n < nvar; ++n) {
          state_rhs(m, n, k, j, i) = NAN;
        }
        return;
      }
      const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};
      Real hhat[4];          // NOLINT(runtime/arrays)
      Real theta[4];         // NOLINT(runtime/arrays)
      Real upsilon[3];       // NOLINT(runtime/arrays)
      Real d_hhat[3][4];     // NOLINT(runtime/arrays)
      ReferenceGaugeBaseline baseline{};
      if (gauge_reference_subtraction) {
        baseline = ComputeProductionReferenceGaugeBaseline(reference);
        if (!baseline.valid) {
          for (int n = kHhatOffset; n < nvar; ++n) {
            state_rhs(m, n, k, j, i) = NAN;
          }
          return;
        }
      }
      for (int A = 0; A < 4; ++A) {
        hhat[A] = state(m, kHhatOffset + A, k, j, i)
                  + (gauge_reference_subtraction ? baseline.hhat[A] : 0.0);
        theta[A] = state(m, kThetaOffset + A, k, j, i)
                   + (gauge_reference_subtraction ? baseline.theta[A] : 0.0);
        for (int p = 0; p < 3; ++p) {
          d_hhat[p][A] = Dx<FDNG>(p, idx, state, m, kHhatOffset + A,
                                  k, j, i)
                          + (gauge_reference_subtraction
                                 ? baseline.d_hhat[p + 1][A] : 0.0);
        }
      }
      for (int p = 0; p < 3; ++p) {
        upsilon[p] = state(m, kUpsilonOffset + p, k, j, i);
      }
      PhysicalGaugeTarget target;
      if (!ComputePhysicalGaugeTarget(metric, d_metric, geometry, reference,
                                      upsilon, shift_nu, shift_eta, target)) {
        for (int n = kHhatOffset; n < nvar; ++n) {
          state_rhs(m, n, k, j, i) = NAN;
        }
        return;
      }
      const GaugeDriverRhs gauge_rhs = ComputeGaugeDriverRhs(
          reference, hhat, theta, upsilon, d_hhat, geometry.shift,
          target.frame, target.conformal_gamma, gauge_mu, gauge_eta,
          shift_eta);
      for (int A = 0; A < 4; ++A) {
        state_rhs(m, kHhatOffset + A, k, j, i) = gauge_rhs.hhat[A]
            - (gauge_reference_subtraction ? baseline.d_hhat[0][A] : 0.0);
        state_rhs(m, kThetaOffset + A, k, j, i) = gauge_rhs.theta[A]
            - ((gauge_reference_subtraction && reference_time_dependent)
                   ? ProductionReferenceDtTheta(reference, A) : 0.0);
      }
      for (int p = 0; p < 3; ++p) {
        state_rhs(m, kUpsilonOffset + p, k, j, i) = gauge_rhs.upsilon[p];
      }
    });
    DebugFence("ref_gh CalcRHS gauge_driver");
  }

  // The lean source and principal Pi update share the same reconstructed point
  // geometry.  Keep the source implementation in a separate inline function so
  // its large contraction temporaries end before the Pi working set begins.
  par_for("ref_gh scalar source and pi rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const auto reference = MakeTypedProductionReferencePoint<Analytic>(
        reference_cache, reference_extra, analytic_static, analytic_stage,
        m, k, j, i, x, y, z, center_x, center_y, center_z);
    Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4]; // NOLINT
    Real d_psi[4][4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    if (!LoadProductionPointGeometry(state, reference, m, k, j, i, psi, pi,
                                     phi, d_psi, metric, d_metric, geometry,
                                     determinant)) {
      for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
      return;
    }
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    if (reference_backend == 1 && gauge_driver_enabled) {
      Real hhat[4];          // NOLINT(runtime/arrays)
      Real theta[4];         // NOLINT(runtime/arrays)
      Real upsilon[3];       // NOLINT(runtime/arrays)
      Real d_hhat[3][4];     // NOLINT(runtime/arrays)
      ReferenceGaugeBaseline baseline{};
      if (gauge_reference_subtraction) {
        baseline = ComputeProductionReferenceGaugeBaseline(reference);
        if (!baseline.valid) {
          for (int n = kHhatOffset; n < nvar; ++n) {
            state_rhs(m, n, k, j, i) = NAN;
          }
          return;
        }
      }
      for (int A = 0; A < 4; ++A) {
        hhat[A] = state(m, kHhatOffset + A, k, j, i)
                  + (gauge_reference_subtraction ? baseline.hhat[A] : 0.0);
        theta[A] = state(m, kThetaOffset + A, k, j, i)
                   + (gauge_reference_subtraction ? baseline.theta[A] : 0.0);
        for (int p = 0; p < 3; ++p) {
          d_hhat[p][A] = Dx<FDNG>(p, idx, state, m, kHhatOffset + A,
                                  k, j, i)
                          + (gauge_reference_subtraction
                                 ? baseline.d_hhat[p + 1][A] : 0.0);
        }
      }
      for (int p = 0; p < 3; ++p) {
        upsilon[p] = state(m, kUpsilonOffset + p, k, j, i);
      }
      PhysicalGaugeTarget target;
      if (!ComputePhysicalGaugeTarget(metric, d_metric, geometry, reference,
                                      upsilon, shift_nu, shift_eta, target)) {
        for (int n = kHhatOffset; n < nvar; ++n) {
          state_rhs(m, n, k, j, i) = NAN;
        }
        return;
      }
      const GaugeDriverRhs gauge_rhs = ComputeGaugeDriverRhs(
          reference, hhat, theta, upsilon, d_hhat, geometry.shift,
          target.frame, target.conformal_gamma, gauge_mu, gauge_eta,
          shift_eta);
      for (int A = 0; A < 4; ++A) {
        state_rhs(m, kHhatOffset + A, k, j, i) = gauge_rhs.hhat[A]
            - (gauge_reference_subtraction ? baseline.d_hhat[0][A] : 0.0);
        state_rhs(m, kThetaOffset + A, k, j, i) = gauge_rhs.theta[A]
            - ((gauge_reference_subtraction && reference_time_dependent)
                   ? ProductionReferenceDtTheta(reference, A) : 0.0);
      }
      for (int p = 0; p < 3; ++p) {
        state_rhs(m, kUpsilonOffset + p, k, j, i) = gauge_rhs.upsilon[p];
      }
    }
    Real scalar_source[4][4];  // NOLINT(runtime/arrays)
    if constexpr (Analytic) {
      if (!ProductionCovariantScalarWaveSource(
              psi, pi, phi, reference, geometry, gamma0, scalar_source)) {
        for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
        return;
      }
    } else if (source_kind == 0) {
      if (!ProductionCovariantScalarWaveSource(
              psi, pi, phi, reference, geometry, gamma0, scalar_source)) {
        for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
        return;
      }
    } else {
      Real partial_source[4][4];  // NOLINT(runtime/arrays)
      StandardGhPartialWaveSource(metric, d_metric, reference,
                                  geometry, gamma0, partial_source);
      TransformPartialWaveSource(metric, d_metric, partial_source, d_psi,
                                 reference, geometry, scalar_source);
    }
    if (gauge_driver_enabled) {
      Real hhat[4];       // NOLINT(runtime/arrays)
      Real d_hhat[4][4];  // NOLINT(runtime/arrays)
      ReferenceGaugeBaseline baseline{};
      if (gauge_reference_subtraction) {
        baseline = ComputeProductionReferenceGaugeBaseline(reference);
        if (!baseline.valid) {
          for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
          return;
        }
      }
      for (int A = 0; A < 4; ++A) {
        hhat[A] = state(m, kHhatOffset + A, k, j, i)
                  + (gauge_reference_subtraction ? baseline.hhat[A] : 0.0);
        d_hhat[0][A] = state_rhs(m, kHhatOffset + A, k, j, i)
                       + (gauge_reference_subtraction
                              ? baseline.d_hhat[0][A] : 0.0);
        for (int p = 0; p < 3; ++p) {
          d_hhat[p + 1][A] = Dx<FDNG>(
              p, idx, state, m, kHhatOffset + A, k, j, i)
              + (gauge_reference_subtraction
                     ? baseline.d_hhat[p + 1][A] : 0.0);
        }
      }
      AddProductionOrdinaryGaugeSource(
          metric, d_metric, reference, geometry, hhat, d_hhat, gamma0,
          scalar_source);
    }
    Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
    Real spatial_determinant = 0.0;
    if (!InvertSpatial3(metric, spatial_inverse, spatial_determinant)) {
      for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
      return;
    }
    Real spatial_connection[3][3][3];  // NOLINT(runtime/arrays)
    for (int q = 0; q < 3; ++q) {
      for (int p = 0; p < 3; ++p) {
        for (int r = 0; r < 3; ++r) {
          spatial_connection[q][p][r] = 0.0;
          for (int ell = 0; ell < 3; ++ell) {
            spatial_connection[q][p][r] += 0.5*spatial_inverse[q][ell]
              *(d_metric[p + 1][ell + 1][r + 1]
                + d_metric[r + 1][ell + 1][p + 1]
                - d_metric[ell + 1][p + 1][r + 1]);
          }
        }
      }
    }
    Real trace_k = 0.0;
    for (int p = 0; p < 3; ++p) {
      for (int q = 0; q < 3; ++q) {
        trace_k -= geometry.lapse*spatial_inverse[p][q]
                   *geometry.christoffel[0][p + 1][q + 1];
      }
    }
    Real d_alpha[3];  // NOLINT(runtime/arrays)
    for (int p = 0; p < 3; ++p) {
      Real d_inverse_00 = 0.0;
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          d_inverse_00 -= geometry.inverse_metric[0][a]
                          *geometry.inverse_metric[0][b]*d_metric[p + 1][a][b];
        }
      }
      d_alpha[p] = 0.5*geometry.lapse*geometry.lapse*geometry.lapse*d_inverse_00;
    }

    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        Real divergence = 0.0;
        Real lapse_gradient_term = 0.0;
        for (int p = 0; p < 3; ++p) {
          for (int q = 0; q < 3; ++q) {
            Real partial_tilde_phi = 0.0;
            Real tilde_phi_q = 0.0;
            for (int I = 0; I < 3; ++I) {
              partial_tilde_phi +=
                  CoframeDerivative(reference, p + 1, I + 1, q + 1)
                    *phi[I][a][b]
                  + ReferenceSpatialCoframe(reference, I, q)
                    *Dx<FDNG>(p, idx, state, m, PhiIndex(I, a, b), k, j, i);
              tilde_phi_q +=
                  ReferenceSpatialCoframe(reference, I, q)*phi[I][a][b];
            }
            Real covariant_derivative = partial_tilde_phi;
            for (int r = 0; r < 3; ++r) {
              Real tilde_phi_r = 0.0;
              for (int I = 0; I < 3; ++I) {
                tilde_phi_r +=
                    ReferenceSpatialCoframe(reference, I, r)*phi[I][a][b];
              }
              covariant_derivative -= spatial_connection[r][p][q]*tilde_phi_r;
            }
            divergence += spatial_inverse[p][q]*covariant_derivative;
            lapse_gradient_term += spatial_inverse[p][q]*d_alpha[p]*tilde_phi_q;
          }
        }
        Real pi_rhs = geometry.lapse*(trace_k*pi[a][b] - divergence
                                      + scalar_source[a][b])
                      - lapse_gradient_term;
        for (int p = 0; p < 3; ++p) {
          pi_rhs += geometry.shift[p]
                    *Dx<FDNG>(p, idx, state, m, PiIndex(a, b), k, j, i);
        }
        state_rhs(m, PiIndex(a, b), k, j, i) = pi_rhs;
      }
    }
  });
  DebugFence("ref_gh CalcRHS primary");

  if (phi_ordering == 0) {
    // Preserve the qualified compatible kernel and its arithmetic exactly.
    par_for("ref_gh compatible phi rhs", DevExeSpace(), 0, nmb - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const auto reference = MakeTypedProductionReferencePoint<Analytic>(
          reference_cache, reference_extra, analytic_static, analytic_stage,
          m, k, j, i, x, y, z, center_x, center_y, center_z);
      const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};
      for (int I = 0; I < 3; ++I) {
        for (int component = 0; component < kSymmetric4Size; ++component) {
          Real phi_rhs = 0.0;
          for (int p = 0; p < 3; ++p) {
            phi_rhs += ReferenceSpatialFrame(reference, I, p)
                         *Dx<FDNG>(p, idx, state_rhs, m,
                                    kPsiOffset + component, k, j, i);
            Real coordinate_d_psi = 0.0;
            for (int J = 0; J < 3; ++J) {
              coordinate_d_psi += ReferenceSpatialCoframe(reference, J, p)
                  *state(m, kPhiOffset + J*kSymmetric4Size + component, k, j, i);
            }
            phi_rhs += ReferenceDtSpatialFrame(reference, I, p)*coordinate_d_psi;
          }
          state_rhs(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i) = phi_rhs;
        }
      }
    });
    DebugFence("ref_gh CalcRHS compatible_phi");
  } else {
    par_for("ref_gh standard phi rhs", DevExeSpace(), 0, nmb - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const auto reference = MakeTypedProductionReferencePoint<Analytic>(
          reference_cache, reference_extra, analytic_static, analytic_stage,
          m, k, j, i, x, y, z, center_x, center_y, center_z);
      const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};

      // The standard correction needs the physical shift expressed in the
      // reference spatial frame. Reconstruct it once per cell.
      Real psi[4][4], metric[4][4], inverse[4][4];  // NOLINT(runtime/arrays)
      LoadSymmetric(state, kPsiOffset, m, k, j, i, psi);
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          metric[a][b] = 0.0;
          for (int A = 0; A < 4; ++A) {
            for (int B = 0; B < 4; ++B) {
              metric[a][b] += ReferenceCoframe(reference, A, a)
                              *ReferenceCoframe(reference, B, b)*psi[A][B];
            }
          }
        }
      }
      Real determinant = 0.0;
      if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
        for (int n = kPhiOffset; n < nvar; ++n) state_rhs(m, n, k, j, i) = NAN;
        return;
      }
      const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
      Real beta_frame[3] = {};  // NOLINT(runtime/arrays)
      for (int J = 0; J < 3; ++J) {
        for (int p = 0; p < 3; ++p) {
          const Real beta_coordinate = lapse*lapse*inverse[0][p + 1];
          beta_frame[J] += ReferenceSpatialCoframe(reference, J, p)*beta_coordinate;
        }
      }
      Real structure[3][3][3];  // NOLINT(runtime/arrays)
      for (int I = 0; I < 3; ++I) {
        for (int J = 0; J < 3; ++J) {
          for (int K = 0; K < 3; ++K) {
            structure[I][J][K] = ReferenceStructure(reference, I, J, K);
          }
        }
      }

      for (int component = 0; component < kSymmetric4Size; ++component) {
        Real phi[3];                    // NOLINT(runtime/arrays)
        Real coordinate_d_phi[3][3];   // NOLINT(runtime/arrays)
        Real frame_derivative[3][3];   // NOLINT(runtime/arrays)
        for (int J = 0; J < 3; ++J) {
          for (int p = 0; p < 3; ++p) {
            coordinate_d_phi[J][p] = Dx<FDNG>(
                p, idx, state, m, kPhiOffset + J*kSymmetric4Size + component,
                k, j, i);
          }
          phi[J] = state(m, kPhiOffset + J*kSymmetric4Size + component, k, j, i);
        }
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            frame_derivative[I][J] = 0.0;
            for (int p = 0; p < 3; ++p) {
              frame_derivative[I][J] += ReferenceSpatialFrame(reference, I, p)
                                        *coordinate_d_phi[J][p];
            }
          }
        }
        for (int I = 0; I < 3; ++I) {
          Real phi_rhs = 0.0;
          for (int p = 0; p < 3; ++p) {
            phi_rhs += ReferenceSpatialFrame(reference, I, p)
                         *Dx<FDNG>(p, idx, state_rhs, m,
                                    kPsiOffset + component, k, j, i);
            Real coordinate_d_psi = 0.0;
            for (int J = 0; J < 3; ++J) {
              coordinate_d_psi += ReferenceSpatialCoframe(reference, J, p)
                  *state(m, kPhiOffset + J*kSymmetric4Size + component, k, j, i);
            }
            phi_rhs += ReferenceDtSpatialFrame(reference, I, p)*coordinate_d_psi;
          }
          phi_rhs -= StandardPhiOrderingCorrection(
              I, beta_frame, frame_derivative, structure, phi);
          state_rhs(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i) = phi_rhs;
        }
      }
    });
    DebugFence("ref_gh CalcRHS standard_phi");
  }

  // Complete standard first-order GH reduction-constraint damping for
  // gamma1=-1.  In frame notation C_IAB=E_I(Psi_AB)-Phi_IAB, and the additions
  // are -gamma2 beta^I C_IAB in Pi_t and +alpha gamma2 C_IAB in Phi_IAB,t.
  // Computing the coordinate representative first makes the contraction exact
  // for a non-coordinate, time-dependent reference frame:
  //   beta^I C_I = beta^i(partial_i Psi-theta^I_i Phi_I).
  if (gamma2 > 0.0) {
    par_for("ref_gh gamma2 reduction damping", DevExeSpace(), 0, nmb - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                 size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                 size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                 size.d_view(m).x3min, size.d_view(m).x3max);
      const auto reference = MakeTypedProductionReferencePoint<Analytic>(
          reference_cache, reference_extra, analytic_static, analytic_stage,
          m, k, j, i, x, y, z, center_x, center_y, center_z);
      const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};
      Real psi[4][4];     // NOLINT(runtime/arrays)
      Real metric[4][4];  // NOLINT(runtime/arrays)
      Real inverse[4][4]; // NOLINT(runtime/arrays)
      LoadSymmetric(state, kPsiOffset, m, k, j, i, psi);
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          metric[a][b] = 0.0;
          for (int A = 0; A < 4; ++A) {
            for (int B = 0; B < 4; ++B) {
              metric[a][b] += ReferenceCoframe(reference, A, a)
                              *ReferenceCoframe(reference, B, b)*psi[A][B];
            }
          }
        }
      }
      Real determinant = 0.0;
      if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
        for (int n = kPiOffset; n < nvar; ++n) {
          state_rhs(m, n, k, j, i) = NAN;
        }
        return;
      }
      const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
      Real shift[3];  // NOLINT(runtime/arrays)
      for (int p = 0; p < 3; ++p) {
        shift[p] = lapse*lapse*inverse[0][p + 1];
      }
      for (int component = 0; component < kSymmetric4Size; ++component) {
        Real coordinate_reduction[3];  // NOLINT(runtime/arrays)
        Real spatial_frame[3][3];      // NOLINT(runtime/arrays)
        for (int p = 0; p < 3; ++p) {
          coordinate_reduction[p] =
              Dx<FDNG>(p, idx, state, m, kPsiOffset + component, k, j, i);
          for (int I = 0; I < 3; ++I) {
            coordinate_reduction[p] -= ReferenceSpatialCoframe(reference, I, p)
                *state(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i);
          }
          for (int I = 0; I < 3; ++I) {
            spatial_frame[I][p] = ReferenceSpatialFrame(reference, I, p);
          }
        }
        const Gamma2DampingRhs damping = ComputeGamma2DampingRhs(
            lapse, shift, coordinate_reduction, spatial_frame, gamma2);
        state_rhs(m, kPiOffset + component, k, j, i) += damping.pi;
        for (int I = 0; I < 3; ++I) {
          state_rhs(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i) +=
              damping.phi[I];
        }
      }
    });
    DebugFence("ref_gh CalcRHS gamma2");
  }

  if (opt.diss > 0.0) {
    const Real sign = (FDNG % 2 == 0) ? -1.0 : 1.0;
    const Real coefficient = opt.diss*std::pow(2.0, -2.0*FDNG)*sign;
    par_for("ref_gh dissipation", DevExeSpace(), 0, nmb - 1, 0, nref_gh - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};
      for (int p = 0; p < 3; ++p) {
        state_rhs(m, n, k, j, i) += coefficient
            *Diss<FDNG>(p, idx, state, m, n, k, j, i);
      }
    });
    DebugFence("ref_gh CalcRHS dissipation");
  }
  DebugFence("ref_gh CalcRHS");
  return TaskStatus::complete;
}

template TaskStatus RefGh::CalcRHS<2>(Driver *, int);
template TaskStatus RefGh::CalcRHS<3>(Driver *, int);
template TaskStatus RefGh::CalcRHS<4>(Driver *, int);

template <int FDNG>
void RefGh::CalcConstraints() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const auto constraints = u_con;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto analytic_static = reference_static;
  const auto analytic_stage = reference_stage;
  const int reference_backend = opt.reference_backend;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const int source_kind = opt.source_kind;
  const bool gauge_driver_enabled = opt.gauge_driver_enabled;
  const bool gauge_reference_subtraction = opt.gauge_reference_subtraction;
  const int active_dimensions = pmy_pack->pmesh->one_d ? 1
      : (pmy_pack->pmesh->two_d ? 2 : 3);
  Kokkos::deep_copy(constraints, 0.0);
  par_for("ref_gh flat constraints", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const ProductionReferencePoint reference = MakeProductionReferencePoint(
        reference_backend, reference_cache, reference_extra, analytic_static,
        analytic_stage, m, k, j, i, x, y, z, center_x, center_y, center_z);
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4]; // NOLINT
    Real d_psi[4][4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    if (!LoadProductionPointGeometry(state, reference, m, k, j, i, psi, pi,
                                     phi, d_psi, metric, d_metric, geometry,
                                     determinant)) {
      for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
      return;
    }
    Real scalar_source[4][4];  // NOLINT(runtime/arrays)
    CovariantSourceSectors source_sectors;
    if (!ProductionCovariantScalarWaveSourceDiagnostics(
            psi, pi, phi, reference, geometry, 0.0, scalar_source,
            source_sectors)) {
      for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
      return;
    }
    if (source_kind == 0) {
      ReferenceGaugeBaseline baseline{};
      if (gauge_driver_enabled && gauge_reference_subtraction) {
        baseline = ComputeProductionReferenceGaugeBaseline(reference);
        if (!baseline.valid) {
          for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
          return;
        }
      }
      for (int A = 0; A < 4; ++A) {
        constraints(m, A, k, j, i) = source_sectors.delta[A];
        if (gauge_driver_enabled) {
          Real baseline_frame = 0.0;
          for (int a = 0; a < 4; ++a) {
            baseline_frame += ReferenceFrame(reference, A, a)
                              *geometry.gauge_source[a];
          }
          constraints(m, A, k, j, i) +=
              state(m, kHhatOffset + A, k, j, i)
              + (gauge_reference_subtraction ? baseline.hhat[A] : 0.0)
              - baseline_frame;
        }
      }
    } else {
      ReferenceGaugeBaseline baseline{};
      if (gauge_driver_enabled && gauge_reference_subtraction) {
        baseline = ComputeProductionReferenceGaugeBaseline(reference);
        if (!baseline.valid) {
          for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
          return;
        }
      }
      for (int a = 0; a < 4; ++a) {
        constraints(m, a, k, j, i) = geometry.gauge_constraint[a];
        if (gauge_driver_enabled) {
          Real hhat_coordinate = 0.0;
          for (int A = 0; A < 4; ++A) {
            hhat_coordinate += ReferenceCoframe(reference, A, a)
                               *(state(m, kHhatOffset + A, k, j, i)
                                 + (gauge_reference_subtraction
                                        ? baseline.hhat[A] : 0.0));
          }
          constraints(m, a, k, j, i) +=
              hhat_coordinate - geometry.gauge_source[a];
        }
      }
    }

    Real q2 = 0.0;
    Real delta2 = 0.0;
    Real frame_ricci2 = 0.0;
    Real coordinate_ricci2 = 0.0;
    Real curvature2 = 0.0;
    Real qq2 = 0.0;
    Real delta_product2 = 0.0;
    Real damping2 = 0.0;
    Real frame_correction2 = 0.0;
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        if (reference_backend == 0) {
          frame_ricci2 += ReferenceRicci(reference.generic, A, B)
                          *ReferenceRicci(reference.generic, A, B);
        }
        curvature2 += source_sectors.curvature[A][B]*source_sectors.curvature[A][B];
        qq2 += source_sectors.qq[A][B]*source_sectors.qq[A][B];
        delta_product2 += source_sectors.delta_product[A][B]
                          *source_sectors.delta_product[A][B];
        damping2 += source_sectors.damping[A][B]*source_sectors.damping[A][B];
        frame_correction2 += source_sectors.frame_correction[A][B]
                             *source_sectors.frame_correction[A][B];
        for (int C = 0; C < 4; ++C) {
          q2 += source_sectors.q[C][A][B]*source_sectors.q[C][A][B];
          delta2 += source_sectors.delta_lower[A][B][C]
                    *source_sectors.delta_lower[A][B][C];
        }
        if (reference_backend == 0) {
          Real coordinate_ricci = 0.0;
          for (int C = 0; C < 4; ++C) {
            coordinate_ricci +=
                ReferenceDChristoffel(reference.generic, C, C, A, B)
                - ReferenceDChristoffel(reference.generic, B, C, A, C);
            for (int D = 0; D < 4; ++D) {
              coordinate_ricci +=
                  ReferenceChristoffel(reference.generic, C, C, D)
                    *ReferenceChristoffel(reference.generic, D, A, B)
                  - ReferenceChristoffel(reference.generic, C, B, D)
                    *ReferenceChristoffel(reference.generic, D, A, C);
            }
          }
          coordinate_ricci2 += coordinate_ricci*coordinate_ricci;
        }
      }
    }
    constraints(m, kDiagnosticOffset + 0, k, j, i) = Kokkos::sqrt(q2);
    constraints(m, kDiagnosticOffset + 1, k, j, i) = Kokkos::sqrt(delta2);
    constraints(m, kDiagnosticOffset + 2, k, j, i) = Kokkos::sqrt(frame_ricci2);
    constraints(m, kDiagnosticOffset + 3, k, j, i) =
        Kokkos::sqrt(coordinate_ricci2);
    constraints(m, kDiagnosticOffset + 4, k, j, i) = Kokkos::sqrt(curvature2);
    constraints(m, kDiagnosticOffset + 5, k, j, i) = Kokkos::sqrt(qq2);
    constraints(m, kDiagnosticOffset + 6, k, j, i) = Kokkos::sqrt(delta_product2);
    constraints(m, kDiagnosticOffset + 7, k, j, i) = Kokkos::sqrt(damping2);
    constraints(m, kDiagnosticOffset + 8, k, j, i) =
        Kokkos::sqrt(frame_correction2);
    constraints(m, kMetricConditionDiagnostic, k, j, i) =
        ReferenceSpatialFrame(reference, 0, 0);
    Real reduction2 = 0.0;
    Real curl2 = 0.0;
    for (int I = 0; I < 3; ++I) {
      for (int component = 0; component < kSymmetric4Size; ++component) {
        Real reduction =
            -state(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i);
        for (int p = 0; p < active_dimensions; ++p) {
          reduction += ReferenceSpatialFrame(reference, I, p)
              *Dx<FDNG>(p, idx, state, m, kPsiOffset + component, k, j, i);
        }
        reduction2 += reduction*reduction;
        for (int J = I + 1; J < 3; ++J) {
          Real curl = 0.0;
          for (int p = 0; p < active_dimensions; ++p) {
            curl += ReferenceSpatialFrame(reference, I, p)
                      *Dx<FDNG>(p, idx, state, m,
                                kPhiOffset + J*kSymmetric4Size + component,
                                k, j, i)
                    - ReferenceSpatialFrame(reference, J, p)
                      *Dx<FDNG>(p, idx, state, m,
                                kPhiOffset + I*kSymmetric4Size + component,
                                k, j, i);
          }
          for (int K = 0; K < 3; ++K) {
            curl -= ReferenceStructure(reference, I, J, K)
                    *state(m, kPhiOffset + K*kSymmetric4Size + component,
                           k, j, i);
          }
          curl2 += curl*curl;
        }
      }
    }
    constraints(m, 4, k, j, i) = Kokkos::sqrt(reduction2);
    constraints(m, 5, k, j, i) = Kokkos::sqrt(curl2);
  });
}

template void RefGh::CalcConstraints<2>();
template void RefGh::CalcConstraints<3>();
template void RefGh::CalcConstraints<4>();

}  // namespace ref_gh
