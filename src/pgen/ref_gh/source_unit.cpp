//========================================================================================
//! \file source_unit.cpp
//! \brief Device regressions for flat and nonflat covariant Ref-GH sources.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/gamma2_damping.hpp"
#include "ref_gh/gauge_driver.hpp"
#include "ref_gh/phi_ordering.hpp"
#include "ref_gh/physical_gauge_target.hpp"
#include "ref_gh/puncture_exponent.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_characteristics.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_generic_singular.hpp"
#include "ref_gh/reference_time_dependent_spatial.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace {

void CheckCoframeDerivativeIdentity() {
  ref_gh::ReferenceGeometry reference;
  ref_gh::ZeroReferenceGeometry(reference);
  const Real frame[4][4] = {
    {1.2, 0.17, -0.08, 0.11},
    {-0.09, 0.94, 0.13, -0.04},
    {0.06, -0.12, 1.11, 0.15},
    {-0.07, 0.05, -0.10, 0.88}};
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!ref_gh::Invert4(frame, inverse, determinant)) {
    std::cout << "### FATAL ERROR: coframe derivative oracle frame is singular."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      reference.frame[A][a] = frame[A][a];
      reference.coframe[A][a] = inverse[a][A];
      for (int p = 0; p < 4; ++p) {
        reference.d_frame[p][A][a] =
            0.007*static_cast<Real>(1 + 3*p - 2*A + 5*a)
            + 0.003*static_cast<Real>((p + A + a) % 3);
      }
    }
  }
  Real maximum = 0.0;
  for (int p = 0; p < 4; ++p) {
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        Real derivative_of_identity = 0.0;
        for (int a = 0; a < 4; ++a) {
          derivative_of_identity +=
              reference.d_frame[p][A][a]*reference.coframe[B][a]
              + reference.frame[A][a]
                    *ref_gh::CoframeDerivative(reference, p, B, a);
        }
        maximum = fmax(maximum, Kokkos::abs(derivative_of_identity));
      }
    }
  }
  if (maximum > 2.0e-15) {
    std::cout << "### FATAL ERROR: inverse-coframe derivative identity failed: "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH inverse-coframe derivative identity passed: max error = "
            << maximum << std::endl;
}

void CheckGaugeDriverAlgebra() {
  constexpr int nsamples = 512;
  Real maximum = 0.0;
  Real source_maximum = 0.0;
  Real target_maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh gauge driver algebra",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum,
                    Real &local_source_maximum, Real &local_target_maximum) {
        const Real scale = static_cast<Real>(sample % 41 - 20)/20.0;
        const Real time = 0.3 + 0.013*static_cast<Real>(sample % 67);
        const Real x = 0.17 + 0.03*scale;
        const Real y = -0.21 + 0.02*scale;
        const Real z = 0.11 - 0.01*scale;
        ref_gh::ReferenceGeometry reference;
        ref_gh::TimeDependentSpatialReference().Populate(
            time, x, y, z, reference);
        const Real shift[3] = {0.13, -0.07 + 0.01*scale, 0.09};
        const Real h_coordinate[4] = {
          0.11 - 0.02*scale, -0.08, 0.14 + 0.01*scale, -0.05};
        const Real theta_coordinate[4] = {
          -0.04, 0.07 + 0.01*scale, -0.09, 0.12 - 0.02*scale};
        const Real target_coordinate[4] = {
          0.03 + 0.01*scale, -0.02, 0.06, -0.01*scale};
        Real d_h_coordinate[3][4];  // NOLINT(runtime/arrays)
        for (int p = 0; p < 3; ++p) {
          for (int a = 0; a < 4; ++a) {
            d_h_coordinate[p][a] =
                0.01*static_cast<Real>(1 + 3*p - 2*a) + 0.003*scale;
          }
        }
        Real hhat[4] = {};       // NOLINT(runtime/arrays)
        Real theta[4] = {};      // NOLINT(runtime/arrays)
        Real target[4] = {};     // NOLINT(runtime/arrays)
        Real d_hhat[3][4] = {};  // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          for (int a = 0; a < 4; ++a) {
            hhat[A] += reference.frame[A][a]*h_coordinate[a];
            theta[A] += reference.frame[A][a]*theta_coordinate[a];
            target[A] += reference.frame[A][a]*target_coordinate[a];
          }
          for (int p = 0; p < 3; ++p) {
            for (int a = 0; a < 4; ++a) {
              d_hhat[p][A] += reference.d_frame[p + 1][A][a]
                                  *h_coordinate[a]
                              + reference.frame[A][a]
                                  *d_h_coordinate[p][a];
            }
          }
        }
        const Real upsilon[3] = {0.05, -0.03 + 0.01*scale, 0.08};
        const Real conformal_gamma[3] = {
          -0.02, 0.07 - 0.01*scale, -0.04};
        const Real mu = 0.8;
        const Real eta = 1.3;
        const Real eta_beta = 0.9;
        const ref_gh::GaugeDriverRhs rhs = ref_gh::ComputeGaugeDriverRhs(
            reference, hhat, theta, upsilon, d_hhat, shift, target,
            conformal_gamma, mu, eta, eta_beta);

        Real h_coordinate_rhs[4];      // NOLINT(runtime/arrays)
        Real theta_coordinate_rhs[4];  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          Real advection = 0.0;
          for (int p = 0; p < 3; ++p) {
            advection += shift[p]*d_h_coordinate[p][a];
          }
          h_coordinate_rhs[a] = advection
              - mu*(h_coordinate[a] - target_coordinate[a])
              + theta_coordinate[a];
          theta_coordinate_rhs[a] = -eta*theta_coordinate[a] - eta*advection;
        }
        for (int A = 0; A < 4; ++A) {
          Real expected_h = 0.0;
          Real expected_theta = 0.0;
          for (int a = 0; a < 4; ++a) {
            expected_h += reference.d_frame[0][A][a]*h_coordinate[a]
                          + reference.frame[A][a]*h_coordinate_rhs[a];
            expected_theta +=
                reference.d_frame[0][A][a]*theta_coordinate[a]
                + reference.frame[A][a]*theta_coordinate_rhs[a];
          }
          local_maximum = fmax(
              local_maximum, Kokkos::abs(rhs.hhat[A] - expected_h));
          local_maximum = fmax(
              local_maximum, Kokkos::abs(rhs.theta[A] - expected_theta));
        }
        for (int p = 0; p < 3; ++p) {
          const Real expected = conformal_gamma[p] - eta_beta*upsilon[p];
          local_maximum = fmax(
              local_maximum, Kokkos::abs(rhs.upsilon[p] - expected));
        }

        // Independently invert the target definitions on an exact, moving
        // reference metric.  This checks both physical gauge identities rather
        // than comparing the helper with a duplicate expression.
        ref_gh::CoordinateGhGeometry geometry;
        Real determinant = 0.0;
        if (!ref_gh::ComputeCoordinateGhGeometry(
                reference.metric, reference.d_metric, reference, geometry,
                determinant)) {
          local_maximum = fmax(local_maximum, 1.0);
          return;
        }
        Real source_with_hhat[4][4] = {};  // NOLINT(runtime/arrays)
        Real all_d_hhat[4][4];             // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          all_d_hhat[0][A] = rhs.hhat[A];
          for (int p = 0; p < 3; ++p) {
            all_d_hhat[p + 1][A] = d_hhat[p][A];
          }
        }
        constexpr Real gamma0 = 0.6;
        ref_gh::AddOrdinaryGaugePartialWaveSource(
            reference.metric, reference.d_metric, reference, geometry, hhat,
            all_d_hhat, gamma0, source_with_hhat);
        Real d_inverse[4][4][4];   // NOLINT(runtime/arrays)
        Real d_base_upper[4][4];   // NOLINT(runtime/arrays)
        Real d_base_lower[4][4];   // NOLINT(runtime/arrays)
        for (int p = 0; p < 4; ++p) {
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              d_inverse[p][a][b] = 0.0;
              for (int c = 0; c < 4; ++c) {
                for (int d = 0; d < 4; ++d) {
                  d_inverse[p][a][b] -= geometry.inverse_metric[a][c]
                      *geometry.inverse_metric[b][d]
                      *reference.d_metric[p][c][d];
                }
              }
            }
          }
        }
        for (int p = 0; p < 4; ++p) {
          for (int a = 0; a < 4; ++a) {
            d_base_upper[p][a] = 0.0;
            for (int b = 0; b < 4; ++b) {
              for (int c = 0; c < 4; ++c) {
                d_base_upper[p][a] -= d_inverse[p][b][c]
                    *reference.christoffel[a][b][c]
                    + geometry.inverse_metric[b][c]
                        *reference.d_christoffel[p][a][b][c];
              }
            }
          }
        }
        for (int p = 0; p < 4; ++p) {
          for (int a = 0; a < 4; ++a) {
            d_base_lower[p][a] = 0.0;
            for (int b = 0; b < 4; ++b) {
              d_base_lower[p][a] += reference.d_metric[p][a][b]
                      *geometry.gauge_source_upper[b]
                  + reference.metric[a][b]*d_base_upper[p][b];
            }
          }
        }
        Real coordinate_extra[4][4];  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            const Real d_h_ab = ((a == 0) ? h_coordinate_rhs[b]
                                           : d_h_coordinate[a - 1][b])
                                - d_base_lower[a][b];
            const Real d_h_ba = ((b == 0) ? h_coordinate_rhs[a]
                                           : d_h_coordinate[b - 1][a])
                                - d_base_lower[b][a];
            Real expected = -d_h_ab - d_h_ba;
            for (int c = 0; c < 4; ++c) {
              const Real increment =
                  h_coordinate[c] - geometry.gauge_source[c];
              expected += 2.0*geometry.christoffel[c][a][b]
                          *increment;
              const Real projector =
                  ((c == a) ? geometry.normal_lower[b] : 0.0)
                  + ((c == b) ? geometry.normal_lower[a] : 0.0)
                  - reference.metric[a][b]*geometry.normal_upper[c];
              expected += gamma0*projector*increment;
            }
            coordinate_extra[a][b] = expected;
          }
        }
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            Real expected = 0.0;
            for (int a = 0; a < 4; ++a) {
              for (int b = 0; b < 4; ++b) {
                expected += reference.frame[A][a]*reference.frame[B][b]
                            *coordinate_extra[a][b];
              }
            }
            local_source_maximum = fmax(
                local_source_maximum,
                Kokkos::abs(source_with_hhat[A][B] - expected));
          }
        }
        ref_gh::PhysicalGaugeTarget physical_target;
        constexpr Real nu = 0.75;
        if (!ref_gh::ComputePhysicalGaugeTarget(
                reference.metric, reference.d_metric, geometry, reference,
                upsilon, nu, eta_beta, physical_target)) {
          local_maximum = fmax(local_maximum, 1.0);
          return;
        }
        const Real normal_target =
            (physical_target.coordinate[0]
             - geometry.shift[0]*physical_target.coordinate[1]
             - geometry.shift[1]*physical_target.coordinate[2]
             - geometry.shift[2]*physical_target.coordinate[3])/geometry.lapse;
        local_target_maximum = fmax(
            local_target_maximum,
            Kokkos::abs(normal_target
                        - (2.0/geometry.lapse - 1.0)*physical_target.trace_k));
        Real inverse_spatial[3][3];  // NOLINT(runtime/arrays)
        Real spatial_determinant = 0.0;
        if (!ref_gh::InvertSpatial3(
                reference.metric, inverse_spatial, spatial_determinant)) {
          local_maximum = fmax(local_maximum, 1.0);
          return;
        }
        Real contracted_spatial_connection[3] = {};  // NOLINT(runtime/arrays)
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
              contracted_spatial_connection[i] += 0.5*inverse_spatial[j][k]
                  *(reference.d_metric[j + 1][i + 1][k + 1]
                    + reference.d_metric[k + 1][i + 1][j + 1]
                    - reference.d_metric[i + 1][j + 1][k + 1]);
            }
          }
        }
        for (int i = 0; i < 3; ++i) {
          Real recovered_d0_shift = 0.0;
          for (int j = 0; j < 3; ++j) {
            recovered_d0_shift += geometry.lapse*geometry.lapse
                *inverse_spatial[i][j]
                *(physical_target.coordinate[j + 1]
                  - physical_target.d_alpha[j]/geometry.lapse
                  + contracted_spatial_connection[j]);
          }
          Real recovered_dt_shift = recovered_d0_shift;
          for (int p = 0; p < 3; ++p) {
            recovered_dt_shift +=
                geometry.shift[p]*physical_target.d_shift[p][i];
          }
          const Real expected_dt_shift = nu*(physical_target.conformal_gamma[i]
                                             - eta_beta*upsilon[i]);
          local_target_maximum = fmax(
              local_target_maximum,
              Kokkos::abs(recovered_dt_shift - expected_dt_shift));
        }
      }, Kokkos::Max<Real>(maximum), Kokkos::Max<Real>(source_maximum),
         Kokkos::Max<Real>(target_maximum));
  Kokkos::fence();
  if (!(maximum < 3.0e-13) || !(source_maximum < 3.0e-13)
      || !(target_maximum < 3.0e-13)) {
    std::cout << "### FATAL ERROR: reference-GH gauge-driver algebra failed: "
              << maximum << ", source error=" << source_maximum
              << ", target error=" << target_maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH improved gauge-driver frame and physical-target "
               "algebra passed: max error = "
            << maximum << ", source error = " << source_maximum
            << ", target error = " << target_maximum << std::endl;
}

void CheckGamma2Algebra() {
  constexpr int nsamples = 1024;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh gamma2 algebra", Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real scale = static_cast<Real>(sample % 37 - 18)/18.0;
        const Real alpha = 0.63 + 0.17*static_cast<Real>(sample % 19)/18.0;
        const Real gamma2 = 0.2 + 1.3*static_cast<Real>(sample % 23)/22.0;
        const Real inverse_spatial_metric[3][3] = {
          {1.2, 0.1, -0.04}, {0.1, 0.9, 0.06}, {-0.04, 0.06, 1.1}};
        Real s_cov[3] = {0.31 + 0.02*scale, -0.47, 0.79 - 0.01*scale};
        Real norm2 = 0.0;
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            norm2 += s_cov[I]*inverse_spatial_metric[I][J]*s_cov[J];
          }
        }
        const Real beta[3] = {0.19, -0.08 + 0.01*scale, 0.11};
        const Real coordinate_reduction[3] = {
          0.13 - 0.01*scale, -0.07, 0.21 + 0.02*scale};
        const Real spatial_frame[3][3] = {
          {1.1, 0.03, -0.02}, {0.0, 0.9, 0.04}, {0.01, -0.05, 1.2}};
        const ref_gh::Gamma2DampingRhs damping =
            ref_gh::ComputeGamma2DampingRhs(
                alpha, beta, coordinate_reduction, spatial_frame, gamma2);
        Real expected_pi = 0.0;
        for (int p = 0; p < 3; ++p) {
          expected_pi -= gamma2*beta[p]*coordinate_reduction[p];
        }
        local_maximum = fmax(
            local_maximum, Kokkos::abs(damping.pi - expected_pi));
        for (int I = 0; I < 3; ++I) {
          Real expected_phi = 0.0;
          for (int p = 0; p < 3; ++p) {
            expected_phi += alpha*gamma2*spatial_frame[I][p]
                            *coordinate_reduction[p];
          }
          local_maximum = fmax(
              local_maximum, Kokkos::abs(damping.phi[I] - expected_phi));
        }
        const Real inverse_norm = 1.0/Kokkos::sqrt(norm2);
        for (int I = 0; I < 3; ++I) s_cov[I] *= inverse_norm;
        Real beta_s = 0.0;
        Real s_upper[3] = {};  // NOLINT(runtime/arrays)
        for (int I = 0; I < 3; ++I) {
          beta_s += beta[I]*s_cov[I];
          for (int J = 0; J < 3; ++J) {
            s_upper[I] += inverse_spatial_metric[I][J]*s_cov[J];
          }
        }
        const Real psi = 0.27 - 0.09*scale;
        const Real pi = -0.14 + 0.07*scale;
        const Real phi[3] = {0.12 + 0.03*scale, -0.21, 0.08 - 0.02*scale};
        const ref_gh::GhCharacteristicFields characteristic =
            ref_gh::ToGhCharacteristicFields(
                psi, pi, phi, gamma2, inverse_spatial_metric, s_cov);
        Real recovered_psi = 0.0;
        Real recovered_pi = 0.0;
        Real recovered_phi[3];  // NOLINT(runtime/arrays)
        ref_gh::FromGhCharacteristicFields(
            characteristic, gamma2, s_cov, recovered_psi, recovered_pi,
            recovered_phi);
        local_maximum = fmax(local_maximum, Kokkos::abs(recovered_psi - psi));
        local_maximum = fmax(local_maximum, Kokkos::abs(recovered_pi - pi));
        Real transverse_normal = 0.0;
        for (int I = 0; I < 3; ++I) {
          local_maximum = fmax(
              local_maximum, Kokkos::abs(recovered_phi[I] - phi[I]));
          transverse_normal += s_upper[I]*characteristic.transverse[I];
        }
        local_maximum = fmax(local_maximum, Kokkos::abs(transverse_normal));

        // Frozen-coefficient principal symbol for gamma1=-1.
        Real a_phi[3];  // NOLINT(runtime/arrays)
        Real normal_phi = 0.0;
        for (int I = 0; I < 3; ++I) normal_phi += s_upper[I]*phi[I];
        const Real a_psi = 0.0;
        const Real a_pi = gamma2*beta_s*psi - beta_s*pi + alpha*normal_phi;
        for (int I = 0; I < 3; ++I) {
          a_phi[I] = -alpha*gamma2*s_cov[I]*psi + alpha*s_cov[I]*pi
                     - beta_s*phi[I];
        }
        const ref_gh::GhCharacteristicFields a_characteristic =
            ref_gh::ToGhCharacteristicFields(
                a_psi, a_pi, a_phi, gamma2, inverse_spatial_metric, s_cov);
        local_maximum = fmax(local_maximum, Kokkos::abs(a_characteristic.metric));
        local_maximum = fmax(
            local_maximum,
            Kokkos::abs(a_characteristic.plus
                        - (-beta_s + alpha)*characteristic.plus));
        local_maximum = fmax(
            local_maximum,
            Kokkos::abs(a_characteristic.minus
                        - (-beta_s - alpha)*characteristic.minus));
        for (int I = 0; I < 3; ++I) {
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(a_characteristic.transverse[I]
                          + beta_s*characteristic.transverse[I]));
        }

        // The standard symmetrizer must satisfy H A(s)=(H A(s))^T when
        // Lambda^2>gamma2^2.
        Real principal[5][5] = {};   // NOLINT(runtime/arrays)
        Real symmetrizer[5][5] = {}; // NOLINT(runtime/arrays)
        principal[1][0] = gamma2*beta_s;
        principal[1][1] = -beta_s;
        for (int I = 0; I < 3; ++I) {
          principal[1][I + 2] = alpha*s_upper[I];
          principal[I + 2][0] = -alpha*gamma2*s_cov[I];
          principal[I + 2][1] = alpha*s_cov[I];
          principal[I + 2][I + 2] = -beta_s;
        }
        symmetrizer[0][0] = gamma2*gamma2 + 1.0;
        symmetrizer[0][1] = symmetrizer[1][0] = -gamma2;
        symmetrizer[1][1] = 1.0;
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            symmetrizer[I + 2][J + 2] = inverse_spatial_metric[I][J];
          }
        }
        Real product[5][5] = {};  // NOLINT(runtime/arrays)
        for (int row = 0; row < 5; ++row) {
          for (int column = 0; column < 5; ++column) {
            for (int inner = 0; inner < 5; ++inner) {
              product[row][column] +=
                  symmetrizer[row][inner]*principal[inner][column];
            }
          }
        }
        for (int row = 0; row < 5; ++row) {
          for (int column = 0; column < 5; ++column) {
            local_maximum = fmax(
                local_maximum,
                Kokkos::abs(product[row][column] - product[column][row]));
          }
        }

        // Independent reduction and curl subsidiary-system identities.
        Real d_psi[3] = {0.17, -0.12 + 0.01*scale, 0.09};
        Real d_phi[3][3];   // NOLINT(runtime/arrays)
        Real dd_psi[3][3];  // NOLINT(runtime/arrays)
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            d_phi[I][J] = 0.03*static_cast<Real>(2*I - J + 1) + 0.01*scale;
            dd_psi[I][J] = 0.04*static_cast<Real>(I + J + 1);
          }
        }
        for (int I = 0; I < 3; ++I) {
          const Real reduction = d_psi[I] - phi[I];
          const Real reduction_rhs = -alpha*gamma2*reduction;
          const Real phi_rhs = alpha*gamma2*reduction;
          local_maximum = fmax(
              local_maximum, Kokkos::abs(reduction_rhs + phi_rhs));
          for (int J = I + 1; J < 3; ++J) {
            const Real curl = d_phi[I][J] - d_phi[J][I];
            const Real d_phi_rhs_ij =
                alpha*gamma2*(dd_psi[I][J] - d_phi[I][J]);
            const Real d_phi_rhs_ji =
                alpha*gamma2*(dd_psi[J][I] - d_phi[J][I]);
            const Real curl_rhs = d_phi_rhs_ij - d_phi_rhs_ji;
            local_maximum = fmax(
                local_maximum,
                Kokkos::abs(curl_rhs + alpha*gamma2*curl));
          }
        }
      }, Kokkos::Max<Real>(maximum));
  Kokkos::fence();
  if (!(maximum < 2.0e-13)) {
    std::cout << "### FATAL ERROR: reference-GH gamma2 algebra failed: "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH gamma2 characteristic, symmetrizer, reduction, "
               "and curl algebra passed: max error = "
            << maximum << std::endl;
}

void CheckCombinedGaugeCharacteristics() {
  constexpr int nsamples = 512;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh combined characteristics",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real scale = static_cast<Real>(sample % 31 - 15)/15.0;
        const Real alpha = 0.71 + 0.09*static_cast<Real>(sample % 17)/16.0;
        const Real beta_s = -0.16 + 0.04*scale;
        const Real gamma2 = 0.4 + 0.3*static_cast<Real>(sample % 13)/12.0;
        const Real eta = 1.1;
        const Real inverse_spatial[3][3] = {
          {1.15, 0.07, -0.03}, {0.07, 0.92, 0.05}, {-0.03, 0.05, 1.08}};
        Real s_cov[3] = {0.34 + 0.01*scale, -0.42, 0.73};
        Real norm2 = 0.0;
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            norm2 += s_cov[I]*inverse_spatial[I][J]*s_cov[J];
          }
        }
        const Real inverse_norm = 1.0/Kokkos::sqrt(norm2);
        for (int I = 0; I < 3; ++I) s_cov[I] *= inverse_norm;
        const Real s_frame[4] = {
          0.21 - 0.02*scale, s_cov[0], s_cov[1], s_cov[2]};
        Real psi[ref_gh::kSymmetric4Size];          // NOLINT(runtime/arrays)
        Real pi[ref_gh::kSymmetric4Size];           // NOLINT(runtime/arrays)
        Real phi[3][ref_gh::kSymmetric4Size];       // NOLINT(runtime/arrays)
        Real hhat[4];                               // NOLINT(runtime/arrays)
        Real theta[4];                              // NOLINT(runtime/arrays)
        Real upsilon[3];                            // NOLINT(runtime/arrays)
        for (int component = 0; component < ref_gh::kSymmetric4Size;
             ++component) {
          psi[component] = 0.02*static_cast<Real>(component - 4) + 0.01*scale;
          pi[component] = -0.03*static_cast<Real>(component - 3) + 0.02*scale;
          for (int I = 0; I < 3; ++I) {
            phi[I][component] =
                0.01*static_cast<Real>(2*component - 3*I + 1) - 0.01*scale;
          }
        }
        for (int A = 0; A < 4; ++A) {
          hhat[A] = 0.04*static_cast<Real>(A - 1) + 0.01*scale;
          theta[A] = -0.03*static_cast<Real>(A - 2) - 0.02*scale;
        }
        for (int I = 0; I < 3; ++I) {
          upsilon[I] = 0.05*static_cast<Real>(I - 1) + 0.01*scale;
        }
        const ref_gh::CombinedGhCharacteristicFields characteristic =
            ref_gh::ToCombinedGhCharacteristicFields(
                psi, pi, phi, hhat, theta, upsilon, gamma2, eta,
                inverse_spatial, s_cov, s_frame);
        Real recovered_psi[ref_gh::kSymmetric4Size];     // NOLINT
        Real recovered_pi[ref_gh::kSymmetric4Size];      // NOLINT
        Real recovered_phi[3][ref_gh::kSymmetric4Size];  // NOLINT
        Real recovered_hhat[4], recovered_theta[4], recovered_upsilon[3];
        ref_gh::FromCombinedGhCharacteristicFields(
            characteristic, gamma2, eta, s_cov, s_frame, recovered_psi,
            recovered_pi, recovered_phi, recovered_hhat, recovered_theta,
            recovered_upsilon);
        for (int component = 0; component < ref_gh::kSymmetric4Size;
             ++component) {
          local_maximum = fmax(
              local_maximum, Kokkos::abs(recovered_psi[component]
                                         - psi[component]));
          local_maximum = fmax(
              local_maximum, Kokkos::abs(recovered_pi[component]
                                         - pi[component]));
          for (int I = 0; I < 3; ++I) {
            local_maximum = fmax(
                local_maximum, Kokkos::abs(recovered_phi[I][component]
                                           - phi[I][component]));
          }
        }
        for (int A = 0; A < 4; ++A) {
          local_maximum = fmax(
              local_maximum, Kokkos::abs(recovered_hhat[A] - hhat[A]));
          local_maximum = fmax(
              local_maximum, Kokkos::abs(recovered_theta[A] - theta[A]));
        }
        for (int I = 0; I < 3; ++I) {
          local_maximum = fmax(
              local_maximum, Kokkos::abs(recovered_upsilon[I] - upsilon[I]));
        }

        Real a_psi[ref_gh::kSymmetric4Size] = {};     // NOLINT
        Real a_pi[ref_gh::kSymmetric4Size];           // NOLINT
        Real a_phi[3][ref_gh::kSymmetric4Size];       // NOLINT
        Real a_hhat[4], a_theta[4];                   // NOLINT
        Real a_upsilon[3] = {};                       // NOLINT
        Real s_upper[3] = {};                         // NOLINT
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            s_upper[I] += inverse_spatial[I][J]*s_cov[J];
          }
        }
        for (int A = 0; A < 4; ++A) {
          a_hhat[A] = -beta_s*hhat[A];
          a_theta[A] = eta*beta_s*hhat[A];
        }
        for (int A = 0; A < 4; ++A) {
          for (int B = A; B < 4; ++B) {
            const int component = ref_gh::Symmetric4Index(A, B);
            Real normal_phi = 0.0;
            for (int I = 0; I < 3; ++I) {
              normal_phi += s_upper[I]*phi[I][component];
            }
            const Real gauge_coupling =
                s_frame[A]*hhat[B] + s_frame[B]*hhat[A];
            a_pi[component] = gamma2*beta_s*psi[component]
                              - beta_s*pi[component]
                              + alpha*normal_phi + alpha*gauge_coupling;
            for (int I = 0; I < 3; ++I) {
              a_phi[I][component] =
                  -alpha*gamma2*s_cov[I]*psi[component]
                  + alpha*s_cov[I]*pi[component]
                  - beta_s*phi[I][component];
            }
          }
        }
        const ref_gh::CombinedGhCharacteristicFields a_characteristic =
            ref_gh::ToCombinedGhCharacteristicFields(
                a_psi, a_pi, a_phi, a_hhat, a_theta, a_upsilon, gamma2,
                eta, inverse_spatial, s_cov, s_frame);
        for (int component = 0; component < ref_gh::kSymmetric4Size;
             ++component) {
          local_maximum = fmax(
              local_maximum, Kokkos::abs(a_characteristic.metric[component]));
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(a_characteristic.plus[component]
                          - (-beta_s + alpha)*characteristic.plus[component]));
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(a_characteristic.minus[component]
                          - (-beta_s - alpha)*characteristic.minus[component]));
          for (int I = 0; I < 3; ++I) {
            local_maximum = fmax(
                local_maximum,
                Kokkos::abs(a_characteristic.transverse[I][component]
                            + beta_s*characteristic.transverse[I][component]));
          }
        }
        for (int A = 0; A < 4; ++A) {
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(a_characteristic.gauge_advective[A]
                          + beta_s*characteristic.gauge_advective[A]));
          local_maximum = fmax(
              local_maximum, Kokkos::abs(a_characteristic.gauge_zero[A]));
        }
        for (int I = 0; I < 3; ++I) {
          local_maximum = fmax(
              local_maximum, Kokkos::abs(a_characteristic.upsilon_zero[I]));
        }
      }, Kokkos::Max<Real>(maximum));
  Kokkos::fence();
  if (!(maximum < 4.0e-13)) {
    std::cout << "### FATAL ERROR: reference-GH combined characteristic "
                 "oracle failed: " << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH combined Einstein/gauge characteristic and "
               "inverse oracle passed: max error = "
            << maximum << std::endl;
}

struct ExponentSample {
  Real q_state;
  Real p_state;
  Real q_exact;
  Real p_exact;
  Real q_fd;
  bool valid;
};

KOKKOS_INLINE_FUNCTION
void IsotropicMetricPoint(const Real alpha, const Real psi2,
                          const Real d_alpha[3], const Real d_psi2[3],
                          Real metric[4][4], Real d_metric[4][4][4]) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = 0.0;
      for (int p = 0; p < 4; ++p) d_metric[p][a][b] = 0.0;
    }
  }
  metric[0][0] = -alpha*alpha;
  for (int i = 0; i < 3; ++i) {
    metric[i + 1][i + 1] = psi2*psi2;
    d_metric[i + 1][0][0] = -2.0*alpha*d_alpha[i];
    for (int j = 0; j < 3; ++j) {
      d_metric[i + 1][j + 1][j + 1] = 2.0*psi2*d_psi2[i];
    }
  }
}

KOKKOS_INLINE_FUNCTION
Real WormholePsi2(const Real mass, const Real x, const Real y, const Real z) {
  const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
  const Real psi = 1.0 + mass/(2.0*radius);
  return psi*psi;
}

KOKKOS_INLINE_FUNCTION
Real TrumpetPsi2(const DvceArray2D<Real> &table, const Real mass,
                 const Real x, const Real y, const Real z) {
  const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
  return ref_gh::ArealRadiusToPsi2(
      ref_gh::InterpolateTrumpetProfile(
          table, ref_gh::kCoeffArealRadius, radius/mass), radius/mass).value;
}

KOKKOS_INLINE_FUNCTION
Real FourthOrderSpatialMetricExponent(
    const int geometry_kind, const DvceArray2D<Real> &table,
    const Real mass, const Real h, const Real x, const Real y, const Real z) {
  const Real position[3] = {x, y, z};
  Real psi2_center = 1.0;
  if (geometry_kind == 1) {
    psi2_center = WormholePsi2(mass, x, y, z);
  } else if (geometry_kind == 2) {
    psi2_center = TrumpetPsi2(table, mass, x, y, z);
  }
  const Real inverse_diagonal_metric = 1.0/(psi2_center*psi2_center);
  Real radial_metric_derivative = 0.0;
  for (int direction = 0; direction < 3; ++direction) {
    Real diagonal_metric[4];  // NOLINT(runtime/arrays)
    for (int sample = 0; sample < 4; ++sample) {
      Real shifted[3] = {position[0], position[1], position[2]};
      const int offset = sample < 2 ? sample - 2 : sample - 1;
      shifted[direction] += static_cast<Real>(offset)*h;
      Real psi2 = 1.0;
      if (geometry_kind == 1) {
        psi2 = WormholePsi2(mass, shifted[0], shifted[1], shifted[2]);
      } else if (geometry_kind == 2) {
        psi2 = TrumpetPsi2(table, mass, shifted[0], shifted[1], shifted[2]);
      }
      diagonal_metric[sample] = psi2*psi2;
    }
    const Real derivative = (diagonal_metric[0] - 8.0*diagonal_metric[1]
                             + 8.0*diagonal_metric[2] - diagonal_metric[3])
                            /(12.0*h);
    // gamma^{ij} partial_k gamma_ij is three times the derivative
    // of the common isotropic diagonal component.
    radial_metric_derivative +=
        3.0*position[direction]*inverse_diagonal_metric*derivative;
  }
  return -radial_metric_derivative/6.0;
}

KOKKOS_INLINE_FUNCTION
ExponentSample EvaluateExponentSample(const int geometry_kind,
                                      const DvceArray2D<Real> &table,
                                      const Real mass, const Real h,
                                      const Real x, const Real y, const Real z) {
  const Real displacement[3] = {x, y, z};
  const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
  Real metric[4][4];             // NOLINT(runtime/arrays)
  Real d_metric[4][4][4];       // NOLINT(runtime/arrays)
  Real q_exact = 0.0;
  Real p_exact = 0.0;
  if (geometry_kind == 0) {
    const Real zero[3] = {0.0, 0.0, 0.0};
    IsotropicMetricPoint(1.0, 1.0, zero, zero, metric, d_metric);
  } else if (geometry_kind == 1) {
    const Real psi = 1.0 + mass/(2.0*radius);
    const Real psi2 = psi*psi;
    const Real alpha = 1.0/psi2;
    Real d_alpha[3];  // NOLINT(runtime/arrays)
    Real d_psi2[3];   // NOLINT(runtime/arrays)
    for (int k = 0; k < 3; ++k) {
      const Real d_psi = -0.5*mass*displacement[k]/(radius*radius*radius);
      d_psi2[k] = 2.0*psi*d_psi;
      d_alpha[k] = -d_psi2[k]/(psi2*psi2);
    }
    IsotropicMetricPoint(alpha, psi2, d_alpha, d_psi2, metric, d_metric);
    q_exact = mass/(radius + 0.5*mass);
    p_exact = q_exact;
  } else {
    ref_gh::ReferenceGeometry reference;
    const ref_gh::TrumpetSchwarzschildReference provider{
        table, mass, {0.0, 0.0, 0.0}};
    provider.Populate(0.0, x, y, z, reference);
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        metric[a][b] = reference.metric[a][b];
        for (int p = 0; p < 4; ++p) {
          d_metric[p][a][b] = reference.d_metric[p][a][b];
        }
      }
    }
    const Real rho = radius/mass;
    const ref_gh::RadialProfile alpha = ref_gh::InterpolateTrumpetProfile(
        table, ref_gh::kCoeffAlpha, rho);
    const ref_gh::RadialProfile psi2 = ref_gh::ArealRadiusToPsi2(
        ref_gh::InterpolateTrumpetProfile(
            table, ref_gh::kCoeffArealRadius, rho), rho);
    q_exact = -rho*psi2.d1/psi2.value;
    p_exact = rho*alpha.d1/alpha.value;
  }
  const ref_gh::LocalPunctureExponents exponents =
      ref_gh::ComputeLocalPunctureExponents(metric, d_metric, displacement);
  return {exponents.q, exponents.p, q_exact, p_exact,
          FourthOrderSpatialMetricExponent(
              geometry_kind, table, mass, h, x, y, z),
          exponents.spatial_valid && exponents.lapse_valid};
}

void CheckLocalPunctureExponentEstimator(const DvceArray2D<Real> &table,
                                         const bool strict_gate) {
  {
    const Real spacing[3] = {1.0, 0.5, 0.25};
    const Real overlapping[3] = {2.0, 1.0, 0.5};
    const Real clear_in_x[3] = {2.0 + 1.0e-12, 0.0, 0.0};
    const Real clear_in_y[3] = {0.0, 1.0 + 1.0e-12, 0.0};
    if (ref_gh::PunctureStencilIsClear(overlapping, spacing, 2)
        || !ref_gh::PunctureStencilIsClear(clear_in_x, spacing, 2)
        || !ref_gh::PunctureStencilIsClear(clear_in_y, spacing, 2)) {
      std::cout << "### FATAL ERROR: puncture stencil-footprint mask failed."
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  constexpr int nside = 20;
  constexpr int ncells = nside*nside*nside;
  constexpr int nresolution = 5;
  constexpr Real inverse_resolutions[nresolution] = {
      16.0, 24.0, 32.0, 48.0, 64.0};
  constexpr Real mass = 1.0;
  constexpr Real gaussian_width = 3.0;
  Real previous_wormhole = 0.0;
  Real previous_trumpet = 0.0;
  Real first_fd_difference[3] = {0.0, 0.0, 0.0};
  Real final_fd_difference[3] = {0.0, 0.0, 0.0};
  Real first_fixed_fd_difference[3] = {0.0, 0.0, 0.0};
  Real final_fixed_fd_difference[3] = {0.0, 0.0, 0.0};

  for (int geometry_kind = 0; geometry_kind < 3; ++geometry_kind) {
    for (int resolution = 0; resolution < nresolution; ++resolution) {
      const Real h = mass/inverse_resolutions[resolution];
      Real sum_w = 0.0;
      Real sum_w2 = 0.0;
      Real sum_wq = 0.0;
      Real sum_wq2 = 0.0;
      Real safe_sum_w = 0.0;
      Real safe_sum_w2 = 0.0;
      Real safe_sum_wq = 0.0;
      Real sum_wq_fd = 0.0;
      Real maximum_state_error = 0.0;
      Real maximum_fd_error = 0.0;
      int count = 0;
      int safe_count = 0;
      Kokkos::parallel_reduce(
          "ref_gh local puncture exponent estimator",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
          KOKKOS_LAMBDA(const int index, Real &local_sum_w,
                        Real &local_sum_w2, Real &local_sum_wq,
                        Real &local_sum_wq2, Real &local_safe_sum_w,
                        Real &local_safe_sum_w2, Real &local_safe_sum_wq,
                        Real &local_sum_wq_fd,
                        Real &local_maximum_state_error,
                        Real &local_maximum_fd_error, int &local_count,
                        int &local_safe_count) {
            int work = index;
            const int ix = work % nside; work /= nside;
            const int iy = work % nside;
            const int iz = work/nside;
            const Real x = (static_cast<Real>(ix) - 0.5*(nside - 1))*h;
            const Real y = (static_cast<Real>(iy) - 0.5*(nside - 1))*h;
            const Real z = (static_cast<Real>(iz) - 0.5*(nside - 1))*h;
            const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
            if (!ref_gh::InPunctureEstimatorShell(
                    radius, h, gaussian_width)) return;
            const ExponentSample sample = EvaluateExponentSample(
                geometry_kind, table, mass, h, x, y, z);
            if (!sample.valid) {
              local_maximum_state_error =
                  std::numeric_limits<Real>::infinity();
              return;
            }
            const Real weight = ref_gh::PunctureEstimatorWeight(radius, h);
            local_sum_w += weight;
            local_sum_w2 += weight*weight;
            local_sum_wq += weight*sample.q_state;
            local_sum_wq2 += weight*sample.q_state*sample.q_state;
            local_maximum_state_error = fmax(
                local_maximum_state_error,
                Kokkos::abs(sample.q_state - sample.q_exact));
            local_maximum_state_error = fmax(
                local_maximum_state_error,
                Kokkos::abs(sample.p_state - sample.p_exact));
            ++local_count;
            const Real displacement[3] = {x, y, z};
            if (ref_gh::PunctureStencilIsClear(displacement, h, 2)) {
              local_safe_sum_w += weight;
              local_safe_sum_w2 += weight*weight;
              local_safe_sum_wq += weight*sample.q_state;
              local_sum_wq_fd += weight*sample.q_fd;
              local_maximum_fd_error = fmax(
                  local_maximum_fd_error,
                  Kokkos::abs(sample.q_fd - sample.q_exact));
              ++local_safe_count;
            }
          }, Kokkos::Sum<Real>(sum_w), Kokkos::Sum<Real>(sum_w2),
          Kokkos::Sum<Real>(sum_wq), Kokkos::Sum<Real>(sum_wq2),
          Kokkos::Sum<Real>(safe_sum_w), Kokkos::Sum<Real>(safe_sum_w2),
          Kokkos::Sum<Real>(safe_sum_wq),
          Kokkos::Sum<Real>(sum_wq_fd), Kokkos::Max<Real>(maximum_state_error),
          Kokkos::Max<Real>(maximum_fd_error), Kokkos::Sum<int>(count),
          Kokkos::Sum<int>(safe_count));
      const Real q_est = sum_wq/sum_w;
      const Real safe_q_est = safe_sum_wq/safe_sum_w;
      const Real q_fd_est = sum_wq_fd/safe_sum_w;
      const Real variance = fmax(0.0, sum_wq2/sum_w - q_est*q_est);
      const Real n_eff = sum_w*sum_w/sum_w2;
      const Real safe_n_eff = safe_sum_w*safe_sum_w/safe_sum_w2;
      if (count <= 0 || safe_count <= 0 || !(n_eff > 1.0)
          || !(safe_n_eff > 1.0) || !std::isfinite(variance)
          || maximum_state_error > 2.0e-11
          || (geometry_kind == 0 && maximum_fd_error > 2.0e-13)) {
        std::cout << "### FATAL ERROR: local puncture exponent estimator failed: "
                  << "geometry=" << geometry_kind << " h=" << h
                  << " count=" << count << " N_eff=" << n_eff
                  << " state-error=" << maximum_state_error
                  << " FD-error=" << maximum_fd_error << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real fd_difference = Kokkos::abs(q_fd_est - safe_q_est);
      Real fixed_fd_difference = 0.0;
      constexpr int nfixed_samples = 48;
      Kokkos::parallel_reduce(
          "ref_gh fixed-coordinate direct-FD exponent comparison",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, nfixed_samples),
          KOKKOS_LAMBDA(const int sample, Real &maximum) {
            const int permutation = sample % 6;
            const int signs = sample/6;
            const Real a = 0.75 + 0.5*h;
            const Real b = 0.25 + 0.5*h;
            const Real c = 0.125 + 0.5*h;
            Real coordinate[3];  // NOLINT(runtime/arrays)
            if (permutation == 0) {
              coordinate[0] = a; coordinate[1] = b; coordinate[2] = c;
            } else if (permutation == 1) {
              coordinate[0] = a; coordinate[1] = c; coordinate[2] = b;
            } else if (permutation == 2) {
              coordinate[0] = b; coordinate[1] = a; coordinate[2] = c;
            } else if (permutation == 3) {
              coordinate[0] = b; coordinate[1] = c; coordinate[2] = a;
            } else if (permutation == 4) {
              coordinate[0] = c; coordinate[1] = a; coordinate[2] = b;
            } else {
              coordinate[0] = c; coordinate[1] = b; coordinate[2] = a;
            }
            for (int p = 0; p < 3; ++p) {
              if ((signs & (1 << p)) != 0) coordinate[p] = -coordinate[p];
            }
            if (!ref_gh::PunctureStencilIsClear(coordinate, h, 2)) {
              maximum = std::numeric_limits<Real>::infinity();
              return;
            }
            const ExponentSample sample_value = EvaluateExponentSample(
                geometry_kind, table, mass, h, coordinate[0], coordinate[1],
                coordinate[2]);
            if (!sample_value.valid) {
              maximum = std::numeric_limits<Real>::infinity();
              return;
            }
            maximum = fmax(
                maximum,
                Kokkos::abs(sample_value.q_fd - sample_value.q_state));
          }, Kokkos::Max<Real>(fixed_fd_difference));
      if (resolution == 0) first_fd_difference[geometry_kind] = fd_difference;
      if (resolution == 0) {
        first_fixed_fd_difference[geometry_kind] = fixed_fd_difference;
      }
      if (resolution == nresolution - 1) {
        final_fd_difference[geometry_kind] = fd_difference;
        final_fixed_fd_difference[geometry_kind] = fixed_fd_difference;
      }
      if (geometry_kind == 1) {
        if (resolution > 0 && !(q_est > previous_wormhole)) {
          std::cout << "### FATAL ERROR: wormhole q_est did not approach 2."
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        previous_wormhole = q_est;
      }
      if (geometry_kind == 2) {
        if (resolution > 0 && !(Kokkos::abs(q_est - 1.0)
                                < Kokkos::abs(previous_trumpet - 1.0))) {
          std::cout << "### FATAL ERROR: trumpet q_est did not approach 1."
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        previous_trumpet = q_est;
      }
      std::cout << "reference-GH local exponent: geometry=" << geometry_kind
                << " h=" << h << " q_est=" << q_est
                << " safe_q_est=" << safe_q_est
                << " q_fd_est=" << q_fd_est << " variance=" << variance
                << " N_eff=" << n_eff << " samples=" << count
                << " safe_N_eff=" << safe_n_eff
                << " safe_samples=" << safe_count
                << " state-error=" << maximum_state_error
                << " FD-error=" << maximum_fd_error
                << " fixed-coordinate-FD-error=" << fixed_fd_difference
                << std::endl;
    }
  }
  const bool direct_fd_converged =
      final_fd_difference[0] <= 2.0e-13
      && final_fd_difference[1] < 0.5*first_fd_difference[1]
      && final_fd_difference[2] < 0.5*first_fd_difference[2];
  const bool fixed_direct_fd_converged =
      final_fixed_fd_difference[0] <= 2.0e-13
      && final_fixed_fd_difference[1]
             < 0.02*first_fixed_fd_difference[1]
      && final_fixed_fd_difference[2]
             < 0.02*first_fixed_fd_difference[2];
  std::cout << "reference-GH first-order-state puncture-exponent estimator passed; "
            << "direct-FD same-shell convergence="
            << (direct_fd_converged ? "PASS" : "FAIL")
            << " wormhole(initial,final)=(" << first_fd_difference[1] << ","
            << final_fd_difference[1] << ") trumpet(initial,final)=("
            << first_fd_difference[2] << "," << final_fd_difference[2] << ")"
            << " fixed-coordinate="
            << (fixed_direct_fd_converged ? "PASS" : "FAIL")
            << " wormhole(initial,final)=("
            << first_fixed_fd_difference[1] << ","
            << final_fixed_fd_difference[1] << ") trumpet(initial,final)=("
            << first_fixed_fd_difference[2] << ","
            << final_fixed_fd_difference[2] << ")"
            << std::endl;
  if (strict_gate && !direct_fd_converged) {
    std::cout << "### FATAL ERROR: the direct-FD estimator does not converge to "
                 "the first-order-state estimator on the prescribed r/h shell."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void ScanGenericSingularReference(ParameterInput *pin) {
  constexpr int nside = 8;
  constexpr int ncells = nside*nside*nside;
  constexpr int nmeasures = 16;
  constexpr int nresolutions = 6;
  constexpr Real inverse_resolutions[nresolutions] = {
      16.0, 24.0, 32.0, 48.0, 64.0, 128.0};
  constexpr int ntransition_times = 3;
  constexpr Real transition_times[ntransition_times] = {4.0, 8.0, 16.0};
  constexpr int nwidths = 3;
  constexpr Real widths[nwidths] = {2.0, 3.0, 4.0};
  const char *measure_names[nmeasures] = {
    "qdot-W-logrho", "qdot2-W2-logrho2", "dt-frame", "dtt-frame",
    "spin", "spin-derivative", "reference-Ricci", "reference-Riemann",
    "source-q", "source-delta", "source-curvature", "source-qq",
    "source-delta-product", "source-damping", "source-frame-correction",
    "source-total"
  };
  const std::string filename = pin->GetOrAddString(
      "problem", "generic_reference_scan_file",
      pin->GetString("job", "basename") + "-generic-reference-scan.tsv");
  FILE *file = nullptr;
  if (global_variable::my_rank == 0) {
    file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# mode\ttau_M\tR_G_M\th_M\tmeasure\tmaximum\tradius_M\t"
                       "radius_over_h\n");
  }
  DvceArray2D<Real> samples("generic singular reference scan", nmeasures, ncells);
  for (int tau_index = 0; tau_index < ntransition_times; ++tau_index) {
    for (int width_index = 0; width_index < nwidths; ++width_index) {
      for (int mode = 0; mode < 2; ++mode) {
        for (int resolution = 0; resolution < nresolutions; ++resolution) {
          const Real tau = transition_times[tau_index];
          const Real width = widths[width_index];
          const Real h = 1.0/inverse_resolutions[resolution];
          const Real time = 0.5*tau;
          const ref_gh::GenericSingularReferenceParameters params{
              1.0, {0.0, 0.0, 0.0}, width,
              mode == 0 ? 2.0 : 1.5, mode == 0 ? 1.0 : 1.5, tau};
          Kokkos::parallel_for(
              "ref_gh generic singular reference scan",
              Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
              KOKKOS_LAMBDA(const int index) {
                int work = index;
                const int ix = work % nside; work /= nside;
                const int iy = work % nside;
                const int iz = work/nside;
                const Real x = (static_cast<Real>(ix) - 0.5*(nside - 1))*h;
                const Real y = (static_cast<Real>(iy) - 0.5*(nside - 1))*h;
                const Real z = (static_cast<Real>(iz) - 0.5*(nside - 1))*h;
                const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
                ref_gh::ReferenceJet alpha;
                ref_gh::ReferenceJet spatial_cholesky;
                ref_gh::ReferenceJet shift_q;
                ref_gh::ReferenceJet q;
                ref_gh::ReferenceJet window;
                ref_gh::GenericSingularProfileJets(
                    params, time, x, y, z, alpha, spatial_cholesky, shift_q,
                    &q, &window);
                const Real logarithm = Kokkos::log(radius);
                samples(0, index) = Kokkos::abs(q.d[0]*window.value*logarithm);
                samples(1, index) = q.d[0]*q.d[0]*window.value*window.value
                                    *logarithm*logarithm;

                ref_gh::ReferenceGeometry reference;
                const ref_gh::GenericSingularReference provider{params};
                provider.Populate(time, x, y, z, reference);
                Real dt_frame = 0.0;
                Real dtt_frame = 0.0;
                Real spin = 0.0;
                Real spin_derivative = 0.0;
                Real ricci = 0.0;
                Real riemann = 0.0;
                for (int A = 0; A < 4; ++A) {
                  for (int a = 0; a < 4; ++a) {
                    dt_frame = fmax(dt_frame,
                                    Kokkos::abs(reference.d_frame[0][A][a]));
                    dtt_frame = fmax(
                        dtt_frame, Kokkos::abs(reference.dd_frame[0][0][A][a]));
                  }
                  for (int B = 0; B < 4; ++B) {
                    ricci = fmax(ricci, Kokkos::abs(reference.ricci_frame[A][B]));
                    for (int C = 0; C < 4; ++C) {
                      spin = fmax(spin, Kokkos::abs(reference.spin[A][B][C]));
                      for (int D = 0; D < 4; ++D) {
                        spin_derivative = fmax(
                            spin_derivative,
                            Kokkos::abs(reference.spin_derivative[A][B][C][D]));
                        riemann = fmax(
                            riemann,
                            Kokkos::abs(reference.riemann_frame[A][B][C][D]));
                      }
                    }
                  }
                }
                samples(2, index) = dt_frame;
                samples(3, index) = dtt_frame;
                samples(4, index) = spin;
                samples(5, index) = spin_derivative;
                samples(6, index) = ricci;
                samples(7, index) = riemann;

                Real psi[4][4] = {};       // NOLINT(runtime/arrays)
                Real pi[4][4] = {};        // NOLINT(runtime/arrays)
                Real phi[3][4][4] = {};   // NOLINT(runtime/arrays)
                for (int A = 0; A < 4; ++A) psi[A][A] = A == 0 ? -1.0 : 1.0;
                ref_gh::CoordinateGhGeometry geometry;
                Real determinant = 0.0;
                Real source[4][4];  // NOLINT(runtime/arrays)
                ref_gh::CovariantSourceSectors sectors;
                if (!ref_gh::ComputeCoordinateGhGeometry(
                        reference.metric, reference.d_metric, reference,
                        geometry, determinant)
                    || !ref_gh::CovariantGhScalarWaveSource(
                        psi, pi, phi, reference, geometry, 1.0, source, sectors)) {
                  for (int measure = 8; measure < nmeasures; ++measure) {
                    samples(measure, index) =
                        std::numeric_limits<Real>::infinity();
                  }
                  return;
                }
                Real source_q = 0.0;
                Real source_delta = 0.0;
                Real source_curvature = 0.0;
                Real source_qq = 0.0;
                Real source_delta_product = 0.0;
                Real source_damping = 0.0;
                Real source_frame_correction = 0.0;
                Real source_total = 0.0;
                for (int A = 0; A < 4; ++A) {
                  source_delta = fmax(
                      source_delta, Kokkos::abs(sectors.delta[A]));
                  for (int B = 0; B < 4; ++B) {
                    source_curvature = fmax(
                        source_curvature, Kokkos::abs(sectors.curvature[A][B]));
                    source_qq = fmax(source_qq, Kokkos::abs(sectors.qq[A][B]));
                    source_delta_product = fmax(
                        source_delta_product,
                        Kokkos::abs(sectors.delta_product[A][B]));
                    source_damping = fmax(
                        source_damping, Kokkos::abs(sectors.damping[A][B]));
                    source_frame_correction = fmax(
                        source_frame_correction,
                        Kokkos::abs(sectors.frame_correction[A][B]));
                    source_total = fmax(source_total, Kokkos::abs(source[A][B]));
                    for (int C = 0; C < 4; ++C) {
                      source_q = fmax(source_q,
                                      Kokkos::abs(sectors.q[A][B][C]));
                    }
                  }
                }
                samples(8, index) = source_q;
                samples(9, index) = source_delta;
                samples(10, index) = source_curvature;
                samples(11, index) = source_qq;
                samples(12, index) = source_delta_product;
                samples(13, index) = source_damping;
                samples(14, index) = source_frame_correction;
                samples(15, index) = source_total;
              });
          Kokkos::fence();
          using MaxLoc = Kokkos::MaxLoc<Real, int>;
          for (int measure = 0; measure < nmeasures; ++measure) {
            MaxLoc::value_type maximum;
            Kokkos::parallel_reduce(
                "ref_gh generic singular reference maximum",
                Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
                KOKKOS_LAMBDA(const int index, MaxLoc::value_type &local_maximum) {
                  const Real value = samples(measure, index);
                  if (value >= local_maximum.val) {
                    local_maximum.val = value;
                    local_maximum.loc = index;
                  }
                }, MaxLoc(maximum));
            int location = maximum.loc;
            const int ix = location % nside; location /= nside;
            const int iy = location % nside;
            const int iz = location/nside;
            const Real x = (static_cast<Real>(ix) - 0.5*(nside - 1))*h;
            const Real y = (static_cast<Real>(iy) - 0.5*(nside - 1))*h;
            const Real z = (static_cast<Real>(iz) - 0.5*(nside - 1))*h;
            const Real radius = std::sqrt(x*x + y*y + z*z);
            if (file != nullptr) {
              std::fprintf(file, "%s\t%.17e\t%.17e\t%.17e\t%s\t%.17e\t%.17e\t"
                                 "%.17e\n", mode == 0 ? "dynamic" : "static",
                           tau, width, h,
                           measure_names[measure], maximum.val, radius, radius/h);
            }
          }
        }
      }
    }
  }
  if (file != nullptr) std::fclose(file);
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH generic singular reference scan written to "
              << filename << std::endl;
  }
}

void CheckPhiOrderingAlgebra() {
  constexpr int nsamples = 1024;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh phi ordering algebra",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real scale = static_cast<Real>(sample % 29 - 14)/14.0;
        const Real alpha = 0.61 + 0.19*static_cast<Real>(sample % 17)/16.0;
        // Include shifts both below and well above the lapse.
        const Real beta[3] = {
          2.4*scale, -1.7 + 0.03*static_cast<Real>(sample % 11),
          0.9 - 0.02*static_cast<Real>(sample % 13)};
        Real phi[3];                    // NOLINT(runtime/arrays)
        Real frame_derivative[3][3];   // NOLINT(runtime/arrays)
        Real structure[3][3][3];       // NOLINT(runtime/arrays)
        for (int I = 0; I < 3; ++I) {
          phi[I] = 0.13*static_cast<Real>(I + 1) - 0.07*scale;
          for (int J = 0; J < 3; ++J) {
            frame_derivative[I][J] =
                0.11*static_cast<Real>(2*I - J + 1) + 0.04*scale;
            for (int K = 0; K < 3; ++K) {
              structure[I][J][K] = 0.017*static_cast<Real>(I - J)
                                    *static_cast<Real>(K + 1);
            }
          }
        }

        // The helper must reproduce the exact rewrite
        // E_I Phi_J = E_J Phi_I + c^K_IJ Phi_K + C_IJ.
        for (int I = 0; I < 3; ++I) {
          const Real grad_pi = 0.09*static_cast<Real>(I + 1) - 0.03*scale;
          Real compatible = -alpha*grad_pi;
          Real expected_standard = -alpha*grad_pi;
          for (int J = 0; J < 3; ++J) {
            compatible += beta[J]*frame_derivative[I][J];
            Real commutator = 0.0;
            for (int K = 0; K < 3; ++K) {
              commutator += structure[I][J][K]*phi[K];
            }
            expected_standard += beta[J]
                *(frame_derivative[J][I] + commutator);
          }
          const Real standard = compatible - ref_gh::StandardPhiOrderingCorrection(
              I, beta, frame_derivative, structure, phi);
          local_maximum = fmax(local_maximum,
                               Kokkos::abs(standard - expected_standard));
        }

        // On an integrable state, construct derivatives whose antisymmetric
        // part is exactly the non-coordinate-frame commutator.  Compatible and
        // standard orderings must then agree.
        Real constrained_derivative[3][3];  // NOLINT(runtime/arrays)
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            Real commutator = 0.0;
            for (int K = 0; K < 3; ++K) {
              commutator += structure[I][J][K]*phi[K];
            }
            const Real symmetric = 0.08*static_cast<Real>(I + J + 1) - 0.02*scale;
            constrained_derivative[I][J] = symmetric + 0.5*commutator;
          }
        }
        for (int I = 0; I < 3; ++I) {
          local_maximum = fmax(local_maximum, Kokkos::abs(
              ref_gh::StandardPhiOrderingCorrection(
                  I, beta, constrained_derivative, structure, phi)));
        }

        // Manufactured plane-wave principal symbol.  The standard correction
        // must turn beta^J n_I Phi_J into beta^J n_J Phi_I for arbitrary shift.
        const Real n[3] = {0.36, -0.48, 0.80};
        const Real pi_amplitude = 0.37 - 0.05*scale;
        Real plane_derivative[3][3];  // NOLINT(runtime/arrays)
        Real zero_structure[3][3][3] = {};  // NOLINT(runtime/arrays)
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            plane_derivative[I][J] = n[I]*phi[J];
          }
        }
        Real beta_n = 0.0;
        for (int J = 0; J < 3; ++J) beta_n += beta[J]*n[J];
        for (int I = 0; I < 3; ++I) {
          Real compatible = -alpha*n[I]*pi_amplitude;
          for (int J = 0; J < 3; ++J) {
            compatible += beta[J]*plane_derivative[I][J];
          }
          const Real standard = compatible - ref_gh::StandardPhiOrderingCorrection(
              I, beta, plane_derivative, zero_structure, phi);
          const Real expected = -alpha*n[I]*pi_amplitude + beta_n*phi[I];
          local_maximum = fmax(local_maximum, Kokkos::abs(standard - expected));
        }

        // The normal (Pi,Phi_n) block [[beta_n,-alpha],[-alpha,beta_n]]
        // has the complete real eigenbasis (1,1), (1,-1), even for |beta_n|>alpha.
        const Real lambda_minus = beta_n - alpha;
        const Real lambda_plus = beta_n + alpha;
        const Real minus_lhs_pi = beta_n*1.0 - alpha*1.0;
        const Real minus_lhs_phi = -alpha*1.0 + beta_n*1.0;
        const Real plus_lhs_pi = beta_n*1.0 - alpha*(-1.0);
        const Real plus_lhs_phi = -alpha*1.0 + beta_n*(-1.0);
        local_maximum = fmax(local_maximum,
            Kokkos::abs(minus_lhs_pi - lambda_minus));
        local_maximum = fmax(local_maximum,
            Kokkos::abs(minus_lhs_phi - lambda_minus));
        local_maximum = fmax(local_maximum,
            Kokkos::abs(plus_lhs_pi - lambda_plus));
        local_maximum = fmax(local_maximum,
            Kokkos::abs(plus_lhs_phi + lambda_plus));
      }, Kokkos::Max<Real>(maximum));
  Kokkos::fence();
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH Phi-ordering algebra maximum error = "
              << maximum << std::endl;
  }
  if (!(maximum < 2.0e-13)) {
    std::cout << "### FATAL ERROR: reference-GH Phi-ordering algebra failed."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

KOKKOS_INLINE_FUNCTION
ref_gh::ReferenceJet SpatialJet(const Real value, const Real dx, const Real dy,
                                const Real dz, const Real dxx, const Real dyy,
                                const Real dzz, const Real dxy) {
  ref_gh::ReferenceJet result = ref_gh::ConstantJet(value);
  result.d[1] = dx;
  result.d[2] = dy;
  result.d[3] = dz;
  result.dd[1][1] = dxx;
  result.dd[2][2] = dyy;
  result.dd[3][3] = dzz;
  result.dd[1][2] = dxy;
  result.dd[2][1] = dxy;
  return result;
}

// A stationary, foliation-adapted manufactured reference with nonzero shift,
// spin connection, and curvature. The production frame-completion routine
// generates its spin derivative and Riemann data from one coordinate 2-jet.
KOKKOS_INLINE_FUNCTION
void ManufacturedNonflatReference(const int sample, ref_gh::ReferenceGeometry &reference) {
  ref_gh::ZeroReferenceGeometry(reference);
  const Real perturbation = 0.002*static_cast<Real>(sample % 17 - 8);
  const ref_gh::ReferenceJet lapse = SpatialJet(
      0.87 + perturbation, 0.071, -0.039, 0.022, 0.031, -0.017, 0.013, 0.019);
  const ref_gh::ReferenceJet scale_x = SpatialJet(
      1.14 - perturbation, -0.058, 0.043, -0.021, 0.026, 0.014, -0.019, -0.016);
  const ref_gh::ReferenceJet scale_y = SpatialJet(
      1.08 + 0.5*perturbation, 0.037, 0.062, 0.018, -0.022, 0.029, 0.015, 0.011);
  const ref_gh::ReferenceJet scale_z = SpatialJet(
      1.19 - 0.5*perturbation, -0.029, 0.024, 0.067, 0.018, -0.021, 0.034, -0.013);
  const ref_gh::ReferenceJet shift = SpatialJet(
      0.16 + 0.25*perturbation, 0.046, -0.033, 0.028, 0.014, -0.018, 0.012, 0.017);
  const ref_gh::ReferenceJet inverse_lapse = ref_gh::Reciprocal(lapse);
  const ref_gh::ReferenceJet inverse_scale_x = ref_gh::Reciprocal(scale_x);
  const ref_gh::ReferenceJet inverse_scale_y = ref_gh::Reciprocal(scale_y);
  const ref_gh::ReferenceJet inverse_scale_z = ref_gh::Reciprocal(scale_z);
  ref_gh::ReferenceJet coframe[4][4];  // NOLINT(runtime/arrays)
  ref_gh::ReferenceJet frame[4][4];    // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      coframe[A][a] = ref_gh::ConstantJet(0.0);
      frame[A][a] = ref_gh::ConstantJet(0.0);
    }
  }
  coframe[0][0] = lapse;
  coframe[1][0] = scale_x*shift;
  coframe[1][1] = scale_x;
  coframe[2][2] = scale_y;
  coframe[3][3] = scale_z;
  frame[0][0] = inverse_lapse;
  frame[0][1] = -(shift*inverse_lapse);
  frame[1][1] = inverse_scale_x;
  frame[2][2] = inverse_scale_y;
  frame[3][3] = inverse_scale_z;
  ref_gh::ReferenceJet metric[4][4];          // NOLINT(runtime/arrays)
  ref_gh::ReferenceJet inverse_metric[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = -(coframe[0][a]*coframe[0][b]);
      inverse_metric[a][b] = -(frame[0][a]*frame[0][b]);
      for (int I = 1; I < 4; ++I) {
        metric[a][b] = metric[a][b] + coframe[I][a]*coframe[I][b];
        inverse_metric[a][b] = inverse_metric[a][b] + frame[I][a]*frame[I][b];
      }
      reference.metric[a][b] = metric[a][b].value;
      reference.inverse_metric[a][b] = inverse_metric[a][b].value;
      reference.coframe[a][b] = coframe[a][b].value;
      reference.frame[a][b] = frame[a][b].value;
      for (int c = 0; c < 4; ++c) {
        reference.d_metric[c][a][b] = metric[a][b].d[c];
        reference.d_frame[c][a][b] = frame[a][b].d[c];
        for (int d = 0; d < 4; ++d) {
          reference.dd_metric[c][d][a][b] = metric[a][b].dd[c][d];
          reference.dd_frame[c][d][a][b] = frame[a][b].dd[c][d];
        }
      }
    }
  }
  Real first_kind[4][4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        first_kind[a][b][c] = 0.5*(reference.d_metric[b][a][c]
                                   + reference.d_metric[c][a][b]
                                   - reference.d_metric[a][b][c]);
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        for (int ell = 0; ell < 4; ++ell) {
          reference.christoffel[a][b][c] +=
              reference.inverse_metric[a][ell]*first_kind[ell][b][c];
        }
        for (int p = 0; p < 4; ++p) {
          for (int ell = 0; ell < 4; ++ell) {
            const Real d_first = 0.5*(reference.dd_metric[p][b][ell][c]
                                      + reference.dd_metric[p][c][ell][b]
                                      - reference.dd_metric[p][ell][b][c]);
            reference.d_christoffel[p][a][b][c] +=
                inverse_metric[a][ell].d[p]*first_kind[ell][b][c]
                + reference.inverse_metric[a][ell]*d_first;
          }
        }
      }
    }
  }
  ref_gh::CompleteReferenceFrameGeometry(reference);
}

KOKKOS_INLINE_FUNCTION
void ManufacturedFrameState(const int sample, Real psi[4][4], Real p[4][4][4]) {
  Real spatial[3][3];  // NOLINT(runtime/arrays)
  const Real scale = static_cast<Real>(sample % 19 - 9)/9.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) spatial[i][j] = 0.0;
  }
  spatial[0][0] = 1.17 + 0.04*scale;
  spatial[1][1] = 1.29 - 0.03*scale;
  spatial[2][2] = 1.36 + 0.02*scale;
  spatial[0][1] = spatial[1][0] = 0.031*scale;
  spatial[0][2] = spatial[2][0] = -0.019*scale;
  spatial[1][2] = spatial[2][1] = 0.014*scale;
  const Real lapse = 0.79 + 0.05*static_cast<Real>(sample % 11)/10.0;
  const Real shift[3] = {0.071*scale, -0.053*scale, 0.037*scale};
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      psi[A][B] = 0.0;
      for (int C = 0; C < 4; ++C) p[C][A][B] = 0.0;
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      psi[i + 1][j + 1] = spatial[i][j];
      psi[0][i + 1] += spatial[i][j]*shift[j];
    }
    psi[i + 1][0] = psi[0][i + 1];
    psi[0][0] += shift[i]*psi[0][i + 1];
  }
  psi[0][0] -= lapse*lapse;
  for (int C = 0; C < 4; ++C) {
    for (int A = 0; A < 4; ++A) {
      for (int B = A; B < 4; ++B) {
        const Real value = 4.0e-3*(static_cast<Real>(C + 1)*(A + B + 2)
            - 0.13*static_cast<Real>((sample + 2*A + 3*B + 5*C) % 23));
        p[C][A][B] = value;
        p[C][B][A] = value;
      }
    }
  }
}

KOKKOS_INLINE_FUNCTION
void CoordinateStateFromFrame(const ref_gh::ReferenceGeometry &reference,
                              const Real psi[4][4], const Real p[4][4][4],
                              Real d_psi[4][4][4], Real metric[4][4],
                              Real d_metric[4][4][4]) {
  for (int c = 0; c < 4; ++c) {
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        d_psi[c][A][B] = 0.0;
        for (int C = 0; C < 4; ++C) d_psi[c][A][B] += reference.coframe[C][c]
                                                           *p[C][A][B];
      }
    }
  }
  Real d_coframe[4][4][4];  // NOLINT(runtime/arrays)
  for (int c = 0; c < 4; ++c) {
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        d_coframe[c][A][a] = 0.0;
        for (int B = 0; B < 4; ++B) {
          for (int b = 0; b < 4; ++b) {
            d_coframe[c][A][a] -= reference.coframe[B][a]
                                      *reference.d_frame[c][B][b]
                                      *reference.coframe[A][b];
          }
        }
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = 0.0;
      for (int c = 0; c < 4; ++c) d_metric[c][a][b] = 0.0;
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          metric[a][b] += psi[A][B]*reference.coframe[A][a]*reference.coframe[B][b];
          for (int c = 0; c < 4; ++c) {
            d_metric[c][a][b] += d_psi[c][A][B]*reference.coframe[A][a]
                                  *reference.coframe[B][b]
                                + psi[A][B]*d_coframe[c][A][a]
                                  *reference.coframe[B][b]
                                + psi[A][B]*reference.coframe[A][a]*d_coframe[c][B][b];
          }
        }
      }
    }
  }
}

void CheckFlatCovariantSource() {
  constexpr int nsamples = 1000;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh flat covariant source", Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        ref_gh::ReferenceGeometry reference = ref_gh::MinkowskiReference{}(0.0, 0.0, 0.0,
                                                                              0.0);
        Real psi[4][4];       // NOLINT(runtime/arrays)
        Real d_psi[4][4][4]; // NOLINT(runtime/arrays)
        Real pi[4][4];        // NOLINT(runtime/arrays)
        Real phi[3][4][4];    // NOLINT(runtime/arrays)
        Real gamma[3][3];     // NOLINT(runtime/arrays)
        Real beta[3];         // NOLINT(runtime/arrays)
        const Real lapse = 0.73 + 0.11*static_cast<Real>(sample % 17)/16.0;
        for (int i = 0; i < 3; ++i) {
          beta[i] = 0.055*static_cast<Real>((sample + 3*i) % 13 - 6)/6.0;
          for (int j = 0; j < 3; ++j) gamma[i][j] = 0.0;
        }
        gamma[0][0] = 1.13 + 0.09*static_cast<Real>(sample % 7)/6.0;
        gamma[1][1] = 1.27 + 0.07*static_cast<Real>(sample % 11)/10.0;
        gamma[2][2] = 1.41 + 0.06*static_cast<Real>(sample % 5)/4.0;
        gamma[0][1] = gamma[1][0] = 0.018*static_cast<Real>((sample % 9) - 4)/4.0;
        gamma[0][2] = gamma[2][0] = -0.014*static_cast<Real>((sample % 8) - 3)/4.0;
        gamma[1][2] = gamma[2][1] = 0.011*static_cast<Real>((sample % 6) - 2)/3.0;
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            psi[a][b] = 0.0;
            for (int c = 0; c < 4; ++c) d_psi[c][a][b] = 0.0;
          }
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            psi[i + 1][j + 1] = gamma[i][j];
            psi[0][i + 1] += gamma[i][j]*beta[j];
          }
          psi[i + 1][0] = psi[0][i + 1];
          psi[0][0] += beta[i]*psi[0][i + 1];
        }
        psi[0][0] -= lapse*lapse;
        for (int c = 0; c < 4; ++c) {
          for (int a = 0; a < 4; ++a) {
            for (int b = a; b < 4; ++b) {
              const Real derivative = 2.5e-3*(
                  static_cast<Real>(c + 1)*static_cast<Real>(a + b + 2)
                  - 0.17*static_cast<Real>((sample + 3*a + 5*b + 7*c) % 19));
              d_psi[c][a][b] = derivative;
              d_psi[c][b][a] = derivative;
            }
          }
        }
        ref_gh::CoordinateGhGeometry geometry;
        Real determinant = 0.0;
        if (!ref_gh::ComputeCoordinateGhGeometry(psi, d_psi, reference, geometry,
                                                  determinant)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            pi[a][b] = 0.0;
            for (int c = 0; c < 4; ++c) pi[a][b] -= geometry.normal_upper[c]
                                                        *d_psi[c][a][b];
            for (int I = 0; I < 3; ++I) phi[I][a][b] = d_psi[I + 1][a][b];
          }
        }
        ref_gh::CovariantSourceSectors sectors;
        Real covariant[4][4];      // NOLINT(runtime/arrays)
        Real production[4][4];     // NOLINT(runtime/arrays)
        Real coordinate_partial[4][4];  // NOLINT(runtime/arrays)
        Real coordinate[4][4];     // NOLINT(runtime/arrays)
        if (!ref_gh::CovariantGhScalarWaveSource(psi, pi, phi, reference, geometry,
                                                  1.3, covariant, sectors)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        if (!ref_gh::CovariantGhScalarWaveSourceProduction(
                psi, pi, phi, reference, geometry, 1.3, production)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        ref_gh::StandardGhPartialWaveSource(psi, d_psi, reference, geometry, 1.3,
                                             coordinate_partial);
        ref_gh::TransformPartialWaveSource(psi, d_psi, coordinate_partial, d_psi,
                                            reference, geometry, coordinate);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            if (a <= b) {
              const Real production_error =
                  Kokkos::abs(production[a][b] - covariant[a][b]);
              const Real production_tolerance =
                  256.0*std::numeric_limits<Real>::epsilon()
                  *(1.0 + Kokkos::abs(covariant[a][b]));
              if (!(production_error <= production_tolerance)) {
                local_maximum = fmax(local_maximum, 1.0e30);
              }
            }
            const Real error = Kokkos::abs(covariant[a][b] - coordinate[a][b]);
            if (!Kokkos::isfinite(error)) {
              local_maximum = fmax(local_maximum, 1.0e30);
            } else {
              local_maximum = fmax(local_maximum, error);
            }
          }
        }
      }, Kokkos::Max<Real>(maximum));
  if (maximum > 1.0e-11) {
    std::cout << "reference-GH flat covariant source unit failed: max error = "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH flat covariant source unit passed: samples = "
            << nsamples << ", max error = " << maximum << std::endl;
}

void CheckNonflatCovariantSource() {
  constexpr int nsamples = 128;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh nonflat covariant source", Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        ref_gh::ReferenceGeometry reference;
        ManufacturedNonflatReference(sample, reference);
        Real psi[4][4];       // NOLINT(runtime/arrays)
        Real p[4][4][4];      // NOLINT(runtime/arrays)
        Real d_psi[4][4][4];  // NOLINT(runtime/arrays)
        Real metric[4][4];    // NOLINT(runtime/arrays)
        Real d_metric[4][4][4];  // NOLINT(runtime/arrays)
        ManufacturedFrameState(sample, psi, p);
        CoordinateStateFromFrame(reference, psi, p, d_psi, metric, d_metric);
        ref_gh::CoordinateGhGeometry geometry;
        Real determinant = 0.0;
        if (!ref_gh::ComputeCoordinateGhGeometry(metric, d_metric, reference, geometry,
                                                  determinant)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        Real normal[4];  // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          normal[A] = 0.0;
          for (int a = 0; a < 4; ++a) normal[A] += reference.coframe[A][a]
                                                     *geometry.normal_upper[a];
        }
        Real pi[4][4];       // NOLINT(runtime/arrays)
        Real phi[3][4][4];   // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            pi[A][B] = 0.0;
            for (int C = 0; C < 4; ++C) pi[A][B] -= normal[C]*p[C][A][B];
            for (int I = 0; I < 3; ++I) phi[I][A][B] = p[I + 1][A][B];
          }
        }
        ref_gh::CovariantSourceSectors sectors;
        Real covariant[4][4];       // NOLINT(runtime/arrays)
        Real production[4][4];      // NOLINT(runtime/arrays)
        Real coordinate_partial[4][4];  // NOLINT(runtime/arrays)
        Real coordinate[4][4];      // NOLINT(runtime/arrays)
        if (!ref_gh::CovariantGhScalarWaveSource(psi, pi, phi, reference, geometry,
                                                  1.3, covariant, sectors)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        if (!ref_gh::CovariantGhScalarWaveSourceProduction(
                psi, pi, phi, reference, geometry, 1.3, production)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        ref_gh::StandardGhPartialWaveSource(metric, d_metric, reference, geometry, 1.3,
                                             coordinate_partial);
        ref_gh::TransformPartialWaveSource(metric, d_metric, coordinate_partial, d_psi,
                                            reference, geometry, coordinate);
        Real reference_scale = 0.0;
        Real spin_scale = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            for (int C = 0; C < 4; ++C) {
              for (int D = 0; D < 4; ++D) {
                reference_scale = fmax(reference_scale,
                    Kokkos::abs(reference.riemann_frame[A][B][C][D]));
              }
              spin_scale = fmax(spin_scale, Kokkos::abs(reference.spin[A][B][C]));
            }
            const Real error = Kokkos::abs(covariant[A][B] - coordinate[A][B]);
            local_maximum = fmax(local_maximum, error);
            if (A <= B) {
              const Real production_error =
                  Kokkos::abs(production[A][B] - covariant[A][B]);
              const Real production_tolerance =
                  256.0*std::numeric_limits<Real>::epsilon()
                  *(1.0 + Kokkos::abs(covariant[A][B]));
              if (!(production_error <= production_tolerance)) {
                local_maximum = fmax(local_maximum, 1.0e30);
              }
            }
          }
        }
        if (!(reference_scale > 1.0e-5) || !(spin_scale > 1.0e-5)) {
          local_maximum = fmax(local_maximum, 1.0e30);
        }
      }, Kokkos::Max<Real>(maximum));
  if (maximum > 1.0e-10) {
    std::cout << "reference-GH nonflat covariant source unit failed: max error = "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH nonflat covariant source unit passed: samples = "
            << nsamples << ", max error = " << maximum << std::endl;
}

void CheckDynamicSpatialReference() {
  constexpr int nsamples = 128;
  Real curvature_error = 0.0;
  Real curvature_scale = 0.0;
  Real spin_scale = 0.0;
  Real dt_frame_scale = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh dynamic spatial curvature error",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real time = -0.37 + 0.013*static_cast<Real>(sample);
        const Real x = -0.81 + 0.017*static_cast<Real>(sample % 31);
        const Real y = 0.63 - 0.021*static_cast<Real>(sample % 29);
        const Real z = -0.44 + 0.019*static_cast<Real>(sample % 23);
        ref_gh::ReferenceGeometry reference;
        ref_gh::TimeDependentSpatialReference().Populate(time, x, y, z,
                                                          reference);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            for (int C = 0; C < 4; ++C) {
              for (int D = 0; D < 4; ++D) {
                const Real coordinate = ref_gh::CoordinateReferenceRiemannFrame(
                    reference, A, B, C, D);
                const Real error = Kokkos::abs(
                    reference.riemann_frame[A][B][C][D] - coordinate);
                local_maximum = fmax(local_maximum, error);
              }
            }
          }
        }
      }, Kokkos::Max<Real>(curvature_error));
  Kokkos::parallel_reduce(
      "ref_gh dynamic spatial curvature scale",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real time = -0.37 + 0.013*static_cast<Real>(sample);
        const Real x = -0.81 + 0.017*static_cast<Real>(sample % 31);
        const Real y = 0.63 - 0.021*static_cast<Real>(sample % 29);
        const Real z = -0.44 + 0.019*static_cast<Real>(sample % 23);
        ref_gh::ReferenceGeometry reference;
        ref_gh::TimeDependentSpatialReference().Populate(time, x, y, z,
                                                          reference);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            for (int C = 0; C < 4; ++C) {
              for (int D = 0; D < 4; ++D) {
                local_maximum = fmax(local_maximum,
                    Kokkos::abs(ref_gh::CoordinateReferenceRiemannFrame(
                        reference, A, B, C, D)));
              }
            }
          }
        }
      }, Kokkos::Max<Real>(curvature_scale));
  Kokkos::parallel_reduce(
      "ref_gh dynamic spatial spin scale",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real time = -0.37 + 0.013*static_cast<Real>(sample);
        const Real x = -0.81 + 0.017*static_cast<Real>(sample % 31);
        const Real y = 0.63 - 0.021*static_cast<Real>(sample % 29);
        const Real z = -0.44 + 0.019*static_cast<Real>(sample % 23);
        ref_gh::ReferenceGeometry reference;
        ref_gh::TimeDependentSpatialReference().Populate(time, x, y, z,
                                                          reference);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            for (int C = 0; C < 4; ++C) {
              local_maximum = fmax(local_maximum,
                  Kokkos::abs(reference.spin[A][B][C]));
            }
          }
        }
      }, Kokkos::Max<Real>(spin_scale));
  Kokkos::parallel_reduce(
      "ref_gh dynamic spatial dt frame scale",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real time = -0.37 + 0.013*static_cast<Real>(sample);
        ref_gh::ReferenceGeometry reference;
        ref_gh::TimeDependentSpatialReference().Populate(time, 0.31, -0.27,
                                                          0.19, reference);
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            local_maximum = fmax(local_maximum,
                Kokkos::abs(reference.dt_spatial_frame[I][J]));
          }
        }
      }, Kokkos::Max<Real>(dt_frame_scale));

  Real source_error = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh dynamic spatial source oracle",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real time = -0.37 + 0.013*static_cast<Real>(sample);
        const Real x = -0.81 + 0.017*static_cast<Real>(sample % 31);
        const Real y = 0.63 - 0.021*static_cast<Real>(sample % 29);
        const Real z = -0.44 + 0.019*static_cast<Real>(sample % 23);
        ref_gh::ReferenceGeometry reference;
        ref_gh::TimeDependentSpatialReference().Populate(time, x, y, z,
                                                          reference);
        Real psi[4][4];       // NOLINT(runtime/arrays)
        Real p[4][4][4];      // NOLINT(runtime/arrays)
        Real d_psi[4][4][4];  // NOLINT(runtime/arrays)
        Real metric[4][4];    // NOLINT(runtime/arrays)
        Real d_metric[4][4][4];  // NOLINT(runtime/arrays)
        ManufacturedFrameState(sample, psi, p);
        CoordinateStateFromFrame(reference, psi, p, d_psi, metric, d_metric);
        ref_gh::CoordinateGhGeometry geometry;
        Real determinant = 0.0;
        if (!ref_gh::ComputeCoordinateGhGeometry(
                metric, d_metric, reference, geometry, determinant)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        Real normal[4];  // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          normal[A] = 0.0;
          for (int a = 0; a < 4; ++a) {
            normal[A] += reference.coframe[A][a]*geometry.normal_upper[a];
          }
        }
        Real pi[4][4];       // NOLINT(runtime/arrays)
        Real phi[3][4][4];   // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            pi[A][B] = 0.0;
            for (int C = 0; C < 4; ++C) pi[A][B] -= normal[C]*p[C][A][B];
            for (int I = 0; I < 3; ++I) phi[I][A][B] = p[I + 1][A][B];
          }
        }
        ref_gh::CovariantSourceSectors sectors;
        Real covariant[4][4];            // NOLINT(runtime/arrays)
        Real production[4][4];           // NOLINT(runtime/arrays)
        Real coordinate_partial[4][4];   // NOLINT(runtime/arrays)
        Real coordinate_source[4][4];    // NOLINT(runtime/arrays)
        if (!ref_gh::CovariantGhScalarWaveSource(
                psi, pi, phi, reference, geometry, 1.3, covariant, sectors)
            || !ref_gh::CovariantGhScalarWaveSourceProduction(
                psi, pi, phi, reference, geometry, 1.3, production)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        ref_gh::StandardGhPartialWaveSource(
            metric, d_metric, reference, geometry, 1.3, coordinate_partial);
        ref_gh::TransformPartialWaveSource(
            metric, d_metric, coordinate_partial, d_psi,
            reference, geometry, coordinate_source);
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            local_maximum = fmax(local_maximum,
                Kokkos::abs(covariant[A][B] - coordinate_source[A][B]));
            if (A <= B) {
              local_maximum = fmax(local_maximum,
                  Kokkos::abs(covariant[A][B] - production[A][B]));
            }
          }
        }
      }, Kokkos::Max<Real>(source_error));

  constexpr Real kCurvatureTolerance = 2.0e-13;
  constexpr Real kSourceTolerance = 1.0e-10;
  if (!(curvature_error <= kCurvatureTolerance)
      || !(curvature_scale <= kCurvatureTolerance)
      || !(source_error <= kSourceTolerance)
      || !(spin_scale <= kCurvatureTolerance)
      || !(dt_frame_scale > 1.0e-5)) {
    std::cout << "reference-GH dynamic spatial oracle failed: frame-coordinate "
              << "curvature Linf=" << curvature_error
              << ", coordinate curvature Linf=" << curvature_scale
              << ", source Linf=" << source_error
              << ", spin scale=" << spin_scale
              << ", dt spatial frame scale=" << dt_frame_scale << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH dynamic spatial oracle passed: frame-coordinate "
            << "curvature Linf=" << curvature_error
            << ", coordinate curvature Linf=" << curvature_scale
            << ", source Linf=" << source_error
            << ", spin scale=" << spin_scale
            << ", dt spatial frame scale=" << dt_frame_scale << std::endl;
}

void FillTrumpetTable(DvceArray2D<Real> &table) {
  Kokkos::realloc(table, ref_gh::kTrumpetProfiles, ref_gh::kTrumpetTableSize);
  auto host = Kokkos::create_mirror_view(table);
  for (int i = 0; i < ref_gh::kTrumpetTableSize; ++i) {
    host(ref_gh::kProfileAlpha, i) = ref_gh::kTrumpetAlpha[i];
    host(ref_gh::kProfileAlphaDy, i) = ref_gh::kTrumpetAlphaDy[i];
    host(ref_gh::kProfileAlphaDyy, i) = ref_gh::kTrumpetAlphaDyy[i];
    host(ref_gh::kProfileArealRadius, i) = ref_gh::kTrumpetArealRadius[i];
    host(ref_gh::kProfileArealRadiusDy, i) = ref_gh::kTrumpetArealRadiusDy[i];
    host(ref_gh::kProfileArealRadiusDyy, i) = ref_gh::kTrumpetArealRadiusDyy[i];
    host(ref_gh::kProfileShiftQ, i) = ref_gh::kTrumpetShiftQ[i];
    host(ref_gh::kProfileShiftQDy, i) = ref_gh::kTrumpetShiftQDy[i];
    host(ref_gh::kProfileShiftQDyy, i) = ref_gh::kTrumpetShiftQDyy[i];
    host(ref_gh::kCoeffAlpha, i) = ref_gh::kTrumpetAlphaA0[i];
    host(ref_gh::kCoeffAlpha + 1, i) = ref_gh::kTrumpetAlphaA1[i];
    host(ref_gh::kCoeffAlpha + 2, i) = ref_gh::kTrumpetAlphaA2[i];
    host(ref_gh::kCoeffAlpha + 3, i) = ref_gh::kTrumpetAlphaA3[i];
    host(ref_gh::kCoeffAlpha + 4, i) = ref_gh::kTrumpetAlphaA4[i];
    host(ref_gh::kCoeffAlpha + 5, i) = ref_gh::kTrumpetAlphaA5[i];
    host(ref_gh::kCoeffArealRadius, i) = ref_gh::kTrumpetArealRadiusA0[i];
    host(ref_gh::kCoeffArealRadius + 1, i) = ref_gh::kTrumpetArealRadiusA1[i];
    host(ref_gh::kCoeffArealRadius + 2, i) = ref_gh::kTrumpetArealRadiusA2[i];
    host(ref_gh::kCoeffArealRadius + 3, i) = ref_gh::kTrumpetArealRadiusA3[i];
    host(ref_gh::kCoeffArealRadius + 4, i) = ref_gh::kTrumpetArealRadiusA4[i];
    host(ref_gh::kCoeffArealRadius + 5, i) = ref_gh::kTrumpetArealRadiusA5[i];
    host(ref_gh::kCoeffShiftQ, i) = ref_gh::kTrumpetShiftQA0[i];
    host(ref_gh::kCoeffShiftQ + 1, i) = ref_gh::kTrumpetShiftQA1[i];
    host(ref_gh::kCoeffShiftQ + 2, i) = ref_gh::kTrumpetShiftQA2[i];
    host(ref_gh::kCoeffShiftQ + 3, i) = ref_gh::kTrumpetShiftQA3[i];
    host(ref_gh::kCoeffShiftQ + 4, i) = ref_gh::kTrumpetShiftQA4[i];
    host(ref_gh::kCoeffShiftQ + 5, i) = ref_gh::kTrumpetShiftQA5[i];
  }
  Kokkos::deep_copy(table, host);
}

void ScanReferencePaths(ParameterInput *pin) {
  constexpr int kSamples = 32769;
  constexpr int kMeasures = 7;
  constexpr Real times[] = {0.0, 0.5, 1.0, 1.25, 1.4,
                            1.5, 1.6, 1.7, 2.0};
  constexpr const char *path_names[] = {
    "shrinking_width", "frozen_wormhole", "fixed_core"
  };
  constexpr const char *measure_names[kMeasures] = {
    "Ricci", "Riemann", "spin", "spin_derivative",
    "matched_source", "dB_dr", "d2B_dr2"
  };
  DvceArray2D<Real> table("ref_gh path scan trumpet table", 1, 1);
  FillTrumpetTable(table);
  DvceArray2D<Real> samples("ref_gh path scan samples", kMeasures, kSamples);
  const ref_gh::ControlledReferenceParameters base{
      1.0, {0.0, 0.0, 0.0}, 0.30, 1.5, 1.0,
      ref_gh::kShrinkingWidthPath, 0.20, 4.0,
      ref_gh::kLegacyTimeActivation, 0.0, 0.0, 0.0, 0.50, 0.60,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  const std::string filename =
      pin->GetString("job", "basename") + ".reference_path_scan.tsv";
  FILE *file = nullptr;
  if (global_variable::my_rank == 0) {
    file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) {
      std::cout << "### FATAL ERROR: cannot open reference path scan "
                << filename << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::fprintf(file, "path\ttime\tr_core\tr_min\tr_max\tsamples\t"
                       "measure\tmaximum\tradius\tr_over_r_core\n");
  }

  for (int path = 0; path < 3; ++path) {
    for (const Real time : times) {
      ref_gh::ControlledReferenceParameters params = base;
      if (path == 2) params.transition_path = ref_gh::kFixedCorePath;
      const Real r_core = path == 2 ? 0.30 : 0.30*std::exp(-time/1.5);
      const Real r_min = 0.5*r_core;
      const Real r_max = std::fmax(0.6, 3.0*r_core);
      const bool frozen = path == 1;
      Kokkos::parallel_for(
          "ref_gh reference path scan",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, kSamples),
          KOKKOS_LAMBDA(const int sample) {
            const Real radius = r_min + (r_max - r_min)
                *static_cast<Real>(sample)/static_cast<Real>(kSamples - 1);
            ref_gh::ReferenceGeometry reference;
            ref_gh::ReferenceJet core_blend = ref_gh::ConstantJet(0.0);
            if (frozen) {
              const ref_gh::WormholeSchwarzschildReference wormhole{
                  1.0, {0.0, 0.0, 0.0}};
              wormhole.Populate(time, radius, 0.0, 0.0, reference);
            } else {
              ref_gh::ReferenceJet alpha;
              ref_gh::ReferenceJet psi2;
              ref_gh::ReferenceJet shift;
              ref_gh::ControlledTransitionProfileJets(
                  table, params, time, radius, 0.0, 0.0,
                  alpha, psi2, shift, nullptr, nullptr, &core_blend);
              ref_gh::PopulateIsotropicReferenceGeometry(
                  alpha, psi2, shift, radius, 0.0, 0.0,
                  0.0, 0.0, 0.0, reference);
            }
            Real ricci2 = 0.0;
            Real riemann2 = 0.0;
            Real spin2 = 0.0;
            Real spin_derivative2 = 0.0;
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                ricci2 += reference.ricci_frame[A][B]
                          *reference.ricci_frame[A][B];
                for (int C = 0; C < 4; ++C) {
                  spin2 += reference.spin[A][B][C]*reference.spin[A][B][C];
                  for (int D = 0; D < 4; ++D) {
                    riemann2 += reference.riemann_frame[A][B][C][D]
                                *reference.riemann_frame[A][B][C][D];
                    spin_derivative2 += reference.spin_derivative[D][A][B][C]
                        *reference.spin_derivative[D][A][B][C];
                  }
                }
              }
            }

            Real psi[4][4] = {};       // NOLINT(runtime/arrays)
            Real d_psi[4][4][4] = {}; // NOLINT(runtime/arrays)
            Real pi[4][4] = {};        // NOLINT(runtime/arrays)
            Real phi[3][4][4] = {};   // NOLINT(runtime/arrays)
            for (int A = 0; A < 4; ++A) psi[A][A] = A == 0 ? -1.0 : 1.0;
            ref_gh::CoordinateGhGeometry geometry;
            Real determinant = 0.0;
            Real source[4][4];  // NOLINT(runtime/arrays)
            ref_gh::CovariantSourceSectors sectors;
            Real source2 = std::numeric_limits<Real>::max();
            if (ref_gh::ComputeCoordinateGhGeometry(
                    psi, d_psi, reference, geometry, determinant)
                && ref_gh::CovariantGhScalarWaveSource(
                    psi, pi, phi, reference, geometry, 0.0,
                    source, sectors)) {
              source2 = 0.0;
              for (int A = 0; A < 4; ++A) {
                for (int B = 0; B < 4; ++B) source2 += source[A][B]*source[A][B];
              }
            }
            samples(0, sample) = Kokkos::sqrt(ricci2);
            samples(1, sample) = Kokkos::sqrt(riemann2);
            samples(2, sample) = Kokkos::sqrt(spin2);
            samples(3, sample) = Kokkos::sqrt(spin_derivative2);
            samples(4, sample) = Kokkos::sqrt(source2);
            samples(5, sample) = Kokkos::abs(core_blend.d[1]);
            samples(6, sample) = Kokkos::abs(core_blend.dd[1][1]);
          });
      Kokkos::fence();
      using MaxLoc = Kokkos::MaxLoc<Real, int>;
      for (int measure = 0; measure < kMeasures; ++measure) {
        MaxLoc::value_type maximum;
        Kokkos::parallel_reduce(
            "ref_gh reference path scan maximum",
            Kokkos::RangePolicy<>(DevExeSpace(), 0, kSamples),
            KOKKOS_LAMBDA(const int sample,
                          MaxLoc::value_type &local_maximum) {
              const Real value = samples(measure, sample);
              if (value >= local_maximum.val) {
                local_maximum.val = value;
                local_maximum.loc = sample;
              }
            }, MaxLoc(maximum));
        if (global_variable::my_rank == 0) {
          const Real radius = r_min + (r_max - r_min)
              *static_cast<Real>(maximum.loc)/static_cast<Real>(kSamples - 1);
          std::fprintf(file,
              "%s\t%.17e\t%.17e\t%.17e\t%.17e\t%d\t%s\t%.17e\t"
              "%.17e\t%.17e\n",
              path_names[path], time, r_core, r_min, r_max, kSamples,
              measure_names[measure], maximum.val, radius, radius/r_core);
        }
      }
    }
  }
  if (file != nullptr) std::fclose(file);
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH reference-only path scan written to "
              << filename << std::endl;
  }
}

}  // namespace

void ProblemGenerator::RefGhSourceUnit(ParameterInput *pin, const bool restart) {
  CheckCoframeDerivativeIdentity();
  CheckGaugeDriverAlgebra();
  CheckGamma2Algebra();
  CheckCombinedGaugeCharacteristics();
  CheckPhiOrderingAlgebra();
  CheckFlatCovariantSource();
  CheckNonflatCovariantSource();
  CheckDynamicSpatialReference();
  if (pin->GetOrAddBoolean("problem", "puncture_exponent_gate", false)) {
    CheckLocalPunctureExponentEstimator(
        pmy_mesh_->pmb_pack->prefgh->reference_table,
        pin->GetOrAddBoolean("problem", "puncture_exponent_gate_strict", true));
  }
  if (pin->GetOrAddBoolean("problem", "generic_reference_scan", false)) {
    ScanGenericSingularReference(pin);
  }
  if (pin->GetOrAddBoolean("problem", "reference_path_scan", false)) {
    ScanReferencePaths(pin);
  }
  // Leave a valid exact state for the zero-time AthenaK task sequence.
  RefGhMinkowski(pin, restart);
}
