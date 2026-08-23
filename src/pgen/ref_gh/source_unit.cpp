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
#include "ref_gh/phi_ordering.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_time_dependent_spatial.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace {

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
            d_coframe[c][A][a] -= reference.coframe[A][b]
                                      *reference.d_frame[c][B][b]
                                      *reference.coframe[B][a];
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
  CheckPhiOrderingAlgebra();
  CheckFlatCovariantSource();
  CheckNonflatCovariantSource();
  CheckDynamicSpatialReference();
  if (pin->GetOrAddBoolean("problem", "reference_path_scan", false)) {
    ScanReferencePaths(pin);
  }
  // Leave a valid exact state for the zero-time AthenaK task sequence.
  RefGhMinkowski(pin, restart);
}
