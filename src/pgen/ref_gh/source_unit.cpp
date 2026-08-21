//========================================================================================
//! \file source_unit.cpp
//! \brief Device regressions for flat and nonflat covariant Ref-GH sources.
//========================================================================================
#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace {

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

}  // namespace

void ProblemGenerator::RefGhSourceUnit(ParameterInput *pin, const bool restart) {
  CheckFlatCovariantSource();
  CheckNonflatCovariantSource();
  // Leave a valid exact state for the zero-time AthenaK task sequence.
  RefGhMinkowski(pin, restart);
}
