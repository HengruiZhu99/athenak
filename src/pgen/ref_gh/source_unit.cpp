//========================================================================================
//! \file source_unit.cpp
//! \brief Device regression for the flat-reference covariant Ref-GH source.
//========================================================================================
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace {

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
        Real coordinate_partial[4][4];  // NOLINT(runtime/arrays)
        Real coordinate[4][4];     // NOLINT(runtime/arrays)
        if (!ref_gh::CovariantGhScalarWaveSource(psi, pi, phi, reference, geometry,
                                                  1.3, covariant, sectors)) {
          local_maximum = fmax(local_maximum, 1.0e30);
          return;
        }
        ref_gh::StandardGhPartialWaveSource(psi, d_psi, reference, geometry, 1.3,
                                             coordinate_partial);
        ref_gh::TransformPartialWaveSource(psi, d_psi, coordinate_partial, d_psi,
                                            reference, geometry, coordinate);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
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

}  // namespace

void ProblemGenerator::RefGhSourceUnit(ParameterInput *pin, const bool restart) {
  CheckFlatCovariantSource();
  // Leave a valid exact state for the zero-time AthenaK task sequence.
  RefGhMinkowski(pin, restart);
}
