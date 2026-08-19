//========================================================================================
// Frame-native background-covariant GH lower-order source.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_COVARIANT_GH_SOURCE_HPP_
#define REF_GH_COVARIANT_GH_SOURCE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct CovariantSourceSectors {
  Real q[4][4][4];                 // NOLINT(runtime/arrays)
  Real delta_lower[4][4][4];       // NOLINT(runtime/arrays)
  Real delta_upper[4][4][4];       // NOLINT(runtime/arrays)
  Real delta[4];                   // NOLINT(runtime/arrays)
  Real curvature[4][4];            // NOLINT(runtime/arrays)
  Real qq[4][4];                   // NOLINT(runtime/arrays)
  Real delta_product[4][4];        // NOLINT(runtime/arrays)
  Real damping[4][4];              // NOLINT(runtime/arrays)
  Real frame_correction[4][4];     // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
bool CovariantGhScalarWaveSource(const Real psi[4][4], const Real pi[4][4],
                                 const Real phi[3][4][4],
                                 const ReferenceGeometry &reference,
                                 const CoordinateGhGeometry &geometry,
                                 const Real gamma0, Real source[4][4],
                                 CovariantSourceSectors &sectors) {
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!Invert4(psi, inverse, determinant)) return false;

  Real normal[4];  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    normal[A] = 0.0;
    for (int a = 0; a < 4; ++a) {
      normal[A] += reference.coframe[A][a]*geometry.normal_upper[a];
    }
  }
  if (!(normal[0] > 0.0) || !Kokkos::isfinite(normal[0])) return false;

  Real p[4][4][4];  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int I = 0; I < 3; ++I) p[I + 1][A][B] = phi[I][A][B];
      p[0][A][B] = -pi[A][B];
      for (int I = 0; I < 3; ++I) {
        p[0][A][B] -= normal[I + 1]*phi[I][A][B];
      }
      p[0][A][B] /= normal[0];
    }
  }

  for (int C = 0; C < 4; ++C) {
    sectors.delta[C] = 0.0;
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        sectors.q[C][A][B] = p[C][A][B];
        for (int D = 0; D < 4; ++D) {
          sectors.q[C][A][B] -=
              reference.spin[D][A][C]*psi[D][B]
              + reference.spin[D][B][C]*psi[A][D];
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        sectors.delta_lower[A][B][C] = 0.5*(
            sectors.q[B][A][C] + sectors.q[C][A][B] - sectors.q[A][B][C]);
        sectors.delta_upper[A][B][C] = 0.0;
        for (int D = 0; D < 4; ++D) {
          sectors.delta_upper[A][B][C] +=
              inverse[A][D]*sectors.delta_lower[D][B][C];
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        sectors.delta[A] += inverse[B][C]*sectors.delta_lower[A][B][C];
      }
    }
  }

  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      sectors.curvature[A][B] = 0.0;
      sectors.qq[A][B] = 0.0;
      sectors.delta_product[A][B] = 0.0;
      sectors.damping[A][B] = 0.0;
      sectors.frame_correction[A][B] = 0.0;
      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          for (int E = 0; E < 4; ++E) {
            sectors.curvature[A][B] -= inverse[C][D]*(
                reference.riemann_frame[E][C][D][A]*psi[B][E]
                + reference.riemann_frame[E][C][D][B]*psi[A][E]);
            for (int F = 0; F < 4; ++F) {
              sectors.qq[A][B] += 2.0*inverse[C][D]*inverse[E][F]
                  *sectors.q[E][C][A]*sectors.q[F][D][B];
              sectors.delta_product[A][B] -=
                  2.0*inverse[C][D]*inverse[E][F]
                  *sectors.delta_lower[A][C][E]
                  *sectors.delta_lower[B][D][F];
            }
          }
        }
        // n_A is required in the reference frame, not the coordinate frame.
        Real normal_lower_A = 0.0;
        Real normal_lower_B = 0.0;
        for (int D = 0; D < 4; ++D) {
          normal_lower_A += psi[A][D]*normal[D];
          normal_lower_B += psi[B][D]*normal[D];
        }
        const Real frame_projector = ((C == A) ? normal_lower_B : 0.0)
                                     + ((C == B) ? normal_lower_A : 0.0)
                                     - psi[A][B]*normal[C];
        sectors.damping[A][B] += gamma0*frame_projector*sectors.delta[C];
      }

      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          Real f_cdab = 0.0;
          for (int E = 0; E < 4; ++E) {
            f_cdab -= (reference.spin[E][D][C]
                       + sectors.delta_upper[E][D][C])*p[E][A][B];
            f_cdab += reference.spin_derivative[C][E][A][D]*psi[E][B]
                      + reference.spin[E][A][D]*p[C][E][B]
                      + reference.spin_derivative[C][E][B][D]*psi[A][E]
                      + reference.spin[E][B][D]*p[C][A][E]
                      + reference.spin[E][D][C]*sectors.q[E][A][B]
                      + reference.spin[E][A][C]*sectors.q[D][E][B]
                      + reference.spin[E][B][C]*sectors.q[D][A][E];
          }
          sectors.frame_correction[A][B] += inverse[C][D]*f_cdab;
        }
      }
      source[A][B] = sectors.curvature[A][B] + sectors.qq[A][B]
                     + sectors.delta_product[A][B] + sectors.damping[A][B]
                     + sectors.frame_correction[A][B];
    }
  }
  return true;
}

}  // namespace ref_gh

#endif  // REF_GH_COVARIANT_GH_SOURCE_HPP_
