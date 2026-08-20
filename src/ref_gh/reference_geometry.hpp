//========================================================================================
// Prescribed reference geometry interface for reference-frame FO-GH.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REFERENCE_GEOMETRY_HPP_
#define REF_GH_REFERENCE_GEOMETRY_HPP_

#include "athena.hpp"

namespace ref_gh {

// Minimal reference data required by the Psi evolution.  Keeping this separate
// avoids constructing the full reference two-jet in kernels that use only the
// coframe and spatial derivative map.
struct ReferencePsiKinematics {
  Real coframe[4][4];          // NOLINT(runtime/arrays)
  Real spatial_coframe[3][3];  // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
void ZeroReferencePsiKinematics(ReferencePsiKinematics &reference) {
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) reference.coframe[A][a] = 0.0;
  }
  for (int I = 0; I < 3; ++I) {
    for (int i = 0; i < 3; ++i) reference.spatial_coframe[I][i] = 0.0;
  }
}

// All arrays have fixed extent and may live in a pointwise Kokkos kernel.  Derivative
// index 0 is coordinate time and 1..3 are Cartesian space.  The provider owns no storage
// and performs no allocation.
struct ReferenceGeometry {
  Real metric[4][4];                 // NOLINT(runtime/arrays)
  Real inverse_metric[4][4];         // NOLINT(runtime/arrays)
  Real d_metric[4][4][4];            // NOLINT(runtime/arrays)
  Real dd_metric[4][4][4][4];        // NOLINT(runtime/arrays)
  Real christoffel[4][4][4];         // NOLINT(runtime/arrays)
  Real d_christoffel[4][4][4][4];    // NOLINT(runtime/arrays)
  Real coframe[4][4];                // NOLINT(runtime/arrays)
  Real frame[4][4];                  // NOLINT(runtime/arrays)
  Real d_frame[4][4][4];             // NOLINT(runtime/arrays)
  Real dd_frame[4][4][4][4];         // NOLINT(runtime/arrays)
  Real spatial_frame[3][3];          // NOLINT(runtime/arrays)
  Real spatial_coframe[3][3];        // NOLINT(runtime/arrays)
  Real dt_spatial_frame[3][3];       // NOLINT(runtime/arrays)
  Real structure[3][3][3];           // NOLINT(runtime/arrays)
  Real spin[4][4][4];                // NOLINT(runtime/arrays)
  Real spin_derivative[4][4][4][4];  // NOLINT(runtime/arrays)
  Real structure4[4][4][4];          // NOLINT(runtime/arrays)
  Real riemann_frame[4][4][4][4];    // NOLINT(runtime/arrays)
  Real ricci_frame[4][4];            // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
void ZeroReferenceGeometry(ReferenceGeometry &reference) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      reference.metric[a][b] = 0.0;
      reference.inverse_metric[a][b] = 0.0;
      reference.coframe[a][b] = 0.0;
      reference.frame[a][b] = 0.0;
      reference.ricci_frame[a][b] = 0.0;
      for (int c = 0; c < 4; ++c) {
        reference.d_metric[c][a][b] = 0.0;
        reference.christoffel[a][b][c] = 0.0;
        reference.d_frame[c][a][b] = 0.0;
        reference.spin[a][b][c] = 0.0;
        reference.structure4[a][b][c] = 0.0;
        for (int d = 0; d < 4; ++d) {
          reference.dd_metric[c][d][a][b] = 0.0;
          reference.d_christoffel[d][a][b][c] = 0.0;
          reference.dd_frame[c][d][a][b] = 0.0;
          reference.spin_derivative[a][b][c][d] = 0.0;
          reference.riemann_frame[a][b][c][d] = 0.0;
        }
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      reference.spatial_frame[i][j] = 0.0;
      reference.spatial_coframe[i][j] = 0.0;
      reference.dt_spatial_frame[i][j] = 0.0;
      for (int k = 0; k < 3; ++k) reference.structure[i][j][k] = 0.0;
    }
  }
}

// Complete regular frame data from one internally consistent coordinate 2-jet.
// spin_derivative[C][A][B][D] stores e_C(omega^A_BD).
KOKKOS_INLINE_FUNCTION
void CompleteReferenceFrameGeometry(ReferenceGeometry &reference) {
  Real d_coframe[4][4][4];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 4; ++p) {
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        d_coframe[p][A][a] = 0.0;
        for (int B = 0; B < 4; ++B) {
          for (int b = 0; b < 4; ++b) {
            d_coframe[p][A][a] -= reference.coframe[A][b]
                *reference.d_frame[p][B][b]*reference.coframe[B][a];
          }
        }
      }
    }
  }

  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        for (int a = 0; a < 4; ++a) {
          for (int c = 0; c < 4; ++c) {
            Real derivative = reference.d_frame[c][B][a];
            for (int d = 0; d < 4; ++d) {
              derivative += reference.christoffel[a][c][d]*reference.frame[B][d];
            }
            reference.spin[A][B][C] +=
                reference.coframe[A][a]*reference.frame[C][c]*derivative;
          }
        }
      }
    }
  }

  for (int A = 0; A < 4; ++A) {
    const Real eta_A = (A == 0) ? -1.0 : 1.0;
    for (int B = A; B < 4; ++B) {
      const Real eta_B = (B == 0) ? -1.0 : 1.0;
      for (int C = 0; C < 4; ++C) {
        const Real projected = 0.5*(eta_A*reference.spin[A][B][C]
                                    - eta_B*reference.spin[B][A][C]);
        reference.spin[A][B][C] = eta_A*projected;
        reference.spin[B][A][C] = -eta_B*projected;
      }
    }
  }

  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        for (int p = 0; p < 4; ++p) {
          Real coordinate_derivative = 0.0;
          for (int a = 0; a < 4; ++a) {
            for (int c = 0; c < 4; ++c) {
              Real frame_covariant_derivative = reference.d_frame[c][B][a];
              Real d_frame_covariant_derivative = reference.dd_frame[p][c][B][a];
              for (int d = 0; d < 4; ++d) {
                frame_covariant_derivative +=
                    reference.christoffel[a][c][d]*reference.frame[B][d];
                d_frame_covariant_derivative +=
                    reference.d_christoffel[p][a][c][d]*reference.frame[B][d]
                    + reference.christoffel[a][c][d]
                        *reference.d_frame[p][B][d];
              }
              coordinate_derivative +=
                  (d_coframe[p][A][a]*reference.frame[C][c]
                   + reference.coframe[A][a]*reference.d_frame[p][C][c])
                    *frame_covariant_derivative
                  + reference.coframe[A][a]*reference.frame[C][c]
                    *d_frame_covariant_derivative;
            }
          }
          for (int D = 0; D < 4; ++D) {
            reference.spin_derivative[D][A][B][C] +=
                reference.frame[D][p]*coordinate_derivative;
          }
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    const Real eta_A = (A == 0) ? -1.0 : 1.0;
    for (int B = A; B < 4; ++B) {
      const Real eta_B = (B == 0) ? -1.0 : 1.0;
      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          const Real projected =
              0.5*(eta_A*reference.spin_derivative[D][A][B][C]
                   - eta_B*reference.spin_derivative[D][B][A][C]);
          reference.spin_derivative[D][A][B][C] = eta_A*projected;
          reference.spin_derivative[D][B][A][C] = -eta_B*projected;
        }
      }
    }
  }

  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = B; C < 4; ++C) {
        Real value = 0.0;
        for (int a = 0; a < 4; ++a) {
          for (int p = 0; p < 4; ++p) {
            value += reference.coframe[A][a]
                *(reference.frame[B][p]*reference.d_frame[p][C][a]
                  - reference.frame[C][p]*reference.d_frame[p][B][a]);
          }
        }
        reference.structure4[A][B][C] = value;
        reference.structure4[A][C][B] = -value;
      }
    }
  }

  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          Real value = reference.spin_derivative[C][A][B][D]
                       - reference.spin_derivative[D][A][B][C];
          for (int E = 0; E < 4; ++E) {
            value += reference.spin[A][E][C]*reference.spin[E][B][D]
                     - reference.spin[A][E][D]*reference.spin[E][B][C]
                     - reference.structure4[E][C][D]*reference.spin[A][B][E];
          }
          reference.riemann_frame[A][B][C][D] = value;
        }
      }
    }
  }
  for (int B = 0; B < 4; ++B) {
    for (int D = 0; D < 4; ++D) {
      for (int A = 0; A < 4; ++A) {
        reference.ricci_frame[B][D] += reference.riemann_frame[A][B][A][D];
      }
    }
  }
}

// Identity Cartesian tetrad on Minkowski spacetime.  This is the exact flat-reference
// oracle and makes the stored variables identical to ordinary coordinate FO-GH fields.
struct MinkowskiReference {
  KOKKOS_INLINE_FUNCTION
  void PopulatePsiKinematics(const Real /*time*/, const Real /*x*/,
                             const Real /*y*/, const Real /*z*/,
                             ReferencePsiKinematics &reference) const {
    ZeroReferencePsiKinematics(reference);
    for (int a = 0; a < 4; ++a) reference.coframe[a][a] = 1.0;
    for (int i = 0; i < 3; ++i) reference.spatial_coframe[i][i] = 1.0;
  }

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real /*time*/, const Real /*x*/, const Real /*y*/,
                const Real /*z*/, ReferenceGeometry &reference) const {
    ZeroReferenceGeometry(reference);
    for (int a = 0; a < 4; ++a) {
      const Real sign = (a == 0) ? -1.0 : 1.0;
      reference.metric[a][a] = sign;
      reference.inverse_metric[a][a] = sign;
      reference.coframe[a][a] = 1.0;
      reference.frame[a][a] = 1.0;
    }
    for (int i = 0; i < 3; ++i) {
      reference.spatial_frame[i][i] = 1.0;
      reference.spatial_coframe[i][i] = 1.0;
    }
    CompleteReferenceFrameGeometry(reference);
  }

  KOKKOS_INLINE_FUNCTION
  ReferenceGeometry operator()(const Real time, const Real x,
                               const Real y, const Real z) const {
    ReferenceGeometry reference;
    Populate(time, x, y, z, reference);
    return reference;
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_GEOMETRY_HPP_
