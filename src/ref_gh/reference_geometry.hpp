//========================================================================================
// Prescribed reference geometry interface for reference-frame FO-GH.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REFERENCE_GEOMETRY_HPP_
#define REF_GH_REFERENCE_GEOMETRY_HPP_

#include "athena.hpp"

namespace ref_gh {

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
};

KOKKOS_INLINE_FUNCTION
void ZeroReferenceGeometry(ReferenceGeometry &reference) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      reference.metric[a][b] = 0.0;
      reference.inverse_metric[a][b] = 0.0;
      reference.coframe[a][b] = 0.0;
      reference.frame[a][b] = 0.0;
      for (int c = 0; c < 4; ++c) {
        reference.d_metric[c][a][b] = 0.0;
        reference.christoffel[a][b][c] = 0.0;
        reference.d_frame[c][a][b] = 0.0;
        for (int d = 0; d < 4; ++d) {
          reference.dd_metric[c][d][a][b] = 0.0;
          reference.d_christoffel[d][a][b][c] = 0.0;
          reference.dd_frame[c][d][a][b] = 0.0;
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

// Identity Cartesian tetrad on Minkowski spacetime.  This is the exact flat-reference
// oracle and makes the stored variables identical to ordinary coordinate FO-GH fields.
struct MinkowskiReference {
  KOKKOS_INLINE_FUNCTION
  ReferenceGeometry operator()(const Real /*time*/, const Real /*x*/,
                               const Real /*y*/, const Real /*z*/) const {
    ReferenceGeometry reference;
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
    return reference;
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_GEOMETRY_HPP_
