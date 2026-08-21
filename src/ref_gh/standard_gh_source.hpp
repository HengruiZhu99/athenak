//========================================================================================
// Coordinate-space standard GH wave source and reference-frame product-rule transform.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_STANDARD_GH_SOURCE_HPP_
#define REF_GH_STANDARD_GH_SOURCE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_geometry.hpp"

namespace ref_gh {

struct CoordinateGhGeometry {
  Real inverse_metric[4][4];       // NOLINT(runtime/arrays)
  Real christoffel_first[4][4][4]; // NOLINT(runtime/arrays)
  Real christoffel[4][4][4];       // NOLINT(runtime/arrays)
  Real contracted_first[4];        // NOLINT(runtime/arrays)
  Real contracted_upper[4];        // NOLINT(runtime/arrays)
  Real gauge_source_upper[4];      // NOLINT(runtime/arrays)
  Real gauge_source[4];            // NOLINT(runtime/arrays)
  Real gauge_constraint[4];        // NOLINT(runtime/arrays)
  Real lapse;
  Real shift[3];                   // NOLINT(runtime/arrays)
  Real normal_upper[4];            // NOLINT(runtime/arrays)
  Real normal_lower[4];            // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
bool Invert4(const Real matrix[4][4], Real inverse[4][4], Real &determinant) {
  Real augmented[4][8];  // NOLINT(runtime/arrays)
  determinant = 1.0;
  int parity = 1;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      augmented[i][j] = matrix[i][j];
      augmented[i][j + 4] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (int column = 0; column < 4; ++column) {
    int pivot = column;
    Real maximum = Kokkos::abs(augmented[column][column]);
    for (int row = column + 1; row < 4; ++row) {
      const Real candidate = Kokkos::abs(augmented[row][column]);
      if (candidate > maximum) {
        maximum = candidate;
        pivot = row;
      }
    }
    if (!(maximum > 0.0) || !Kokkos::isfinite(maximum)) {
      determinant = 0.0;
      return false;
    }
    if (pivot != column) {
      for (int j = 0; j < 8; ++j) {
        const Real temporary = augmented[column][j];
        augmented[column][j] = augmented[pivot][j];
        augmented[pivot][j] = temporary;
      }
      parity = -parity;
    }
    const Real diagonal = augmented[column][column];
    determinant *= diagonal;
    for (int j = 0; j < 8; ++j) augmented[column][j] /= diagonal;
    for (int row = 0; row < 4; ++row) {
      if (row == column) continue;
      const Real factor = augmented[row][column];
      for (int j = 0; j < 8; ++j) {
        augmented[row][j] -= factor*augmented[column][j];
      }
    }
  }
  determinant *= parity;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) inverse[i][j] = augmented[i][j + 4];
  }
  return Kokkos::isfinite(determinant);
}

KOKKOS_INLINE_FUNCTION
bool InvertSpatial3(const Real metric[4][4], Real inverse[3][3], Real &determinant) {
  const Real a = metric[1][1];
  const Real b = metric[1][2];
  const Real c = metric[1][3];
  const Real d = metric[2][2];
  const Real e = metric[2][3];
  const Real f = metric[3][3];
  determinant = a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);
  if (!(determinant > 0.0) || !Kokkos::isfinite(determinant)) return false;
  inverse[0][0] = (d*f - e*e)/determinant;
  inverse[0][1] = inverse[1][0] = (c*e - b*f)/determinant;
  inverse[0][2] = inverse[2][0] = (b*e - c*d)/determinant;
  inverse[1][1] = (a*f - c*c)/determinant;
  inverse[1][2] = inverse[2][1] = (b*c - a*e)/determinant;
  inverse[2][2] = (a*d - b*b)/determinant;
  return true;
}

// Construct all first-derivative coordinate geometry and the background-covariant
// wave-map constraint C_a = H_a + Gamma_a, with H^a=-g^{bc} barGamma^a_bc.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool ComputeCoordinateGhGeometry(const Real metric[4][4],
                                 const Real d_metric[4][4][4],
                                 const Reference &reference,
                                 CoordinateGhGeometry &geometry,
                                 Real &determinant) {
  if (!Invert4(metric, geometry.inverse_metric, determinant)) return false;
  if (!(geometry.inverse_metric[0][0] < 0.0)) return false;
  geometry.lapse = 1.0/Kokkos::sqrt(-geometry.inverse_metric[0][0]);
  for (int i = 0; i < 3; ++i) {
    geometry.shift[i] = geometry.lapse*geometry.lapse
                        *geometry.inverse_metric[0][i + 1];
  }
  geometry.normal_upper[0] = 1.0/geometry.lapse;
  geometry.normal_lower[0] = -geometry.lapse;
  for (int i = 0; i < 3; ++i) {
    geometry.normal_upper[i + 1] = -geometry.shift[i]/geometry.lapse;
    geometry.normal_lower[i + 1] = 0.0;
  }

  for (int a = 0; a < 4; ++a) {
    geometry.contracted_first[a] = 0.0;
    geometry.contracted_upper[a] = 0.0;
    geometry.gauge_source_upper[a] = 0.0;
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        geometry.christoffel_first[a][b][c] =
            0.5*(d_metric[b][a][c] + d_metric[c][a][b] - d_metric[a][b][c]);
        geometry.christoffel[a][b][c] = 0.0;
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          geometry.christoffel[a][b][c] += geometry.inverse_metric[a][d]
                                             *geometry.christoffel_first[d][b][c];
        }
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        geometry.contracted_first[a] += geometry.inverse_metric[b][c]
                                          *geometry.christoffel_first[a][b][c];
        geometry.contracted_upper[a] += geometry.inverse_metric[b][c]
                                          *geometry.christoffel[a][b][c];
        geometry.gauge_source_upper[a] -= geometry.inverse_metric[b][c]
                                             *ReferenceChristoffel(reference, a, b, c);
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    geometry.gauge_source[a] = 0.0;
    for (int b = 0; b < 4; ++b) {
      geometry.gauge_source[a] += metric[a][b]*geometry.gauge_source_upper[b];
    }
    geometry.gauge_constraint[a] = geometry.gauge_source[a]
                                    + geometry.contracted_first[a];
  }
  return Kokkos::isfinite(geometry.lapse) && geometry.lapse > 0.0;
}

// Evaluate Eq. (18) of Lindblom et al., arXiv:gr-qc/0512093v3.  The derivative
// of H_a is analytic in g, dg, barGamma, and d(barGamma); no finite difference of H
// or of a coordinate metric is used.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
void StandardGhPartialWaveSource(const Real metric[4][4],
                                 const Real d_metric[4][4][4],
                                 const Reference &reference,
                                 const CoordinateGhGeometry &geometry,
                                 const Real gamma0,
                                 Real source[4][4]) {
  Real d_inverse[4][4][4];  // NOLINT(runtime/arrays)
  Real d_h_upper[4][4];     // NOLINT(runtime/arrays)
  Real d_h_lower[4][4];     // NOLINT(runtime/arrays)
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_inverse[p][a][b] = 0.0;
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] -= geometry.inverse_metric[a][c]
                                    *geometry.inverse_metric[b][d]*d_metric[p][c][d];
          }
        }
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      d_h_upper[p][a] = 0.0;
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          d_h_upper[p][a] -= d_inverse[p][b][c]
                                  *ReferenceChristoffel(reference, a, b, c)
                              + geometry.inverse_metric[b][c]
                                  *ReferenceDChristoffel(reference, p, a, b, c);
        }
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      d_h_lower[p][a] = 0.0;
      for (int b = 0; b < 4; ++b) {
        d_h_lower[p][a] += d_metric[p][a][b]*geometry.gauge_source_upper[b]
                            + metric[a][b]*d_h_upper[p][b];
      }
    }
  }

  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real nabla_h_ab = d_h_lower[a][b];
      Real nabla_h_ba = d_h_lower[b][a];
      for (int c = 0; c < 4; ++c) {
        nabla_h_ab -= geometry.christoffel[c][a][b]*geometry.gauge_source[c];
        nabla_h_ba -= geometry.christoffel[c][b][a]*geometry.gauge_source[c];
      }
      Real value = -nabla_h_ab - nabla_h_ba;
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          for (int e = 0; e < 4; ++e) {
            for (int f = 0; f < 4; ++f) {
              value += 2.0*geometry.inverse_metric[c][d]
                       *geometry.inverse_metric[e][f]
                       *(d_metric[e][c][a]*d_metric[f][d][b]
                         - geometry.christoffel_first[a][c][e]
                           *geometry.christoffel_first[b][d][f]);
            }
          }
        }
      }
      for (int c = 0; c < 4; ++c) {
        const Real projector = ((c == a) ? geometry.normal_lower[b] : 0.0)
                               + ((c == b) ? geometry.normal_lower[a] : 0.0)
                               - metric[a][b]*geometry.normal_upper[c];
        value += gamma0*projector*geometry.gauge_constraint[c];
      }
      source[a][b] = value;
    }
  }
}

// Transform the coordinate partial-wave equation by Psi_AB=e_A^a e_B^b g_ab,
// then convert to the covariant scalar-wave source.  d_psi contains partial_c Psi_AB.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
void TransformPartialWaveSource(const Real metric[4][4],
                                const Real d_metric[4][4][4],
                                const Real coordinate_source[4][4],
                                const Real d_psi[4][4][4],
                                const Reference &reference,
                                const CoordinateGhGeometry &geometry,
                                Real source[4][4]) {
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      Real value = 0.0;
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          const Real tensor = ReferenceFrame(reference, A, a)
                              *ReferenceFrame(reference, B, b);
          value += tensor*coordinate_source[a][b];
          for (int c = 0; c < 4; ++c) {
            const Real d_tensor_c = ReferenceDFrame(reference, c, A, a)
                                      *ReferenceFrame(reference, B, b)
                                    + ReferenceFrame(reference, A, a)
                                      *ReferenceDFrame(reference, c, B, b);
            for (int d = 0; d < 4; ++d) {
              const Real dd_tensor =
                  ReferenceDDFrame(reference, c, d, A, a)
                    *ReferenceFrame(reference, B, b)
                  + ReferenceDFrame(reference, c, A, a)
                    *ReferenceDFrame(reference, d, B, b)
                  + ReferenceDFrame(reference, d, A, a)
                    *ReferenceDFrame(reference, c, B, b)
                  + ReferenceFrame(reference, A, a)
                    *ReferenceDDFrame(reference, c, d, B, b);
              value += 2.0*geometry.inverse_metric[c][d]*d_tensor_c
                         *d_metric[d][a][b]
                       + geometry.inverse_metric[c][d]*dd_tensor*metric[a][b];
            }
          }
        }
      }
      for (int c = 0; c < 4; ++c) value -= geometry.contracted_upper[c]*d_psi[c][A][B];
      source[A][B] = value;
    }
  }
}

}  // namespace ref_gh

#endif  // REF_GH_STANDARD_GH_SOURCE_HPP_
