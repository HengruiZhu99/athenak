//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file curvature_diagnostics.hpp
//! \brief Pointwise vacuum curvature invariants used by Z4c diagnostics.

#ifndef Z4C_CURVATURE_DIAGNOSTICS_HPP_
#define Z4C_CURVATURE_DIAGNOSTICS_HPP_

#include "athena.hpp"
#include "athena_tensor.hpp"
#include "coordinates/adm.hpp"
#include "utils/finite_diff.hpp"

class Mesh;

struct Z4cCurvatureDiagnostics {
  bool valid = false;
  Real kretschmann = 0.0;
  Real electric[6] = {};
  Real magnetic[6] = {};
  Real poynting[3] = {};
};

KOKKOS_INLINE_FUNCTION
int Z4cSymmetricComponent(const int a, const int b) {
  const int low = (a < b) ? a : b;
  const int high = (a < b) ? b : a;
  if (low == 0) return high;
  if (low == 1) return high + 2;
  return 5;
}

template <int NGHOST, bool COMPUTE_POYNTING = true, typename MetricView,
          typename ExtrinsicCurvatureView>
KOKKOS_INLINE_FUNCTION Z4cCurvatureDiagnostics ComputeZ4cCurvatureDiagnostics(
    MetricView metric, ExtrinsicCurvatureView extrinsic_curvature,
    const Real inverse_spacing[3], const int m, const int k, const int j, const int i) {
  Z4cCurvatureDiagnostics result;

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse_metric;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ricci;
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> mixed_extrinsic_curvature;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> metric_derivative;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> extrinsic_derivative;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> christoffel_lower;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> christoffel_upper;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> covariant_extrinsic_derivative;
  AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> metric_second_derivative;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> electric;
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> magnetic;

  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      mixed_extrinsic_curvature(a, b) = 0.0;
      for (int c = 0; c < 3; ++c) {
        metric_derivative(c, a, b) = 0.0;
        extrinsic_derivative(c, a, b) = 0.0;
        christoffel_lower(c, a, b) = 0.0;
        christoffel_upper(c, a, b) = 0.0;
        covariant_extrinsic_derivative(c, a, b) = 0.0;
        for (int d = c; d < 3; ++d) {
          metric_second_derivative(c, d, a, b) = 0.0;
        }
      }
    }
    for (int b = a; b < 3; ++b) {
      inverse_metric(a, b) = 0.0;
      ricci(a, b) = 0.0;
      electric(a, b) = 0.0;
      magnetic(a, b) = 0.0;
    }
  }

  const Real determinant =
      adm::SpatialDet(metric(m, 0, 0, k, j, i), metric(m, 0, 1, k, j, i),
                      metric(m, 0, 2, k, j, i), metric(m, 1, 1, k, j, i),
                      metric(m, 1, 2, k, j, i), metric(m, 2, 2, k, j, i));
  if (!(determinant > 1.0e-10) || !Kokkos::isfinite(determinant)) return result;

  adm::SpatialInv(
      1.0 / determinant, metric(m, 0, 0, k, j, i), metric(m, 0, 1, k, j, i),
      metric(m, 0, 2, k, j, i), metric(m, 1, 1, k, j, i),
      metric(m, 1, 2, k, j, i), metric(m, 2, 2, k, j, i),
      &inverse_metric(0, 0), &inverse_metric(0, 1), &inverse_metric(0, 2),
      &inverse_metric(1, 1), &inverse_metric(1, 2), &inverse_metric(2, 2));

  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        metric_derivative(c, a, b) =
            Dx<NGHOST>(c, inverse_spacing, metric, m, a, b, k, j, i);
        extrinsic_derivative(c, a, b) =
            Dx<NGHOST>(c, inverse_spacing, extrinsic_curvature, m, a, b, k, j, i);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = c; d < 3; ++d) {
          metric_second_derivative(a, b, c, d) =
              (a == b)
                  ? Dxx<NGHOST>(a, inverse_spacing, metric, m, c, d, k, j, i)
                  : Dxy<NGHOST>(a, b, inverse_spacing, metric, m, c, d, k, j, i);
        }
      }
    }
  }

  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        christoffel_lower(c, a, b) =
            0.5 * (metric_derivative(a, b, c) + metric_derivative(b, a, c) -
                   metric_derivative(c, a, b));
      }
    }
  }
  for (int c = 0; c < 3; ++c) {
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        for (int d = 0; d < 3; ++d) {
          christoffel_upper(c, a, b) +=
              inverse_metric(c, d) * christoffel_lower(d, a, b);
        }
      }
    }
  }

  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          for (int e = 0; e < 3; ++e) {
            ricci(a, b) += inverse_metric(c, d) * christoffel_upper(e, a, c) *
                           christoffel_lower(e, b, d);
            ricci(a, b) -= inverse_metric(c, d) * christoffel_upper(e, a, b) *
                           christoffel_lower(e, c, d);
          }
          ricci(a, b) +=
              0.5 * inverse_metric(c, d) *
              (-metric_second_derivative(c, d, a, b) -
               metric_second_derivative(a, b, c, d) +
               metric_second_derivative(a, c, b, d) +
               metric_second_derivative(b, c, a, d));
        }
      }
    }
  }

  Real trace_extrinsic_curvature = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        mixed_extrinsic_curvature(a, b) +=
            inverse_metric(a, c) * extrinsic_curvature(m, c, b, k, j, i);
      }
    }
    trace_extrinsic_curvature += mixed_extrinsic_curvature(a, a);
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        covariant_extrinsic_derivative(a, b, c) =
            extrinsic_derivative(a, b, c);
        for (int d = 0; d < 3; ++d) {
          covariant_extrinsic_derivative(a, b, c) -=
              christoffel_upper(d, a, b) * extrinsic_curvature(m, d, c, k, j, i);
          covariant_extrinsic_derivative(a, b, c) -=
              christoffel_upper(d, a, c) * extrinsic_curvature(m, b, d, k, j, i);
        }
      }
    }
  }

  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      electric(a, b) =
          ricci(a, b) +
          trace_extrinsic_curvature * extrinsic_curvature(m, a, b, k, j, i);
      for (int c = 0; c < 3; ++c) {
        electric(a, b) -= extrinsic_curvature(m, a, c, k, j, i) *
                          mixed_extrinsic_curvature(c, b);
      }
    }
  }

  const Real square_root_determinant = Kokkos::sqrt(determinant);
  Real levi_civita[3][3][3] = {};
  levi_civita[0][1][2] = levi_civita[1][2][0] = levi_civita[2][0][1] = 1.0;
  levi_civita[0][2][1] = levi_civita[2][1][0] = levi_civita[1][0][2] = -1.0;
  Real epsilon_lower_upper_upper[3][3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          for (int e = 0; e < 3; ++e) {
            epsilon_lower_upper_upper[a][b][c] +=
                inverse_metric(b, d) * inverse_metric(c, e) *
                levi_civita[a][d][e] * square_root_determinant;
          }
        }
      }
    }
  }

  Real magnetic_unsymmetrized[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          magnetic_unsymmetrized[a][b] +=
              epsilon_lower_upper_upper[a][c][d] *
              covariant_extrinsic_derivative(c, d, b);
        }
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      magnetic(a, b) =
          0.5 * (magnetic_unsymmetrized[a][b] + magnetic_unsymmetrized[b][a]);
    }
  }

  Real electric_upper[3][3] = {};
  Real magnetic_upper[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          electric_upper[a][b] +=
              inverse_metric(a, c) * inverse_metric(b, d) *
              electric((c < d) ? c : d, (c < d) ? d : c);
          magnetic_upper[a][b] +=
              inverse_metric(a, c) * inverse_metric(b, d) *
              magnetic((c < d) ? c : d, (c < d) ? d : c);
        }
      }
      result.kretschmann +=
          8.0 * (electric((a < b) ? a : b, (a < b) ? b : a) *
                     electric_upper[a][b] -
                 magnetic((a < b) ? a : b, (a < b) ? b : a) *
                     magnetic_upper[a][b]);
    }
  }

  if constexpr (COMPUTE_POYNTING) {
    Real epsilon_upper[3][3][3] = {};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            epsilon_upper[a][b][c] +=
                inverse_metric(a, d) * epsilon_lower_upper_upper[d][b][c];
          }
        }
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          for (int d = 0; d < 3; ++d) {
            for (int e = 0; e < 3; ++e) {
              result.poynting[a] -=
                  epsilon_upper[a][b][c] *
                  electric((b < d) ? b : d, (b < d) ? d : b) *
                  inverse_metric(d, e) *
                  magnetic((c < e) ? c : e, (c < e) ? e : c);
            }
          }
        }
      }
    }
  }

  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      const int component = Z4cSymmetricComponent(a, b);
      result.electric[component] = electric(a, b);
      result.magnetic[component] = magnetic(a, b);
    }
  }
  result.valid = Kokkos::isfinite(result.kretschmann);
  for (int component = 0; component < 6; ++component) {
    result.valid = result.valid &&
                   Kokkos::isfinite(result.electric[component]) &&
                   Kokkos::isfinite(result.magnetic[component]);
  }
  if constexpr (COMPUTE_POYNTING) {
    for (int component = 0; component < 3; ++component) {
      result.valid =
          result.valid && Kokkos::isfinite(result.poynting[component]);
    }
  }
  return result;
}

struct Z4cGlobalCurvatureMaxima {
  Real max_abs_k = 0.0;
  Real max_kretschmann = 0.0;
  bool finite = true;
};

// Compute MPI-global active-zone maxima. This is intended for infrequent
// stopping-condition checks; regular history output retains its own local
// reductions and lets HistoryOutput perform the MPI reduction.
Z4cGlobalCurvatureMaxima ComputeZ4cGlobalCurvatureMaxima(Mesh *pm);

#endif  // Z4C_CURVATURE_DIAGNOSTICS_HPP_
