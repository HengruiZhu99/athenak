//========================================================================================
//! \file analytic_radial_q_source.hpp
//! \brief Contracted covariant GH source for the compact radial-q reference.
//========================================================================================
#ifndef REF_GH_ANALYTIC_RADIAL_Q_SOURCE_HPP_
#define REF_GH_ANALYTIC_RADIAL_Q_SOURCE_HPP_

#include "athena.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/generated/analytic_radial_q_source.hpp"
#include "ref_gh/reference_analytic_radial_q.hpp"
#include "ref_gh/reference_residual.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct CompactAnalyticCoordinateGeometry {
  CoordinateGhGeometry geometry;
  Real d_reference_gauge[4][4];  // NOLINT(runtime/arrays)
  Real d_gauge_source_upper[4][4];  // NOLINT(runtime/arrays)
};

// Reconstruct physical point geometry once while obtaining the implicit
// reference gauge contractions from generated compact expressions.  This is
// the analytic counterpart of ComputeCoordinateGhGeometry plus
// ImplicitGaugeSourceDerivative; no reference connection is materialized.
KOKKOS_INLINE_FUNCTION
bool ComputeCompactAnalyticCoordinateGeometry(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const AnalyticRadialQPoint &reference,
    CompactAnalyticCoordinateGeometry &result, Real &determinant) {
  CoordinateGhGeometry &geometry = result.geometry;
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
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        geometry.christoffel_first[a][b][c] =
            0.5*(d_metric[b][a][c] + d_metric[c][a][b]
                 - d_metric[a][b][c]);
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
      }
    }
  }
  Real d_inverse[4][4][4] = {};  // NOLINT(runtime/arrays)
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] -= geometry.inverse_metric[a][c]
                *geometry.inverse_metric[b][d]*d_metric[p][c][d];
          }
        }
      }
    }
  }
  Real h_upper[4];       // NOLINT(runtime/arrays)
  Real d_h_upper[4][4];  // NOLINT(runtime/arrays)
  GeneratedAnalyticRadialQPhysicalGaugeUpper(
      reference.alpha, reference.l, reference.b, reference.displacement,
      reference.radius, geometry.inverse_metric, d_inverse,
      h_upper, d_h_upper);
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      result.d_gauge_source_upper[p][a] = d_h_upper[p][a];
    }
  }
  for (int a = 0; a < 4; ++a) {
    geometry.gauge_source[a] = 0.0;
    geometry.gauge_source_upper[a] = h_upper[a];
    for (int p = 0; p < 4; ++p) result.d_reference_gauge[p][a] = 0.0;
    for (int b = 0; b < 4; ++b) {
      geometry.gauge_source[a] += metric[a][b]*h_upper[b];
      for (int p = 0; p < 4; ++p) {
        result.d_reference_gauge[p][a] +=
            d_metric[p][a][b]*h_upper[b] + metric[a][b]*d_h_upper[p][b];
      }
    }
    geometry.gauge_constraint[a] = geometry.gauge_source[a]
                                   + geometry.contracted_first[a];
  }
  return Kokkos::isfinite(geometry.lapse) && geometry.lapse > 0.0;
}

// Cancellation-free compact analytic implementation of
//
//   J_a=E^A_a delta_H_A-[B_a(g;gbar)-B_a(gbar;gbar)].
//
// GeneratedAnalyticRadialQPhysicalGaugeUpper is linear in g^{ab} and in
// partial_p g^{ab}.  Supplying their exact residuals therefore evaluates the
// upper-index gauge-source residual directly, without materializing the
// reference connection or subtracting two singular full sources.
KOKKOS_INLINE_FUNCTION
bool AddCompactAnalyticOrdinaryGaugeResidualSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const Real metric[4][4],
    const Real d_metric[4][4][4], const AnalyticRadialQPoint &reference,
    const CompactAnalyticCoordinateGeometry &compact_geometry,
    const Real delta_hhat[4], const Real d_delta_hhat[4][4],
    const Real gamma0, Real source[4][4]) {
  const CoordinateGhGeometry &geometry = compact_geometry.geometry;
  ReferenceRelativeCoordinateData relative;
  if (!BuildReferenceRelativeCoordinateData(
          psi, pi, phi, metric, d_metric, geometry, reference, relative)) {
    return false;
  }

  ReferenceResidualValue inverse[4][4];  // NOLINT(runtime/arrays)
  ReferenceResidualValue dg[4][4][4];    // NOLINT(runtime/arrays)
  ReferenceResidualValue d_inverse[4][4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real delta_inverse = 0.0;
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          delta_inverse -= geometry.inverse_metric[a][c]
              *relative.delta_metric[c][d]*relative.reference_inverse[d][b];
        }
      }
      inverse[a][b] = MakeReferenceResidual(
          relative.reference_inverse[a][b], geometry.inverse_metric[a][b],
          delta_inverse);
      for (int p = 0; p < 4; ++p) {
        dg[p][a][b] = MakeReferenceResidual(
            relative.reference_d_metric[p][a][b], d_metric[p][a][b],
            relative.delta_d_metric[p][a][b]);
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_inverse[p][a][b] = ReferenceResidualConstant(0.0);
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] = d_inverse[p][a][b]
                - inverse[a][c]*inverse[b][d]*dg[p][c][d];
          }
        }
      }
    }
  }

  Real delta_inverse[4][4];       // NOLINT(runtime/arrays)
  Real delta_d_inverse[4][4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      delta_inverse[a][b] = inverse[a][b].delta;
      for (int p = 0; p < 4; ++p) {
        delta_d_inverse[p][a][b] = d_inverse[p][a][b].delta;
      }
    }
  }
  Real delta_base_upper[4];       // NOLINT(runtime/arrays)
  Real d_delta_base_upper[4][4];  // NOLINT(runtime/arrays)
  GeneratedAnalyticRadialQPhysicalGaugeUpper(
      reference.alpha, reference.l, reference.b, reference.displacement,
      reference.radius, delta_inverse, delta_d_inverse,
      delta_base_upper, d_delta_base_upper);

  Real delta_base[4] = {};       // NOLINT(runtime/arrays)
  Real d_delta_base[4][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      delta_base[a] +=
          relative.delta_metric[a][b]*geometry.gauge_source_upper[b]
          + relative.reference_metric[a][b]*delta_base_upper[b];
      for (int p = 0; p < 4; ++p) {
        d_delta_base[p][a] +=
            relative.delta_d_metric[p][a][b]
                *geometry.gauge_source_upper[b]
            + relative.reference_d_metric[p][a][b]
                *delta_base_upper[b]
            + relative.delta_metric[a][b]
                *compact_geometry.d_gauge_source_upper[p][b]
            + relative.reference_metric[a][b]
                *d_delta_base_upper[p][b];
      }
    }
  }

  Real j[4] = {};       // NOLINT(runtime/arrays)
  Real d_j[4][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    j[a] = -delta_base[a];
    for (int A = 0; A < 4; ++A) {
      j[a] += ReferenceCoframe(reference, A, a)*delta_hhat[A];
    }
    for (int p = 0; p < 4; ++p) {
      d_j[p][a] = -d_delta_base[p][a];
      for (int A = 0; A < 4; ++A) {
        d_j[p][a] += AnalyticDCoframe(reference, p, A, a)*delta_hhat[A]
                      + ReferenceCoframe(reference, A, a)
                          *d_delta_hhat[p][A];
      }
    }
  }

  Real coordinate_extra[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real nabla_ab = d_j[a][b];
      Real nabla_ba = d_j[b][a];
      for (int c = 0; c < 4; ++c) {
        nabla_ab -= geometry.christoffel[c][a][b]*j[c];
        nabla_ba -= geometry.christoffel[c][b][a]*j[c];
      }
      coordinate_extra[a][b] = -nabla_ab - nabla_ba;
      for (int c = 0; c < 4; ++c) {
        const Real projector = ((c == a) ? geometry.normal_lower[b] : 0.0)
                               + ((c == b) ? geometry.normal_lower[a] : 0.0)
                               - metric[a][b]*geometry.normal_upper[c];
        coordinate_extra[a][b] += gamma0*projector*j[c];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          source[A][B] += ReferenceFrame(reference, A, a)
                          *ReferenceFrame(reference, B, b)
                          *coordinate_extra[a][b];
        }
      }
    }
  }
  return true;
}

// This is the analytic all-source oracle and future production entry point.
// It consumes only the 12+8 radial coefficients at a point.  In particular it
// never calls PopulateGeneratedAnalyticRadialQGeometry, ReferenceSpin,
// ReferenceSpinDerivative, or ReferenceRiemann, and never materializes their
// full tensors.  The generated functions provide the already-contracted
// reference pieces of the ten covariant scalar-wave sources.
KOKKOS_INLINE_FUNCTION
bool CompactAnalyticRadialQScalarWaveSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const AnalyticRadialQPoint &reference,
    const CoordinateGhGeometry &geometry, const Real gamma0,
    Real source[4][4], CovariantSourceSectors *sectors = nullptr) {
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!Invert4(psi, inverse, determinant)) return false;

  Real normal[4] = {};  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      normal[A] += ReferenceCoframe(reference, A, a)
                   *geometry.normal_upper[a];
    }
  }
  if (!(normal[0] > 0.0) || !Kokkos::isfinite(normal[0])) return false;

  Real p[4][4][4];             // NOLINT(runtime/arrays)
  Real q[4][4][4];             // NOLINT(runtime/arrays)
  Real q_correction[4][4][4];  // NOLINT(runtime/arrays)
  Real delta_lower[4][4][4];   // NOLINT(runtime/arrays)
  Real delta_upper[4][4][4];   // NOLINT(runtime/arrays)
  Real delta[4] = {};           // NOLINT(runtime/arrays)
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
  GeneratedAnalyticRadialQQCorrection(
      reference.alpha, reference.l, reference.b, reference.displacement,
      reference.radius, psi, q_correction);
  for (int C = 0; C < 4; ++C) {
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        q[C][A][B] = p[C][A][B] - q_correction[C][A][B];
        if (sectors != nullptr) sectors->q[C][A][B] = q[C][A][B];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        delta_lower[A][B][C] =
            0.5*(q[B][A][C] + q[C][A][B] - q[A][B][C]);
        if (sectors != nullptr) {
          sectors->delta_lower[A][B][C] = delta_lower[A][B][C];
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        delta_upper[A][B][C] = 0.0;
        for (int D = 0; D < 4; ++D) {
          delta_upper[A][B][C] += inverse[A][D]*delta_lower[D][B][C];
        }
        if (sectors != nullptr) {
          sectors->delta_upper[A][B][C] = delta_upper[A][B][C];
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        delta[A] += inverse[B][C]*delta_lower[A][B][C];
      }
    }
  }
  if (sectors != nullptr) {
    for (int A = 0; A < 4; ++A) sectors->delta[A] = delta[A];
  }

  Real curvature[4][4];        // NOLINT(runtime/arrays)
  Real frame_correction[4][4]; // NOLINT(runtime/arrays)
  GeneratedAnalyticRadialQCurvatureSource(
      reference.alpha, reference.l, reference.b, reference.displacement,
      reference.radius, inverse, psi, curvature);
  GeneratedAnalyticRadialQFrameCorrection(
      reference.alpha, reference.l, reference.b, reference.displacement,
      reference.radius, inverse, psi, p, q, delta_upper, frame_correction);
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      Real qq = 0.0;
      Real delta_product = 0.0;
      Real damping = 0.0;
      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          for (int E = 0; E < 4; ++E) {
            for (int F = 0; F < 4; ++F) {
              qq += 2.0*inverse[C][D]*inverse[E][F]
                    *q[E][C][A]*q[F][D][B];
              delta_product -= 2.0*inverse[C][D]*inverse[E][F]
                               *delta_lower[A][C][E]
                               *delta_lower[B][D][F];
            }
          }
        }
        Real normal_lower_A = 0.0;
        Real normal_lower_B = 0.0;
        for (int D = 0; D < 4; ++D) {
          normal_lower_A += psi[A][D]*normal[D];
          normal_lower_B += psi[B][D]*normal[D];
        }
        const Real frame_projector = ((C == A) ? normal_lower_B : 0.0)
                                     + ((C == B) ? normal_lower_A : 0.0)
                                     - psi[A][B]*normal[C];
        damping += gamma0*frame_projector*delta[C];
      }
      const Real value = curvature[A][B] + qq + delta_product + damping
                         + frame_correction[A][B];
      source[A][B] = value;
      source[B][A] = value;
      if (sectors != nullptr) {
        sectors->curvature[A][B] = sectors->curvature[B][A] = curvature[A][B];
        sectors->qq[A][B] = sectors->qq[B][A] = qq;
        sectors->delta_product[A][B] = sectors->delta_product[B][A] =
            delta_product;
        sectors->damping[A][B] = sectors->damping[B][A] = damping;
        sectors->frame_correction[A][B] =
            sectors->frame_correction[B][A] = frame_correction[A][B];
      }
    }
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
void AddCompactAnalyticOrdinaryGaugeSource(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const AnalyticRadialQPoint &reference,
    const CompactAnalyticCoordinateGeometry &compact_geometry,
    const Real hhat[4], const Real d_hhat[4][4], const Real gamma0,
    Real source[4][4]) {
  const CoordinateGhGeometry &geometry = compact_geometry.geometry;
  Real coordinate_hhat[4] = {};       // NOLINT(runtime/arrays)
  Real d_coordinate_hhat[4][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int A = 0; A < 4; ++A) {
      coordinate_hhat[a] += ReferenceCoframe(reference, A, a)*hhat[A];
      for (int p = 0; p < 4; ++p) {
        d_coordinate_hhat[p][a] +=
            AnalyticDCoframe(reference, p, A, a)*hhat[A]
            + ReferenceCoframe(reference, A, a)*d_hhat[p][A];
      }
    }
  }
  Real increment[4];       // NOLINT(runtime/arrays)
  Real d_increment[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    increment[a] = coordinate_hhat[a] - geometry.gauge_source[a];
    for (int p = 0; p < 4; ++p) {
      d_increment[p][a] = d_coordinate_hhat[p][a]
                          - compact_geometry.d_reference_gauge[p][a];
    }
  }
  Real coordinate_extra[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real nabla_ab = d_increment[a][b];
      Real nabla_ba = d_increment[b][a];
      for (int c = 0; c < 4; ++c) {
        nabla_ab -= geometry.christoffel[c][a][b]*increment[c];
        nabla_ba -= geometry.christoffel[c][b][a]*increment[c];
      }
      coordinate_extra[a][b] = -nabla_ab - nabla_ba;
      for (int c = 0; c < 4; ++c) {
        const Real projector = ((c == a) ? geometry.normal_lower[b] : 0.0)
                               + ((c == b) ? geometry.normal_lower[a] : 0.0)
                               - metric[a][b]*geometry.normal_upper[c];
        coordinate_extra[a][b] += gamma0*projector*increment[c];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          source[A][B] += ReferenceFrame(reference, A, a)
                          *ReferenceFrame(reference, B, b)
                          *coordinate_extra[a][b];
        }
      }
    }
  }
}

}  // namespace ref_gh

#endif  // REF_GH_ANALYTIC_RADIAL_Q_SOURCE_HPP_
