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
#include <type_traits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/analytic_radial_q_source.hpp"
#include "ref_gh/exact_matched_state.hpp"
#include "ref_gh/gamma2_damping.hpp"
#include "ref_gh/generated/analytic_radial_q_geometry.hpp"
#include "ref_gh/generated/analytic_radial_q_gauge.hpp"
#include "ref_gh/gauge_driver.hpp"
#include "ref_gh/phi_ordering.hpp"
#include "ref_gh/physical_gauge_target.hpp"
#include "ref_gh/puncture_exponent.hpp"
#include "ref_gh/q_relaxed_controller.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_characteristics.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/reference_analytic_radial_q.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_generic_singular.hpp"
#include "ref_gh/reference_projection.hpp"
#include "ref_gh/reference_gauge_baseline.hpp"
#include "ref_gh/residual_gauge_source.hpp"
#include "ref_gh/reference_time_dependent_spatial.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"
#include "ref_gh/reference_trumpet_q_controlled.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace {

constexpr int kAnalyticOracleRadiusCount = 10;
constexpr int kAnalyticOracleDirectionCount = 4;
constexpr int kAnalyticOraclePointCount =
    kAnalyticOracleRadiusCount*kAnalyticOracleDirectionCount;
constexpr int kGeneratedAnalyticOriginalPointCount = 4;
constexpr int kGeneratedAnalyticOraclePointCount =
    kGeneratedAnalyticOriginalPointCount + kAnalyticOraclePointCount;

void CheckExactMatchedQ1Predicate() {
  const bool exact = ref_gh::IsExactMatchedQ1StaticReference(
      true, false, false, false, 1.0, 0.0, 0.0);
  // Use the smallest normal probe for the rate/acceleration false cases.
  // Accelerator validation builds may flush subnormals to zero, which would
  // test the compiler mode rather than this exact production predicate.
  const bool false_cases[] = {  // NOLINT(runtime/arrays)
      ref_gh::IsExactMatchedQ1StaticReference(
          false, false, false, false, 1.0, 0.0, 0.0),
      ref_gh::IsExactMatchedQ1StaticReference(
          true, true, false, false, 1.0, 0.0, 0.0),
      ref_gh::IsExactMatchedQ1StaticReference(
          true, false, true, false, 1.0, 0.0, 0.0),
      ref_gh::IsExactMatchedQ1StaticReference(
          true, false, false, true, 1.0, 0.0, 0.0),
      ref_gh::IsExactMatchedQ1StaticReference(
          true, false, false, false,
          std::nextafter(1.0, 2.0), 0.0, 0.0),
      ref_gh::IsExactMatchedQ1StaticReference(
          true, false, false, false, 1.0,
          std::numeric_limits<Real>::min(), 0.0),
      ref_gh::IsExactMatchedQ1StaticReference(
          true, false, false, false, 1.0, 0.0,
          std::numeric_limits<Real>::min()),
  };
  bool valid = exact;
  for (const bool value : false_cases) valid = valid && !value;
  if (!valid) {
    std::cout << "### FATAL ERROR: exact matched q=1 predicate admitted a "
                 "controlled, moving, or nonidentical reference."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH exact matched q=1 predicate passed" << std::endl;
}

KOKKOS_INLINE_FUNCTION
void AnalyticOraclePoint(const int point, Real &x, Real &y, Real &z) {
  // Fixed before running the oracle.  Every direction is non-axis-aligned and
  // normalized.  The radii cover the puncture-adjacent logarithm, the RG=3M
  // Gaussian transition, and the far-field q-correction tail without r=0.
  constexpr Real radii[kAnalyticOracleRadiusCount] = {
      0.03, 0.05, 0.08, 0.125, 0.2, 0.4, 0.8, 1.5, 3.0, 5.0};
  constexpr Real inv_sqrt_14 = 0.26726124191242438468;
  constexpr Real directions[kAnalyticOracleDirectionCount][3] = {
      {inv_sqrt_14, 2.0*inv_sqrt_14, 3.0*inv_sqrt_14},
      {-2.0/3.0, 1.0/3.0, 2.0/3.0},
      {2.0*inv_sqrt_14, 3.0*inv_sqrt_14, -inv_sqrt_14},
      {3.0*inv_sqrt_14, -inv_sqrt_14, -2.0*inv_sqrt_14}};
  const int direction = point % kAnalyticOracleDirectionCount;
  const int radial = point/kAnalyticOracleDirectionCount;
  x = radii[radial]*directions[direction][0];
  y = radii[radial]*directions[direction][1];
  z = radii[radial]*directions[direction][2];
}

KOKKOS_INLINE_FUNCTION
void GeneratedAnalyticOraclePoint(const int point, Real &x, Real &y, Real &z) {
  if (point < kGeneratedAnalyticOriginalPointCount) {
    // Preserve the exact four-point matrix used by the original 216-sample
    // generated-geometry and recursive-accessor checkpoint.
    x = 0.31 + 0.071*point;
    y = -0.43 + 0.053*point;
    z = 0.27 - 0.037*point;
    return;
  }
  AnalyticOraclePoint(point - kGeneratedAnalyticOriginalPointCount, x, y, z);
}

KOKKOS_INLINE_FUNCTION
Real GeneratedRawSpinCondition(const ref_gh::ReferenceGeometry &reference,
                               const int A, const int B, const int C) {
  Real condition = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int c = 0; c < 4; ++c) {
      Real derivative = reference.d_frame[c][B][a];
      for (int d = 0; d < 4; ++d) {
        derivative += reference.christoffel[a][c][d]
                      *reference.frame[B][d];
      }
      condition += Kokkos::abs(
          reference.coframe[A][a]*reference.frame[C][c]*derivative);
    }
  }
  return condition;
}

KOKKOS_INLINE_FUNCTION
Real GeneratedSpinCondition(const ref_gh::ReferenceGeometry &reference,
                            const int A, const int B, const int C) {
  if (A == B) return 1.0;
  return 0.5*(GeneratedRawSpinCondition(reference, A, B, C)
              + GeneratedRawSpinCondition(reference, B, A, C));
}

KOKKOS_INLINE_FUNCTION
Real GeneratedRawSpinDerivativeCondition(
    const ref_gh::ReferenceGeometry &reference, const int D,
    const int A, const int B, const int C) {
  Real condition = 0.0;
  for (int p = 0; p < 4; ++p) {
    Real coordinate_condition = 0.0;
    for (int a = 0; a < 4; ++a) {
      const Real d_coframe =
          ref_gh::ReferenceCoframeDerivative(reference, p, A, a);
      for (int c = 0; c < 4; ++c) {
        Real frame_covariant_derivative = reference.d_frame[c][B][a];
        Real d_frame_covariant_derivative =
            reference.dd_frame[p][c][B][a];
        for (int d = 0; d < 4; ++d) {
          frame_covariant_derivative +=
              reference.christoffel[a][c][d]*reference.frame[B][d];
          d_frame_covariant_derivative +=
              reference.d_christoffel[p][a][c][d]
                *reference.frame[B][d]
              + reference.christoffel[a][c][d]
                *reference.d_frame[p][B][d];
        }
        const Real first_factor =
            d_coframe*reference.frame[C][c]
            + reference.coframe[A][a]*reference.d_frame[p][C][c];
        const Real second_factor =
            reference.coframe[A][a]*reference.frame[C][c];
        coordinate_condition +=
            Kokkos::abs(first_factor*frame_covariant_derivative)
            + Kokkos::abs(second_factor*d_frame_covariant_derivative);
      }
    }
    condition += Kokkos::abs(reference.frame[D][p])*coordinate_condition;
  }
  return condition;
}

KOKKOS_INLINE_FUNCTION
Real GeneratedSpinDerivativeCondition(
    const ref_gh::ReferenceGeometry &reference, const int D,
    const int A, const int B, const int C) {
  if (A == B) return 1.0;
  return 0.5*(GeneratedRawSpinDerivativeCondition(reference, D, A, B, C)
              + GeneratedRawSpinDerivativeCondition(
                    reference, D, B, A, C));
}

KOKKOS_INLINE_FUNCTION
Real GeneratedRiemannCondition(const ref_gh::ReferenceGeometry &reference,
                               const int A, const int B,
                               const int C, const int D) {
  // Propagate the inner spin/spin-derivative contraction scales.  Near the
  // puncture a final O(1) curvature component can inherit roundoff from much
  // larger differentiated-frame terms even when the outer Cartan sum itself
  // is not strongly cancelling.
  Real condition = GeneratedSpinDerivativeCondition(reference, C, A, B, D)
                   + GeneratedSpinDerivativeCondition(reference, D, A, B, C);
  for (int E = 0; E < 4; ++E) {
    const Real first_left = reference.spin[A][E][C];
    const Real first_right = reference.spin[E][B][D];
    condition += Kokkos::abs(first_left)
                   *GeneratedSpinCondition(reference, E, B, D)
                 + Kokkos::abs(first_right)
                   *GeneratedSpinCondition(reference, A, E, C);
    const Real second_left = reference.spin[A][E][D];
    const Real second_right = reference.spin[E][B][C];
    condition += Kokkos::abs(second_left)
                   *GeneratedSpinCondition(reference, E, B, C)
                 + Kokkos::abs(second_right)
                   *GeneratedSpinCondition(reference, A, E, D);
    condition += Kokkos::abs(reference.structure4[E][C][D])
                 *GeneratedSpinCondition(reference, A, B, E);
  }
  return condition;
}

KOKKOS_INLINE_FUNCTION
Real GeneratedRicciCondition(const ref_gh::ReferenceGeometry &reference,
                             const int A, const int B) {
  Real condition = 0.0;
  for (int C = 0; C < 4; ++C) {
    condition += GeneratedRiemannCondition(reference, C, A, C, B);
  }
  return condition;
}

// Independent generic-jet oracle for the mixed-third portion of the moving
// reference gauge subtraction.  This deliberately follows the mature cache
// update staging and is never used by the analytic production backend.
KOKKOS_INLINE_FUNCTION
bool GenericReferenceDtThetaOracle(
    const ref_gh::ReferenceJet &alpha, const ref_gh::ReferenceJet &psi2,
    const ref_gh::ReferenceJet &shift_q, const Real displacement[3],
    const ref_gh::ReferenceGeometry &reference, Real dt_theta[4],
    Real hhat_condition[4], Real d_hhat_condition[4][4],
    Real theta_condition[4], Real dt_theta_condition[4]) {
  const ref_gh::ReferenceJet inverse_alpha = ref_gh::Reciprocal(alpha);
  const ref_gh::ReferenceJet inverse_psi2 = ref_gh::Reciprocal(psi2);
  const ref_gh::ReferenceJet coordinates[3] = {
      ref_gh::CoordinateJet(displacement[0], 1),
      ref_gh::CoordinateJet(displacement[1], 2),
      ref_gh::CoordinateJet(displacement[2], 3)};
  ref_gh::ReferenceJet coframe[4][4];        // NOLINT(runtime/arrays)
  ref_gh::ReferenceJet frame[4][4];          // NOLINT(runtime/arrays)
  ref_gh::ReferenceJet metric[4][4];         // NOLINT(runtime/arrays)
  ref_gh::ReferenceJet inverse_metric[4][4]; // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      coframe[A][a] = ref_gh::ConstantJet(0.0);
      frame[A][a] = ref_gh::ConstantJet(0.0);
    }
  }
  coframe[0][0] = alpha;
  frame[0][0] = inverse_alpha;
  for (int I = 0; I < 3; ++I) {
    coframe[I + 1][0] = psi2*shift_q*coordinates[I];
    coframe[I + 1][I + 1] = psi2;
    frame[0][I + 1] = -(shift_q*coordinates[I]*inverse_alpha);
    frame[I + 1][I + 1] = inverse_psi2;
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = -(coframe[0][a]*coframe[0][b]);
      inverse_metric[a][b] = -(frame[0][a]*frame[0][b]);
      for (int I = 1; I < 4; ++I) {
        metric[a][b] = metric[a][b] + coframe[I][a]*coframe[I][b];
        inverse_metric[a][b] =
            inverse_metric[a][b] + frame[I][a]*frame[I][b];
      }
    }
  }

  Real h_upper[4] = {};       // NOLINT(runtime/arrays)
  Real d_h_upper[4][4] = {};  // NOLINT(runtime/arrays)
  Real dt_di_h_upper[3][4] = {};  // NOLINT(runtime/arrays)
  Real h_upper_condition[4] = {};  // NOLINT(runtime/arrays)
  Real d_h_upper_condition[4][4] = {};  // NOLINT(runtime/arrays)
  Real dt_di_h_upper_condition[3][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        h_upper[a] -= inverse_metric[b][c].value
                      *reference.christoffel[a][b][c];
        h_upper_condition[a] += Kokkos::abs(
            inverse_metric[b][c].value*reference.christoffel[a][b][c]);
        for (int p = 0; p < 4; ++p) {
          d_h_upper[p][a] -=
              inverse_metric[b][c].d[p]*reference.christoffel[a][b][c]
              + inverse_metric[b][c].value
                *reference.d_christoffel[p][a][b][c];
          d_h_upper_condition[p][a] += Kokkos::abs(
              inverse_metric[b][c].d[p]*reference.christoffel[a][b][c])
              + Kokkos::abs(inverse_metric[b][c].value
                            *reference.d_christoffel[p][a][b][c]);
        }
        for (int spatial = 0; spatial < 3; ++spatial) {
          const int s = spatial + 1;
          Real dt_di_christoffel = 0.0;
          Real dt_di_christoffel_condition = 0.0;
          for (int ell = 0; ell < 4; ++ell) {
            const Real first = 0.5*(
                metric[ell][c].d[b] + metric[ell][b].d[c]
                - metric[b][c].d[ell]);
            const Real first_t = 0.5*(
                metric[ell][c].dd[0][b] + metric[ell][b].dd[0][c]
                - metric[b][c].dd[0][ell]);
            const Real first_i = 0.5*(
                metric[ell][c].dd[s][b] + metric[ell][b].dd[s][c]
                - metric[b][c].dd[s][ell]);
            const Real first_ti = 0.5*(
                metric[ell][c].dt_dd[spatial][b]
                + metric[ell][b].dt_dd[spatial][c]
                - metric[b][c].dt_dd[spatial][ell]);
            dt_di_christoffel +=
                inverse_metric[a][ell].dd[0][s]*first
                + inverse_metric[a][ell].d[s]*first_t
                + inverse_metric[a][ell].d[0]*first_i
                + inverse_metric[a][ell].value*first_ti;
            dt_di_christoffel_condition += Kokkos::abs(
                inverse_metric[a][ell].dd[0][s]*first)
                + Kokkos::abs(inverse_metric[a][ell].d[s]*first_t)
                + Kokkos::abs(inverse_metric[a][ell].d[0]*first_i)
                + Kokkos::abs(inverse_metric[a][ell].value*first_ti);
          }
          dt_di_h_upper[spatial][a] -=
              inverse_metric[b][c].dd[0][s]
                *reference.christoffel[a][b][c]
              + inverse_metric[b][c].d[s]
                *reference.d_christoffel[0][a][b][c]
              + inverse_metric[b][c].d[0]
                *reference.d_christoffel[s][a][b][c]
              + inverse_metric[b][c].value*dt_di_christoffel;
          dt_di_h_upper_condition[spatial][a] += Kokkos::abs(
              inverse_metric[b][c].dd[0][s]
                *reference.christoffel[a][b][c])
              + Kokkos::abs(inverse_metric[b][c].d[s]
                            *reference.d_christoffel[0][a][b][c])
              + Kokkos::abs(inverse_metric[b][c].d[0]
                            *reference.d_christoffel[s][a][b][c])
              + Kokkos::abs(inverse_metric[b][c].value)
                *(Kokkos::abs(dt_di_christoffel)
                  + dt_di_christoffel_condition);
        }
      }
    }
  }
  Real h_lower[4] = {};             // NOLINT(runtime/arrays)
  Real d_h_lower[4][4] = {};        // NOLINT(runtime/arrays)
  Real dt_di_h_lower[3][4] = {};    // NOLINT(runtime/arrays)
  Real h_lower_condition[4] = {};   // NOLINT(runtime/arrays)
  Real d_h_lower_condition[4][4] = {};  // NOLINT(runtime/arrays)
  Real dt_di_h_lower_condition[3][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      h_lower[a] += metric[a][b].value*h_upper[b];
      h_lower_condition[a] += Kokkos::abs(metric[a][b].value)
          *(Kokkos::abs(h_upper[b]) + h_upper_condition[b]);
      for (int p = 0; p < 4; ++p) {
        d_h_lower[p][a] += metric[a][b].d[p]*h_upper[b]
                           + metric[a][b].value*d_h_upper[p][b];
        d_h_lower_condition[p][a] +=
            Kokkos::abs(metric[a][b].d[p])
              *(Kokkos::abs(h_upper[b]) + h_upper_condition[b])
            + Kokkos::abs(metric[a][b].value)
              *(Kokkos::abs(d_h_upper[p][b])
                + d_h_upper_condition[p][b]);
      }
      for (int spatial = 0; spatial < 3; ++spatial) {
        const int s = spatial + 1;
        dt_di_h_lower[spatial][a] +=
            metric[a][b].dd[0][s]*h_upper[b]
            + metric[a][b].d[s]*d_h_upper[0][b]
            + metric[a][b].d[0]*d_h_upper[s][b]
            + metric[a][b].value*dt_di_h_upper[spatial][b];
        dt_di_h_lower_condition[spatial][a] +=
            Kokkos::abs(metric[a][b].dd[0][s])
              *(Kokkos::abs(h_upper[b]) + h_upper_condition[b])
            + Kokkos::abs(metric[a][b].d[s])
              *(Kokkos::abs(d_h_upper[0][b])
                + d_h_upper_condition[0][b])
            + Kokkos::abs(metric[a][b].d[0])
              *(Kokkos::abs(d_h_upper[s][b])
                + d_h_upper_condition[s][b])
            + Kokkos::abs(metric[a][b].value)
              *(Kokkos::abs(dt_di_h_upper[spatial][b])
                + dt_di_h_upper_condition[spatial][b]);
      }
    }
  }
  Real hhat[4] = {};             // NOLINT(runtime/arrays)
  Real d_hhat[4][4] = {};        // NOLINT(runtime/arrays)
  Real dt_di_hhat[3][4] = {};    // NOLINT(runtime/arrays)
  Real dt_di_hhat_condition[3][4] = {};  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    hhat_condition[A] = 0.0;
    theta_condition[A] = 0.0;
    dt_theta_condition[A] = 0.0;
    for (int p = 0; p < 4; ++p) d_hhat_condition[p][A] = 0.0;
    for (int a = 0; a < 4; ++a) {
      hhat[A] += reference.frame[A][a]*h_lower[a];
      hhat_condition[A] += Kokkos::abs(reference.frame[A][a])
          *(Kokkos::abs(h_lower[a]) + h_lower_condition[a]);
      for (int p = 0; p < 4; ++p) {
        d_hhat[p][A] += reference.d_frame[p][A][a]*h_lower[a]
                        + reference.frame[A][a]*d_h_lower[p][a];
        d_hhat_condition[p][A] +=
            Kokkos::abs(reference.d_frame[p][A][a])
              *(Kokkos::abs(h_lower[a]) + h_lower_condition[a])
            + Kokkos::abs(reference.frame[A][a])
              *(Kokkos::abs(d_h_lower[p][a])
                + d_h_lower_condition[p][a]);
      }
      for (int spatial = 0; spatial < 3; ++spatial) {
        const int s = spatial + 1;
        dt_di_hhat[spatial][A] +=
            reference.dd_frame[0][s][A][a]*h_lower[a]
            + reference.d_frame[s][A][a]*d_h_lower[0][a]
            + reference.d_frame[0][A][a]*d_h_lower[s][a]
            + reference.frame[A][a]*dt_di_h_lower[spatial][a];
        dt_di_hhat_condition[spatial][A] +=
            Kokkos::abs(reference.dd_frame[0][s][A][a])
              *(Kokkos::abs(h_lower[a]) + h_lower_condition[a])
            + Kokkos::abs(reference.d_frame[s][A][a])
              *(Kokkos::abs(d_h_lower[0][a])
                + d_h_lower_condition[0][a])
            + Kokkos::abs(reference.d_frame[0][A][a])
              *(Kokkos::abs(d_h_lower[s][a])
                + d_h_lower_condition[s][a])
            + Kokkos::abs(reference.frame[A][a])
              *(Kokkos::abs(dt_di_h_lower[spatial][a])
                + dt_di_h_lower_condition[spatial][a]);
      }
    }
  }
  if (!(inverse_metric[0][0].value < 0.0)) return false;
  const Real lapse = 1.0/Kokkos::sqrt(-inverse_metric[0][0].value);
  const Real dt_lapse = 0.5*lapse*lapse*lapse*inverse_metric[0][0].d[0];
  Real shift[3];     // NOLINT(runtime/arrays)
  Real dt_shift[3];  // NOLINT(runtime/arrays)
  for (int spatial = 0; spatial < 3; ++spatial) {
    const Real inverse0i = inverse_metric[0][spatial + 1].value;
    shift[spatial] = lapse*lapse*inverse0i;
    dt_shift[spatial] = 2.0*lapse*dt_lapse*inverse0i
        + lapse*lapse*inverse_metric[0][spatial + 1].d[0];
  }
  for (int A = 0; A < 4; ++A) {
    dt_theta[A] = 0.0;
    for (int spatial = 0; spatial < 3; ++spatial) {
      dt_theta[A] -= dt_shift[spatial]*d_hhat[spatial + 1][A]
                     + shift[spatial]*dt_di_hhat[spatial][A];
      theta_condition[A] += Kokkos::abs(shift[spatial])
          *(Kokkos::abs(d_hhat[spatial + 1][A])
            + d_hhat_condition[spatial + 1][A]);
      dt_theta_condition[A] +=
          Kokkos::abs(dt_shift[spatial])
            *(Kokkos::abs(d_hhat[spatial + 1][A])
              + d_hhat_condition[spatial + 1][A])
          + Kokkos::abs(shift[spatial])
            *(Kokkos::abs(dt_di_hhat[spatial][A])
              + dt_di_hhat_condition[spatial][A]);
    }
    for (int B = 0; B < 4; ++B) {
      Real motion = ref_gh::ReferenceFrameMotion(reference, A, 0, B);
      Real dt_motion = ref_gh::ReferenceDtFrameMotion(reference, A, 0, B);
      Real motion_condition = Kokkos::abs(motion);
      Real dt_motion_condition = Kokkos::abs(dt_motion);
      for (int spatial = 0; spatial < 3; ++spatial) {
        const Real spatial_motion = ref_gh::ReferenceFrameMotion(
            reference, A, spatial + 1, B);
        motion -= shift[spatial]*spatial_motion;
        const Real dt_spatial_motion = ref_gh::ReferenceDtFrameMotion(
            reference, A, spatial + 1, B);
        dt_motion -= dt_shift[spatial]*spatial_motion
                     + shift[spatial]*dt_spatial_motion;
        motion_condition += Kokkos::abs(
            shift[spatial]*spatial_motion);
        dt_motion_condition += Kokkos::abs(
            dt_shift[spatial]*spatial_motion)
            + Kokkos::abs(shift[spatial]*dt_spatial_motion);
      }
      dt_theta[A] -= dt_motion*hhat[B] + motion*d_hhat[0][B];
      theta_condition[A] += motion_condition
          *(Kokkos::abs(hhat[B]) + hhat_condition[B]);
      dt_theta_condition[A] +=
          dt_motion_condition*(Kokkos::abs(hhat[B]) + hhat_condition[B])
          + motion_condition*(Kokkos::abs(d_hhat[0][B])
                              + d_hhat_condition[0][B]);
    }
    if (!Kokkos::isfinite(dt_theta[A])) return false;
  }
  return true;
}

template <typename Maximum>
KOKKOS_INLINE_FUNCTION
void UpdateGeneratedAnalyticOracleMaximum(const Real generated,
                                          const Real generic,
                                          const int category,
                                          Maximum &maximum,
                                          const Real conditioning_scale = 1.0,
                                          const int diagnostic_location = -1) {
  Real scale = 1.0;
  scale = fmax(scale, Kokkos::abs(generated));
  scale = fmax(scale, Kokkos::abs(generic));
  scale = fmax(scale, conditioning_scale);
  // Keep the production-cache oracle's established contraction-depth scales.
  const Real operation_scale =
      (category == 14 || category == 15) ? 256.0
      : ((category == 16) ? 4.0
         : ((category == 17 || category == 19) ? 16.0
            : ((category == 18) ? 32.0 : 1.0)));
  const Real error = Kokkos::abs(generated - generic)
                     /(scale*operation_scale);
  if (error > maximum.val) {
    maximum.val = error;
    maximum.loc = diagnostic_location >= 0 ? diagnostic_location : category;
  }
}

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

        // General time-dependent residual-variable equivalence.  These
        // reference and delta values are independent, so this exercises every
        // explicit S_H/S_theta, delta-beta, and frame-motion term.
        Real reference_hhat[4];        // NOLINT(runtime/arrays)
        Real reference_theta[4];       // NOLINT(runtime/arrays)
        Real reference_d_hhat[4][4];   // NOLINT(runtime/arrays)
        Real reference_dt_theta[4];    // NOLINT(runtime/arrays)
        Real reference_target[4];      // NOLINT(runtime/arrays)
        Real delta_hhat[4];            // NOLINT(runtime/arrays)
        Real delta_theta[4];           // NOLINT(runtime/arrays)
        Real delta_target[4];          // NOLINT(runtime/arrays)
        Real d_delta_hhat[3][4];       // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          reference_hhat[A] = 0.021*static_cast<Real>(A - 1) + 0.004*scale;
          reference_theta[A] = -0.017*static_cast<Real>(A + 1)
                               + 0.003*scale;
          reference_dt_theta[A] = 0.013*static_cast<Real>(2 - A)
                                  - 0.002*scale;
          reference_target[A] = -0.019*static_cast<Real>(A - 2)
                                + 0.005*scale;
          delta_hhat[A] = hhat[A] - reference_hhat[A];
          delta_theta[A] = theta[A] - reference_theta[A];
          delta_target[A] = target[A] - reference_target[A];
          reference_d_hhat[0][A] =
              0.015*static_cast<Real>(A + 1) - 0.003*scale;
          for (int p = 0; p < 3; ++p) {
            reference_d_hhat[p + 1][A] =
                0.009*static_cast<Real>(1 + A - 2*p) + 0.002*scale;
            d_delta_hhat[p][A] =
                d_hhat[p][A] - reference_d_hhat[p + 1][A];
          }
        }
        const Real delta_shift[3] = {
          0.012, -0.008 + 0.001*scale, 0.006};
        Real reference_shift[3];  // NOLINT(runtime/arrays)
        for (int p = 0; p < 3; ++p) {
          reference_shift[p] = shift[p] - delta_shift[p];
        }
        const ref_gh::GaugeDriverRhs residual_rhs =
            ref_gh::ComputeGaugeDriverResidualRhs(
                reference, reference_hhat, reference_theta,
                reference_d_hhat, reference_dt_theta, delta_hhat,
                delta_theta, upsilon, d_delta_hhat, shift, reference_shift,
                delta_shift, delta_target, reference_target,
                conformal_gamma, mu, eta, eta_beta, false);
        for (int A = 0; A < 4; ++A) {
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(residual_rhs.hhat[A]
                          - (rhs.hhat[A] - reference_d_hhat[0][A])));
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(residual_rhs.theta[A]
                          - (rhs.theta[A] - reference_dt_theta[A])));
        }
        for (int p = 0; p < 3; ++p) {
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(residual_rhs.upsilon[p] - rhs.upsilon[p]));
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

void CheckRelativeExponentIdentity() {
  constexpr int nsamples = 7;
  Real maximum_error = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh relative exponent identity",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real displacement[3] = {
            0.21 + 0.037*sample, -0.33 + 0.019*sample,
            0.27 + 0.011*sample};
        const Real radius2 = displacement[0]*displacement[0]
                             + displacement[1]*displacement[1]
                             + displacement[2]*displacement[2];
        const Real radius = Kokkos::sqrt(radius2);
        const Real q_reference = 0.65 + 0.09*sample;
        const Real epsilon_exact = -0.18 + 0.055*sample;
        const Real lambda = Kokkos::pow(radius, -q_reference);
        const Real relative_scale = Kokkos::pow(radius, -2.0*epsilon_exact);
        const Real seed[3][3] = {
            {1.4, 0.12, -0.07}, {0.12, 0.9, 0.08},
            {-0.07, 0.08, 1.2}};
        Real relative_metric[4][4] = {};  // NOLINT(runtime/arrays)
        relative_metric[0][0] = -1.0;
        Real phi[3][4][4] = {};  // NOLINT(runtime/arrays)
        Real spatial_coframe[3][3] = {};  // NOLINT(runtime/arrays)
        Real physical_metric[4][4] = {};  // NOLINT(runtime/arrays)
        Real d_physical_metric[4][4][4] = {};  // NOLINT(runtime/arrays)
        physical_metric[0][0] = -1.0;
        for (int I = 0; I < 3; ++I) {
          spatial_coframe[I][I] = lambda;
          for (int J = 0; J < 3; ++J) {
            relative_metric[I + 1][J + 1] = relative_scale*seed[I][J];
            physical_metric[I + 1][J + 1] =
                lambda*lambda*relative_metric[I + 1][J + 1];
          }
        }
        for (int K = 0; K < 3; ++K) {
          for (int I = 0; I < 3; ++I) {
            for (int J = 0; J < 3; ++J) {
              const Real d_g = -2.0*epsilon_exact*displacement[K]/radius2
                               *relative_metric[I + 1][J + 1];
              phi[K][I + 1][J + 1] = d_g/lambda;
            }
          }
        }
        for (int k = 0; k < 3; ++k) {
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              d_physical_metric[k + 1][i + 1][j + 1] =
                  -2.0*(q_reference + epsilon_exact)
                  *displacement[k]/radius2
                  *physical_metric[i + 1][j + 1];
            }
          }
        }
        Real epsilon_g = NAN;
        const bool valid = ref_gh::ComputeRelativeSpatialExponentMismatch(
            relative_metric, phi, spatial_coframe, displacement, epsilon_g);
        const ref_gh::LocalPunctureExponents physical =
            ref_gh::ComputeLocalPunctureExponents(
                physical_metric, d_physical_metric, displacement);
        if (!valid || !physical.spatial_valid) {
          local_maximum = std::numeric_limits<Real>::infinity();
          return;
        }
        local_maximum = fmax(
            local_maximum, Kokkos::abs(epsilon_g - epsilon_exact));
        local_maximum = fmax(
            local_maximum,
            Kokkos::abs(physical.q - (q_reference + epsilon_g)));
      }, Kokkos::Max<Real>(maximum_error));
  if (!(maximum_error <= 2.0e-13)) {
    std::cout << "### FATAL ERROR: relative exponent identity failed: error="
              << maximum_error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH relative exponent identity passed: max-error="
            << maximum_error << std::endl;
}

void CheckTrumpetQControlledReference(const DvceArray2D<Real> &table) {
  constexpr int nsamples = 64;
  Real maximum_identity_error = 0.0;
  Real maximum_profile_error = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh q-controlled trumpet reference",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_identity,
                    Real &local_profile) {
        const Real x = 0.41 + 0.013*(sample % 4);
        const Real y = -0.36 + 0.017*((sample/4) % 4);
        const Real z = 0.29 + 0.019*((sample/16) % 4);
        const Real displacement[3] = {x, y, z};
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
        const Real rho = radius;

        const ref_gh::TrumpetQControlledReferenceParameters identity_params{
            1.0, {0.0, 0.0, 0.0}, 3.0, 1.0, 0.0, 0.0};
        ref_gh::ReferenceGeometry controlled_identity;
        const ref_gh::TrumpetQControlledReference identity_provider{
            table, identity_params};
        identity_provider.Populate(0.0, x, y, z, controlled_identity);
        ref_gh::ReferenceGeometry exact;
        const ref_gh::TrumpetSchwarzschildReference exact_provider{
            table, 1.0, {0.0, 0.0, 0.0}};
        exact_provider.Populate(0.0, x, y, z, exact);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            local_identity = fmax(
                local_identity,
                Kokkos::abs(controlled_identity.metric[a][b]
                            - exact.metric[a][b]));
            for (int p = 0; p < 4; ++p) {
              local_identity = fmax(
                  local_identity,
                  Kokkos::abs(controlled_identity.d_metric[p][a][b]
                              - exact.d_metric[p][a][b]));
            }
          }
        }

        const Real q_value = 0.75 + 0.5*static_cast<Real>(sample % 5)/4.0;
        const Real q_dot = -0.08 + 0.04*static_cast<Real>((sample/5) % 5);
        const Real q_ddot = 0.03 - 0.015*static_cast<Real>((sample/25) % 3);
        const ref_gh::TrumpetQControlledReferenceParameters params{
            1.0, {0.0, 0.0, 0.0}, 3.0, q_value, q_dot, q_ddot};
        ref_gh::ReferenceJet alpha;
        ref_gh::ReferenceJet spatial_cholesky;
        ref_gh::ReferenceJet shift_q;
        ref_gh::ReferenceJet q;
        ref_gh::ReferenceJet window;
        ref_gh::TrumpetQControlledProfileJets(
            table, params, x, y, z, alpha, spatial_cholesky, shift_q,
            &q, &window);
        const ref_gh::RadialProfile alpha_profile =
            ref_gh::InterpolateTrumpetProfile(
                table, ref_gh::kCoeffAlpha, rho);
        const ref_gh::RadialProfile psi2_profile =
            ref_gh::ArealRadiusToPsi2(
                ref_gh::InterpolateTrumpetProfile(
                    table, ref_gh::kCoeffArealRadius, rho), rho);
        ref_gh::RadialProfile shift_profile =
            ref_gh::InterpolateTrumpetProfile(
                table, ref_gh::kCoeffShiftQ, rho);
        const Real expected_window = Kokkos::exp(-(radius/3.0)*(radius/3.0));
        const Real expected_factor = Kokkos::exp(
            -(q_value - 1.0)*expected_window*Kokkos::log(rho));
        local_profile = fmax(
            local_profile, Kokkos::abs(alpha.value - alpha_profile.value));
        local_profile = fmax(
            local_profile, Kokkos::abs(shift_q.value - shift_profile.value));
        local_profile = fmax(
            local_profile,
            Kokkos::abs(spatial_cholesky.value
                        - psi2_profile.value*expected_factor));
        local_profile = fmax(local_profile, Kokkos::abs(q.value - q_value));
        local_profile = fmax(local_profile, Kokkos::abs(q.d[0] - q_dot));
        local_profile = fmax(local_profile, Kokkos::abs(q.dd[0][0] - q_ddot));
        local_profile = fmax(
            local_profile, Kokkos::abs(window.value - expected_window));
        for (int p = 1; p < 4; ++p) {
          local_profile = fmax(local_profile, Kokkos::abs(alpha.d[0]));
          local_profile = fmax(local_profile, Kokkos::abs(shift_q.d[0]));
          local_profile = fmax(local_profile, Kokkos::abs(q.d[p]));
        }
      }, Kokkos::Max<Real>(maximum_identity_error),
      Kokkos::Max<Real>(maximum_profile_error));
  if (!(maximum_identity_error <= 2.0e-13)
      || !(maximum_profile_error <= 2.0e-13)) {
    std::cout << "### FATAL ERROR: q-controlled trumpet reference oracle failed: "
              << "q=1 identity=" << maximum_identity_error
              << " profile=" << maximum_profile_error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH q-controlled trumpet provider passed: q=1 identity="
            << maximum_identity_error
            << " profile=" << maximum_profile_error << std::endl;
}

void CheckAnalyticRadialQCoefficients(const DvceArray2D<Real> &table) {
  // Preserve the accepted 216-point raw-absolute oracle unchanged.  The
  // expanded radial gate below is additional coverage, not a replacement.
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = 4;
  constexpr int nsamples = nq*nrate*nacceleration*npoints;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh analytic radial-q coefficients",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const Real q = q_values[work % nq];
        work /= nq;
        const Real q_dot = q_dot_values[work % nrate];
        work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration];
        work /= nacceleration;
        const Real point = static_cast<Real>(work);
        const Real x = 0.31 + 0.071*point;
        const Real y = -0.43 + 0.053*point;
        const Real z = 0.27 - 0.037*point;
        const Real displacement[3] = {x, y, z};
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);

        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};

        const ref_gh::TrumpetQControlledReferenceParameters parameters{
            1.0, {0.0, 0.0, 0.0}, 3.0, q, q_dot, q_ddot};
        ref_gh::ReferenceJet alpha;
        ref_gh::ReferenceJet spatial_cholesky;
        ref_gh::ReferenceJet shift_b;
        ref_gh::TrumpetQControlledProfileJets(
            table, parameters, x, y, z, alpha, spatial_cholesky, shift_b);
        const ref_gh::AnalyticRadialScalar analytic[3] = {
            analytic_alpha, analytic_l, analytic_b};
        const ref_gh::ReferenceJet generic[3] = {
            alpha, spatial_cholesky, shift_b};
        for (int scalar = 0; scalar < 3; ++scalar) {
          local_maximum = fmax(
              local_maximum,
              Kokkos::abs(analytic[scalar].value - generic[scalar].value));
          for (int p = 0; p < 4; ++p) {
            local_maximum = fmax(
                local_maximum,
                Kokkos::abs(analytic[scalar].D(displacement, radius, p)
                            - generic[scalar].d[p]));
            for (int r = 0; r < 4; ++r) {
              local_maximum = fmax(
                  local_maximum,
                  Kokkos::abs(analytic[scalar].DD(
                                  displacement, radius, p, r)
                              - generic[scalar].dd[p][r]));
            }
          }
          for (int i = 0; i < 3; ++i) {
            for (int p = 0; p < 4; ++p) {
              local_maximum = fmax(
                  local_maximum,
                  Kokkos::abs(analytic[scalar].DtDD(
                                  displacement, radius, i, p)
                              - generic[scalar].dt_dd[i][p]));
            }
          }
        }
      }, Kokkos::Max<Real>(maximum));
  Kokkos::fence();
  if (!(maximum <= 2.0e-13)) {
    std::cout << "### FATAL ERROR: analytic radial-q coefficient oracle failed: "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH analytic radial-q coefficient oracle passed: "
            << "samples=" << nsamples << " max error=" << maximum
            << std::endl;
}

KOKKOS_INLINE_FUNCTION
Real ConditionedAnalyticCoefficientError(const Real analytic,
                                         const Real generic) {
  const Real scale = fmax(1.0, fmax(Kokkos::abs(analytic),
                                   Kokkos::abs(generic)));
  return Kokkos::abs(analytic - generic)/scale;
}

void CheckExpandedAnalyticRadialQCoefficients(
    const DvceArray2D<Real> &table) {
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = kAnalyticOraclePointCount;
  constexpr int nsamples = nq*nrate*nacceleration*npoints;
  Real maximum = 0.0;
  Real near_puncture_maximum = 0.0;
  Real gaussian_transition_maximum = 0.0;
  Real far_field_maximum = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh analytic radial-q coefficients",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum,
                    Real &local_near_puncture,
                    Real &local_gaussian_transition,
                    Real &local_far_field) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const Real q = q_values[work % nq];
        work /= nq;
        const Real q_dot = q_dot_values[work % nrate];
        work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration];
        work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        AnalyticOraclePoint(work, x, y, z);
        const Real displacement[3] = {x, y, z};
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);

        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};

        const ref_gh::TrumpetQControlledReferenceParameters parameters{
            1.0, {0.0, 0.0, 0.0}, 3.0, q, q_dot, q_ddot};
        ref_gh::ReferenceJet alpha;
        ref_gh::ReferenceJet spatial_cholesky;
        ref_gh::ReferenceJet shift_b;
        ref_gh::TrumpetQControlledProfileJets(
            table, parameters, x, y, z, alpha, spatial_cholesky, shift_b);
        const ref_gh::AnalyticRadialScalar analytic[3] = {
            analytic_alpha, analytic_l, analytic_b};
        const ref_gh::ReferenceJet generic[3] = {
            alpha, spatial_cholesky, shift_b};
        Real sample_maximum = 0.0;
        for (int scalar = 0; scalar < 3; ++scalar) {
          sample_maximum = fmax(
              sample_maximum,
              ConditionedAnalyticCoefficientError(
                  analytic[scalar].value, generic[scalar].value));
          for (int p = 0; p < 4; ++p) {
            sample_maximum = fmax(
                sample_maximum,
                ConditionedAnalyticCoefficientError(
                    analytic[scalar].D(displacement, radius, p),
                    generic[scalar].d[p]));
            for (int r = 0; r < 4; ++r) {
              sample_maximum = fmax(
                  sample_maximum,
                  ConditionedAnalyticCoefficientError(
                      analytic[scalar].DD(displacement, radius, p, r),
                      generic[scalar].dd[p][r]));
            }
          }
          for (int i = 0; i < 3; ++i) {
            for (int p = 0; p < 4; ++p) {
              sample_maximum = fmax(
                  sample_maximum,
                  ConditionedAnalyticCoefficientError(
                      analytic[scalar].DtDD(
                          displacement, radius, i, p),
                      generic[scalar].dt_dd[i][p]));
            }
          }
        }
        local_maximum = fmax(local_maximum, sample_maximum);
        const int radial = work/kAnalyticOracleDirectionCount;
        if (radial <= 3) {
          local_near_puncture = fmax(local_near_puncture, sample_maximum);
        }
        if (radial == 8) {
          local_gaussian_transition =
              fmax(local_gaussian_transition, sample_maximum);
        }
        if (radial == 9) {
          local_far_field = fmax(local_far_field, sample_maximum);
        }
      }, Kokkos::Max<Real>(maximum),
      Kokkos::Max<Real>(near_puncture_maximum),
      Kokkos::Max<Real>(gaussian_transition_maximum),
      Kokkos::Max<Real>(far_field_maximum));
  Kokkos::fence();
  if (!(maximum <= 2.0e-13)) {
    std::cout << "### FATAL ERROR: expanded analytic radial-q coefficient "
                 "oracle failed: "
              << maximum << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH expanded analytic radial-q coefficient oracle "
               "passed: samples=" << nsamples
            << " conditioned error=" << maximum
            << " near-r<=0.125=" << near_puncture_maximum
            << " gaussian-r=3=" << gaussian_transition_maximum
            << " far-r=5=" << far_field_maximum
            << std::endl;
}

void ReportGeneratedGeometryMismatch(const DvceArray2D<Real> &table,
                                     const int encoded_location) {
  const int diagnostic_kind = encoded_location/1000000;
  if (diagnostic_kind < 1 || diagnostic_kind > 4) return;
  int work = encoded_location - diagnostic_kind*1000000;
  const int sample = work/256;
  work %= 256;
  const int a = work/64; work %= 64;
  const int b = work/16; work %= 16;
  const int c = work/4;
  const int d = work % 4;
  DvceArray1D<Real> values("ref_gh generated mismatch values", 12);
  Kokkos::parallel_for(
      "ref_gh generated mismatch detail", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int) {
        constexpr int nq = 6;
        constexpr int nrate = 3;
        constexpr int nacceleration = 3;
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int sample_work = sample;
        const Real q = q_values[sample_work % nq]; sample_work /= nq;
        const Real q_dot = q_dot_values[sample_work % nrate];
        sample_work /= nrate;
        const Real q_ddot = q_ddot_values[sample_work % nacceleration];
        sample_work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        GeneratedAnalyticOraclePoint(sample_work, x, y, z);
        const Real displacement[3] = {x, y, z};
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        ref_gh::ReferenceGeometry generated;
        ref_gh::PopulateGeneratedAnalyticRadialQGeometry(
            analytic_alpha, analytic_l, analytic_b, displacement, radius,
            generated);
        const ref_gh::AnalyticRadialQPoint accessor{
            analytic_alpha, analytic_l, analytic_b, {x, y, z}, radius};
        ref_gh::ReferenceGeometry generic;
        ref_gh::PopulateIsotropicReferenceGeometry(
            ref_gh::AnalyticRadialScalarOracleJet(accessor, analytic_alpha),
            ref_gh::AnalyticRadialScalarOracleJet(accessor, analytic_l),
            ref_gh::AnalyticRadialScalarOracleJet(accessor, analytic_b),
            x, y, z, 0.0, 0.0, 0.0, generic);
        if (diagnostic_kind == 2) {
          values(0) = generated.spin_derivative[a][b][c][d];
          values(1) = generic.spin_derivative[a][b][c][d];
          values(2) = GeneratedSpinDerivativeCondition(generic, a, b, c, d);
        } else if (diagnostic_kind == 1) {
          values(0) = generated.riemann_frame[a][b][c][d];
          values(1) = generic.riemann_frame[a][b][c][d];
          values(2) = GeneratedRiemannCondition(generic, a, b, c, d);
        } else {
          values(0) = diagnostic_kind == 3
              ? generated.d_christoffel[a][b][c][d]
              : ref_gh::ReferenceDChristoffel(accessor, a, b, c, d);
          values(1) = generic.d_christoffel[a][b][c][d];
          values(2) = 1.0;
        }
        values(3) = radius;
        for (int n = 4; n < 12; ++n) values(n) = 0.0;
        for (int p = 0; p < 4; ++p) {
          for (int aa = 0; aa < 4; ++aa) {
            for (int bb = 0; bb < 4; ++bb) {
              values(4) = fmax(values(4), Kokkos::abs(
                  generated.metric[aa][bb] - generic.metric[aa][bb]));
              values(5) = fmax(values(5), Kokkos::abs(
                  generated.d_metric[p][aa][bb]
                  - generic.d_metric[p][aa][bb]));
              values(7) = fmax(values(7), Kokkos::abs(
                  generated.frame[aa][bb] - generic.frame[aa][bb]));
              values(8) = fmax(values(8), Kokkos::abs(
                  generated.d_frame[p][aa][bb]
                  - generic.d_frame[p][aa][bb]));
              for (int qq = 0; qq < 4; ++qq) {
                values(6) = fmax(values(6), Kokkos::abs(
                    generated.dd_metric[p][qq][aa][bb]
                    - generic.dd_metric[p][qq][aa][bb]));
                values(9) = fmax(values(9), Kokkos::abs(
                    generated.dd_frame[p][qq][aa][bb]
                    - generic.dd_frame[p][qq][aa][bb]));
              }
              values(10) = fmax(values(10), Kokkos::abs(
                  generated.christoffel[aa][p][bb]
                  - generic.christoffel[aa][p][bb]));
              for (int qq = 0; qq < 4; ++qq) {
                values(11) = fmax(values(11), Kokkos::abs(
                    generated.d_christoffel[qq][aa][p][bb]
                    - generic.d_christoffel[qq][aa][p][bb]));
              }
            }
          }
        }
      });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), values);
  const char *kind_name = diagnostic_kind == 1 ? "Riemann"
      : (diagnostic_kind == 2 ? "spin derivative"
         : (diagnostic_kind == 3 ? "dChristoffel" : "accessor dChristoffel"));
  std::cout << "generated " << kind_name
            << " mismatch detail: sample=" << sample
            << " component=" << a << b << c << d
            << " radius=" << host(3)
            << " generated=" << host(0)
            << " generic=" << host(1)
            << " condition=" << host(2)
            << " primitive_abs(metric,dmetric,ddmetric,frame,dframe,ddframe,"
               "Gamma,dGamma)="
            << host(4) << "," << host(5) << "," << host(6) << ","
            << host(7) << "," << host(8) << "," << host(9) << ","
            << host(10) << "," << host(11) << std::endl;
}

void CheckGeneratedAnalyticRadialQGeometry(const DvceArray2D<Real> &table) {
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = kGeneratedAnalyticOraclePointCount;
  constexpr int nsamples = nq*nrate*nacceleration*npoints;
  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  MaxLoc::value_type maximum;
  Kokkos::parallel_reduce(
      "ref_gh generated analytic radial-q geometry",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, MaxLoc::value_type &local_maximum) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const Real q = q_values[work % nq];
        work /= nq;
        const Real q_dot = q_dot_values[work % nrate];
        work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration];
        work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        GeneratedAnalyticOraclePoint(work, x, y, z);
        const Real displacement[3] = {x, y, z};
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        ref_gh::ReferenceGeometry generated;
        ref_gh::PopulateGeneratedAnalyticRadialQGeometry(
            analytic_alpha, analytic_l, analytic_b, displacement, radius,
            generated);
        const ref_gh::AnalyticRadialQPoint accessor{
            analytic_alpha, analytic_l, analytic_b, {x, y, z}, radius};
        ref_gh::ReferenceGeometry generic;
        ref_gh::PopulateIsotropicReferenceGeometry(
            ref_gh::AnalyticRadialScalarOracleJet(accessor, analytic_alpha),
            ref_gh::AnalyticRadialScalarOracleJet(accessor, analytic_l),
            ref_gh::AnalyticRadialScalarOracleJet(accessor, analytic_b),
            x, y, z, 0.0, 0.0, 0.0, generic);
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            UpdateGeneratedAnalyticOracleMaximum(
                generated.metric[a][b], generic.metric[a][b], 20,
                local_maximum);
            UpdateGeneratedAnalyticOracleMaximum(
                generated.inverse_metric[a][b], generic.inverse_metric[a][b],
                20, local_maximum);
            UpdateGeneratedAnalyticOracleMaximum(
                generated.ricci_frame[a][b], generic.ricci_frame[a][b], 15,
                local_maximum, GeneratedRicciCondition(generic, a, b));
            UpdateGeneratedAnalyticOracleMaximum(
                generated.coframe[a][b], generic.coframe[a][b], 0,
                local_maximum);
            UpdateGeneratedAnalyticOracleMaximum(
                generated.frame[a][b], generic.frame[a][b], 1,
                local_maximum);
            if (work < kGeneratedAnalyticOriginalPointCount) {
              UpdateGeneratedAnalyticOracleMaximum(
                  ref_gh::ReferenceCoframe(accessor, a, b),
                  generic.coframe[a][b], 0, local_maximum);
              UpdateGeneratedAnalyticOracleMaximum(
                  ref_gh::ReferenceFrame(accessor, a, b), generic.frame[a][b],
                  1, local_maximum);
            }
            for (int c = 0; c < 4; ++c) {
              UpdateGeneratedAnalyticOracleMaximum(
                  generated.d_metric[c][a][b], generic.d_metric[c][a][b], 20,
                  local_maximum);
              UpdateGeneratedAnalyticOracleMaximum(
                  generated.d_frame[c][a][b], generic.d_frame[c][a][b], 2,
                  local_maximum);
              UpdateGeneratedAnalyticOracleMaximum(
                  generated.christoffel[a][b][c],
                  generic.christoffel[a][b][c], 16, local_maximum);
              UpdateGeneratedAnalyticOracleMaximum(
                  generated.spin[a][b][c], generic.spin[a][b][c], 18,
                  local_maximum,
                  GeneratedSpinCondition(generic, a, b, c));
              UpdateGeneratedAnalyticOracleMaximum(
                  generated.structure4[a][b][c], generic.structure4[a][b][c],
                  20, local_maximum);
              if (work < kGeneratedAnalyticOriginalPointCount) {
                UpdateGeneratedAnalyticOracleMaximum(
                    ref_gh::ReferenceDFrame(accessor, c, a, b),
                    generic.d_frame[c][a][b], 2, local_maximum);
                UpdateGeneratedAnalyticOracleMaximum(
                    ref_gh::ReferenceChristoffel(accessor, a, b, c),
                    generic.christoffel[a][b][c], 16, local_maximum);
              }
              if (sample < 4) {
                UpdateGeneratedAnalyticOracleMaximum(
                    ref_gh::ReferenceSpin(accessor, a, b, c),
                    generic.spin[a][b][c], 18, local_maximum,
                    GeneratedSpinCondition(generic, a, b, c));
              }
              for (int d = 0; d < 4; ++d) {
                UpdateGeneratedAnalyticOracleMaximum(
                    generated.dd_metric[c][d][a][b],
                    generic.dd_metric[c][d][a][b], 20, local_maximum);
                UpdateGeneratedAnalyticOracleMaximum(
                    generated.d_christoffel[d][a][b][c],
                    generic.d_christoffel[d][a][b][c], 17, local_maximum,
                    1.0, 3000000 + 256*sample + 64*d + 16*a + 4*b + c);
                UpdateGeneratedAnalyticOracleMaximum(
                    generated.dd_frame[c][d][a][b],
                    generic.dd_frame[c][d][a][b], 11, local_maximum);
                UpdateGeneratedAnalyticOracleMaximum(
                    generated.spin_derivative[d][a][b][c],
                    generic.spin_derivative[d][a][b][c], 19, local_maximum,
                    GeneratedSpinDerivativeCondition(
                        generic, d, a, b, c),
                    2000000 + 256*sample + 64*d + 16*a + 4*b + c);
                UpdateGeneratedAnalyticOracleMaximum(
                    generated.riemann_frame[a][b][c][d],
                    generic.riemann_frame[a][b][c][d], 14, local_maximum,
                    GeneratedRiemannCondition(generic, a, b, c, d),
                    1000000 + 256*sample + 64*a + 16*b + 4*c + d);
                if (work < kGeneratedAnalyticOriginalPointCount) {
                  UpdateGeneratedAnalyticOracleMaximum(
                      ref_gh::ReferenceDDFrame(accessor, c, d, a, b),
                      generic.dd_frame[c][d][a][b], 11, local_maximum);
                  UpdateGeneratedAnalyticOracleMaximum(
                      ref_gh::ReferenceDChristoffel(accessor, d, a, b, c),
                      generic.d_christoffel[d][a][b][c], 17, local_maximum,
                      1.0, 4000000 + 256*sample + 64*d + 16*a + 4*b + c);
                }
                if (sample == 0) {
                  UpdateGeneratedAnalyticOracleMaximum(
                      ref_gh::ReferenceSpinDerivative(
                          accessor, d, a, b, c),
                      generic.spin_derivative[d][a][b][c], 19,
                      local_maximum,
                      GeneratedSpinDerivativeCondition(
                          generic, d, a, b, c));
                  UpdateGeneratedAnalyticOracleMaximum(
                      ref_gh::ReferenceRiemann(accessor, a, b, c, d),
                      generic.riemann_frame[a][b][c][d], 14, local_maximum,
                      GeneratedRiemannCondition(generic, a, b, c, d));
                }
              }
            }
          }
        }
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            UpdateGeneratedAnalyticOracleMaximum(
                generated.spatial_frame[i][j], generic.spatial_frame[i][j], 7,
                local_maximum);
            UpdateGeneratedAnalyticOracleMaximum(
                generated.spatial_coframe[i][j],
                generic.spatial_coframe[i][j], 8, local_maximum);
            UpdateGeneratedAnalyticOracleMaximum(
                generated.dt_spatial_frame[i][j],
                generic.dt_spatial_frame[i][j], 9, local_maximum);
            if (work < kGeneratedAnalyticOriginalPointCount) {
              UpdateGeneratedAnalyticOracleMaximum(
                  ref_gh::ReferenceSpatialFrame(accessor, i, j),
                  generic.spatial_frame[i][j], 7, local_maximum);
              UpdateGeneratedAnalyticOracleMaximum(
                  ref_gh::ReferenceSpatialCoframe(accessor, i, j),
                  generic.spatial_coframe[i][j], 8, local_maximum);
              UpdateGeneratedAnalyticOracleMaximum(
                  ref_gh::ReferenceDtSpatialFrame(accessor, i, j),
                  generic.dt_spatial_frame[i][j], 9, local_maximum);
            }
            for (int k = 0; k < 3; ++k) {
              UpdateGeneratedAnalyticOracleMaximum(
                  generated.structure[i][j][k], generic.structure[i][j][k],
                  10, local_maximum);
              if (work < kGeneratedAnalyticOriginalPointCount) {
                UpdateGeneratedAnalyticOracleMaximum(
                    ref_gh::ReferenceStructure(accessor, i, j, k),
                    generic.structure[i][j][k], 10, local_maximum);
              }
            }
          }
        }
      }, MaxLoc(maximum));
  Kokkos::fence();
  constexpr Real tolerance = 256.0*std::numeric_limits<Real>::epsilon();
  if (!(maximum.val <= tolerance)) {
    ReportGeneratedGeometryMismatch(table, maximum.loc);
    std::cout << "### FATAL ERROR: generated analytic radial-q geometry "
              << "oracle failed: " << maximum.val
              << " category=" << maximum.loc
              << " tolerance=" << tolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH generated analytic radial-q geometry oracle "
            << "passed: samples=" << nsamples << " conditioned error="
            << maximum.val << " category=" << maximum.loc
            << std::endl;
}

KOKKOS_INLINE_FUNCTION
Real ScaledGaugeOracleError(const Real analytic, const Real generic,
                            const Real operation_condition = 1.0) {
  return Kokkos::abs(analytic - generic)
         /fmax(operation_condition,
               fmax(1.0, fmax(Kokkos::abs(analytic), Kokkos::abs(generic))));
}

void CheckGeneratedAnalyticRadialQGauge(const DvceArray2D<Real> &table) {
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = kAnalyticOraclePointCount;
  constexpr int nsamples = nq*nrate*nacceleration*npoints;
  Real maximum_hhat = 0.0;
  Real maximum_d_hhat = 0.0;
  Real maximum_reference_k = 0.0;
  Real maximum_theta = 0.0;
  Real maximum_dt_theta = 0.0;
  Real maximum_motion = 0.0;
  Real maximum_r_ge_02 = 0.0;
  using GaugeMaxLoc = Kokkos::MaxLoc<Real, int>;
  GaugeMaxLoc::value_type maximum_location;
  DvceArray1D<Real> gauge_diagnostic("ref_gh gauge diagnostic", 12);
  Kokkos::parallel_reduce(
      "ref_gh generated analytic radial-q gauge",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_hhat,
                    Real &local_d_hhat, Real &local_reference_k,
                    Real &local_theta,
                    Real &local_dt_theta, Real &local_motion,
                    Real &local_r_ge_02,
                    GaugeMaxLoc::value_type &local_location) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const Real q = q_values[work % nq]; work /= nq;
        const Real q_dot = q_dot_values[work % nrate]; work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration]; work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        const int radial_index = work/kAnalyticOracleDirectionCount;
        AnalyticOraclePoint(work, x, y, z);
        const Real displacement[3] = {x, y, z};
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialQGaugeBaseline analytic =
            ref_gh::PopulateGeneratedAnalyticRadialQGauge(
                analytic_alpha, analytic_l, analytic_b, displacement, radius);
        const ref_gh::AnalyticRadialQPoint analytic_point{
            analytic_alpha, analytic_l, analytic_b, {x, y, z}, radius};
        const ref_gh::ReferenceJet generic_alpha =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_point, analytic_alpha);
        const ref_gh::ReferenceJet generic_l =
            ref_gh::AnalyticRadialScalarOracleJet(analytic_point, analytic_l);
        const ref_gh::ReferenceJet generic_b =
            ref_gh::AnalyticRadialScalarOracleJet(analytic_point, analytic_b);
        ref_gh::ReferenceGeometry generic_geometry;
        ref_gh::PopulateIsotropicReferenceGeometry(
            generic_alpha, generic_l, generic_b, x, y, z,
            0.0, 0.0, 0.0, generic_geometry);
        const ref_gh::ReferenceGaugeBaseline generic =
            ref_gh::ComputeReferenceGaugeBaseline(generic_geometry);
        Real generic_dt_theta[4];  // NOLINT(runtime/arrays)
        Real hhat_condition[4];  // NOLINT(runtime/arrays)
        Real d_hhat_condition[4][4];  // NOLINT(runtime/arrays)
        Real theta_condition[4];  // NOLINT(runtime/arrays)
        Real dt_theta_condition[4];  // NOLINT(runtime/arrays)
        const bool generic_dt_valid = GenericReferenceDtThetaOracle(
            generic_alpha, generic_l, generic_b, displacement,
            generic_geometry, generic_dt_theta, hhat_condition,
            d_hhat_condition, theta_condition, dt_theta_condition);
        if (!analytic.valid || !generic.valid || !generic_dt_valid) {
          local_dt_theta = std::numeric_limits<Real>::infinity();
          return;
        }
        if (sample == 143) {
          gauge_diagnostic(0) = analytic.d_hhat[0][0];
          gauge_diagnostic(1) = generic.d_hhat[0][0];
          gauge_diagnostic(2) = analytic.hhat[0];
          gauge_diagnostic(3) = generic.hhat[0];
          gauge_diagnostic(4) = analytic.theta[0];
          gauge_diagnostic(5) = generic.theta[0];
          gauge_diagnostic(6) = analytic.dt_theta[0];
          gauge_diagnostic(7) = generic_dt_theta[0];
          gauge_diagnostic(8) = d_hhat_condition[0][0];
          gauge_diagnostic(9) = hhat_condition[0];
          gauge_diagnostic(10) = theta_condition[0];
          gauge_diagnostic(11) = dt_theta_condition[0];
        }
        for (int A = 0; A < 4; ++A) {
          const Real hhat_error = ScaledGaugeOracleError(
              analytic.hhat[A], generic.hhat[A], hhat_condition[A]);
          const Real theta_error = ScaledGaugeOracleError(
              analytic.theta[A], generic.theta[A], theta_condition[A]);
          const Real dt_theta_error = ScaledGaugeOracleError(
              analytic.dt_theta[A], generic_dt_theta[A],
              dt_theta_condition[A]);
          local_hhat = fmax(local_hhat, hhat_error);
          local_theta = fmax(local_theta, theta_error);
          local_dt_theta = fmax(local_dt_theta, dt_theta_error);
          if (hhat_error > local_location.val) {
            local_location.val = hhat_error;
            local_location.loc = 100*sample + A;
          }
          if (theta_error > local_location.val) {
            local_location.val = theta_error;
            local_location.loc = 100*sample + 40 + A;
          }
          if (dt_theta_error > local_location.val) {
            local_location.val = dt_theta_error;
            local_location.loc = 100*sample + 50 + A;
          }
          if (radial_index >= 4) {
            local_r_ge_02 = fmax(local_r_ge_02, ScaledGaugeOracleError(
                analytic.hhat[A], generic.hhat[A], hhat_condition[A]));
            local_r_ge_02 = fmax(local_r_ge_02, ScaledGaugeOracleError(
                analytic.theta[A], generic.theta[A], theta_condition[A]));
            local_r_ge_02 = fmax(local_r_ge_02, ScaledGaugeOracleError(
                analytic.dt_theta[A], generic_dt_theta[A],
                dt_theta_condition[A]));
          }
          for (int p = 0; p < 4; ++p) {
            const Real d_hhat_error = ScaledGaugeOracleError(
                analytic.d_hhat[p][A], generic.d_hhat[p][A],
                d_hhat_condition[p][A]);
            local_d_hhat = fmax(local_d_hhat, d_hhat_error);
            if (d_hhat_error > local_location.val) {
              local_location.val = d_hhat_error;
              local_location.loc = 100*sample + 10 + 4*p + A;
            }
            if (radial_index >= 4) {
              local_r_ge_02 = fmax(local_r_ge_02, ScaledGaugeOracleError(
                  analytic.d_hhat[p][A], generic.d_hhat[p][A],
                  d_hhat_condition[p][A]));
            }
            for (int B = 0; B < 4; ++B) {
              local_motion = fmax(local_motion, ScaledGaugeOracleError(
                  ref_gh::GeneratedAnalyticRadialQFrameMotion(
                      analytic_alpha, analytic_l, analytic_b, displacement,
                      radius, A, p, B),
                  ref_gh::ReferenceFrameMotion(generic_geometry, A, p, B)));
              local_motion = fmax(local_motion, ScaledGaugeOracleError(
                  ref_gh::GeneratedAnalyticRadialQDtFrameMotion(
                      analytic_alpha, analytic_l, analytic_b, displacement,
                      radius, A, p, B),
                  ref_gh::ReferenceDtFrameMotion(
                      generic_geometry, A, p, B)));
            }
          }
          for (int p = 0; p < 3; ++p) {
            Real reference_k_condition =
                Kokkos::abs(generic.d_hhat[p + 1][A])
                + d_hhat_condition[p + 1][A];
            for (int B = 0; B < 4; ++B) {
              reference_k_condition += Kokkos::abs(
                  ref_gh::ReferenceFrameMotion(
                      generic_geometry, A, p + 1, B)*generic.hhat[B]);
            }
            const Real reference_k_error = ScaledGaugeOracleError(
                analytic.reference_k[p][A], generic.reference_k[p][A],
                reference_k_condition);
            local_reference_k = fmax(
                local_reference_k, reference_k_error);
            if (reference_k_error > local_location.val) {
              local_location.val = reference_k_error;
              local_location.loc = 100*sample + 60 + 4*p + A;
            }
            if (radial_index >= 4) {
              local_r_ge_02 = fmax(local_r_ge_02, reference_k_error);
            }
          }
        }
      }, Kokkos::Max<Real>(maximum_hhat),
      Kokkos::Max<Real>(maximum_d_hhat),
      Kokkos::Max<Real>(maximum_reference_k),
      Kokkos::Max<Real>(maximum_theta),
      Kokkos::Max<Real>(maximum_dt_theta), Kokkos::Max<Real>(maximum_motion),
      Kokkos::Max<Real>(maximum_r_ge_02), GaugeMaxLoc(maximum_location));
  Kokkos::fence();
  constexpr Real tolerance =
      256.0*std::numeric_limits<Real>::epsilon();
  if (!(maximum_hhat <= tolerance) || !(maximum_d_hhat <= tolerance)
      || !(maximum_reference_k <= tolerance)
      || !(maximum_theta <= tolerance) || !(maximum_dt_theta <= tolerance)
      || !(maximum_motion <= tolerance)) {
    const auto diagnostic_host = Kokkos::create_mirror_view_and_copy(
        HostMemSpace(), gauge_diagnostic);
    std::cout << "generated gauge diagnostic sample=143 dH00="
              << diagnostic_host(0) << "/" << diagnostic_host(1)
              << " H0=" << diagnostic_host(2) << "/" << diagnostic_host(3)
              << " theta0=" << diagnostic_host(4) << "/" << diagnostic_host(5)
              << " dtTheta0=" << diagnostic_host(6) << "/"
              << diagnostic_host(7)
              << " conditions(dH,H,theta,dtTheta)="
              << diagnostic_host(8) << "," << diagnostic_host(9) << ","
              << diagnostic_host(10) << "," << diagnostic_host(11)
              << std::endl;
    std::cout << "### FATAL ERROR: generated analytic radial-q moving gauge "
              "oracle failed: Hhat=" << maximum_hhat
              << " dHhat=" << maximum_d_hhat
              << " Kref=" << maximum_reference_k
              << " theta=" << maximum_theta
              << " dtTheta=" << maximum_dt_theta
              << " motion=" << maximum_motion
              << " r>=0.2=" << maximum_r_ge_02
              << " worst=" << maximum_location.val
              << " location=" << maximum_location.loc
              << " tolerance=" << tolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH generated analytic radial-q moving gauge oracle "
               "passed: samples=" << nsamples
            << " Hhat=" << maximum_hhat
            << " dHhat=" << maximum_d_hhat
            << " Kref=" << maximum_reference_k
            << " theta=" << maximum_theta
            << " dtTheta=" << maximum_dt_theta
            << " motion=" << maximum_motion << std::endl;
}

KOKKOS_INLINE_FUNCTION
Real BoundaryMetricProjectionCondition(
    const ref_gh::ReferenceGeometry &physical,
    const ref_gh::ReferenceGeometry &current, const int A, const int B,
    const int field, const int spatial = 0) {
  Real psi_condition = 0.0;
  Real d_psi[4] = {};            // NOLINT(runtime/arrays)
  Real d_psi_condition[4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      psi_condition += Kokkos::abs(
          current.frame[A][a]*current.frame[B][b]*physical.metric[a][b]);
      for (int p = 0; p < 4; ++p) {
        const Real frame_term =
            (current.d_frame[p][A][a]*current.frame[B][b]
             + current.frame[A][a]*current.d_frame[p][B][b])
            *physical.metric[a][b];
        const Real metric_term = current.frame[A][a]*current.frame[B][b]
                                 *physical.d_metric[p][a][b];
        d_psi[p] += frame_term + metric_term;
        d_psi_condition[p] +=
            Kokkos::abs(frame_term) + Kokkos::abs(metric_term);
      }
    }
  }
  if (field == 0) return psi_condition;
  const Real lapse = 1.0/Kokkos::sqrt(-physical.inverse_metric[0][0]);
  Real shift[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    shift[i] = lapse*lapse*physical.inverse_metric[0][i + 1];
  }
  if (field == 1) {
    Real condition = (Kokkos::abs(d_psi[0]) + d_psi_condition[0])/lapse;
    for (int i = 0; i < 3; ++i) {
      condition += Kokkos::abs(shift[i])
                   *(Kokkos::abs(d_psi[i + 1])
                     + d_psi_condition[i + 1])/lapse;
    }
    return condition;
  }
  Real condition = 0.0;
  for (int i = 0; i < 3; ++i) {
    condition += Kokkos::abs(current.spatial_frame[spatial][i])
                 *(Kokkos::abs(d_psi[i + 1])
                   + d_psi_condition[i + 1]);
  }
  return condition;
}

void CheckCompactAnalyticRadialQBoundaryProjection(
    const DvceArray2D<Real> &table) {
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = kAnalyticOraclePointCount;
  constexpr int nsamples = nq*nrate*nacceleration*npoints;
  using BoundaryMaxLoc = Kokkos::MaxLoc<Real, int>;
  BoundaryMaxLoc::value_type maximum_metric;
  BoundaryMaxLoc::value_type maximum_gauge;
  BoundaryMaxLoc::value_type maximum_subtracted_gauge;
  Kokkos::parallel_reduce(
      "ref_gh compact analytic radial-q boundary projection",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample,
                    BoundaryMaxLoc::value_type &local_metric,
                    BoundaryMaxLoc::value_type &local_gauge,
                    BoundaryMaxLoc::value_type &local_subtracted_gauge) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const Real q = q_values[work % nq]; work /= nq;
        const Real q_dot = q_dot_values[work % nrate]; work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration];
        work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        AnalyticOraclePoint(work, x, y, z);
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);

        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real current_stage[ref_gh::kAnalyticRadialQStageSize];
        Real physical_stage[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, current_stage);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, 1.0, 0.0, 0.0, physical_stage);
        const ref_gh::AnalyticRadialScalar alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar shift_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar current_l{
            current_stage[ref_gh::kAnalyticL],
            current_stage[ref_gh::kAnalyticLT],
            current_stage[ref_gh::kAnalyticLR],
            current_stage[ref_gh::kAnalyticLTT],
            current_stage[ref_gh::kAnalyticLTR],
            current_stage[ref_gh::kAnalyticLRR],
            current_stage[ref_gh::kAnalyticLTTR],
            current_stage[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar physical_l{
            physical_stage[ref_gh::kAnalyticL],
            physical_stage[ref_gh::kAnalyticLT],
            physical_stage[ref_gh::kAnalyticLR],
            physical_stage[ref_gh::kAnalyticLTT],
            physical_stage[ref_gh::kAnalyticLTR],
            physical_stage[ref_gh::kAnalyticLRR],
            physical_stage[ref_gh::kAnalyticLTTR],
            physical_stage[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialQPoint analytic_current{
            alpha, current_l, shift_b, {x, y, z}, radius};
        const ref_gh::AnalyticRadialQPoint analytic_physical{
            alpha, physical_l, shift_b, {x, y, z}, radius};

        // The independent coefficient/provider agreement is the preceding
        // 2160-point gate.  Feed those already-qualified compact coefficients
        // into the generic geometry builder here so this strict 256-epsilon
        // gate isolates only boundary projection and subtraction algebra.
        const ref_gh::ReferenceJet physical_alpha =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_physical, analytic_physical.alpha);
        const ref_gh::ReferenceJet physical_l_jet =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_physical, analytic_physical.l);
        const ref_gh::ReferenceJet physical_b =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_physical, analytic_physical.b);
        const ref_gh::ReferenceJet current_alpha =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_current, analytic_current.alpha);
        const ref_gh::ReferenceJet current_l_jet =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_current, analytic_current.l);
        const ref_gh::ReferenceJet current_b =
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_current, analytic_current.b);
        ref_gh::ReferenceGeometry generic_physical;
        ref_gh::PopulateIsotropicReferenceGeometry(
            physical_alpha, physical_l_jet, physical_b,
            x, y, z, 0.0, 0.0, 0.0, generic_physical);
        ref_gh::ReferenceGeometry generic_current;
        ref_gh::PopulateIsotropicReferenceGeometry(
            current_alpha, current_l_jet, current_b,
            x, y, z, 0.0, 0.0, 0.0, generic_current);

        const ref_gh::ProjectedFirstOrderMetric compact_metric =
            ref_gh::ProjectAnalyticPhysicalMetricToReference(
                analytic_physical, analytic_current);
        const ref_gh::ProjectedFirstOrderMetric generic_metric =
            ref_gh::ProjectPhysicalMetricToReference(
                generic_physical.metric, generic_physical.d_metric,
                generic_current);
        const ref_gh::ProjectedStationaryGaugeState compact_gauge =
            ref_gh::ProjectAnalyticStationaryPhysicalGaugeToReference(
                analytic_physical, analytic_current);
        const ref_gh::ProjectedStationaryGaugeState generic_gauge =
            ref_gh::ProjectStationaryPhysicalGaugeToReference(
                generic_physical, generic_current);
        const ref_gh::AnalyticRadialQGaugeBaseline compact_baseline =
            ref_gh::PopulateGeneratedAnalyticRadialQGauge(
                analytic_current.alpha, analytic_current.l,
                analytic_current.b, analytic_current.displacement,
                analytic_current.radius);
        const ref_gh::ReferenceGaugeBaseline generic_baseline =
            ref_gh::ComputeReferenceGaugeBaseline(generic_current);
        const ref_gh::ReferenceGaugeBaseline generic_physical_baseline =
            ref_gh::ComputeReferenceGaugeBaseline(generic_physical);
        if (!compact_metric.valid || !generic_metric.valid
            || !compact_gauge.valid || !generic_gauge.valid
            || !compact_baseline.valid || !generic_baseline.valid) {
          local_metric.val = std::numeric_limits<Real>::infinity();
          local_metric.loc = sample*100;
          return;
        }

        Real unused_dt_theta[4];  // NOLINT(runtime/arrays)
        Real physical_hhat_condition[4];  // NOLINT(runtime/arrays)
        Real physical_d_hhat_condition[4][4];  // NOLINT(runtime/arrays)
        Real physical_theta_condition[4];  // NOLINT(runtime/arrays)
        Real physical_dt_theta_condition[4];  // NOLINT(runtime/arrays)
        Real current_hhat_condition[4];  // NOLINT(runtime/arrays)
        Real current_d_hhat_condition[4][4];  // NOLINT(runtime/arrays)
        Real current_theta_condition[4];  // NOLINT(runtime/arrays)
        Real current_dt_theta_condition[4];  // NOLINT(runtime/arrays)
        if (!GenericReferenceDtThetaOracle(
                physical_alpha, physical_l_jet, physical_b,
                analytic_physical.displacement, generic_physical,
                unused_dt_theta, physical_hhat_condition,
                physical_d_hhat_condition, physical_theta_condition,
                physical_dt_theta_condition)
            || !GenericReferenceDtThetaOracle(
                current_alpha, current_l_jet, current_b,
                analytic_current.displacement, generic_current,
                unused_dt_theta, current_hhat_condition,
                current_d_hhat_condition, current_theta_condition,
                current_dt_theta_condition)) {
          local_gauge.val = std::numeric_limits<Real>::infinity();
          local_gauge.loc = sample*100;
          return;
        }
        for (int A = 0; A < 4; ++A) {
          Real projected_hhat_condition = 0.0;
          Real projected_theta_condition = 0.0;
          for (int a = 0; a < 4; ++a) {
            Real coordinate_hhat_condition = 0.0;
            Real coordinate_theta_condition = 0.0;
            for (int P = 0; P < 4; ++P) {
              coordinate_hhat_condition +=
                  Kokkos::abs(generic_physical.coframe[P][a])
                  *(Kokkos::abs(generic_physical_baseline.hhat[P])
                    + physical_hhat_condition[P]);
              coordinate_theta_condition +=
                  Kokkos::abs(generic_physical.coframe[P][a])
                  *(Kokkos::abs(generic_physical_baseline.theta[P])
                    + physical_theta_condition[P]);
            }
            projected_hhat_condition +=
                Kokkos::abs(generic_current.frame[A][a])
                *coordinate_hhat_condition;
            projected_theta_condition +=
                Kokkos::abs(generic_current.frame[A][a])
                *coordinate_theta_condition;
          }
          Real error = ScaledGaugeOracleError(
              compact_gauge.hhat[A], generic_gauge.hhat[A],
              projected_hhat_condition);
          if (error > local_gauge.val) {
            local_gauge.val = error;
            local_gauge.loc = sample*100 + A;
          }
          error = ScaledGaugeOracleError(
              compact_gauge.theta[A], generic_gauge.theta[A],
              projected_theta_condition);
          if (error > local_gauge.val) {
            local_gauge.val = error;
            local_gauge.loc = sample*100 + 10 + A;
          }
          error = ScaledGaugeOracleError(
              compact_gauge.hhat[A] - compact_baseline.hhat[A],
              generic_gauge.hhat[A] - generic_baseline.hhat[A],
              projected_hhat_condition + current_hhat_condition[A]
              + Kokkos::abs(generic_gauge.hhat[A])
              + Kokkos::abs(generic_baseline.hhat[A]));
          if (error > local_subtracted_gauge.val) {
            local_subtracted_gauge.val = error;
            local_subtracted_gauge.loc = sample*100 + A;
          }
          error = ScaledGaugeOracleError(
              compact_gauge.theta[A] - compact_baseline.theta[A],
              generic_gauge.theta[A] - generic_baseline.theta[A],
              projected_theta_condition + current_theta_condition[A]
              + Kokkos::abs(generic_gauge.theta[A])
              + Kokkos::abs(generic_baseline.theta[A]));
          if (error > local_subtracted_gauge.val) {
            local_subtracted_gauge.val = error;
            local_subtracted_gauge.loc = sample*100 + 10 + A;
          }
          for (int B = A; B < 4; ++B) {
            error = ScaledGaugeOracleError(
                compact_metric.psi[A][B], generic_metric.psi[A][B],
                BoundaryMetricProjectionCondition(
                    generic_physical, generic_current, A, B, 0));
            if (error > local_metric.val) {
              local_metric.val = error;
              local_metric.loc = sample*100 + 10*A + B;
            }
            error = ScaledGaugeOracleError(
                compact_metric.pi[A][B], generic_metric.pi[A][B],
                BoundaryMetricProjectionCondition(
                    generic_physical, generic_current, A, B, 1));
            if (error > local_metric.val) {
              local_metric.val = error;
              local_metric.loc = sample*100 + 20 + 10*A + B;
            }
            for (int I = 0; I < 3; ++I) {
              error = ScaledGaugeOracleError(
                  compact_metric.phi[I][A][B],
                  generic_metric.phi[I][A][B],
                  BoundaryMetricProjectionCondition(
                      generic_physical, generic_current, A, B, 2, I));
              if (error > local_metric.val) {
                local_metric.val = error;
                local_metric.loc = sample*100 + 40 + 20*I + 4*A + B;
              }
            }
          }
        }
      }, BoundaryMaxLoc(maximum_metric), BoundaryMaxLoc(maximum_gauge),
      BoundaryMaxLoc(maximum_subtracted_gauge));
  Kokkos::fence();
  constexpr Real tolerance =
      256.0*std::numeric_limits<Real>::epsilon();
  if (!(maximum_metric.val <= tolerance)
      || !(maximum_gauge.val <= tolerance)
      || !(maximum_subtracted_gauge.val <= tolerance)) {
    std::cout << "### FATAL ERROR: compact analytic radial-q boundary "
                 "projection oracle failed: metric=" << maximum_metric.val
              << "@" << maximum_metric.loc
              << " gauge=" << maximum_gauge.val << "@" << maximum_gauge.loc
              << " subtracted-gauge=" << maximum_subtracted_gauge.val
              << "@" << maximum_subtracted_gauge.loc
              << " tolerance=" << tolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH compact analytic radial-q boundary projection "
               "oracle passed: samples=" << nsamples
            << " metric=" << maximum_metric.val
            << " gauge=" << maximum_gauge.val
            << " subtracted-gauge=" << maximum_subtracted_gauge.val
            << std::endl;
}

template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool BuildRhsOracleMetricData(
    const Reference &reference, const Real psi[4][4],
    const Real pi[4][4], const Real phi[3][4][4],
    Real metric[4][4], Real d_metric[4][4][4]) {
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = 0.0;
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          metric[a][b] += ref_gh::ReferenceCoframe(reference, A, a)
                          *ref_gh::ReferenceCoframe(reference, B, b)
                          *psi[A][B];
        }
      }
    }
  }
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!ref_gh::Invert4(metric, inverse, determinant)
      || !(inverse[0][0] < 0.0)) return false;
  const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real shift[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) {
    shift[p] = lapse*lapse*inverse[0][p + 1];
  }
  Real d_psi[4][4][4];  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int p = 0; p < 3; ++p) {
        d_psi[p + 1][A][B] = 0.0;
        for (int I = 0; I < 3; ++I) {
          d_psi[p + 1][A][B] +=
              ref_gh::ReferenceSpatialCoframe(reference, I, p)
              *phi[I][A][B];
        }
      }
      d_psi[0][A][B] = -lapse*pi[A][B];
      for (int p = 0; p < 3; ++p) {
        d_psi[0][A][B] += shift[p]*d_psi[p + 1][A][B];
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    Real frame_corrected[4][4];  // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        frame_corrected[A][B] = d_psi[p][A][B];
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            frame_corrected[A][B] -=
                (ref_gh::ReferenceDFrame(reference, p, A, a)
                   *ref_gh::ReferenceFrame(reference, B, b)
                 + ref_gh::ReferenceFrame(reference, A, a)
                   *ref_gh::ReferenceDFrame(reference, p, B, b))*metric[a][b];
          }
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_metric[p][a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            d_metric[p][a][b] +=
                ref_gh::ReferenceCoframe(reference, A, a)
                *ref_gh::ReferenceCoframe(reference, B, b)
                *frame_corrected[A][B];
          }
        }
      }
    }
  }
  return true;
}

template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool BuildGenericRhsOraclePointGeometry(
    const Reference &reference, const Real psi[4][4],
    const Real pi[4][4], const Real phi[3][4][4],
    Real metric[4][4], Real d_metric[4][4][4],
    ref_gh::CoordinateGhGeometry &geometry, Real &determinant) {
  return BuildRhsOracleMetricData(
             reference, psi, pi, phi, metric, d_metric)
         && ref_gh::ComputeCoordinateGhGeometry(
             metric, d_metric, reference, geometry, determinant);
}

template <bool CompactBackend, typename Reference>
KOKKOS_INLINE_FUNCTION
bool EvaluateRhsOraclePoint(
    const Reference &reference,
    const ref_gh::AnalyticRadialQPoint &analytic_reference,
    const int phi_ordering, const bool exact_matched_static, const int seed,
    Real rhs[ref_gh::nvar], Real rhs_condition[ref_gh::nvar]) {
  constexpr Real gamma0 = 0.73;
  constexpr Real gamma2 = 0.41;
  constexpr Real gauge_mu = 0.62;
  constexpr Real gauge_eta = 0.57;
  constexpr Real shift_nu = 0.83;
  constexpr Real shift_eta = 0.36;
  Real psi[4][4], pi[4][4], phi[3][4][4];  // NOLINT(runtime/arrays)
  Real d_pi[3][4][4], d_phi[3][3][4][4];  // NOLINT(runtime/arrays)
  Real d_psi_rhs[3][4][4];                 // NOLINT(runtime/arrays)
  for (int n = 0; n < ref_gh::nvar; ++n) {
    rhs[n] = 0.0;
    rhs_condition[n] = 1.0;
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      const int code = 1 + A + 3*B + seed;
      const Real perturb = 2.0e-4*static_cast<Real>((code % 9) - 4);
      const Real diagonal = A == B ? (A == 0 ? -1.0 : 1.0) : 0.0;
      psi[A][B] = psi[B][A] = diagonal + perturb;
      pi[A][B] = pi[B][A] =
          3.0e-4*static_cast<Real>(((2*code + 1) % 11) - 5);
      for (int I = 0; I < 3; ++I) {
        phi[I][A][B] = phi[I][B][A] =
            2.0e-4*static_cast<Real>(((code + 4*I) % 13) - 6);
        d_psi_rhs[I][A][B] = d_psi_rhs[I][B][A] =
            1.0e-4*static_cast<Real>(((3*code + I) % 15) - 7);
        d_pi[I][A][B] = d_pi[I][B][A] =
            1.0e-4*static_cast<Real>(((code + 5*I) % 17) - 8);
        for (int J = 0; J < 3; ++J) {
          d_phi[I][J][A][B] = d_phi[I][J][B][A] =
              7.0e-5*static_cast<Real>(
                  ((2*code + 3*I + J) % 19) - 9);
        }
      }
    }
  }
  Real metric[4][4], d_metric[4][4][4];  // NOLINT(runtime/arrays)
  ref_gh::CoordinateGhGeometry geometry;
  ref_gh::CompactAnalyticCoordinateGeometry compact_geometry;
  Real determinant = 0.0;
  if (!BuildRhsOracleMetricData(
          reference, psi, pi, phi, metric, d_metric)) return false;
  if constexpr (CompactBackend) {
    if (!ref_gh::ComputeCompactAnalyticCoordinateGeometry(
            metric, d_metric, analytic_reference,
            compact_geometry, determinant)) return false;
    geometry = compact_geometry.geometry;
  } else if (!ref_gh::ComputeCoordinateGhGeometry(
                 metric, d_metric, reference, geometry, determinant)) {
    return false;
  }

  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      Real value = -geometry.lapse*pi[A][B];
      for (int p = 0; p < 3; ++p) {
        Real coordinate_d_psi = 0.0;
        for (int I = 0; I < 3; ++I) {
          coordinate_d_psi += ref_gh::ReferenceSpatialCoframe(
              reference, I, p)*phi[I][A][B];
        }
        value += geometry.shift[p]*coordinate_d_psi;
      }
      rhs[ref_gh::PsiIndex(A, B)] = value;
    }
  }

  Real delta_hhat[4], delta_theta[4], upsilon[3];  // NOLINT(runtime/arrays)
  Real d_delta_hhat[3][4];                         // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    delta_hhat[A] = 1.0e-3*static_cast<Real>(((seed + 2*A) % 9) - 4);
    delta_theta[A] = 8.0e-4*static_cast<Real>(((seed + 3*A) % 11) - 5);
    for (int p = 0; p < 3; ++p) {
      d_delta_hhat[p][A] =
          6.0e-4*static_cast<Real>(((seed + A + 2*p) % 13) - 6);
    }
  }
  for (int p = 0; p < 3; ++p) {
    upsilon[p] = 7.0e-4*static_cast<Real>(((seed + 4*p) % 7) - 3);
  }
  Real hhat[4], theta[4], d_hhat_spatial[3][4];  // NOLINT(runtime/arrays)
  Real baseline_hhat[4], baseline_theta[4];      // NOLINT(runtime/arrays)
  Real baseline_d_hhat[4][4];                    // NOLINT(runtime/arrays)
  Real baseline_reference_k[3][4];               // NOLINT(runtime/arrays)
  Real baseline_dt_theta[4];                     // NOLINT(runtime/arrays)
  Real gauge_h_condition[4] = {};                // NOLINT(runtime/arrays)
  Real gauge_dh_condition[4][4] = {};             // NOLINT(runtime/arrays)
  Real gauge_theta_condition[4] = {};             // NOLINT(runtime/arrays)
  Real gauge_dt_theta_condition[4] = {};          // NOLINT(runtime/arrays)
  if constexpr (CompactBackend) {
    const ref_gh::AnalyticRadialQGaugeBaseline baseline =
        ref_gh::PopulateGeneratedAnalyticRadialQGauge(
            analytic_reference.alpha, analytic_reference.l,
            analytic_reference.b, analytic_reference.displacement,
            analytic_reference.radius);
    if (!baseline.valid) return false;
    for (int A = 0; A < 4; ++A) {
      hhat[A] = delta_hhat[A] + baseline.hhat[A];
      theta[A] = delta_theta[A] + baseline.theta[A];
      baseline_hhat[A] = baseline.hhat[A];
      baseline_theta[A] = baseline.theta[A];
      baseline_dt_theta[A] = baseline.dt_theta[A];
      for (int p = 0; p < 4; ++p) {
        baseline_d_hhat[p][A] = baseline.d_hhat[p][A];
      }
      for (int p = 0; p < 3; ++p) {
        baseline_reference_k[p][A] = baseline.reference_k[p][A];
      }
    }
  } else {
    const ref_gh::ReferenceGaugeBaseline baseline =
        ref_gh::ComputeReferenceGaugeBaseline(reference);
    if (!baseline.valid) return false;
    Real generic_dt_theta[4];  // NOLINT(runtime/arrays)
    const ref_gh::ReferenceJet generic_alpha =
        ref_gh::AnalyticRadialScalarOracleJet(
            analytic_reference, analytic_reference.alpha);
    const ref_gh::ReferenceJet generic_l =
        ref_gh::AnalyticRadialScalarOracleJet(
            analytic_reference, analytic_reference.l);
    const ref_gh::ReferenceJet generic_b =
        ref_gh::AnalyticRadialScalarOracleJet(
            analytic_reference, analytic_reference.b);
    if (!GenericReferenceDtThetaOracle(
            generic_alpha, generic_l, generic_b,
            analytic_reference.displacement, reference, generic_dt_theta,
            gauge_h_condition, gauge_dh_condition,
            gauge_theta_condition, gauge_dt_theta_condition)) return false;
    for (int A = 0; A < 4; ++A) {
      hhat[A] = delta_hhat[A] + baseline.hhat[A];
      theta[A] = delta_theta[A] + baseline.theta[A];
      baseline_hhat[A] = baseline.hhat[A];
      baseline_theta[A] = baseline.theta[A];
      baseline_dt_theta[A] = generic_dt_theta[A];
      for (int p = 0; p < 4; ++p) baseline_d_hhat[p][A] = baseline.d_hhat[p][A];
      for (int p = 0; p < 3; ++p) {
        baseline_reference_k[p][A] = baseline.reference_k[p][A];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int p = 0; p < 3; ++p) {
      d_hhat_spatial[p][A] = d_delta_hhat[p][A]
                             + baseline_d_hhat[p + 1][A];
    }
  }
  ref_gh::GaugeDriverRhs gauge_rhs;
  if constexpr (CompactBackend) {
    ref_gh::PhysicalGaugeTargetResidual target;
    if (!ref_gh::ComputePhysicalGaugeTargetResidual(
            psi, pi, phi, metric, d_metric, geometry, analytic_reference,
            upsilon, shift_nu, shift_eta, target)) return false;
    gauge_rhs = ref_gh::ComputeGaugeDriverResidualRhsWithReferenceK(
        analytic_reference, baseline_hhat, baseline_theta, baseline_d_hhat,
        baseline_dt_theta, baseline_reference_k,
        delta_hhat, delta_theta, upsilon, d_delta_hhat,
        target.physical_shift, target.reference_shift, target.delta_shift,
        target.delta_frame, target.reference_frame,
        target.delta_conformal_gamma, gauge_mu, gauge_eta, shift_eta,
        exact_matched_static);
  } else {
    ref_gh::PhysicalGaugeTarget target;
    if (!ref_gh::ComputePhysicalGaugeTarget(
            metric, d_metric, geometry, reference, upsilon, shift_nu,
            shift_eta, target)) return false;
    gauge_rhs = ref_gh::ComputeGaugeDriverRhs(
        reference, hhat, theta, upsilon, d_hhat_spatial,
        geometry.shift, target.frame, target.conformal_gamma,
        gauge_mu, gauge_eta, shift_eta);
  }
  for (int A = 0; A < 4; ++A) {
    rhs[ref_gh::kHhatOffset + A] = CompactBackend
        ? gauge_rhs.hhat[A]
        : gauge_rhs.hhat[A] - baseline_d_hhat[0][A];
    rhs[ref_gh::kThetaOffset + A] = CompactBackend
        ? gauge_rhs.theta[A]
        : gauge_rhs.theta[A] - baseline_dt_theta[A];
    if constexpr (!CompactBackend) {
      Real gauge_condition = gauge_h_condition[A]
                             + gauge_theta_condition[A];
      for (int p = 0; p < 4; ++p) {
        gauge_condition += gauge_dh_condition[p][A];
      }
      rhs_condition[ref_gh::kHhatOffset + A] =
          gauge_condition + gauge_dh_condition[0][A];
      rhs_condition[ref_gh::kThetaOffset + A] =
          gauge_condition + gauge_dt_theta_condition[A];
    }
  }
  for (int p = 0; p < 3; ++p) {
    rhs[ref_gh::kUpsilonOffset + p] = gauge_rhs.upsilon[p];
  }

  Real scalar_source[4][4];  // NOLINT(runtime/arrays)
  Real scalar_source_condition[4][4] = {};  // NOLINT(runtime/arrays)
  if constexpr (CompactBackend) {
    if (!ref_gh::CompactAnalyticRadialQScalarWaveSource(
            psi, pi, phi, analytic_reference, geometry,
            gamma0, scalar_source)) return false;
  } else {
    ref_gh::CovariantSourceSectors sectors;
    if (!ref_gh::CovariantGhScalarWaveSource(
            psi, pi, phi, reference, geometry,
            gamma0, scalar_source, sectors)) return false;
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        scalar_source_condition[A][B] =
            Kokkos::abs(sectors.curvature[A][B])
            + Kokkos::abs(sectors.qq[A][B])
            + Kokkos::abs(sectors.delta_product[A][B])
            + Kokkos::abs(sectors.damping[A][B])
            + Kokkos::abs(sectors.frame_correction[A][B]);
      }
    }
  }
  Real full_d_hhat[4][4];  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    full_d_hhat[0][A] = gauge_rhs.hhat[A];
    for (int p = 0; p < 3; ++p) {
      full_d_hhat[p + 1][A] = CompactBackend
          ? d_delta_hhat[p][A] : d_hhat_spatial[p][A];
    }
  }
  if constexpr (CompactBackend) {
    if (!ref_gh::AddCompactAnalyticOrdinaryGaugeResidualSource(
            psi, pi, phi, metric, d_metric, analytic_reference,
            compact_geometry, delta_hhat, full_d_hhat, gamma0,
            scalar_source)) return false;
  } else {
    ref_gh::AddOrdinaryGaugePartialWaveSource(
        metric, d_metric, reference, geometry, hhat, full_d_hhat,
        gamma0, scalar_source);
    Real gauge_condition_sum = 0.0;
    for (int A = 0; A < 4; ++A) {
      gauge_condition_sum += gauge_h_condition[A]
                             + gauge_theta_condition[A]
                             + gauge_dt_theta_condition[A];
      for (int p = 0; p < 4; ++p) {
        gauge_condition_sum += gauge_dh_condition[p][A];
      }
    }
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        scalar_source_condition[A][B] += gauge_condition_sum;
      }
    }
  }

  Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
  Real spatial_determinant = 0.0;
  if (!ref_gh::InvertSpatial3(metric, spatial_inverse, spatial_determinant)) {
    return false;
  }
  Real spatial_connection[3][3][3] = {};  // NOLINT(runtime/arrays)
  for (int q = 0; q < 3; ++q) {
    for (int p = 0; p < 3; ++p) {
      for (int r = 0; r < 3; ++r) {
        for (int ell = 0; ell < 3; ++ell) {
          spatial_connection[q][p][r] += 0.5*spatial_inverse[q][ell]
              *(d_metric[p + 1][ell + 1][r + 1]
                + d_metric[r + 1][ell + 1][p + 1]
                - d_metric[ell + 1][p + 1][r + 1]);
        }
      }
    }
  }
  Real trace_k = 0.0;
  for (int p = 0; p < 3; ++p) {
    for (int q = 0; q < 3; ++q) {
      trace_k -= geometry.lapse*spatial_inverse[p][q]
                 *geometry.christoffel[0][p + 1][q + 1];
    }
  }
  Real d_alpha[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) {
    Real d_inverse_00 = 0.0;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_inverse_00 -= geometry.inverse_metric[0][a]
                        *geometry.inverse_metric[0][b]
                        *d_metric[p + 1][a][b];
      }
    }
    d_alpha[p] = 0.5*geometry.lapse*geometry.lapse*geometry.lapse*d_inverse_00;
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      Real divergence = 0.0;
      Real lapse_gradient_term = 0.0;
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) {
          Real partial_tilde_phi = 0.0;
          Real tilde_phi_q = 0.0;
          for (int I = 0; I < 3; ++I) {
            partial_tilde_phi +=
                ref_gh::CoframeDerivative(reference, p + 1, I + 1, q + 1)
                  *phi[I][A][B]
                + ref_gh::ReferenceSpatialCoframe(reference, I, q)
                  *d_phi[p][I][A][B];
            tilde_phi_q += ref_gh::ReferenceSpatialCoframe(reference, I, q)
                           *phi[I][A][B];
          }
          Real covariant_derivative = partial_tilde_phi;
          for (int r = 0; r < 3; ++r) {
            Real tilde_phi_r = 0.0;
            for (int I = 0; I < 3; ++I) {
              tilde_phi_r += ref_gh::ReferenceSpatialCoframe(reference, I, r)
                             *phi[I][A][B];
            }
            covariant_derivative -= spatial_connection[r][p][q]*tilde_phi_r;
          }
          divergence += spatial_inverse[p][q]*covariant_derivative;
          lapse_gradient_term += spatial_inverse[p][q]*d_alpha[p]*tilde_phi_q;
        }
      }
      Real pi_rhs = geometry.lapse*(trace_k*pi[A][B] - divergence
                                    + scalar_source[A][B])
                    - lapse_gradient_term;
      for (int p = 0; p < 3; ++p) pi_rhs += geometry.shift[p]*d_pi[p][A][B];
      rhs[ref_gh::PiIndex(A, B)] = pi_rhs;
      if constexpr (!CompactBackend) {
        rhs_condition[ref_gh::PiIndex(A, B)] =
            Kokkos::abs(geometry.lapse)
              *(Kokkos::abs(scalar_source[A][B])
                + scalar_source_condition[A][B])
            + Kokkos::abs(pi_rhs);
      }
    }
  }

  Real beta_frame[3] = {};            // NOLINT(runtime/arrays)
  Real structure[3][3][3];            // NOLINT(runtime/arrays)
  for (int J = 0; J < 3; ++J) {
    for (int p = 0; p < 3; ++p) {
      beta_frame[J] += ref_gh::ReferenceSpatialCoframe(reference, J, p)
                       *geometry.shift[p];
    }
    for (int I = 0; I < 3; ++I) {
      for (int K = 0; K < 3; ++K) {
        structure[I][J][K] = ref_gh::ReferenceStructure(reference, I, J, K);
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      Real frame_derivative[3][3];  // NOLINT(runtime/arrays)
      for (int I = 0; I < 3; ++I) {
        for (int J = 0; J < 3; ++J) {
          frame_derivative[I][J] = 0.0;
          for (int p = 0; p < 3; ++p) {
            frame_derivative[I][J] +=
                ref_gh::ReferenceSpatialFrame(reference, I, p)
                *d_phi[p][J][A][B];
          }
        }
      }
      Real phi_values[3] = {phi[0][A][B], phi[1][A][B], phi[2][A][B]};
      for (int I = 0; I < 3; ++I) {
        Real phi_rhs = 0.0;
        for (int p = 0; p < 3; ++p) {
          phi_rhs += ref_gh::ReferenceSpatialFrame(reference, I, p)
                     *d_psi_rhs[p][A][B];
          Real coordinate_d_psi = 0.0;
          for (int J = 0; J < 3; ++J) {
            coordinate_d_psi += ref_gh::ReferenceSpatialCoframe(
                reference, J, p)*phi[J][A][B];
          }
          phi_rhs += ref_gh::ReferenceDtSpatialFrame(reference, I, p)
                     *coordinate_d_psi;
        }
        if (phi_ordering != 0) {
          phi_rhs -= ref_gh::StandardPhiOrderingCorrection(
              I, beta_frame, frame_derivative, structure, phi_values);
        }
        rhs[ref_gh::PhiIndex(I, A, B)] = phi_rhs;
      }
      Real coordinate_reduction[3];  // NOLINT(runtime/arrays)
      Real spatial_frame[3][3];      // NOLINT(runtime/arrays)
      for (int p = 0; p < 3; ++p) {
        coordinate_reduction[p] =
            9.0e-5*static_cast<Real>(((seed + A + B + p) % 11) - 5);
        for (int I = 0; I < 3; ++I) {
          coordinate_reduction[p] -= ref_gh::ReferenceSpatialCoframe(
              reference, I, p)*phi[I][A][B];
          spatial_frame[I][p] = ref_gh::ReferenceSpatialFrame(reference, I, p);
        }
      }
      const ref_gh::Gamma2DampingRhs damping =
          ref_gh::ComputeGamma2DampingRhs(
              geometry.lapse, geometry.shift, coordinate_reduction,
              spatial_frame, gamma2);
      rhs[ref_gh::PiIndex(A, B)] += damping.pi;
      for (int I = 0; I < 3; ++I) {
        rhs[ref_gh::PhiIndex(I, A, B)] += damping.phi[I];
      }
    }
  }
  for (int n = 0; n < ref_gh::nvar; ++n) {
    if (!Kokkos::isfinite(rhs[n])) return false;
  }
  return true;
}

void CheckAll61AnalyticRadialQRhs(const DvceArray2D<Real> &table) {
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = kAnalyticOraclePointCount;
  constexpr int norderings = 2;
  constexpr int nsamples = nq*nrate*nacceleration*npoints*norderings;
  DvceArray2D<Real> generic_rhs(
      "ref_gh all-61 generic RHS", nsamples, ref_gh::nvar);
  DvceArray2D<Real> generic_condition(
      "ref_gh all-61 generic condition", nsamples, ref_gh::nvar);
  DvceArray2D<Real> compact_rhs(
      "ref_gh all-61 compact RHS", nsamples, ref_gh::nvar);

  // Keep the independent generic and compact evaluators in separate device
  // kernels.  Besides making their independence structurally explicit, this
  // avoids asking PVC's device compiler to inline both complete 61-component
  // evaluators into one monolithic kernel.  The third kernel below performs
  // exactly the same conditioned comparison and reduction as the original
  // combined oracle; no equations, samples, or tolerances change.
  Kokkos::parallel_for(
      "ref_gh all-61 generic radial-q RHS",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const int phi_ordering = work % norderings; work /= norderings;
        const Real q = q_values[work % nq]; work /= nq;
        const Real q_dot = q_dot_values[work % nrate]; work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration];
        work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        AnalyticOraclePoint(work, x, y, z);
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialQPoint analytic_reference{
            analytic_alpha, analytic_l, analytic_b, {x, y, z}, radius};
        ref_gh::ReferenceGeometry generic_reference;
        ref_gh::PopulateIsotropicReferenceGeometry(
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_reference, analytic_alpha),
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_reference, analytic_l),
            ref_gh::AnalyticRadialScalarOracleJet(
                analytic_reference, analytic_b),
            x, y, z, 0.0, 0.0, 0.0, generic_reference);
        Real rhs[ref_gh::nvar];        // NOLINT(runtime/arrays)
        Real condition[ref_gh::nvar];  // NOLINT(runtime/arrays)
        const bool exact_matched_static = q == 1.0 && q_dot == 0.0
                                          && q_ddot == 0.0;
        const bool valid = EvaluateRhsOraclePoint<false>(
            generic_reference, analytic_reference,
            phi_ordering, exact_matched_static, sample, rhs, condition);
        for (int n = 0; n < ref_gh::nvar; ++n) {
          generic_rhs(sample, n) = valid
              ? rhs[n] : std::numeric_limits<Real>::infinity();
          generic_condition(sample, n) = valid
              ? condition[n] : std::numeric_limits<Real>::infinity();
        }
      });
  Kokkos::fence();

  Kokkos::parallel_for(
      "ref_gh all-61 compact radial-q RHS",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const int phi_ordering = work % norderings; work /= norderings;
        const Real q = q_values[work % nq]; work /= nq;
        const Real q_dot = q_dot_values[work % nrate]; work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration];
        work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        AnalyticOraclePoint(work, x, y, z);
        const Real radius = Kokkos::sqrt(x*x + y*y + z*z);
        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar analytic_alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar analytic_l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar analytic_b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialQPoint analytic_reference{
            analytic_alpha, analytic_l, analytic_b, {x, y, z}, radius};
        Real rhs[ref_gh::nvar];        // NOLINT(runtime/arrays)
        Real condition[ref_gh::nvar];  // NOLINT(runtime/arrays)
        const bool exact_matched_static = q == 1.0 && q_dot == 0.0
                                          && q_ddot == 0.0;
        const bool valid = EvaluateRhsOraclePoint<true>(
            analytic_reference, analytic_reference,
            phi_ordering, exact_matched_static, sample, rhs, condition);
        for (int n = 0; n < ref_gh::nvar; ++n) {
          compact_rhs(sample, n) = valid
              ? rhs[n] : std::numeric_limits<Real>::infinity();
        }
      });
  Kokkos::fence();

  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  MaxLoc::value_type maximum;
  Kokkos::parallel_reduce(
      "ref_gh all-61 staged radial-q RHS comparison",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples*ref_gh::nvar),
      KOKKOS_LAMBDA(const int index, MaxLoc::value_type &local_maximum) {
        const int sample = index/ref_gh::nvar;
        const int n = index % ref_gh::nvar;
        const Real generic = generic_rhs(sample, n);
        const Real compact = compact_rhs(sample, n);
        const Real condition = generic_condition(sample, n);
        if (!Kokkos::isfinite(generic) || !Kokkos::isfinite(compact)
            || !Kokkos::isfinite(condition)) {
          local_maximum.val = std::numeric_limits<Real>::infinity();
          local_maximum.loc = index;
          return;
        }
        const Real scale = fmax(
            condition, fmax(1.0, fmax(Kokkos::abs(generic),
                                     Kokkos::abs(compact))));
        const Real error = Kokkos::abs(generic - compact)/scale;
        if (error > local_maximum.val) {
          local_maximum.val = error;
          local_maximum.loc = index;
        }
      }, MaxLoc(maximum));
  Kokkos::fence();
  constexpr Real tolerance =
      256.0*std::numeric_limits<Real>::epsilon();
  if (!(maximum.val <= tolerance)) {
    std::cout << "### FATAL ERROR: all-61 legacy/residual radial-q RHS oracle "
                 "failed: "
              << "error=" << maximum.val
              << " sample=" << maximum.loc/ref_gh::nvar
              << " component=" << maximum.loc % ref_gh::nvar
              << " tolerance=" << tolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH all-61 legacy-generic versus fully-subtracted "
               "compact RHS oracle passed: "
            << "samples=" << nsamples << " error=" << maximum.val
            << " compatible+standard Phi" << std::endl;
}

KOKKOS_INLINE_FUNCTION
void UpdateResidualTargetOracleError(
    const Real candidate, const int category, const bool well_conditioned,
    Real &maximum, Real &physical_maximum, Real &delta_maximum,
    int &maximum_category) {
  const bool is_delta = (category >= 20 && category < 30)
                        || (category >= 40 && category < 50)
                        || (category >= 60 && category < 100);
  if (is_delta) {
    delta_maximum = fmax(delta_maximum, candidate);
  } else {
    physical_maximum = fmax(physical_maximum, candidate);
  }
  if (well_conditioned && candidate > maximum) {
    maximum = candidate;
    maximum_category = category;
  }
}

template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real EvaluateResidualPhysicalGaugeTargetOracle(
    const Reference &reference, const int seed, bool &matched_exact,
    const bool well_conditioned, const bool static_q1,
    Real &matched_maximum,
    Real &physical_maximum, Real &delta_maximum, Real &source_maximum,
    Real &compact_source_maximum, Real &raw_driver_maximum,
    int &maximum_category) {
  constexpr Real nu = 0.83;
  constexpr Real eta_beta = 0.36;
  Real psi[4][4] = {};       // NOLINT(runtime/arrays)
  Real pi[4][4] = {};        // NOLINT(runtime/arrays)
  Real phi[3][4][4] = {};    // NOLINT(runtime/arrays)
  Real upsilon[3];           // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      const int code = 1 + A + 3*B + seed;
      const Real diagonal = A == B ? (A == 0 ? -1.0 : 1.0) : 0.0;
      const Real perturb = 2.0e-4*static_cast<Real>((code % 9) - 4);
      psi[A][B] = psi[B][A] = diagonal + perturb;
      pi[A][B] = pi[B][A] =
          3.0e-4*static_cast<Real>(((2*code + 1) % 11) - 5);
      for (int I = 0; I < 3; ++I) {
        phi[I][A][B] = phi[I][B][A] =
            2.0e-4*static_cast<Real>(((code + 4*I) % 13) - 6);
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    upsilon[i] = 7.0e-4*static_cast<Real>(((seed + 4*i) % 7) - 3);
  }
  Real metric[4][4], d_metric[4][4][4];  // NOLINT(runtime/arrays)
  ref_gh::CoordinateGhGeometry geometry;
  Real determinant = 0.0;
  if (!BuildRhsOracleMetricData(
          reference, psi, pi, phi, metric, d_metric)
      || !ref_gh::ComputeCoordinateGhGeometry(
          metric, d_metric, reference, geometry, determinant)) {
    return std::numeric_limits<Real>::infinity();
  }
  ref_gh::PhysicalGaugeTarget legacy;
  ref_gh::PhysicalGaugeTargetResidual residual;
  if (!ref_gh::ComputePhysicalGaugeTarget(
          metric, d_metric, geometry, reference, upsilon, nu, eta_beta,
          legacy)
      || !ref_gh::ComputePhysicalGaugeTargetResidual(
          psi, pi, phi, metric, d_metric, geometry, reference, upsilon,
          nu, eta_beta, residual)) {
    return std::numeric_limits<Real>::infinity();
  }
  Real maximum = 0.0;
  physical_maximum = 0.0;
  delta_maximum = 0.0;
  source_maximum = 0.0;
  compact_source_maximum = 0.0;
  raw_driver_maximum = 0.0;
  maximum_category = -1;
  for (int A = 0; A < 4; ++A) {
    const Real frame_scale = fmax(
        1.0, fmax(Kokkos::abs(legacy.frame[A]),
                  Kokkos::abs(residual.reference_frame[A])));
    UpdateResidualTargetOracleError(
        Kokkos::abs(residual.physical_frame[A] - legacy.frame[A])/frame_scale,
        10 + A, well_conditioned, maximum, physical_maximum, delta_maximum,
        maximum_category);
    UpdateResidualTargetOracleError(
        Kokkos::abs(residual.delta_frame[A]
                    - (legacy.frame[A] - residual.reference_frame[A]))
            /frame_scale, 20 + A, well_conditioned, maximum,
        physical_maximum, delta_maximum, maximum_category);
  }
  for (int i = 0; i < 3; ++i) {
    const Real gamma_scale = fmax(
        1.0, fmax(Kokkos::abs(legacy.conformal_gamma[i]),
                  Kokkos::abs(residual.reference_conformal_gamma[i])));
    UpdateResidualTargetOracleError(
        Kokkos::abs(residual.physical_conformal_gamma[i]
                    - legacy.conformal_gamma[i])/gamma_scale, 30 + i,
        well_conditioned, maximum, physical_maximum, delta_maximum,
        maximum_category);
    UpdateResidualTargetOracleError(
        Kokkos::abs(residual.delta_conformal_gamma[i]
                    - (legacy.conformal_gamma[i]
                       - residual.reference_conformal_gamma[i]))
            /gamma_scale, 40 + i, well_conditioned, maximum,
        physical_maximum, delta_maximum, maximum_category);
    UpdateResidualTargetOracleError(
        Kokkos::abs(residual.physical_shift[i] - geometry.shift[i]), 50 + i,
        well_conditioned, maximum, physical_maximum, delta_maximum,
        maximum_category);
  }

  Real matched_psi[4][4] = {};       // NOLINT(runtime/arrays)
  Real matched_pi[4][4] = {};        // NOLINT(runtime/arrays)
  Real matched_phi[3][4][4] = {};    // NOLINT(runtime/arrays)
  Real zero_upsilon[3] = {};         // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) matched_psi[A][A] = A == 0 ? -1.0 : 1.0;
  Real matched_metric[4][4], matched_d_metric[4][4][4];  // NOLINT
  ref_gh::CoordinateGhGeometry matched_geometry;
  if (!BuildRhsOracleMetricData(
          reference, matched_psi, matched_pi, matched_phi,
          matched_metric, matched_d_metric)
      || !ref_gh::ComputeCoordinateGhGeometry(
          matched_metric, matched_d_metric, reference, matched_geometry,
          determinant)
      || !ref_gh::ComputePhysicalGaugeTargetResidual(
          matched_psi, matched_pi, matched_phi, matched_metric,
          matched_d_metric, matched_geometry, reference, zero_upsilon,
          nu, eta_beta, residual)) {
    return std::numeric_limits<Real>::infinity();
  }
  matched_exact = true;
  matched_maximum = 0.0;
  for (int A = 0; A < 4; ++A) {
    matched_maximum = fmax(matched_maximum,
                           Kokkos::abs(residual.delta_frame[A]));
    matched_maximum = fmax(matched_maximum,
                           Kokkos::abs(residual.delta_coordinate[A]));
    matched_exact = matched_exact && residual.delta_frame[A] == 0.0
                    && residual.delta_coordinate[A] == 0.0;
  }
  for (int i = 0; i < 3; ++i) {
    matched_maximum = fmax(matched_maximum,
                           Kokkos::abs(residual.delta_conformal_gamma[i]));
    matched_maximum = fmax(matched_maximum,
                           Kokkos::abs(residual.delta_shift[i]));
    matched_exact = matched_exact
                    && residual.delta_conformal_gamma[i] == 0.0
                    && residual.delta_shift[i] == 0.0;
  }
  if (static_q1) {
    const ref_gh::ReferenceGaugeBaseline baseline =
        ref_gh::ComputeReferenceGaugeBaseline(reference);
    Real zero_hhat[4] = {};         // NOLINT(runtime/arrays)
    Real zero_theta[4] = {};        // NOLINT(runtime/arrays)
    Real zero_dt_theta[4] = {};     // NOLINT(runtime/arrays)
    Real zero_d_hhat[3][4] = {};    // NOLINT(runtime/arrays)
    if (!baseline.valid) return std::numeric_limits<Real>::infinity();
    const ref_gh::GaugeDriverRhs fixed_rhs =
        ref_gh::ComputeGaugeDriverResidualRhs(
            reference, baseline.hhat, baseline.theta, baseline.d_hhat,
            zero_dt_theta, zero_hhat, zero_theta, zero_upsilon,
            zero_d_hhat, residual.physical_shift, residual.reference_shift,
            residual.delta_shift, residual.delta_frame,
            residual.reference_frame, residual.delta_conformal_gamma,
            0.62, 0.57, eta_beta, true);
    for (int A = 0; A < 4; ++A) {
      matched_maximum = fmax(
          matched_maximum, Kokkos::abs(fixed_rhs.hhat[A]));
      matched_maximum = fmax(
          matched_maximum, Kokkos::abs(fixed_rhs.theta[A]));
      matched_exact = matched_exact && fixed_rhs.hhat[A] == 0.0
                      && fixed_rhs.theta[A] == 0.0;
    }
    for (int i = 0; i < 3; ++i) {
      matched_maximum = fmax(
          matched_maximum, Kokkos::abs(fixed_rhs.upsilon[i]));
      matched_exact = matched_exact && fixed_rhs.upsilon[i] == 0.0;
    }
    Real d_zero_hhat[4][4] = {};  // NOLINT(runtime/arrays)
    Real gauge_source[4][4] = {}; // NOLINT(runtime/arrays)
    ref_gh::OrdinaryGaugeResidualDiagnostics diagnostics;
    if (!ref_gh::AddOrdinaryGaugeResidualPartialWaveSource(
            matched_psi, matched_pi, matched_phi, matched_metric,
            matched_d_metric, reference, matched_geometry, zero_hhat,
            d_zero_hhat, 0.73, gauge_source, &diagnostics)) {
      return std::numeric_limits<Real>::infinity();
    }
    for (int A = 0; A < 4; ++A) {
      matched_maximum = fmax(
          matched_maximum, Kokkos::abs(diagnostics.j[A]));
      matched_maximum = fmax(
          matched_maximum, Kokkos::abs(diagnostics.delta_base[A]));
      matched_exact = matched_exact && diagnostics.j[A] == 0.0
                      && diagnostics.delta_base[A] == 0.0;
      for (int p = 0; p < 4; ++p) {
        matched_maximum = fmax(
            matched_maximum, Kokkos::abs(diagnostics.d_j[p][A]));
        matched_maximum = fmax(
            matched_maximum, Kokkos::abs(diagnostics.d_delta_base[p][A]));
        matched_exact = matched_exact && diagnostics.d_j[p][A] == 0.0
                        && diagnostics.d_delta_base[p][A] == 0.0;
      }
      for (int B = 0; B < 4; ++B) {
        matched_maximum = fmax(
            matched_maximum, Kokkos::abs(gauge_source[A][B]));
        matched_exact = matched_exact && gauge_source[A][B] == 0.0;
      }
    }
    if constexpr (std::is_same_v<Reference,
                                 ref_gh::AnalyticRadialQPoint>) {
      ref_gh::CompactAnalyticCoordinateGeometry compact;
      Real compact_determinant = 0.0;
      Real compact_source[4][4] = {};  // NOLINT(runtime/arrays)
      if (!ref_gh::ComputeCompactAnalyticCoordinateGeometry(
              matched_metric, matched_d_metric, reference, compact,
              compact_determinant)
          || !ref_gh::AddCompactAnalyticOrdinaryGaugeResidualSource(
              matched_psi, matched_pi, matched_phi, matched_metric,
              matched_d_metric, reference, compact, zero_hhat,
              d_zero_hhat, 0.73, compact_source)) {
        return std::numeric_limits<Real>::infinity();
      }
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          matched_maximum = fmax(
              matched_maximum, Kokkos::abs(compact_source[A][B]));
          matched_exact = matched_exact && compact_source[A][B] == 0.0;
        }
      }
    }

    constexpr Real gauge_mu = 0.62;
    constexpr Real gauge_eta = 0.57;
    constexpr Real gamma0 = 0.73;
    Real delta_hhat[4];          // NOLINT(runtime/arrays)
    Real delta_theta[4];         // NOLINT(runtime/arrays)
    Real d_delta_hhat[3][4];     // NOLINT(runtime/arrays)
    Real full_hhat[4];           // NOLINT(runtime/arrays)
    Real full_theta[4];          // NOLINT(runtime/arrays)
    Real d_full_hhat[3][4];      // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      delta_hhat[A] = 1.0e-3*static_cast<Real>(((seed + 2*A) % 9) - 4);
      delta_theta[A] =
          8.0e-4*static_cast<Real>(((seed + 3*A) % 11) - 5);
      full_hhat[A] = baseline.hhat[A] + delta_hhat[A];
      full_theta[A] = baseline.theta[A] + delta_theta[A];
      for (int p = 0; p < 3; ++p) {
        d_delta_hhat[p][A] =
            6.0e-4*static_cast<Real>(((seed + A + 2*p) % 13) - 6);
        d_full_hhat[p][A] =
            baseline.d_hhat[p + 1][A] + d_delta_hhat[p][A];
      }
    }
    const ref_gh::GaugeDriverRhs legacy_driver =
        ref_gh::ComputeGaugeDriverRhs(
            reference, full_hhat, full_theta, upsilon, d_full_hhat,
            geometry.shift, legacy.frame, legacy.conformal_gamma, gauge_mu,
            gauge_eta, eta_beta);
    const ref_gh::GaugeDriverRhs residual_driver =
        ref_gh::ComputeGaugeDriverResidualRhs(
            reference, baseline.hhat, baseline.theta, baseline.d_hhat,
            zero_dt_theta, delta_hhat, delta_theta, upsilon, d_delta_hhat,
            geometry.shift, residual.reference_shift, residual.delta_shift,
            residual.delta_frame, residual.reference_frame,
            residual.delta_conformal_gamma, gauge_mu, gauge_eta, eta_beta,
            true);
    for (int A = 0; A < 4; ++A) {
      const Real h_scale = fmax(1.0, fmax(
          Kokkos::abs(legacy_driver.hhat[A]),
          Kokkos::abs(residual_driver.hhat[A])));
      const Real theta_scale = fmax(1.0, fmax(
          Kokkos::abs(legacy_driver.theta[A]),
          Kokkos::abs(residual_driver.theta[A])));
      // This reconstruction is retained only as a cancellation diagnostic.
      // It is not a truth oracle for the residual driver on the trumpet.
      raw_driver_maximum = fmax(
          raw_driver_maximum,
          Kokkos::abs(residual_driver.hhat[A]
                      - (legacy_driver.hhat[A] - baseline.d_hhat[0][A]))
              /h_scale);
      raw_driver_maximum = fmax(
          raw_driver_maximum,
          Kokkos::abs(residual_driver.theta[A] - legacy_driver.theta[A])
              /theta_scale);
    }
    for (int i = 0; i < 3; ++i) {
      const Real scale = fmax(1.0, fmax(
          Kokkos::abs(legacy_driver.upsilon[i]),
          Kokkos::abs(residual_driver.upsilon[i])));
      raw_driver_maximum = fmax(
          raw_driver_maximum,
          Kokkos::abs(residual_driver.upsilon[i]
                      - legacy_driver.upsilon[i])/scale);
    }
    Real full_d_hhat[4][4];   // NOLINT(runtime/arrays)
    Real delta_d_hhat[4][4];  // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      // Isolate the Einstein-source identity from the ill-conditioned full
      // driver reconstruction.  Both paths receive the same-stage residual
      // time derivative and the analytic reference derivative.
      full_d_hhat[0][A] = baseline.d_hhat[0][A]
                          + residual_driver.hhat[A];
      delta_d_hhat[0][A] = residual_driver.hhat[A];
      for (int p = 0; p < 3; ++p) {
        full_d_hhat[p + 1][A] = d_full_hhat[p][A];
        delta_d_hhat[p + 1][A] = d_delta_hhat[p][A];
      }
    }
    Real legacy_source[4][4] = {};    // NOLINT(runtime/arrays)
    Real residual_source[4][4] = {};  // NOLINT(runtime/arrays)
    ref_gh::OrdinaryGaugeResidualDiagnostics perturbed_diagnostics;
    ref_gh::AddOrdinaryGaugePartialWaveSource(
        metric, d_metric, reference, geometry, full_hhat, full_d_hhat,
        gamma0, legacy_source);
    if (!ref_gh::AddOrdinaryGaugeResidualPartialWaveSource(
            psi, pi, phi, metric, d_metric, reference, geometry, delta_hhat,
            delta_d_hhat, gamma0, residual_source,
            &perturbed_diagnostics)) {
      return std::numeric_limits<Real>::infinity();
    }
    if constexpr (std::is_same_v<Reference,
                                 ref_gh::AnalyticRadialQPoint>) {
      ref_gh::CompactAnalyticCoordinateGeometry compact;
      Real compact_determinant = 0.0;
      Real compact_source[4][4] = {};  // NOLINT(runtime/arrays)
      if (!ref_gh::ComputeCompactAnalyticCoordinateGeometry(
              metric, d_metric, reference, compact, compact_determinant)
          || !ref_gh::AddCompactAnalyticOrdinaryGaugeResidualSource(
              psi, pi, phi, metric, d_metric, reference, compact,
              delta_hhat, delta_d_hhat, gamma0, compact_source)) {
        return std::numeric_limits<Real>::infinity();
      }
      for (int A = 0; A < 4; ++A) {
        for (int B = 0; B < 4; ++B) {
          const Real scale = fmax(
              1.0, fmax(Kokkos::abs(residual_source[A][B]),
                        Kokkos::abs(compact_source[A][B])));
          const Real compact_error =
              Kokkos::abs(compact_source[A][B]
                          - residual_source[A][B])/scale;
          source_maximum = fmax(source_maximum, compact_error);
          compact_source_maximum = fmax(
              compact_source_maximum, compact_error);
          UpdateResidualTargetOracleError(
              compact_error, 90 + (A == B ? A : 4), well_conditioned,
              maximum, physical_maximum, delta_maximum, maximum_category);
        }
      }
    }
    Real legacy_d_base[4][4];  // NOLINT(runtime/arrays)
    ref_gh::ImplicitGaugeSourceDerivative(
        metric, d_metric, reference, geometry, legacy_d_base);
    for (int a = 0; a < 4; ++a) {
      Real coordinate_hhat = 0.0;
      for (int A = 0; A < 4; ++A) {
        coordinate_hhat +=
            ref_gh::ReferenceCoframe(reference, A, a)*full_hhat[A];
      }
      const Real legacy_j = coordinate_hhat - geometry.gauge_source[a];
      const Real j_scale = fmax(
          1.0, fmax(Kokkos::abs(legacy_j),
                    Kokkos::abs(perturbed_diagnostics.j[a])));
      const Real j_error =
          Kokkos::abs(perturbed_diagnostics.j[a] - legacy_j)/j_scale;
      source_maximum = fmax(source_maximum, j_error);
      UpdateResidualTargetOracleError(
          j_error,
          80 + a, well_conditioned, maximum, physical_maximum,
          delta_maximum, maximum_category);
      for (int p = 0; p < 4; ++p) {
        Real d_coordinate_hhat = 0.0;
        for (int A = 0; A < 4; ++A) {
          d_coordinate_hhat +=
              ref_gh::ResidualReferenceCoframeDerivative(
                  reference, p, A, a)*full_hhat[A]
              + ref_gh::ReferenceCoframe(reference, A, a)
                  *full_d_hhat[p][A];
        }
        const Real legacy_d_j = d_coordinate_hhat - legacy_d_base[p][a];
        const Real d_j_scale = fmax(
            1.0, fmax(Kokkos::abs(legacy_d_j),
                      Kokkos::abs(perturbed_diagnostics.d_j[p][a])));
        const Real d_j_error =
            Kokkos::abs(perturbed_diagnostics.d_j[p][a] - legacy_d_j)
                /d_j_scale;
        source_maximum = fmax(source_maximum, d_j_error);
        UpdateResidualTargetOracleError(
            d_j_error,
            84 + p, well_conditioned, maximum, physical_maximum,
            delta_maximum, maximum_category);
      }
    }
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        const Real scale = fmax(1.0, fmax(
            Kokkos::abs(legacy_source[A][B]),
            Kokkos::abs(residual_source[A][B])));
        const Real source_error =
            Kokkos::abs(residual_source[A][B] - legacy_source[A][B])/scale;
        source_maximum = fmax(source_maximum, source_error);
        UpdateResidualTargetOracleError(
            source_error,
            70 + (A == B ? A : 4), well_conditioned, maximum,
            physical_maximum, delta_maximum, maximum_category);
      }
    }
  }
  return maximum;
}

void CheckResidualPhysicalGaugeTarget(const DvceArray2D<Real> &table) {
  constexpr int nq = 6;
  constexpr int nrate = 3;
  constexpr int nacceleration = 3;
  constexpr int npoints = kAnalyticOraclePointCount;
  constexpr int nsamples = nq*nrate*nacceleration*npoints;
  Real maximum_error = 0.0;
  Real maximum_physical = 0.0;
  Real maximum_delta = 0.0;
  Real maximum_source = 0.0;
  Real maximum_compact_source = 0.0;
  Real maximum_raw_driver = 0.0;
  Real maximum_matched = 0.0;
  int matched_failure = 0;
  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  MaxLoc::value_type maximum_location;
  Kokkos::parallel_reduce(
      "ref_gh fully subtracted physical gauge target",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_maximum,
                    Real &local_physical, Real &local_delta,
                    Real &local_source, Real &local_compact_source,
                    Real &local_raw_driver,
                    Real &local_matched, int &local_failure,
                    MaxLoc::value_type &local_location) {
        const Real q_values[nq] = {0.75, 0.9, 1.0, 1.1, 1.25, 2.0};
        const Real q_dot_values[nrate] = {-0.1, 0.0, 0.1};
        const Real q_ddot_values[nacceleration] = {-0.05, 0.0, 0.05};
        int work = sample;
        const Real q = q_values[work % nq]; work /= nq;
        const Real q_dot = q_dot_values[work % nrate]; work /= nrate;
        const Real q_ddot = q_ddot_values[work % nacceleration]; work /= nacceleration;
        Real x = 0.0;
        Real y = 0.0;
        Real z = 0.0;
        AnalyticOraclePoint(work, x, y, z);
        Real static_coefficients[ref_gh::kAnalyticRadialQStaticSize];
        Real stage_coefficients[ref_gh::kAnalyticRadialQStageSize];
        ref_gh::EvaluateAnalyticRadialQStatic(
            table, 1.0, 3.0, x, y, z, 0.0, 0.0, 0.0,
            static_coefficients);
        ref_gh::EvaluateAnalyticRadialQStage(
            static_coefficients, q, q_dot, q_ddot, stage_coefficients);
        const ref_gh::AnalyticRadialScalar alpha{
            static_coefficients[ref_gh::kAnalyticAlpha], 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticAlphaRR], 0.0, 0.0};
        const ref_gh::AnalyticRadialScalar l{
            stage_coefficients[ref_gh::kAnalyticL],
            stage_coefficients[ref_gh::kAnalyticLT],
            stage_coefficients[ref_gh::kAnalyticLR],
            stage_coefficients[ref_gh::kAnalyticLTT],
            stage_coefficients[ref_gh::kAnalyticLTR],
            stage_coefficients[ref_gh::kAnalyticLRR],
            stage_coefficients[ref_gh::kAnalyticLTTR],
            stage_coefficients[ref_gh::kAnalyticLTRR]};
        const ref_gh::AnalyticRadialScalar b{
            static_coefficients[ref_gh::kAnalyticShiftB], 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBR], 0.0, 0.0,
            static_coefficients[ref_gh::kAnalyticShiftBRR], 0.0, 0.0};
        const Real displacement[3] = {x, y, z};
        const ref_gh::AnalyticRadialQPoint analytic{
            alpha, l, b, {x, y, z}, Kokkos::sqrt(x*x + y*y + z*z)};
        ref_gh::ReferenceGeometry generic;
        ref_gh::PopulateIsotropicReferenceGeometry(
            ref_gh::AnalyticRadialScalarOracleJet(analytic, alpha),
            ref_gh::AnalyticRadialScalarOracleJet(analytic, l),
            ref_gh::AnalyticRadialScalarOracleJet(analytic, b),
            displacement[0], displacement[1], displacement[2],
            0.0, 0.0, 0.0, generic);
        bool generic_exact = false;
        bool analytic_exact = false;
        Real generic_matched = 0.0;
        Real analytic_matched = 0.0;
        Real generic_physical = 0.0;
        Real analytic_physical = 0.0;
        Real generic_delta = 0.0;
        Real analytic_delta = 0.0;
        Real generic_source = 0.0;
        Real analytic_source = 0.0;
        Real generic_compact_source = 0.0;
        Real analytic_compact_source = 0.0;
        Real generic_raw_driver = 0.0;
        Real analytic_raw_driver = 0.0;
        int generic_category = -1;
        int analytic_category = -1;
        // Below 0.8M, neither the legacy full target nor F-Fref is a reliable
        // binary64 truth oracle.  Preserve every full/delta discrepancy there
        // as a diagnostic rather than choosing one association as truth.
        // Exact matched zeros remain mandatory everywhere.  All perturbed
        // target/source comparisons remain hard gates at and above the
        // established conditioned radius.  The raw full-driver reconstruction
        // is reported separately.
        const bool well_conditioned = analytic.radius >= 0.8;
        const bool static_q1 = q == 1.0 && q_dot == 0.0 && q_ddot == 0.0;
        const Real generic_error = EvaluateResidualPhysicalGaugeTargetOracle(
            generic, sample, generic_exact, well_conditioned, static_q1,
            generic_matched, generic_physical, generic_delta, generic_source,
            generic_compact_source, generic_raw_driver, generic_category);
        const Real analytic_error = EvaluateResidualPhysicalGaugeTargetOracle(
            analytic, sample, analytic_exact, well_conditioned, static_q1,
            analytic_matched, analytic_physical, analytic_delta,
            analytic_source, analytic_compact_source, analytic_raw_driver,
            analytic_category);
        local_maximum = fmax(local_maximum, fmax(generic_error, analytic_error));
        local_physical = fmax(
            local_physical, fmax(generic_physical, analytic_physical));
        local_delta = fmax(local_delta, fmax(generic_delta, analytic_delta));
        local_source = fmax(
            local_source, fmax(generic_source, analytic_source));
        local_compact_source = fmax(
            local_compact_source,
            fmax(generic_compact_source, analytic_compact_source));
        local_raw_driver = fmax(
            local_raw_driver,
            fmax(generic_raw_driver, analytic_raw_driver));
        local_matched = fmax(local_matched,
                             fmax(generic_matched, analytic_matched));
        if (generic_error > local_location.val) {
          local_location.val = generic_error;
          local_location.loc = 1000*sample + generic_category;
        }
        if (analytic_error > local_location.val) {
          local_location.val = analytic_error;
          local_location.loc = 1000*sample + 100 + analytic_category;
        }
        if (local_failure < 0) local_failure = 0;
        if (!generic_exact) local_failure =
            local_failure > 2*sample + 1 ? local_failure : 2*sample + 1;
        if (!analytic_exact) local_failure =
            local_failure > 2*sample + 2 ? local_failure : 2*sample + 2;
      }, Kokkos::Max<Real>(maximum_error),
      Kokkos::Max<Real>(maximum_physical), Kokkos::Max<Real>(maximum_delta),
      Kokkos::Max<Real>(maximum_source),
      Kokkos::Max<Real>(maximum_compact_source),
      Kokkos::Max<Real>(maximum_raw_driver),
      Kokkos::Max<Real>(maximum_matched), Kokkos::Max<int>(matched_failure),
      MaxLoc(maximum_location));
  Kokkos::fence();
  constexpr Real tolerance =
      1024.0*std::numeric_limits<Real>::epsilon();
  if (!(maximum_error <= tolerance) || matched_failure != 0) {
    std::cout << "### FATAL ERROR: fully subtracted physical gauge target "
              << "oracle failed: error=" << maximum_error
              << " matched_exact=" << (matched_failure == 0)
              << " matched_max=" << maximum_matched
              << " matched_failure_location=" << matched_failure
              << " physical_error=" << maximum_physical
              << " raw_delta_error=" << maximum_delta
              << " source_error=" << maximum_source
              << " compact_source_error=" << maximum_compact_source
              << " raw_driver_error=" << maximum_raw_driver
              << " location=" << maximum_location.loc
              << " tolerance=" << tolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH fully subtracted physical gauge target passed: "
            << "samples=" << 2*nsamples << " error=" << maximum_error
            << " all-radius-physical-diagnostic=" << maximum_physical
            << " raw-delta-diagnostic=" << maximum_delta
            << " source=" << maximum_source
            << " compact-source-diagnostic=" << maximum_compact_source
            << " raw-driver-diagnostic=" << maximum_raw_driver
            << " matched_exact=1" << std::endl;
}

void CheckTrumpetQReprojection(const DvceArray2D<Real> &table) {
  constexpr int nsamples = 96;
  Real maximum_metric_error = 0.0;
  Real maximum_derivative_error = 0.0;
  Real minimum_pi_magnitude = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce(
      "ref_gh q-controlled trumpet reprojection",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_metric,
                    Real &local_derivative, Real &local_minimum_pi) {
        const Real x = 0.38 + 0.011*(sample % 4);
        const Real y = -0.31 + 0.013*((sample/4) % 4);
        const Real z = 0.27 + 0.017*((sample/16) % 6);
        ref_gh::ReferenceGeometry physical;
        const ref_gh::TrumpetSchwarzschildReference physical_provider{
            table, 1.0, {0.0, 0.0, 0.0}};
        physical_provider.Populate(0.0, x, y, z, physical);
        const Real q_value = 0.75 + 0.5*static_cast<Real>(sample % 7)/6.0;
        const ref_gh::TrumpetQControlledReferenceParameters params{
            1.0, {0.0, 0.0, 0.0}, 3.0, q_value, 0.0, 0.0};
        ref_gh::ReferenceGeometry current;
        const ref_gh::TrumpetQControlledReference current_provider{
            table, params};
        current_provider.Populate(0.0, x, y, z, current);
        const ref_gh::ProjectedFirstOrderMetric projected =
            ref_gh::ProjectPhysicalMetricToReference(
                physical.metric, physical.d_metric, current);
        if (!projected.valid) {
          local_metric = std::numeric_limits<Real>::infinity();
          return;
        }
        Real reconstructed[4][4] = {};  // NOLINT(runtime/arrays)
        Real d_reconstructed[4][4][4] = {};  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            for (int A = 0; A < 4; ++A) {
              for (int B = 0; B < 4; ++B) {
                reconstructed[a][b] += current.coframe[A][a]
                                         *current.coframe[B][b]
                                         *projected.psi[A][B];
              }
            }
          }
        }
        Real inverse[4][4];  // NOLINT(runtime/arrays)
        Real determinant = 0.0;
        if (!ref_gh::Invert4(reconstructed, inverse, determinant)
            || !(inverse[0][0] < 0.0)) {
          local_metric = std::numeric_limits<Real>::infinity();
          return;
        }
        const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
        Real shift[3];  // NOLINT(runtime/arrays)
        for (int i = 0; i < 3; ++i) {
          shift[i] = lapse*lapse*inverse[0][i + 1];
        }
        Real recovered_d_psi[4][4][4] = {};  // NOLINT(runtime/arrays)
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            for (int i = 0; i < 3; ++i) {
              for (int I = 0; I < 3; ++I) {
                recovered_d_psi[i + 1][A][B] +=
                    current.spatial_coframe[I][i]
                    *projected.phi[I][A][B];
              }
            }
            recovered_d_psi[0][A][B] = -lapse*projected.pi[A][B];
            for (int i = 0; i < 3; ++i) {
              recovered_d_psi[0][A][B] +=
                  shift[i]*recovered_d_psi[i + 1][A][B];
            }
          }
        }
        for (int p = 0; p < 4; ++p) {
          Real frame_corrected[4][4];  // NOLINT(runtime/arrays)
          for (int A = 0; A < 4; ++A) {
            for (int B = 0; B < 4; ++B) {
              frame_corrected[A][B] = recovered_d_psi[p][A][B];
              for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                  frame_corrected[A][B] -=
                      (current.d_frame[p][A][a]*current.frame[B][b]
                       + current.frame[A][a]*current.d_frame[p][B][b])
                      *reconstructed[a][b];
                }
              }
            }
          }
          for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
              for (int A = 0; A < 4; ++A) {
                for (int B = 0; B < 4; ++B) {
                  d_reconstructed[p][a][b] +=
                      current.coframe[A][a]*current.coframe[B][b]
                      *frame_corrected[A][B];
                }
              }
            }
          }
        }
        Real pi_magnitude = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            pi_magnitude = fmax(
                pi_magnitude, Kokkos::abs(projected.pi[A][B]));
          }
        }
        if (q_value != 1.0) {
          local_minimum_pi = fmin(local_minimum_pi, pi_magnitude);
        }
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            const Real metric_scale = fmax(1.0, Kokkos::abs(
                physical.metric[a][b]));
            local_metric = fmax(
                local_metric,
                Kokkos::abs(reconstructed[a][b] - physical.metric[a][b])
                    /metric_scale);
            for (int p = 0; p < 4; ++p) {
              const Real derivative_scale = fmax(
                  1.0, Kokkos::abs(physical.d_metric[p][a][b]));
              local_derivative = fmax(
                  local_derivative,
                  Kokkos::abs(d_reconstructed[p][a][b]
                              - physical.d_metric[p][a][b])
                      /derivative_scale);
            }
          }
        }
      }, Kokkos::Max<Real>(maximum_metric_error),
      Kokkos::Max<Real>(maximum_derivative_error),
      Kokkos::Min<Real>(minimum_pi_magnitude));
  if (!(maximum_metric_error <= 2.0e-13)
      || !(maximum_derivative_error <= 2.0e-12)
      || !(minimum_pi_magnitude > 1.0e-8)) {
    std::cout << "### FATAL ERROR: q-controlled trumpet reprojection failed: "
              << "metric=" << maximum_metric_error
              << " derivative=" << maximum_derivative_error
              << " min-nontrivial-Pi=" << minimum_pi_magnitude << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH q-controlled trumpet reprojection passed: metric="
            << maximum_metric_error
            << " derivative=" << maximum_derivative_error
            << " min-nontrivial-Pi=" << minimum_pi_magnitude << std::endl;
}

void CheckTrumpetQGaugeReprojection(const DvceArray2D<Real> &table) {
  constexpr int nsamples = 80;
  Real maximum_hhat_error = 0.0;
  Real maximum_theta_error = 0.0;
  Real maximum_subtraction_error = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh q-controlled trumpet gauge reprojection",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nsamples),
      KOKKOS_LAMBDA(const int sample, Real &local_hhat, Real &local_theta,
                    Real &local_subtraction) {
        const Real x = 0.43 + 0.009*(sample % 4);
        const Real y = -0.34 + 0.012*((sample/4) % 4);
        const Real z = 0.25 + 0.014*((sample/16) % 5);
        ref_gh::ReferenceGeometry physical;
        const ref_gh::TrumpetSchwarzschildReference physical_provider{
            table, 1.0, {0.0, 0.0, 0.0}};
        physical_provider.Populate(0.0, x, y, z, physical);
        const Real q_value = 0.75 + 0.5*static_cast<Real>(sample % 9)/8.0;
        const Real q_dot = -0.06 + 0.02*static_cast<Real>((sample/9) % 7);
        const ref_gh::TrumpetQControlledReferenceParameters params{
            1.0, {0.0, 0.0, 0.0}, 3.0, q_value, q_dot, -0.025};
        ref_gh::ReferenceGeometry current;
        const ref_gh::TrumpetQControlledReference current_provider{
            table, params};
        current_provider.Populate(0.0, x, y, z, current);
        const ref_gh::ProjectedStationaryGaugeState projected =
            ref_gh::ProjectStationaryPhysicalGaugeToReference(
                physical, current);
        const ref_gh::ReferenceGaugeBaseline baseline =
            ref_gh::ComputeReferenceGaugeBaseline(current);
        if (!projected.valid || !baseline.valid) {
          local_hhat = std::numeric_limits<Real>::infinity();
          return;
        }
        ref_gh::CoordinateGhGeometry geometry;
        Real determinant = 0.0;
        if (!ref_gh::ComputeCoordinateGhGeometry(
                physical.metric, physical.d_metric, physical, geometry,
                determinant)) {
          local_hhat = std::numeric_limits<Real>::infinity();
          return;
        }
        Real d_hhat_coordinate[4][4] = {};  // NOLINT(runtime/arrays)
        ref_gh::ImplicitGaugeSourceDerivative(
            physical.metric, physical.d_metric, physical, geometry,
            d_hhat_coordinate);
        Real expected_theta_coordinate[4] = {};  // NOLINT(runtime/arrays)
        for (int a = 0; a < 4; ++a) {
          for (int i = 0; i < 3; ++i) {
            expected_theta_coordinate[a] -=
                geometry.shift[i]*d_hhat_coordinate[i + 1][a];
          }
          Real recovered_hhat = 0.0;
          Real recovered_theta = 0.0;
          Real recovered_subtracted_hhat = 0.0;
          Real recovered_subtracted_theta = 0.0;
          for (int A = 0; A < 4; ++A) {
            recovered_hhat += current.coframe[A][a]*projected.hhat[A];
            recovered_theta += current.coframe[A][a]*projected.theta[A];
            const Real delta_hhat = projected.hhat[A] - baseline.hhat[A];
            const Real delta_theta = projected.theta[A] - baseline.theta[A];
            recovered_subtracted_hhat +=
                current.coframe[A][a]*(delta_hhat + baseline.hhat[A]);
            recovered_subtracted_theta +=
                current.coframe[A][a]*(delta_theta + baseline.theta[A]);
          }
          local_hhat = fmax(
              local_hhat,
              Kokkos::abs(recovered_hhat - geometry.gauge_source[a]));
          local_theta = fmax(
              local_theta,
              Kokkos::abs(recovered_theta - expected_theta_coordinate[a]));
          local_subtraction = fmax(
              local_subtraction,
              Kokkos::abs(recovered_subtracted_hhat
                          - geometry.gauge_source[a]));
          local_subtraction = fmax(
              local_subtraction,
              Kokkos::abs(recovered_subtracted_theta
                          - expected_theta_coordinate[a]));
        }
      }, Kokkos::Max<Real>(maximum_hhat_error),
      Kokkos::Max<Real>(maximum_theta_error),
      Kokkos::Max<Real>(maximum_subtraction_error));
  if (!(maximum_hhat_error <= 2.0e-12)
      || !(maximum_theta_error <= 2.0e-12)
      || !(maximum_subtraction_error <= 2.0e-12)) {
    std::cout << "### FATAL ERROR: q-controlled gauge reprojection failed: "
              << "Hhat=" << maximum_hhat_error
              << " theta=" << maximum_theta_error
              << " subtraction=" << maximum_subtraction_error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH q-controlled gauge reprojection passed: Hhat="
            << maximum_hhat_error << " theta=" << maximum_theta_error
            << " subtraction=" << maximum_subtraction_error << std::endl;
}

ref_gh::QRelaxedControllerRhs ManufacturedQControllerRhs(
    const Real q, const Real q_dot, const Real time,
    const bool varying_target) {
  const Real q_est = varying_target ? 1.0 + 0.05*std::sin(0.1*time) : 1.0;
  return ref_gh::EvaluateQRelaxedControllerRhs(
      q, q_dot, q_est, 0.5, 1.0, 1.0e6);
}

void AdvanceManufacturedQController(Real &q, Real &q_dot, Real &time,
                                    const Real dt, const int steps,
                                    const bool varying_target) {
  for (int step = 0; step < steps; ++step) {
    const ref_gh::QRelaxedControllerRhs k1 = ManufacturedQControllerRhs(
        q, q_dot, time, varying_target);
    const ref_gh::QRelaxedControllerRhs k2 = ManufacturedQControllerRhs(
        q + 0.5*dt*k1.q, q_dot + 0.5*dt*k1.q_dot,
        time + 0.5*dt, varying_target);
    const ref_gh::QRelaxedControllerRhs k3 = ManufacturedQControllerRhs(
        q + 0.5*dt*k2.q, q_dot + 0.5*dt*k2.q_dot,
        time + 0.5*dt, varying_target);
    const ref_gh::QRelaxedControllerRhs k4 = ManufacturedQControllerRhs(
        q + dt*k3.q, q_dot + dt*k3.q_dot, time + dt, varying_target);
    q += dt*(k1.q + 2.0*k2.q + 2.0*k3.q + k4.q)/6.0;
    q_dot += dt*(k1.q_dot + 2.0*k2.q_dot
                 + 2.0*k3.q_dot + k4.q_dot)/6.0;
    time += dt;
  }
}

void CheckQRelaxedController() {
  const ref_gh::QRelaxedControllerRhs positive =
      ref_gh::EvaluateQRelaxedControllerRhs(0.9, 0.0, 1.0, 0.5, 1.0, 1.0);
  const ref_gh::QRelaxedControllerRhs negative =
      ref_gh::EvaluateQRelaxedControllerRhs(1.1, 0.0, 1.0, 0.5, 1.0, 1.0);
  if (!(positive.q_dot > 0.0) || !(negative.q_dot < 0.0)) {
    std::cout << "### FATAL ERROR: q-controller sign oracle failed."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  constexpr Real dt = 1.0/256.0;
  constexpr int steps = 2048;
  Real q = 0.75;
  Real q_dot = 0.0;
  Real time = 0.0;
  AdvanceManufacturedQController(q, q_dot, time, dt, steps, false);
  const Real omega = 0.5;
  const Real displacement0 = -0.25;
  const Real exact_q = 1.0
      + (displacement0 + omega*displacement0*time)*std::exp(-omega*time);
  const Real exact_q_dot =
      -omega*omega*displacement0*time*std::exp(-omega*time);
  const Real critical_error = std::max(
      std::abs(q - exact_q), std::abs(q_dot - exact_q_dot));

  Real uninterrupted_q = 1.08;
  Real uninterrupted_q_dot = -0.03;
  Real uninterrupted_time = 0.0;
  AdvanceManufacturedQController(
      uninterrupted_q, uninterrupted_q_dot, uninterrupted_time,
      dt, steps, true);
  Real restarted_q = 1.08;
  Real restarted_q_dot = -0.03;
  Real restarted_time = 0.0;
  AdvanceManufacturedQController(
      restarted_q, restarted_q_dot, restarted_time, dt, steps/2, true);
  const Real checkpoint_q = restarted_q;
  const Real checkpoint_q_dot = restarted_q_dot;
  const Real checkpoint_time = restarted_time;
  restarted_q = checkpoint_q;
  restarted_q_dot = checkpoint_q_dot;
  restarted_time = checkpoint_time;
  AdvanceManufacturedQController(
      restarted_q, restarted_q_dot, restarted_time, dt, steps/2, true);
  const bool restart_exact = restarted_q == uninterrupted_q
      && restarted_q_dot == uninterrupted_q_dot
      && restarted_time == uninterrupted_time;
  const Real final_target = 1.0 + 0.05*std::sin(0.1*uninterrupted_time);
  const Real varying_lag = std::abs(uninterrupted_q - final_target);
  constexpr Real prescribed_duration = 8.0;
  constexpr Real prescribed_time = 2.3;
  constexpr Real prescribed_step = 1.0e-4;
  Real prescribed_error = 0.0;
  const Real prescribed_targets[4] = {0.90, 1.10, 0.75, 1.25};
  for (const Real target : prescribed_targets) {
    const ref_gh::PrescribedQTrajectory start =
        ref_gh::EvaluatePrescribedQTrajectory(
            0.0, target, prescribed_duration);
    const ref_gh::PrescribedQTrajectory peak =
        ref_gh::EvaluatePrescribedQTrajectory(
            0.5*prescribed_duration, target, prescribed_duration);
    const ref_gh::PrescribedQTrajectory end =
        ref_gh::EvaluatePrescribedQTrajectory(
            prescribed_duration, target, prescribed_duration);
    const ref_gh::PrescribedQTrajectory center =
        ref_gh::EvaluatePrescribedQTrajectory(
            prescribed_time, target, prescribed_duration);
    const ref_gh::PrescribedQTrajectory minus =
        ref_gh::EvaluatePrescribedQTrajectory(
            prescribed_time - prescribed_step, target,
            prescribed_duration);
    const ref_gh::PrescribedQTrajectory plus =
        ref_gh::EvaluatePrescribedQTrajectory(
            prescribed_time + prescribed_step, target,
            prescribed_duration);
    const Real fd_q_dot =
        (plus.q - minus.q)/(2.0*prescribed_step);
    const Real fd_q_ddot =
        (plus.q - 2.0*center.q + minus.q)
        /(prescribed_step*prescribed_step);
    prescribed_error = std::max(
        prescribed_error, std::abs(start.q - 1.0));
    prescribed_error = std::max(
        prescribed_error, std::abs(start.q_dot));
    prescribed_error = std::max(
        prescribed_error, std::abs(start.q_ddot));
    prescribed_error = std::max(
        prescribed_error, std::abs(peak.q - target));
    prescribed_error = std::max(
        prescribed_error, std::abs(peak.q_dot));
    prescribed_error = std::max(
        prescribed_error, std::abs(end.q - 1.0));
    prescribed_error = std::max(
        prescribed_error, std::abs(end.q_dot));
    prescribed_error = std::max(
        prescribed_error, std::abs(end.q_ddot));
    prescribed_error = std::max(
        prescribed_error, std::abs(center.q_dot - fd_q_dot));
    prescribed_error = std::max(
        prescribed_error, std::abs(center.q_ddot - fd_q_ddot));
  }
  if (!(critical_error < 2.0e-11) || !restart_exact
      || !(varying_lag < 0.03) || !(uninterrupted_q > 0.5)
      || !(uninterrupted_q < 2.5) || !(prescribed_error < 2.0e-8)) {
    std::cout << "### FATAL ERROR: q-controller manufactured history failed: "
              << "critical-error=" << critical_error
              << " restart-exact=" << restart_exact
              << " varying-lag=" << varying_lag
              << " prescribed-error=" << prescribed_error << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "reference-GH q-controller manufactured histories passed: "
            << "critical-error=" << critical_error
            << " restart-exact=" << restart_exact
            << " varying-lag=" << varying_lag
            << " prescribed-error=" << prescribed_error << std::endl;
}

void CheckLocalPunctureExponentEstimator(const DvceArray2D<Real> &table) {
  {
    const Real spacing[3] = {1.0, 0.5, 0.25};
    const Real overlapping[3] = {2.0, 1.0, 0.5};
    const Real clear_in_x[3] = {2.0 + 1.0e-12, 0.0, 0.0};
    const Real clear_in_y[3] = {0.0, 1.0 + 1.0e-12, 0.0};
    if (ref_gh::PunctureStencilIsClear(overlapping, spacing, 2)
        || !ref_gh::PunctureStencilIsClear(clear_in_x, spacing, 2)
        || !ref_gh::PunctureStencilIsClear(clear_in_y, spacing, 2)
        || ref_gh::PunctureEvolutionStencilRadius(2, 0.0) != 1
        || ref_gh::PunctureEvolutionStencilRadius(2, 0.1) != 2
        || ref_gh::PunctureEvolutionStencilRadius(4, 0.0) != 2
        || ref_gh::PunctureEvolutionStencilRadius(4, 0.1) != 3
        || ref_gh::PunctureEvolutionStencilRadius(6, 0.1) != 4) {
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
      Real sum_wq_exact = 0.0;
      Real safe_sum_w = 0.0;
      Real safe_sum_w2 = 0.0;
      Real safe_sum_wq = 0.0;
      Real safe_sum_wq2 = 0.0;
      Real safe_sum_wq_exact = 0.0;
      Real sum_wq_fd = 0.0;
      Real maximum_state_error = 0.0;
      Real maximum_fd_error = 0.0;
      Real minimum_q = std::numeric_limits<Real>::max();
      Real maximum_q = -std::numeric_limits<Real>::max();
      Real safe_minimum_q = std::numeric_limits<Real>::max();
      Real safe_maximum_q = -std::numeric_limits<Real>::max();
      int count = 0;
      int safe_count = 0;
      Kokkos::parallel_reduce(
          "ref_gh local puncture exponent estimator",
          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
          KOKKOS_LAMBDA(const int index, Real &local_sum_w,
                        Real &local_sum_w2, Real &local_sum_wq,
                        Real &local_sum_wq2, Real &local_safe_sum_w,
                        Real &local_safe_sum_w2, Real &local_safe_sum_wq,
                        Real &local_safe_sum_wq2,
                        Real &local_safe_sum_wq_exact,
                        Real &local_sum_wq_fd, Real &local_sum_wq_exact,
                        Real &local_maximum_state_error,
                        Real &local_maximum_fd_error, Real &local_minimum_q,
                        Real &local_maximum_q, Real &local_safe_minimum_q,
                        Real &local_safe_maximum_q, int &local_count,
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
            local_sum_wq_exact += weight*sample.q_exact;
            local_minimum_q = fmin(local_minimum_q, sample.q_state);
            local_maximum_q = fmax(local_maximum_q, sample.q_state);
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
              local_safe_sum_wq2 += weight*sample.q_state*sample.q_state;
              local_safe_sum_wq_exact += weight*sample.q_exact;
              local_safe_minimum_q = fmin(
                  local_safe_minimum_q, sample.q_state);
              local_safe_maximum_q = fmax(
                  local_safe_maximum_q, sample.q_state);
              local_sum_wq_fd += weight*sample.q_fd;
              local_maximum_fd_error = fmax(
                  local_maximum_fd_error,
                  Kokkos::abs(sample.q_fd - sample.q_exact));
              ++local_safe_count;
            }
          }, Kokkos::Sum<Real>(sum_w), Kokkos::Sum<Real>(sum_w2),
          Kokkos::Sum<Real>(sum_wq), Kokkos::Sum<Real>(sum_wq2),
          Kokkos::Sum<Real>(safe_sum_w), Kokkos::Sum<Real>(safe_sum_w2),
          Kokkos::Sum<Real>(safe_sum_wq), Kokkos::Sum<Real>(safe_sum_wq2),
          Kokkos::Sum<Real>(safe_sum_wq_exact),
          Kokkos::Sum<Real>(sum_wq_fd), Kokkos::Sum<Real>(sum_wq_exact),
          Kokkos::Max<Real>(maximum_state_error),
          Kokkos::Max<Real>(maximum_fd_error), Kokkos::Min<Real>(minimum_q),
          Kokkos::Max<Real>(maximum_q), Kokkos::Min<Real>(safe_minimum_q),
          Kokkos::Max<Real>(safe_maximum_q), Kokkos::Sum<int>(count),
          Kokkos::Sum<int>(safe_count));
      const Real full_q_est = sum_wq/sum_w;
      const Real q_est = safe_sum_wq/safe_sum_w;
      const Real q_analytic = safe_sum_wq_exact/safe_sum_w;
      const Real q_fd_est = sum_wq_fd/safe_sum_w;
      const Real variance = fmax(
          0.0, safe_sum_wq2/safe_sum_w - q_est*q_est);
      const Real n_eff = sum_w*sum_w/sum_w2;
      const Real safe_n_eff = safe_sum_w*safe_sum_w/safe_sum_w2;
      if (count <= 0 || safe_count <= 0 || !(n_eff > 1.0)
          || !(safe_n_eff > 1.0) || !std::isfinite(variance)
          || Kokkos::abs(q_est - q_analytic) > 2.0e-13
          || maximum_state_error > 2.0e-11
          || (geometry_kind == 0 && maximum_fd_error > 2.0e-13)) {
        std::cout << "### FATAL ERROR: local puncture exponent estimator failed: "
                  << "geometry=" << geometry_kind << " h=" << h
                  << " count=" << count << " N_eff=" << n_eff
                  << " state-error=" << maximum_state_error
                  << " FD-error=" << maximum_fd_error << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real fd_difference = Kokkos::abs(q_fd_est - q_est);
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
                << " q_analytic=" << q_analytic
                << " q_fd_est=" << q_fd_est << " variance=" << variance
                << " N_eff=" << safe_n_eff
                << " samples=" << safe_count
                << " q_min=" << safe_minimum_q
                << " q_max=" << safe_maximum_q
                << " unmasked_q_est=" << full_q_est
                << " unmasked_N_eff=" << n_eff
                << " unmasked_samples=" << count
                << " unmasked_q_min=" << minimum_q
                << " unmasked_q_max=" << maximum_q
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
  // Same-r/h direct-FD behavior is intentionally diagnostic-only.  A centered
  // stencil samples identical dimensionless points of a self-similar singular
  // profile as h changes, so its relative bias need not converge to zero.
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

void CheckReferenceJetMixedTimeAlgebra() {
  constexpr Real t_value = 0.3;
  constexpr Real x_value = 0.4;
  constexpr Real y_value = -0.2;
  constexpr Real z_value = 0.1;
  Real product_maximum = 0.0;
  Kokkos::parallel_reduce(
      "reference-GH mixed-time product jet",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int, Real &local_maximum) {
        // Direct polynomial oracle for the product rule:
        // (t*x)*(x*y)=t*x^2*y. Seed the two factor jets directly so this
        // device kernel holds only three 33-Real jets at once.
        ref_gh::ReferenceJet left = ref_gh::ConstantJet(t_value*x_value);
        left.d[0] = x_value;
        left.d[1] = t_value;
        left.dd[0][1] = left.dd[1][0] = 1.0;
        ref_gh::ReferenceJet right = ref_gh::ConstantJet(x_value*y_value);
        right.d[1] = y_value;
        right.d[2] = x_value;
        right.dd[1][2] = right.dd[2][1] = 1.0;
        const ref_gh::ReferenceJet polynomial = left*right;
        const Real expected_polynomial[3][4] = {  // NOLINT(runtime/arrays)
          {0.0, 2.0*y_value, 2.0*x_value, 0.0},
          {0.0, 2.0*x_value, 0.0, 0.0},
          {0.0, 0.0, 0.0, 0.0}
        };
        for (int i = 0; i < 3; ++i) {
          for (int q = 0; q < 4; ++q) {
            local_maximum = fmax(local_maximum, Kokkos::abs(
                polynomial.dt_dd[i][q] - expected_polynomial[i][q]));
          }
        }
      }, Kokkos::Max<Real>(product_maximum));

  Real exponential_maximum = 0.0;
  Kokkos::parallel_reduce(
      "reference-GH mixed-time exponential jet",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
      KOKKOS_LAMBDA(const int, Real &local_maximum) {
        ref_gh::ReferenceJet argument =
            ref_gh::ConstantJet(t_value*x_value + y_value);
        argument.d[0] = x_value;
        argument.d[1] = t_value;
        argument.d[2] = 1.0;
        argument.dd[0][1] = argument.dd[1][0] = 1.0;

        // Closed-form mixed derivatives of exp(t*x+y). These exercise the
        // F'' terms generated by the nonzero u_tx as well as F'''.
        const ref_gh::ReferenceJet exponential = ref_gh::Exp(argument);
        const Real exp_value = Kokkos::exp(t_value*x_value + y_value);
        const Real expected_exp_x[4] = {  // d_t d_x d_q, NOLINT(runtime/arrays)
          exp_value*(2.0*x_value + t_value*x_value*x_value),
          exp_value*(2.0*t_value + t_value*t_value*x_value),
          exp_value*(1.0 + t_value*x_value),
          0.0
        };
        for (int q = 0; q < 4; ++q) {
          local_maximum = fmax(local_maximum, Kokkos::abs(
              exponential.dt_dd[0][q] - expected_exp_x[q]));
        }
      }, Kokkos::Max<Real>(exponential_maximum));

  Real unary_maximum = 0.0;
  Kokkos::parallel_reduce(
      "reference-GH affine mixed-time unary jets",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, 4),
      KOKKOS_LAMBDA(const int function, Real &local_maximum) {
        // For an affine argument, the mixed third derivative of F(v) is
        // exactly F'''(v) v_t v_i v_q. This independently checks all four
        // unary functions and all stored mixed components.
        const Real v = 3.0 + t_value + 2.0*x_value - y_value + 0.5*z_value;
        const Real gradient[4] = {1.0, 2.0, -1.0, 0.5}; // NOLINT
        ref_gh::ReferenceJet affine = ref_gh::ConstantJet(v);
        for (int p = 0; p < 4; ++p) affine.d[p] = gradient[p];
        const Real inverse = 1.0/v;
        const Real unary_third[4] = {  // NOLINT(runtime/arrays)
          -6.0*inverse*inverse*inverse*inverse,
          2.0*inverse*inverse*inverse,
          Kokkos::exp(v),
          0.375/(v*v*Kokkos::sqrt(v))
        };
        ref_gh::ReferenceJet unary = ref_gh::Reciprocal(affine);
        if (function == 1) unary = ref_gh::Log(affine);
        if (function == 2) unary = ref_gh::Exp(affine);
        if (function == 3) unary = ref_gh::Sqrt(affine);
        for (int i = 0; i < 3; ++i) {
          for (int q = 0; q < 4; ++q) {
            const Real expected = unary_third[function]
                                  *gradient[0]*gradient[i + 1]*gradient[q];
            local_maximum = fmax(local_maximum, Kokkos::abs(
                unary.dt_dd[i][q] - expected));
          }
        }
      }, Kokkos::Max<Real>(unary_maximum));
  Kokkos::fence();
  const Real maximum = fmax(product_maximum,
      fmax(exponential_maximum, unary_maximum));
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH mixed-time jet algebra passed: max error = "
              << maximum << std::endl;
  }
  if (!(maximum < 2.0e-13)) {
    std::cout << "### FATAL ERROR: reference-GH mixed-time jet algebra failed."
              << std::endl;
    std::exit(EXIT_FAILURE);
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
  CheckExactMatchedQ1Predicate();
  CheckCoframeDerivativeIdentity();
  CheckGaugeDriverAlgebra();
  CheckGamma2Algebra();
  CheckCombinedGaugeCharacteristics();
  CheckReferenceJetMixedTimeAlgebra();
  CheckPhiOrderingAlgebra();
  CheckFlatCovariantSource();
  CheckNonflatCovariantSource();
  CheckDynamicSpatialReference();
  if (pin->GetOrAddBoolean("problem", "puncture_exponent_gate", false)) {
    CheckRelativeExponentIdentity();
    CheckLocalPunctureExponentEstimator(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
  }
  if (pin->GetOrAddBoolean(
          "problem", "q_controlled_reference_gate", false)) {
    CheckTrumpetQControlledReference(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckAnalyticRadialQCoefficients(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckExpandedAnalyticRadialQCoefficients(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckGeneratedAnalyticRadialQGeometry(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckGeneratedAnalyticRadialQGauge(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckResidualPhysicalGaugeTarget(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckCompactAnalyticRadialQBoundaryProjection(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckAll61AnalyticRadialQRhs(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckTrumpetQReprojection(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
    CheckTrumpetQGaugeReprojection(
        pmy_mesh_->pmb_pack->prefgh->reference_table);
  }
  if (pin->GetOrAddBoolean("problem", "q_controller_self_test", false)) {
    CheckQRelaxedController();
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
