//========================================================================================
//! \file relative_damped_gauge.hpp
//! \brief Regular reference-relative damped-wave gauge for puncture Ref-GH.
//========================================================================================
#ifndef REF_GH_RELATIVE_DAMPED_GAUGE_HPP_
#define REF_GH_RELATIVE_DAMPED_GAUGE_HPP_

#include "athena.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

// The physical gauge is H_a=B_a+bar_theta^A_a W D_A.  The ordinary wave-map
// source B_a is already included by the Ref-GH scalar source.  This structure
// therefore contains only the regular algebraic increment and its coordinate
// derivative.  No singular full gauge source is ever reconstructed here.
struct RelativeDampedGaugeData {
  Real d[4];                    // D_A, NOLINT(runtime/arrays)
  Real d_d[4][4];               // partial_mu D_A, NOLINT(runtime/arrays)
  Real correction[4];           // W D_A, NOLINT(runtime/arrays)
  Real d_correction[4][4];      // partial_mu(W D_A), NOLINT(runtime/arrays)
  Real window;
  bool valid;
};

struct RelativeDampedGaugeDiagnostics {
  Real d_max;
  Real correction_max;
  Real source_max;
};

KOKKOS_INLINE_FUNCTION
void ZeroRelativeDampedGaugeData(RelativeDampedGaugeData &data) {
  data.window = 0.0;
  data.valid = true;
  for (int A = 0; A < 4; ++A) {
    data.d[A] = 0.0;
    data.correction[A] = 0.0;
    for (int p = 0; p < 4; ++p) {
      data.d_d[p][A] = 0.0;
      data.d_correction[p][A] = 0.0;
    }
  }
}

// C2 quintic transition with exact constant branches.  The core branch is
// tested before any relative-metric inverse or logarithm is evaluated.
KOKKOS_INLINE_FUNCTION
void RelativeDampedWindow(const Real displacement[3], const Real radius,
                          const Real r0, const Real r1, Real &window,
                          Real d_window[4]) {
  for (int p = 0; p < 4; ++p) d_window[p] = 0.0;
  if (radius <= r0) {
    window = 0.0;
    return;
  }
  if (radius >= r1) {
    window = 1.0;
    return;
  }
  const Real s = (radius - r0)/(r1 - r0);
  const Real s2 = s*s;
  const Real one_minus_s = 1.0 - s;
  window = s2*s*(10.0 + s*(-15.0 + 6.0*s));
  const Real d_window_dr = 30.0*s2*one_minus_s*one_minus_s/(r1 - r0);
  for (int p = 0; p < 3; ++p) {
    d_window[p + 1] = d_window_dr*displacement[p]/radius;
  }
}

// Compute the relative ADM-like variables and their complete coordinate first
// jet directly from Psi_AB and partial_mu Psi_AB:
//
//   a_R=(-Psi^00)^(-1/2), v_R^I=a_R^2 Psi^0I,
//   L_R=log(sqrt(det Psi_IJ)/a_R), V_A=Psi_AI v_R^I,
//   D_A=mu_L L_R (-a_R,0,0,0)_A-(mu_S/a_R)V_A.
//
// The formulas use only regular relative quantities.  The matched state
// Psi_AB=eta_AB returns D_A=0 exactly in binary arithmetic.
KOKKOS_INLINE_FUNCTION
bool ComputeRelativeDampedGaugeData(
    const Real psi[4][4], const Real d_psi[4][4][4],
    const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z,
    const Real r0, const Real r1, const Real mu_l, const Real mu_s,
    RelativeDampedGaugeData &data) {
  ZeroRelativeDampedGaugeData(data);
  const Real displacement[3] = {  // NOLINT(runtime/arrays)
      x - center_x, y - center_y, z - center_z};
  const Real radius = Kokkos::sqrt(
      displacement[0]*displacement[0]
      + displacement[1]*displacement[1]
      + displacement[2]*displacement[2]);
  Real d_window[4];  // NOLINT(runtime/arrays)
  RelativeDampedWindow(displacement, radius, r0, r1, data.window, d_window);
  if (data.window == 0.0) return true;

  Real inverse[4][4];  // NOLINT(runtime/arrays)
  Real determinant = 0.0;
  if (!Invert4(psi, inverse, determinant) || !(inverse[0][0] < 0.0)) {
    data.valid = false;
    return false;
  }
  Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
  Real spatial_determinant = 0.0;
  if (!InvertSpatial3(psi, spatial_inverse, spatial_determinant)) {
    data.valid = false;
    return false;
  }

  const Real relative_lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
  const Real relative_lapse2 = relative_lapse*relative_lapse;
  Real relative_shift[3];  // NOLINT(runtime/arrays)
  for (int I = 0; I < 3; ++I) {
    relative_shift[I] = relative_lapse2*inverse[0][I + 1];
  }
  const Real log_volume = 0.5*Kokkos::log(spatial_determinant)
                          - Kokkos::log(relative_lapse);
  Real relative_normal[4] = {  // NOLINT(runtime/arrays)
      -relative_lapse, 0.0, 0.0, 0.0};
  Real relative_velocity_lower[4] = {};  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int I = 0; I < 3; ++I) {
      relative_velocity_lower[A] += psi[A][I + 1]*relative_shift[I];
    }
    data.d[A] = mu_l*log_volume*relative_normal[A]
                - (mu_s/relative_lapse)*relative_velocity_lower[A];
  }

  for (int p = 0; p < 4; ++p) {
    Real d_inverse[4][4];  // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        d_inverse[A][B] = 0.0;
        for (int C = 0; C < 4; ++C) {
          for (int D = 0; D < 4; ++D) {
            d_inverse[A][B] -= inverse[A][C]*inverse[B][D]
                                *d_psi[p][C][D];
          }
        }
      }
    }
    const Real d_relative_lapse =
        0.5*relative_lapse*relative_lapse*relative_lapse*d_inverse[0][0];
    Real d_relative_shift[3];  // NOLINT(runtime/arrays)
    for (int I = 0; I < 3; ++I) {
      d_relative_shift[I] =
          2.0*relative_lapse*d_relative_lapse*inverse[0][I + 1]
          + relative_lapse2*d_inverse[0][I + 1];
    }
    Real d_log_volume = -d_relative_lapse/relative_lapse;
    for (int I = 0; I < 3; ++I) {
      for (int J = 0; J < 3; ++J) {
        d_log_volume += 0.5*spatial_inverse[I][J]
                        *d_psi[p][I + 1][J + 1];
      }
    }
    for (int A = 0; A < 4; ++A) {
      Real d_relative_velocity_lower = 0.0;
      for (int I = 0; I < 3; ++I) {
        d_relative_velocity_lower +=
            d_psi[p][A][I + 1]*relative_shift[I]
            + psi[A][I + 1]*d_relative_shift[I];
      }
      const Real d_relative_normal = A == 0 ? -d_relative_lapse : 0.0;
      data.d_d[p][A] =
          mu_l*(d_log_volume*relative_normal[A]
                + log_volume*d_relative_normal)
          - mu_s*(d_relative_velocity_lower/relative_lapse
                  - relative_velocity_lower[A]*d_relative_lapse
                    /relative_lapse2);
      data.correction[A] = data.window*data.d[A];
      data.d_correction[p][A] =
          d_window[p]*data.d[A] + data.window*data.d_d[p][A];
    }
  }

  data.valid = Kokkos::isfinite(relative_lapse)
               && Kokkos::isfinite(log_volume);
  for (int A = 0; A < 4; ++A) {
    data.valid = data.valid && Kokkos::isfinite(data.d[A])
                 && Kokkos::isfinite(data.correction[A]);
    for (int p = 0; p < 4; ++p) {
      data.valid = data.valid && Kokkos::isfinite(data.d_d[p][A])
                   && Kokkos::isfinite(data.d_correction[p][A]);
    }
  }
  return data.valid;
}

template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real RelativeGaugeCoframeDerivative(const Reference &reference, const int p,
                                    const int A, const int a) {
  Real derivative = 0.0;
  for (int B = 0; B < 4; ++B) {
    for (int b = 0; b < 4; ++b) {
      derivative -= ReferenceCoframe(reference, B, a)
                    *ReferenceDFrame(reference, p, B, b)
                    *ReferenceCoframe(reference, A, b);
    }
  }
  return derivative;
}

// Add only the regular J_a=bar_theta^A_a W D_A increment to the already
// assembled wave-map scalar source.  This is the direct derivative path
// required by the relative gauge; it does not call the hyperbolic gauge driver
// and does not form a difference of two singular ordinary gauge sources.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool AddRelativeDampedGaugeSource(
    const Real psi[4][4], const Real d_psi[4][4][4],
    const Real metric[4][4],
    const Reference &reference, const CoordinateGhGeometry &geometry,
    const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z,
    const Real r0, const Real r1, const Real mu_l, const Real mu_s,
    const Real gamma0, Real source[4][4],
    RelativeDampedGaugeDiagnostics *diagnostics = nullptr) {
  RelativeDampedGaugeData data;
  if (!ComputeRelativeDampedGaugeData(
          psi, d_psi, x, y, z, center_x, center_y, center_z,
          r0, r1, mu_l, mu_s, data)) {
    return false;
  }
  Real coordinate_correction[4] = {};       // NOLINT(runtime/arrays)
  Real d_coordinate_correction[4][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int A = 0; A < 4; ++A) {
      coordinate_correction[a] +=
          ReferenceCoframe(reference, A, a)*data.correction[A];
      for (int p = 0; p < 4; ++p) {
        d_coordinate_correction[p][a] +=
            RelativeGaugeCoframeDerivative(reference, p, A, a)
              *data.correction[A]
            + ReferenceCoframe(reference, A, a)*data.d_correction[p][A];
      }
    }
  }

  Real coordinate_extra[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real nabla_ab = d_coordinate_correction[a][b];
      Real nabla_ba = d_coordinate_correction[b][a];
      for (int c = 0; c < 4; ++c) {
        nabla_ab -= geometry.christoffel[c][a][b]*coordinate_correction[c];
        nabla_ba -= geometry.christoffel[c][b][a]*coordinate_correction[c];
      }
      coordinate_extra[a][b] = -nabla_ab - nabla_ba;
      for (int c = 0; c < 4; ++c) {
        const Real projector = ((c == a) ? geometry.normal_lower[b] : 0.0)
                               + ((c == b) ? geometry.normal_lower[a] : 0.0)
                               - metric[a][b]*geometry.normal_upper[c];
        coordinate_extra[a][b] +=
            gamma0*projector*coordinate_correction[c];
      }
    }
  }

  Real source_max = 0.0;
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      Real frame_extra = 0.0;
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          frame_extra += ReferenceFrame(reference, A, a)
                         *ReferenceFrame(reference, B, b)
                         *coordinate_extra[a][b];
        }
      }
      source[A][B] += frame_extra;
      source_max = fmax(source_max, Kokkos::abs(frame_extra));
    }
  }
  if (diagnostics != nullptr) {
    diagnostics->d_max = 0.0;
    diagnostics->correction_max = 0.0;
    diagnostics->source_max = source_max;
    for (int A = 0; A < 4; ++A) {
      diagnostics->d_max = fmax(diagnostics->d_max, Kokkos::abs(data.d[A]));
      diagnostics->correction_max = fmax(
          diagnostics->correction_max, Kokkos::abs(data.correction[A]));
    }
  }
  return true;
}

}  // namespace ref_gh

#endif  // REF_GH_RELATIVE_DAMPED_GAUGE_HPP_
