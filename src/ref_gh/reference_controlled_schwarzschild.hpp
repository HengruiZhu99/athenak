//========================================================================================
//! \file reference_controlled_schwarzschild.hpp
//! \brief Analytic wormhole and localized wormhole-to-trumpet reference profiles.
//========================================================================================
#ifndef REF_GH_REFERENCE_CONTROLLED_SCHWARZSCHILD_HPP_
#define REF_GH_REFERENCE_CONTROLLED_SCHWARZSCHILD_HPP_

#include "athena.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

namespace ref_gh {

struct ControlledReferenceParameters {
  Real mass;
  Real center[3];  // NOLINT(runtime/arrays)
  Real r_core0;
  Real tau_core;
  Real kappa_core;
  int transition_path;
  Real transition_width;
  Real tau_transition;
  int activation_mode;
  Real xi;
  Real xi_dot;
  Real xi_ddot;
  Real regularization_outer_start;
  Real regularization_outer_end;
  Real delta_q;
  Real delta_q_dot;
  Real delta_q_ddot;
  Real delta_p;
  Real delta_p_dot;
  Real delta_p_ddot;
};

enum ControlledReferencePath : int {
  kShrinkingWidthPath = 0,
  kFixedCorePath = 1,
  kFixedWidthPath = 2
};

enum ControlledActivationMode : int {
  kLegacyTimeActivation = 0,
  kContinuationActivation = 1
};

KOKKOS_INLINE_FUNCTION
ReferenceJet QuinticSmoothstep(const ReferenceJet &argument) {
  if (argument.value <= 0.0) return ConstantJet(0.0);
  if (argument.value >= 1.0) return ConstantJet(1.0);
  const ReferenceJet x2 = argument*argument;
  const ReferenceJet x3 = x2*argument;
  const ReferenceJet polynomial = ConstantJet(10.0)
      + argument*(ConstantJet(-15.0) + ConstantJet(6.0)*argument);
  return x3*polynomial;
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ControlledRadiusJet(const Real x, const Real y, const Real z,
                                 const Real center_x, const Real center_y,
                                 const Real center_z) {
  const ReferenceJet dx = CoordinateJet(x - center_x, 1);
  const ReferenceJet dy = CoordinateJet(y - center_y, 2);
  const ReferenceJet dz = CoordinateJet(z - center_z, 3);
  return Sqrt(dx*dx + dy*dy + dz*dz);
}

KOKKOS_INLINE_FUNCTION
void WormholeProfileJets(const Real mass, const Real x, const Real y,
                         const Real z, const Real center_x,
                         const Real center_y, const Real center_z,
                         ReferenceJet &alpha, ReferenceJet &psi2,
                         ReferenceJet &shift_q) {
  const ReferenceJet radius = ControlledRadiusJet(
      x, y, z, center_x, center_y, center_z);
  const ReferenceJet psi_w = ConstantJet(1.0)
      + ConstantJet(0.5*mass)*Reciprocal(radius);
  psi2 = psi_w*psi_w;
  alpha = Reciprocal(psi2);
  shift_q = ConstantJet(0.0);
}

KOKKOS_INLINE_FUNCTION
ReferenceJet ControllerJet(const Real value, const Real first,
                           const Real second) {
  ReferenceJet result = ConstantJet(value);
  result.d[0] = first;
  result.dd[0][0] = second;
  return result;
}

KOKKOS_INLINE_FUNCTION
void ControlledTransitionProfileJets(
    const DvceArray2D<Real> &table, const ControlledReferenceParameters &params,
    const Real time, const Real x, const Real y, const Real z,
    ReferenceJet &alpha, ReferenceJet &psi2, ReferenceJet &shift_q,
    ReferenceJet *r_core_out = nullptr,
    ReferenceJet *activation_out = nullptr,
    ReferenceJet *core_blend_out = nullptr) {
  const Real dx[3] = {x - params.center[0], y - params.center[1],
                      z - params.center[2]};
  const Real radius_value = Kokkos::sqrt(dx[0]*dx[0] + dx[1]*dx[1]
                                         + dx[2]*dx[2]);
  const ReferenceJet radius = ControlledRadiusJet(
      x, y, z, params.center[0], params.center[1], params.center[2]);
  const ReferenceJet time_jet = CoordinateJet(time, 0);
  const ReferenceJet r_core = params.transition_path == kFixedCorePath
      ? ConstantJet(params.r_core0*params.mass)
      : ConstantJet(params.r_core0*params.mass)
          *Exp(ConstantJet(-1.0/(params.tau_core*params.mass))*time_jet);
  const ReferenceJet transition_coordinate =
      params.transition_path == kFixedWidthPath
      ? (radius + (-r_core))*ConstantJet(
            1.0/(params.transition_width*params.mass))
      : (radius*Reciprocal(r_core) + ConstantJet(-1.0))
          *ConstantJet(1.0/params.kappa_core);
  const ReferenceJet core_blend = QuinticSmoothstep(transition_coordinate);
  const ReferenceJet activation_argument =
      params.activation_mode == kContinuationActivation
      ? ControllerJet(params.xi, params.xi_dot, params.xi_ddot)
      : time_jet*ConstantJet(1.0/(params.tau_transition*params.mass));
  const ReferenceJet activation = QuinticSmoothstep(activation_argument);
  const ReferenceJet blend = activation*core_blend;

  ReferenceJet alpha_w;
  ReferenceJet psi2_w;
  ReferenceJet shift_w;
  WormholeProfileJets(params.mass, x, y, z, params.center[0],
                      params.center[1], params.center[2],
                      alpha_w, psi2_w, shift_w);
  const Real rho = radius_value/params.mass;
  const ReferenceJet alpha_t = RadialJet(
      InterpolateTrumpetProfile(table, kCoeffAlpha, rho), params.mass,
      dx, radius_value);
  const ReferenceJet psi2_t = RadialJet(
      ArealRadiusToPsi2(
          InterpolateTrumpetProfile(table, kCoeffArealRadius, rho), rho),
      params.mass, dx, radius_value);
  RadialProfile shift_profile =
      InterpolateTrumpetProfile(table, kCoeffShiftQ, rho);
  shift_profile.value /= params.mass;
  shift_profile.d1 /= params.mass;
  shift_profile.d2 /= params.mass;
  const ReferenceJet shift_t = RadialJet(
      shift_profile, params.mass, dx, radius_value);

  const ReferenceJet one_minus_blend = ConstantJet(1.0) + (-blend);
  ReferenceJet log_alpha = one_minus_blend*Log(alpha_w) + blend*Log(alpha_t);
  ReferenceJet log_psi2 = one_minus_blend*Log(psi2_w) + blend*Log(psi2_t);
  shift_q = blend*shift_t;

  const ReferenceJet outer_coordinate =
      (radius + ConstantJet(-params.regularization_outer_start*params.mass))
      *ConstantJet(1.0/((params.regularization_outer_end
                         - params.regularization_outer_start)*params.mass));
  const ReferenceJet outer_cutoff = ConstantJet(1.0)
                                    + (-QuinticSmoothstep(outer_coordinate));
  const ReferenceJet regularization_window = core_blend*outer_cutoff;
  const ReferenceJet log_radius = Log(
      radius*ConstantJet(1.0/params.mass));
  const ReferenceJet delta_q = ControllerJet(
      params.delta_q, params.delta_q_dot, params.delta_q_ddot);
  const ReferenceJet delta_p = ControllerJet(
      params.delta_p, params.delta_p_dot, params.delta_p_ddot);
  log_psi2 = log_psi2
             + (-(delta_q*regularization_window*log_radius));
  log_alpha = log_alpha + delta_p*regularization_window*log_radius;
  psi2 = Exp(log_psi2);
  alpha = Exp(log_alpha);
  if (r_core_out != nullptr) *r_core_out = r_core;
  if (activation_out != nullptr) *activation_out = activation;
  if (core_blend_out != nullptr) *core_blend_out = core_blend;
}

KOKKOS_INLINE_FUNCTION
void PopulateIsotropicReferenceGeometry(
    const ReferenceJet &alpha, const ReferenceJet &psi2,
    const ReferenceJet &shift_q, const Real x, const Real y, const Real z,
    const Real center_x, const Real center_y, const Real center_z,
    ReferenceGeometry &reference) {
  ZeroReferenceGeometry(reference);
  const ReferenceJet inverse_alpha = Reciprocal(alpha);
  const ReferenceJet inverse_psi2 = Reciprocal(psi2);
  const ReferenceJet coordinates[3] = {
      CoordinateJet(x - center_x, 1), CoordinateJet(y - center_y, 2),
      CoordinateJet(z - center_z, 3)};
  ReferenceJet coframe[4][4];  // NOLINT(runtime/arrays)
  ReferenceJet frame[4][4];    // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      coframe[A][a] = ConstantJet(0.0);
      frame[A][a] = ConstantJet(0.0);
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
  ReferenceJet metric[4][4];          // NOLINT(runtime/arrays)
  ReferenceJet inverse_metric[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = -(coframe[0][a]*coframe[0][b]);
      inverse_metric[a][b] = -(frame[0][a]*frame[0][b]);
      for (int I = 1; I < 4; ++I) {
        metric[a][b] = metric[a][b] + coframe[I][a]*coframe[I][b];
        inverse_metric[a][b] =
            inverse_metric[a][b] + frame[I][a]*frame[I][b];
      }
      reference.metric[a][b] = metric[a][b].value;
      reference.inverse_metric[a][b] = inverse_metric[a][b].value;
      reference.coframe[a][b] = coframe[a][b].value;
      reference.frame[a][b] = frame[a][b].value;
      for (int p = 0; p < 4; ++p) {
        reference.d_metric[p][a][b] = metric[a][b].d[p];
        reference.d_frame[p][a][b] = frame[a][b].d[p];
        for (int q = 0; q < 4; ++q) {
          reference.dd_metric[p][q][a][b] = metric[a][b].dd[p][q];
          reference.dd_frame[p][q][a][b] = frame[a][b].dd[p][q];
        }
      }
    }
  }
  for (int I = 0; I < 3; ++I) {
    reference.spatial_frame[I][I] = inverse_psi2.value;
    reference.spatial_coframe[I][I] = psi2.value;
    reference.dt_spatial_frame[I][I] = inverse_psi2.d[0];
    for (int J = 0; J < 3; ++J) {
      for (int K = 0; K < 3; ++K) {
        reference.structure[I][J][K] =
            ((J == K) ? inverse_psi2.d[I + 1] : 0.0)
            - ((I == K) ? inverse_psi2.d[J + 1] : 0.0);
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
  CompleteReferenceFrameGeometry(reference);
}

struct WormholeSchwarzschildReference {
  Real mass;
  Real center[3];  // NOLINT(runtime/arrays)

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real, const Real x, const Real y, const Real z,
                ReferenceGeometry &reference) const {
    ReferenceJet alpha;
    ReferenceJet psi2;
    ReferenceJet shift_q;
    WormholeProfileJets(mass, x, y, z, center[0], center[1], center[2],
                        alpha, psi2, shift_q);
    PopulateIsotropicReferenceGeometry(alpha, psi2, shift_q, x, y, z,
                                       center[0], center[1], center[2], reference);
  }
};

struct ControlledSchwarzschildReference {
  DvceArray2D<Real> table;
  ControlledReferenceParameters params;

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real time, const Real x, const Real y, const Real z,
                ReferenceGeometry &reference) const {
    ReferenceJet alpha;
    ReferenceJet psi2;
    ReferenceJet shift_q;
    ControlledTransitionProfileJets(table, params, time, x, y, z,
                                    alpha, psi2, shift_q);
    PopulateIsotropicReferenceGeometry(
        alpha, psi2, shift_q, x, y, z, params.center[0], params.center[1],
        params.center[2], reference);
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_CONTROLLED_SCHWARZSCHILD_HPP_
