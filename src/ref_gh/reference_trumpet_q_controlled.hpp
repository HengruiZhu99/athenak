//========================================================================================
//! \file reference_trumpet_q_controlled.hpp
//! \brief Exact trumpet reference with one Gaussian-localized spatial exponent.
//========================================================================================
#ifndef REF_GH_REFERENCE_TRUMPET_Q_CONTROLLED_HPP_
#define REF_GH_REFERENCE_TRUMPET_Q_CONTROLLED_HPP_

#include "athena.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"

namespace ref_gh {

struct TrumpetQControlledReferenceParameters {
  Real mass;
  Real center[3];  // NOLINT(runtime/arrays)
  Real gaussian_width;
  Real q;
  Real q_dot;
  Real q_ddot;
};

// Modify only the exact trumpet spatial Cholesky factor,
//
//   L(q) = L_T exp[-(q-1) exp(-(r/R_G)^2) ln(r/M)].
//
// The exact trumpet lapse, shift, and every finite-radius trumpet profile are
// unchanged.  q, q_dot, and q_ddot are supplied by the coupled RK state.
KOKKOS_INLINE_FUNCTION
void TrumpetQControlledProfileJets(
    const DvceArray2D<Real> &table,
    const TrumpetQControlledReferenceParameters &params,
    const Real x, const Real y, const Real z, ReferenceJet &alpha,
    ReferenceJet &spatial_cholesky, ReferenceJet &shift_q,
    ReferenceJet *q_out = nullptr, ReferenceJet *window_out = nullptr) {
  const Real displacement[3] = {x - params.center[0], y - params.center[1],
                                z - params.center[2]};
  const Real radius_value = Kokkos::sqrt(
      displacement[0]*displacement[0] + displacement[1]*displacement[1]
      + displacement[2]*displacement[2]);
  const Real rho_value = radius_value/params.mass;
  const ReferenceJet radius = ControlledRadiusJet(
      x, y, z, params.center[0], params.center[1], params.center[2]);
  const ReferenceJet rho = radius*ConstantJet(1.0/params.mass);
  const ReferenceJet scaled_radius = radius*ConstantJet(
      1.0/(params.gaussian_width*params.mass));
  const ReferenceJet window = Exp(-(scaled_radius*scaled_radius));
  const ReferenceJet q = ControllerJet(params.q, params.q_dot, params.q_ddot);

  alpha = RadialJet(
      InterpolateTrumpetProfile(table, kCoeffAlpha, rho_value), params.mass,
      displacement, radius_value);
  const ReferenceJet trumpet_cholesky = RadialJet(
      ArealRadiusToPsi2(
          InterpolateTrumpetProfile(table, kCoeffArealRadius, rho_value),
          rho_value),
      params.mass, displacement, radius_value);
  RadialProfile shift_profile =
      InterpolateTrumpetProfile(table, kCoeffShiftQ, rho_value);
  shift_profile.value /= params.mass;
  shift_profile.d1 /= params.mass;
  shift_profile.d2 /= params.mass;
  shift_q = RadialJet(
      shift_profile, params.mass, displacement, radius_value);
  spatial_cholesky = trumpet_cholesky*Exp(
      -((q + ConstantJet(-1.0))*window*Log(rho)));
  if (q_out != nullptr) *q_out = q;
  if (window_out != nullptr) *window_out = window;
}

struct TrumpetQControlledReference {
  DvceArray2D<Real> table;
  TrumpetQControlledReferenceParameters params;

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real /*time*/, const Real x, const Real y, const Real z,
                ReferenceGeometry &reference) const {
    ReferenceJet alpha;
    ReferenceJet spatial_cholesky;
    ReferenceJet shift_q;
    TrumpetQControlledProfileJets(
        table, params, x, y, z, alpha, spatial_cholesky, shift_q);
    PopulateIsotropicReferenceGeometry(
        alpha, spatial_cholesky, shift_q, x, y, z, params.center[0],
        params.center[1], params.center[2], reference);
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_TRUMPET_Q_CONTROLLED_HPP_
