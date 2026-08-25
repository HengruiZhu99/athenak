//========================================================================================
//! \file reference_generic_singular.hpp
//! \brief Gaussian-localized generic singular reference geometry.
//========================================================================================
#ifndef REF_GH_REFERENCE_GENERIC_SINGULAR_HPP_
#define REF_GH_REFERENCE_GENERIC_SINGULAR_HPP_

#include "athena.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"

namespace ref_gh {

struct GenericSingularReferenceParameters {
  Real mass;
  Real center[3];  // NOLINT(runtime/arrays)
  Real gaussian_width;
  Real q_initial;
  Real q_final;
  Real transition_time;
};

KOKKOS_INLINE_FUNCTION
ReferenceJet PrescribedGenericSingularQJet(
    const GenericSingularReferenceParameters &params, const Real time) {
  const ReferenceJet transition_coordinate =
      CoordinateJet(time, 0)*ConstantJet(
          1.0/(params.transition_time*params.mass));
  const ReferenceJet blend = QuinticSmoothstep(transition_coordinate);
  return ConstantJet(params.q_initial)
         + ConstantJet(params.q_final - params.q_initial)*blend;
}

KOKKOS_INLINE_FUNCTION
void GenericSingularProfileJets(
    const GenericSingularReferenceParameters &params, const Real time,
    const Real x, const Real y, const Real z, ReferenceJet &alpha,
    ReferenceJet &spatial_cholesky, ReferenceJet &shift_q,
    ReferenceJet *q_out = nullptr, ReferenceJet *window_out = nullptr) {
  const ReferenceJet radius = ControlledRadiusJet(
      x, y, z, params.center[0], params.center[1], params.center[2]);
  const ReferenceJet rho = radius*ConstantJet(1.0/params.mass);
  const ReferenceJet scaled_radius =
      radius*ConstantJet(1.0/(params.gaussian_width*params.mass));
  const ReferenceJet window = Exp(-(scaled_radius*scaled_radius));
  const ReferenceJet q = PrescribedGenericSingularQJet(params, time);
  spatial_cholesky = Exp(-(q*window*Log(rho)));
  alpha = ConstantJet(1.0);
  shift_q = ConstantJet(0.0);
  if (q_out != nullptr) *q_out = q;
  if (window_out != nullptr) *window_out = window;
}

struct GenericSingularReference {
  GenericSingularReferenceParameters params;

  KOKKOS_INLINE_FUNCTION
  void Populate(const Real time, const Real x, const Real y, const Real z,
                ReferenceGeometry &reference) const {
    ReferenceJet alpha;
    ReferenceJet spatial_cholesky;
    ReferenceJet shift_q;
    GenericSingularProfileJets(params, time, x, y, z, alpha,
                               spatial_cholesky, shift_q);
    PopulateIsotropicReferenceGeometry(
        alpha, spatial_cholesky, shift_q, x, y, z, params.center[0],
        params.center[1], params.center[2], reference);
  }
};

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_GENERIC_SINGULAR_HPP_
