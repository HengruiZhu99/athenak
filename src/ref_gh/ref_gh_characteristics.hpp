//========================================================================================
// Principal-symbol helpers for the 50-field reference-frame FO-GH system.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REF_GH_CHARACTERISTICS_HPP_
#define REF_GH_REF_GH_CHARACTERISTICS_HPP_

#include "athena.hpp"

namespace ref_gh {

struct CharacteristicSpeeds {
  Real metric;
  Real transverse;
  Real plus;
  Real minus;
};

// Characteristic map for one symmetric spacetime component.  The complete
// system consists of ten identical copies.  The covector s_I is normalized by
// G^{IJ}s_I s_J=1, where G^{IJ} is the inverse physical spatial metric in the
// reference frame.
struct GhCharacteristicFields {
  Real metric;
  Real plus;
  Real minus;
  Real transverse[3];  // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
GhCharacteristicFields ToGhCharacteristicFields(
    const Real psi, const Real pi, const Real phi[3], const Real gamma2,
    const Real inverse_spatial_metric[3][3], const Real s_cov[3]) {
  Real s_upper[3] = {};  // NOLINT(runtime/arrays)
  Real normal_phi = 0.0;
  for (int I = 0; I < 3; ++I) {
    for (int J = 0; J < 3; ++J) {
      s_upper[I] += inverse_spatial_metric[I][J]*s_cov[J];
    }
    normal_phi += s_upper[I]*phi[I];
  }
  GhCharacteristicFields characteristic;
  characteristic.metric = psi;
  characteristic.plus = pi + normal_phi - gamma2*psi;
  characteristic.minus = pi - normal_phi - gamma2*psi;
  for (int I = 0; I < 3; ++I) {
    characteristic.transverse[I] = phi[I] - s_cov[I]*normal_phi;
  }
  return characteristic;
}

KOKKOS_INLINE_FUNCTION
void FromGhCharacteristicFields(
    const GhCharacteristicFields &characteristic, const Real gamma2,
    const Real s_cov[3], Real &psi, Real &pi, Real phi[3]) {
  psi = characteristic.metric;
  pi = 0.5*(characteristic.plus + characteristic.minus) + gamma2*psi;
  const Real normal_phi = 0.5*(characteristic.plus - characteristic.minus);
  for (int I = 0; I < 3; ++I) {
    phi[I] = characteristic.transverse[I] + s_cov[I]*normal_phi;
  }
}

KOKKOS_INLINE_FUNCTION
Real GhSymmetrizerEnergy(const Real psi, const Real pi, const Real phi[3],
                         const Real gamma2, const Real lambda_squared,
                         const Real inverse_spatial_metric[3][3]) {
  Real energy = lambda_squared*psi*psi + pi*pi - 2.0*gamma2*psi*pi;
  for (int I = 0; I < 3; ++I) {
    for (int J = 0; J < 3; ++J) {
      energy += inverse_spatial_metric[I][J]*phi[I]*phi[J];
    }
  }
  return energy;
}

// s_cov is normalized with G^{IJ}s_I s_J = 1.  beta_ref contains coframe
// components beta_ref^I, hence beta^s = beta_ref^I s_I.
KOKKOS_INLINE_FUNCTION
CharacteristicSpeeds GetCharacteristicSpeeds(const Real alpha,
                                              const Real beta_ref[3],
                                              const Real s_cov[3]) {
  Real beta_s = 0.0;
  for (int i = 0; i < 3; ++i) beta_s += beta_ref[i]*s_cov[i];
  return {0.0, -beta_s, -beta_s + alpha, -beta_s - alpha};
}

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_CHARACTERISTICS_HPP_
