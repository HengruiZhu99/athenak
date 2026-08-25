//========================================================================================
// Principal-symbol helpers for the 50-field reference-frame FO-GH system.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REF_GH_CHARACTERISTICS_HPP_
#define REF_GH_REF_GH_CHARACTERISTICS_HPP_

#include "athena.hpp"
#include "ref_gh/ref_gh_state.hpp"

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

// Complete Lindblom--Szilagyi (2009) Appendix-B characteristic map for the
// 50 Einstein fields, Hhat_A, theta_A, and the three zero-speed Upsilon fields.
// s_cov[I] is spatial and normalized by G^{IJ}s_I s_J=1.  s_frame[A]
// contains the four reference-frame components of the same physical spacetime
// covector, including its generally nonzero A=0 component.
struct CombinedGhCharacteristicFields {
  Real metric[kSymmetric4Size];                  // NOLINT(runtime/arrays)
  Real plus[kSymmetric4Size];                    // NOLINT(runtime/arrays)
  Real minus[kSymmetric4Size];                   // NOLINT(runtime/arrays)
  Real transverse[3][kSymmetric4Size];           // NOLINT(runtime/arrays)
  Real gauge_advective[4];                       // NOLINT(runtime/arrays)
  Real gauge_zero[4];                            // NOLINT(runtime/arrays)
  Real upsilon_zero[3];                          // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
CombinedGhCharacteristicFields ToCombinedGhCharacteristicFields(
    const Real psi[kSymmetric4Size], const Real pi[kSymmetric4Size],
    const Real phi[3][kSymmetric4Size], const Real hhat[4],
    const Real theta[4], const Real upsilon[3], const Real gamma2,
    const Real eta, const Real inverse_spatial_metric[3][3],
    const Real s_cov[3], const Real s_frame[4]) {
  Real s_upper[3] = {};  // NOLINT(runtime/arrays)
  for (int I = 0; I < 3; ++I) {
    for (int J = 0; J < 3; ++J) {
      s_upper[I] += inverse_spatial_metric[I][J]*s_cov[J];
    }
  }
  CombinedGhCharacteristicFields characteristic{};
  for (int A = 0; A < 4; ++A) {
    characteristic.gauge_advective[A] = hhat[A];
    characteristic.gauge_zero[A] = theta[A] + eta*hhat[A];
  }
  for (int I = 0; I < 3; ++I) {
    characteristic.upsilon_zero[I] = upsilon[I];
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      const int component = Symmetric4Index(A, B);
      Real normal_phi = 0.0;
      for (int I = 0; I < 3; ++I) normal_phi += s_upper[I]*phi[I][component];
      const Real gauge_coupling = s_frame[A]*hhat[B] + s_frame[B]*hhat[A];
      characteristic.metric[component] = psi[component];
      characteristic.plus[component] =
          pi[component] + normal_phi - gamma2*psi[component] + gauge_coupling;
      characteristic.minus[component] =
          pi[component] - normal_phi - gamma2*psi[component] - gauge_coupling;
      for (int I = 0; I < 3; ++I) {
        characteristic.transverse[I][component] =
            phi[I][component] - s_cov[I]*normal_phi;
      }
    }
  }
  return characteristic;
}

KOKKOS_INLINE_FUNCTION
void FromCombinedGhCharacteristicFields(
    const CombinedGhCharacteristicFields &characteristic, const Real gamma2,
    const Real eta, const Real s_cov[3], const Real s_frame[4],
    Real psi[kSymmetric4Size], Real pi[kSymmetric4Size],
    Real phi[3][kSymmetric4Size], Real hhat[4], Real theta[4],
    Real upsilon[3]) {
  for (int A = 0; A < 4; ++A) {
    hhat[A] = characteristic.gauge_advective[A];
    theta[A] = characteristic.gauge_zero[A] - eta*hhat[A];
  }
  for (int I = 0; I < 3; ++I) {
    upsilon[I] = characteristic.upsilon_zero[I];
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = A; B < 4; ++B) {
      const int component = Symmetric4Index(A, B);
      psi[component] = characteristic.metric[component];
      pi[component] = 0.5*(characteristic.plus[component]
                           + characteristic.minus[component])
                      + gamma2*psi[component];
      const Real normal_phi = 0.5*(characteristic.plus[component]
                                   - characteristic.minus[component])
                              - s_frame[A]*hhat[B] - s_frame[B]*hhat[A];
      for (int I = 0; I < 3; ++I) {
        phi[I][component] = characteristic.transverse[I][component]
                            + s_cov[I]*normal_phi;
      }
    }
  }
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
