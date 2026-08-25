//========================================================================================
//! \file gauge_driver.hpp
//! \brief Improved Lindblom--Szilagyi driver in reference-frame components.
//========================================================================================
#ifndef REF_GH_GAUGE_DRIVER_HPP_
#define REF_GH_GAUGE_DRIVER_HPP_

#include "athena.hpp"
#include "ref_gh/reference_cache.hpp"

namespace ref_gh {

struct GaugeDriverRhs {
  Real hhat[4];    // NOLINT(runtime/arrays)
  Real theta[4];   // NOLINT(runtime/arrays)
  Real upsilon[3]; // NOLINT(runtime/arrays)
};

// Omega_{A lambda}^B=(partial_lambda e_A^mu) theta^B_mu is the
// component-rotation matrix for a covector projected on the reference frame.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real ReferenceFrameMotion(const Reference &reference, const int A,
                          const int lambda, const int B) {
  Real omega = 0.0;
  for (int a = 0; a < 4; ++a) {
    omega += ReferenceDFrame(reference, lambda, A, a)
             *ReferenceCoframe(reference, B, a);
  }
  return omega;
}

// Eqs. (9) and (11) of Lindblom--Szilagyi (2009), translated from
// coordinate covectors to Hhat_A=e_A^mu Hhat_mu and theta_A=e_A^mu theta_mu:
//
//   dt Hhat_A = beta^i partial_i Hhat_A - mu(Hhat_A-F_A) + theta_A
//               +(Omega_At^B-beta^i Omega_Ai^B)Hhat_B,
//   dt theta_A = -eta theta_A - eta beta^i partial_i Hhat_A
//                +Omega_At^B theta_B
//                +eta beta^i Omega_Ai^B Hhat_B.
//
// Upsilon is the coordinate-vector auxiliary of Eqs. (60)--(62) in
// Lindblom et al. (2008) and has a zero-speed relaxation equation.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
GaugeDriverRhs ComputeGaugeDriverRhs(
    const Reference &reference, const Real hhat[4], const Real theta[4],
    const Real upsilon[3], const Real d_hhat[3][4], const Real shift[3],
    const Real target_hhat[4], const Real conformal_gamma[3], const Real mu,
    const Real eta, const Real eta_beta) {
  GaugeDriverRhs rhs{};
  for (int A = 0; A < 4; ++A) {
    Real shift_d_hhat = 0.0;
    for (int p = 0; p < 3; ++p) shift_d_hhat += shift[p]*d_hhat[p][A];
    rhs.hhat[A] = shift_d_hhat - mu*(hhat[A] - target_hhat[A]) + theta[A];
    rhs.theta[A] = -eta*theta[A] - eta*shift_d_hhat;
    for (int B = 0; B < 4; ++B) {
      const Real omega_t = ReferenceFrameMotion(reference, A, 0, B);
      rhs.hhat[A] += omega_t*hhat[B];
      rhs.theta[A] += omega_t*theta[B];
      for (int p = 0; p < 3; ++p) {
        const Real beta_omega = shift[p]
            *ReferenceFrameMotion(reference, A, p + 1, B);
        rhs.hhat[A] -= beta_omega*hhat[B];
        rhs.theta[A] += eta*beta_omega*hhat[B];
      }
    }
  }
  for (int p = 0; p < 3; ++p) {
    rhs.upsilon[p] = conformal_gamma[p] - eta_beta*upsilon[p];
  }
  return rhs;
}

}  // namespace ref_gh

#endif  // REF_GH_GAUGE_DRIVER_HPP_
