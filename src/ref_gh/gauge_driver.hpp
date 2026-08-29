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

template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real ReferenceFrameMotion(const Reference &reference, const int A,
                          const int lambda, const int B);

// Direct equations for the stored regular fields delta_Hhat and delta_theta.
// `exact_matched_static` is allowed only for a reference proved to satisfy
// Fref=Href, dt Href=dt theta_ref=Omega_t=0, and
// theta_ref=-beta_ref^i Kref_i.  It removes an analytically zero reference
// source; it is not a near-puncture numerical switch.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
GaugeDriverRhs ComputeGaugeDriverResidualRhs(
    const Reference &reference, const Real reference_hhat[4],
    const Real reference_theta[4],
    const Real reference_d_hhat[4][4],
    const Real reference_dt_theta[4], const Real delta_hhat[4],
    const Real delta_theta[4], const Real upsilon[3],
    const Real d_delta_hhat[3][4], const Real physical_shift[3],
    const Real reference_shift[3], const Real delta_shift[3],
    const Real delta_target_hhat[4], const Real reference_target_hhat[4],
    const Real conformal_gamma_residual[3], const Real mu, const Real eta,
    const Real eta_beta, const bool exact_matched_static) {
  GaugeDriverRhs rhs{};
  Real reference_k[3][4];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    for (int A = 0; A < 4; ++A) {
      reference_k[i][A] = reference_d_hhat[i + 1][A];
      for (int B = 0; B < 4; ++B) {
        reference_k[i][A] -=
            ReferenceFrameMotion(reference, A, i + 1, B)*reference_hhat[B];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    Real advective_delta_hhat = 0.0;
    for (int i = 0; i < 3; ++i) {
      advective_delta_hhat += physical_shift[i]*d_delta_hhat[i][A];
    }
    rhs.hhat[A] = advective_delta_hhat
                  - mu*(delta_hhat[A] - delta_target_hhat[A])
                  + delta_theta[A];
    rhs.theta[A] = -eta*delta_theta[A] - eta*advective_delta_hhat;
    for (int B = 0; B < 4; ++B) {
      const Real omega_t = ReferenceFrameMotion(reference, A, 0, B);
      rhs.hhat[A] += omega_t*delta_hhat[B];
      rhs.theta[A] += omega_t*delta_theta[B];
      for (int i = 0; i < 3; ++i) {
        const Real beta_omega = physical_shift[i]
            *ReferenceFrameMotion(reference, A, i + 1, B);
        rhs.hhat[A] -= beta_omega*delta_hhat[B];
        rhs.theta[A] += eta*beta_omega*delta_hhat[B];
      }
    }
    for (int i = 0; i < 3; ++i) {
      rhs.hhat[A] += delta_shift[i]*reference_k[i][A];
      rhs.theta[A] -= eta*delta_shift[i]*reference_k[i][A];
    }
    if (!exact_matched_static) {
      Real reference_h_source =
          -mu*(reference_hhat[A] - reference_target_hhat[A])
          + reference_theta[A] - reference_d_hhat[0][A];
      Real reference_theta_source =
          -eta*reference_theta[A] - reference_dt_theta[A];
      for (int i = 0; i < 3; ++i) {
        reference_h_source += reference_shift[i]*reference_k[i][A];
        reference_theta_source -=
            eta*reference_shift[i]*reference_k[i][A];
      }
      for (int B = 0; B < 4; ++B) {
        const Real omega_t = ReferenceFrameMotion(reference, A, 0, B);
        reference_h_source += omega_t*reference_hhat[B];
        reference_theta_source += omega_t*reference_theta[B];
      }
      rhs.hhat[A] += reference_h_source;
      rhs.theta[A] += reference_theta_source;
    }
  }
  for (int i = 0; i < 3; ++i) {
    rhs.upsilon[i] = conformal_gamma_residual[i] - eta_beta*upsilon[i];
  }
  return rhs;
}

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
