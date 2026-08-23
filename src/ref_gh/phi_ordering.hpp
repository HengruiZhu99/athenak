//========================================================================================
//! \file phi_ordering.hpp
//! \brief Exact curl-constraint correction between compatible and standard GH ordering.
//========================================================================================
#ifndef REF_GH_PHI_ORDERING_HPP_
#define REF_GH_PHI_ORDERING_HPP_

#include "athena.hpp"

namespace ref_gh {

// For Phi_I = E_I(Psi), integrability in a non-coordinate spatial frame is
//
//   C_IJ = E_I(Phi_J) - E_J(Phi_I) - c^K_IJ Phi_K = 0,
//
// where [E_I,E_J] = c^K_IJ E_K.  Differentiating the Psi equation produces the
// compatible principal term beta^J E_I(Phi_J).  The standard GH ordering uses
// beta^J E_J(Phi_I), so the exact change, including the frame commutator, is
//
//   (partial_t Phi_I)_standard
//       = (partial_t Phi_I)_compatible - beta^J C_IJ.
//
// `frame_derivative[I][J]` stores E_I(Phi_J) for one symmetric spacetime
// component.  The helper is deliberately point-local and backend-portable.
KOKKOS_INLINE_FUNCTION
Real PhiCurlConstraint(const int I, const int J,
                       const Real frame_derivative[3][3],
                       const Real structure[3][3][3],
                       const Real phi[3]) {
  Real curl = frame_derivative[I][J] - frame_derivative[J][I];
  for (int K = 0; K < 3; ++K) curl -= structure[I][J][K]*phi[K];
  return curl;
}

KOKKOS_INLINE_FUNCTION
Real StandardPhiOrderingCorrection(const int I, const Real beta_frame[3],
                                   const Real frame_derivative[3][3],
                                   const Real structure[3][3][3],
                                   const Real phi[3]) {
  Real correction = 0.0;
  for (int J = 0; J < 3; ++J) {
    correction += beta_frame[J]
                  *PhiCurlConstraint(I, J, frame_derivative, structure, phi);
  }
  return correction;
}

}  // namespace ref_gh

#endif  // REF_GH_PHI_ORDERING_HPP_
