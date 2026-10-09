#ifndef RESEARCH_DAMPING_PROFILE_HPP_
#define RESEARCH_DAMPING_PROFILE_HPP_

#include "Kokkos_Core.hpp"

namespace z4c { namespace hyperboloidal {

// Prescribed lower-order damping profile.  The production convention is
// kappa_1=kappa_input/alpha; kappa_2 alone changes here.  Thus
// kappa_input*(1+kappa_2)=2*S/a^2+(kappa_input-2*S/a^2)*Omega.
// This factored form gives kappa_2=0 exactly at Omega=1.  There are no floors
// or data falloff assumptions.  This helper is experimental, not a runtime
// option or a statement of variable-coefficient constraint stability.
template <typename T>
KOKKOS_INLINE_FUNCTION T ResearchKappa2Profile(
    T omega, T scri_radius, T curvature_radius, T kappa_input) {
  return ((T(2)*scri_radius/(curvature_radius*curvature_radius)-kappa_input)
          /kappa_input)*(T(1)-omega);
}

}}  // namespace z4c::hyperboloidal
#endif
