//========================================================================================
//! \file exact_matched_state.hpp
//! \brief Strict predicate for the exact stationary q=1 reference state.
//========================================================================================
#ifndef REF_GH_EXACT_MATCHED_STATE_HPP_
#define REF_GH_EXACT_MATCHED_STATE_HPP_

#include "athena.hpp"

namespace ref_gh {

// This is intentionally an exact predicate, not a tolerance or an inner-radius
// switch.  It may select algebraic exact fills and identities only when the
// physical stationary trumpet and the q-controlled reference are mathematically
// identical and no controller can move the reference during the stage.
KOKKOS_INLINE_FUNCTION
bool IsExactMatchedQ1StaticReference(
    const bool reference_q_controlled, const bool q_controller_enabled,
    const bool q_prescribed_enabled, const bool reference_time_dependent,
    const Real q, const Real q_dot, const Real q_ddot) {
  return reference_q_controlled
      && !q_controller_enabled
      && !q_prescribed_enabled
      && !reference_time_dependent
      && q == 1.0
      && q_dot == 0.0
      && q_ddot == 0.0;
}

}  // namespace ref_gh

#endif  // REF_GH_EXACT_MATCHED_STATE_HPP_
