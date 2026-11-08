//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef DRIVER_CLASSICAL_RK4_HPP_
#define DRIVER_CLASSICAL_RK4_HPP_

#include "athena.hpp"

namespace time_integration {
// Classical four-stage RK4. This is distinct from the existing rk4 2S method.
// Stages are one-based, as in Driver tasks. A module owns an immutable cycle
// base, current stage state, RHS and weighted-sum register. Only independent
// variables are advanced; its algebraic/floor projection follows every stage.
struct ClassicalRK4 {
  static constexpr int stages=4;

  KOKKOS_INLINE_FUNCTION static Real RHSFraction(int stage) {
    KOKKOS_ASSERT(stage>=1 && stage<=stages);
    return stage==1 ? Real(0) : (stage==4 ? Real(1) : Real(.5));
  }
  KOKKOS_INLINE_FUNCTION static Real NextFraction(int stage) {
    KOKKOS_ASSERT(stage>=1 && stage<=stages);
    return stage<3 ? Real(.5) : Real(1);
  }
  KOKKOS_INLINE_FUNCTION static Real Weight(int stage) {
    KOKKOS_ASSERT(stage>=1 && stage<=stages);
    return stage==1 || stage==4 ? Real(1)/6 : Real(1)/3;
  }
  KOKKOS_INLINE_FUNCTION static Real RHSTime(Real time,Real dt,int stage) {
    return stage==1 ? time : time+RHSFraction(stage)*dt;
  }
  KOKKOS_INLINE_FUNCTION static Real StateTime(Real time,Real dt,int stage) {
    return time+NextFraction(stage)*dt;
  }

  KOKKOS_INLINE_FUNCTION static Real Advance(int stage,Real dt,Real base,Real rhs,
                                            Real &weighted_sum) {
    KOKKOS_ASSERT(stage>=1 && stage<=stages);
    // Accumulating unscaled derivatives preserves the original N12 evaluation
    // order k1+2*k2+2*k3+k4, followed by dt*sum/6. N12 spherical_policy does not
    // use the compensated acceptance of the unrelated legacy formulations.
    if (stage==1) weighted_sum=rhs;
    else weighted_sum+=stage==4 ? rhs : 2*rhs;
    if (stage==4) return base+dt*weighted_sum/6;
    return base+(stage<3 ? dt/2 : dt)*rhs;
  }
};
}  // namespace time_integration
#endif  // DRIVER_CLASSICAL_RK4_HPP_
