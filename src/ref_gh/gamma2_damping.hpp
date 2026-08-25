//========================================================================================
//! \file gamma2_damping.hpp
//! \brief Standard gamma2 reduction-constraint damping for gamma1=-1.
//========================================================================================
#ifndef REF_GH_GAMMA2_DAMPING_HPP_
#define REF_GH_GAMMA2_DAMPING_HPP_

#include "athena.hpp"

namespace ref_gh {

struct Gamma2DampingRhs {
  Real pi;
  Real phi[3];  // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
Gamma2DampingRhs ComputeGamma2DampingRhs(
    const Real lapse, const Real shift[3], const Real coordinate_reduction[3],
    const Real spatial_frame[3][3], const Real gamma2) {
  Gamma2DampingRhs rhs{0.0, {0.0, 0.0, 0.0}};
  for (int p = 0; p < 3; ++p) {
    rhs.pi -= gamma2*shift[p]*coordinate_reduction[p];
  }
  for (int I = 0; I < 3; ++I) {
    for (int p = 0; p < 3; ++p) {
      rhs.phi[I] += lapse*gamma2*spatial_frame[I][p]
                    *coordinate_reduction[p];
    }
  }
  return rhs;
}

}  // namespace ref_gh

#endif  // REF_GH_GAMMA2_DAMPING_HPP_
