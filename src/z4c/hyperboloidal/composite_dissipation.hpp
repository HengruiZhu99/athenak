// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_DISSIPATION_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_DISSIPATION_HPP_
#include <Kokkos_Core.hpp>
#ifndef HYPERBOLIC_CONSISTENT_KO
#define HYPERBOLIC_CONSISTENT_KO 0
#endif
namespace z4c::hyperboloidal {
// Q=-epsilon*n/(2^(2p))*V^-1 B_p^T W B_p, with positive line weights.
// Truncating complete lines at patch edges preserves weighted nonpositive
// energy. With p=4 it leaves O(h^3) at edges; p=5 instead gives O(h^4)
// for smooth mapped fields even where the transpose stencil is truncated.
// The taper/physical metric weights are unchanged; this is KO10 in interiors.
// Choosing p=5 does not claim interface stability of the full coupled PDE.
struct CompositeDissipation {
  static constexpr bool consistent=HYPERBOLIC_CONSISTENT_KO;
  static constexpr int order=consistent?5:4;
  static constexpr int points=order+1;
  static constexpr int normalization=1<<(2*order);
  KOKKOS_INLINE_FUNCTION static int Coefficient(int i){
    if constexpr(consistent){const int c[6]={-1,5,-10,10,-5,1};return c[i];}
    else {const int c[5]={1,-4,6,-4,1};return c[i];}
  }
};
}
#endif
