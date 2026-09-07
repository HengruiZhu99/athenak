// AthenaK astrophysical plasma code, 3-clause BSD License (LICENSE).
#ifndef PC_GH_INTRINSIC_TRANSFER_TARGET_HPP_
#define PC_GH_INTRINSIC_TRANSFER_TARGET_HPP_
#include <limits>
#include "athena.hpp"
#include "pc_gh/halo_weights.hpp"
#include "pc_gh/intrinsic_geometry.hpp"
#include "pc_gh/lapse_gradient.hpp"

namespace pc_gh::intrinsic {
template<typename State>
struct StoredPrimary {
  State state;
  int component;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m,int k,int j,int i) const { return state(m,component,k,j,i); }
};

template<typename State>
struct TransferPotential {
  State state;
  int component;  // w,alpha,s[5],beta[3]
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m,int k,int j,int i) const {
    if (component!=1) return state(m,component,k,j,i);
    LapseProduct<StoredPrimary<State>,StoredPrimary<State>> alpha{{state,1},{state,0}};
    return alpha(m,k,j,i);
  }
};

// Explicit available-halo closure, not an exact centered derivative beyond
// available support. Active cells use centered Dx. Ghost stencils shift only
// as far as necessary, sharing the generated legacy polynomial weights.
// This target has no reference metric, factor-two lapse convention, or J/dJ.
template<int Order,typename State>
KOKKOS_INLINE_FUNCTION
Real IntrinsicTransferTarget(State state,int m,int variable,int k,int j,int i,
                             const Real idx[3]) {
  static_assert(Order==2 || Order==4 || Order==6);
  if (variable<P || variable>=nvar) return std::numeric_limits<double>::quiet_NaN();
  int direction,component;
  if (variable<LAPSE_GRADIENT) {direction=variable-P;component=W;}
  else if (variable<S) {direction=variable-LAPSE_GRADIENT;component=1;}
  else if (variable<B) {direction=(variable-S)/5;component=CHART+(variable-S)%5;}
  else {direction=(variable-B)/3;component=BETA+(variable-B)%3;}
  const int extent=state.extent_int(4-direction);
  if (extent==1) return 0;
  if (extent<Order+1) return std::numeric_limits<double>::quiet_NaN();
  const int position=direction==0 ? i : (direction==1 ? j : k);
  TransferPotential<State> potential{state,component};
  if (position>=Order/2 && position+Order/2<extent)
    return Dx<Order/2+1>(direction,idx,potential,m,k,j,i);
  int start=position-Order/2;
  if (start<0) start=0;
  if (start+Order>=extent) start=extent-Order-1;
  Real result=0,center=potential(m,k,j,i);
  for (int node=0;node<=Order;++node) {
    int offset=start+node-position;
    result+=HaloDerivativeWeight<Order>(position-start,node)
      *(potential(m,k+(direction==2 ? offset : 0),j+(direction==1 ? offset : 0),
                  i+(direction==0 ? offset : 0))-center);
  }
  return result*idx[direction];
}
}  // namespace pc_gh::intrinsic
#endif  // PC_GH_INTRINSIC_TRANSFER_TARGET_HPP_
