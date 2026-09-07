// Available-halo derivative target for the explicit residual_shifted experiment.
#ifndef PC_GH_TRANSFER_TARGET_HPP_
#define PC_GH_TRANSFER_TARGET_HPP_
#include "athena.hpp"
#include "pc_gh/pc_gh.hpp"
#include "pc_gh/halo_weights.hpp"
#include "pc_gh/reflection_parity.hpp"
#include "pc_gh/lapse_gradient.hpp"

namespace pc_gh {
struct TransferPrimaryComponent {
  DvceArray5D<Real> state;
  int component;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int m,int k,int j,int i) const {
    return state(m,component,k,j,i);
  }
};
// Centered on every active cell. At ghost cells, shift only as needed to remain
// inside allocated, already synchronized primary ghosts. No out-of-halo reads.
template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real LegacyTransferTarget(const DvceArray5D<Real> &state, int m, int n,
                          int k, int j, int i, const Real idx[3],
                          bool collision_lapse) {
  int direction, primary;
  bool lapse = false;
  if (n < PcGh::I_Q1XX) {
    direction = n-PcGh::I_P1;
    primary = PcGh::I_W;
  } else if (n < PcGh::I_L1) {
    direction = (n-PcGh::I_Q1XX)/6;
    primary = PcGh::I_GTXX+(n-PcGh::I_Q1XX)%6;
  } else if (n < PcGh::I_B11) {
    direction = n-PcGh::I_L1;
    primary = PcGh::I_W;
    lapse = true;
  } else {
    direction = (n-PcGh::I_B11)/3;
    primary = PcGh::I_BETAX+(n-PcGh::I_B11)%3;
  }
  int position = direction == 0 ? i : (direction == 1 ? j : k);
  int extent = state.extent_int(4-direction);
  if (extent == 1) return 0.0;
  if (position>=ORDER/2 && position+ORDER/2<extent) {
    // Match the actual centered operator's arithmetic, particularly before
    // outflow extrapolation can amplify roundoff in E=G-R(u).
    constexpr int stencil=ORDER/2+1;
    if (!lapse) return Dx<stencil>(direction,idx,state,m,primary,k,j,i);
    if (collision_lapse) {
      Real dw=Dx<stencil>(direction,idx,state,m,PcGh::I_W,k,j,i);
      Real drho=Dx<stencil>(direction,idx,state,m,PcGh::I_RHO,k,j,i);
      return 2.0*(state(m,PcGh::I_W,k,j,i)*drho+state(m,PcGh::I_RHO,k,j,i)*dw);
    }
    TransferPrimaryComponent rho{state,PcGh::I_RHO}, w{state,PcGh::I_W};
    return DirectLapseGradient<stencil>(direction,idx,rho,w,m,k,j,i);
  }
  int start = position-ORDER/2;
  if (start < 0) start = 0;
  if (start+ORDER >= extent) start = extent-ORDER-1;
  Real answer = 0.0;
  for (int node = 0; node <= ORDER; ++node) {
    int offset = start+node-position;
    int ii = i+(direction == 0 ? offset : 0);
    int jj = j+(direction == 1 ? offset : 0);
    int kk = k+(direction == 2 ? offset : 0);
    Real center=state(m,primary,k,j,i);
    Real point=state(m,primary,kk,jj,ii);
    Real value = point-center;
    if (lapse) {
      Real rho = state(m,PcGh::I_RHO,kk,jj,ii);
      Real rho_center=state(m,PcGh::I_RHO,k,j,i);
      value = collision_lapse
          ? 2.0*(center*(rho-rho_center)+rho_center*(point-center))
          : 2.0*(rho*point-rho_center*center);
    }
    answer += HaloDerivativeWeight<ORDER>(position-start,node)*value;
  }
  return answer*idx[direction];
}
// Reflection is an explicit extension of the physical primary grid. Fold a
// reflected ghost to its interior mirror before differentiating, then apply the
// complete auxiliary tensor parity. This avoids replacing exact reflection by
// an unrelated shifted normal derivative in the outermost ghost layers.
template <int ORDER>
KOKKOS_INLINE_FUNCTION
Real LegacyBoundaryTransferTarget(const DvceArray5D<Real> &state, int m, int n,
    int k, int j, int i, const Real idx[3], bool collision_lapse,
    const RegionIndcs &ind, const DualArray2D<BoundaryFlag> &bcs) {
  int coordinate[3]={i,j,k};
  int lower[3]={ind.is,ind.js,ind.ks};
  int upper[3]={ind.ie,ind.je,ind.ke};
  Real sign=1.0;
  for (int axis=0; axis<3; ++axis) {
    if (state.extent_int(4-axis)==1) continue;
    bool reflect=false;
    if (coordinate[axis]<lower[axis]
        && bcs.d_view(m,2*axis)==BoundaryFlag::reflect) {
      coordinate[axis]=2*lower[axis]-1-coordinate[axis];
      reflect=true;
    } else if (coordinate[axis]>upper[axis]
        && bcs.d_view(m,2*axis+1)==BoundaryFlag::reflect) {
      coordinate[axis]=2*upper[axis]+1-coordinate[axis];
      reflect=true;
    }
    if (reflect && ReflectOdd(n,axis)) sign=-sign;
  }
  return sign*LegacyTransferTarget<ORDER>(state,m,n,coordinate[2],coordinate[1],
                                         coordinate[0],idx,collision_lapse);
}
}  // namespace pc_gh
#endif
