#ifndef Z4C_Z4C_BOUNDARY_STENCIL_HPP_
#define Z4C_Z4C_BOUNDARY_STENCIL_HPP_

#include "athena.hpp"
#include "mesh/mesh.hpp"

namespace z4c {

// The volume RHS exists only on active cells; its ghosts are neither computed
// nor exchanged. A raised physical-face normal can have tangential components
// for an off-diagonal metric, including at an internal tangential block edge.
// Use this same active-only derivative for the state and its time derivative
// so that the characteristic rate differentiates the same discrete quantity.
// No RHS exchange or synchronization is needed: configuration RHS is immutable
// throughout the characteristic boundary update.
KOKKOS_INLINE_FUNCTION
Real BoundaryCoordinateDerivative2(const DvceArray5D<Real> &u, int m, int n,
                                   int k, int j, int i, int dir,
                                   const RegionIndcs &indcs, const Real idx[3]) {
  const int pos[3] = {i,j,k};
  const int low[3] = {indcs.is,indcs.js,indcs.ks};
  const int high[3] = {indcs.ie,indcs.je,indcs.ke};
  if (high[dir] == low[dir]) return 0.0;  // Inactive spatial direction.
  const int side = pos[dir] == low[dir] ? -1 :
                   (pos[dir] == high[dir] ? 1 : 0);
  if (side != 0) {
    const int di = dir == 0 ? -side : 0;
    const int dj = dir == 1 ? -side : 0;
    const int dk = dir == 2 ? -side : 0;
    return 0.5*side*idx[dir]*(3.0*u(m,n,k,j,i) -
        4.0*u(m,n,k+dk,j+dj,i+di) + u(m,n,k+2*dk,j+2*dj,i+2*di));
  }
  const int di = dir == 0 ? 1 : 0;
  const int dj = dir == 1 ? 1 : 0;
  const int dk = dir == 2 ? 1 : 0;
  return 0.5*idx[dir]*(u(m,n,k+dk,j+dj,i+di) - u(m,n,k-dk,j-dj,i-di));
}

}  // namespace z4c
#endif  // Z4C_Z4C_BOUNDARY_STENCIL_HPP_
