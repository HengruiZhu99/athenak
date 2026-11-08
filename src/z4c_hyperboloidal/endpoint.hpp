// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ENDPOINT_HPP_
#define Z4C_HYPERBOLOIDAL_ENDPOINT_HPP_
#include "z4c_hyperboloidal/point_state.hpp"

namespace z4c_hyperboloidal {
// Stateless polynomial weights from the legacy seven-node endpoint closure.
// Native topology owns donor selection; a radial node list contains six active
// centers and the separately evolved scri endpoint. Angular nodes include peer
// values where required. Never replace the endpoint by a radial ghost value.
KOKKOS_INLINE_FUNCTION mapped::Jet LagrangeJet(Real q,const Real *nodes,
                                              int count,int index,int axis) {
  mapped::Jet basis(1);
  for (int c=0;c<count;++c)if (c!=index)
    basis=basis*(mapped::Jet::Variable(q,axis)-mapped::Jet(nodes[c]))
                /mapped::Jet(nodes[index]-nodes[c]);
  return basis;
}
KOKKOS_INLINE_FUNCTION mapped::Jet EndpointWeight(const mapped::Point &map,
    const Real q[3],const Real nodes[3][7],const int width[3],int i,int j,int k) {
  return CartesianJet(LagrangeJet(q[0],nodes[0],width[0],i,0)
      *LagrangeJet(q[1],nodes[1],width[1],j,1)
      *LagrangeJet(q[2],nodes[2],width[2],k,2),map);
}

// Interpolated correction is a logical jet on the surface tile. Its normal
// extension vanishes at the six interior radial points and at scri, with unit
// radial derivative at scri. This is the same polynomial normal lift as N12.
KOKKOS_INLINE_FUNCTION mapped::Jet NormalLift(const mapped::Point &map,
    const mapped::Jet &logical_correction,Real dr,Real gap) {
  using J=mapped::Jet;
  const J r=mapped::Power(SphericalPolicy::Radius2(map.x),.5);
  const J z=(r-J(1))/J(dr);
  J H=z;Real normalizer=1;
  for (int i=0;i<6;++i) {H=H*(z+J(i+gap));normalizer*=i+gap;}
  H=H*J(dr/normalizer);
  return H*CartesianJet(logical_correction,map);
}
}  // namespace z4c_hyperboloidal
#endif
