// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_TRUMPET_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_TRUMPET_HPP_

#include <map>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "z4c/hyperboloidal/cartesian_radial.hpp"
#include "z4c/hyperboloidal/cmc_trumpet.hpp"

namespace z4c {
namespace hyperboloidal {

inline void InitializeCartesianTrumpet(CartesianConformalPatch &patch,
    const DvceArray5D<Real> &data, Real mass, bool analytic_reconstruction = true) {
  if (patch.reference.layer.enabled) {
    throw std::invalid_argument(
        "CMC trumpet data are incompatible with the layer foliation");
  }
  if (patch.reference.scri_radius != 1 || patch.reference.curvature_radius != 1) {
    throw std::invalid_argument("Cartesian trumpet requires S=a=1");
  }
  const CMCTrumpet trumpet(mass);
  patch.InitializeReference(data);
  const auto nodes = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace(),patch.active);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),data);
  Kokkos::View<Z4cJet<Real> *> jets("Cartesian trumpet jets",nodes.extent(0));
  auto jh = Kokkos::create_mirror_view(jets);
  const auto g = patch.grid;
  std::map<Real,Z4cJet<Real>> radial_cache;
  const int tensor[3][3] = {{0,1,2},{1,3,4},{2,4,5}};
  for (size_t point = 0; point < nodes.extent(0); ++point) {
    const int s = nodes(point), i = s%g.n[0], j = s/g.n[0]%g.n[1];
    const int k = s/(g.n[0]*g.n[1]);
    const Real x[3] = {g.first[0]+i*g.h[0],g.first[1]+j*g.h[1],g.first[2]+k*g.h[2]};
    const Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    if (!(r > 0 && r < 1)) {
      throw std::invalid_argument("Cartesian trumpet grid must exclude r=0 and scri");
    }
    auto found = radial_cache.find(r);
    if (found == radial_cache.end()) {
      found = radial_cache.emplace(r,trumpet.Jet(r)).first;
    }
    const Real n[3] = {x[0]/r,x[1]/r,x[2]/r};
    const auto u = CartesianRadialJet(found->second,r,n);
    jh(point) = u;
    host(0,Z4c::I_Z4C_CHI,k,j,i) = u.chi.value;
    host(0,Z4c::I_Z4C_KHAT,k,j,i) = u.trace.value;
    host(0,Z4c::I_Z4C_THETA,k,j,i) = u.theta.value;
    host(0,Z4c::I_Z4C_ALPHA,k,j,i) = u.alpha.value;
    for (int a = 0; a < 3; ++a) {
      host(0,Z4c::I_Z4C_BETAX+a,k,j,i) = u.beta.value[a];
      host(0,Z4c::I_Z4C_GAMX+a,k,j,i) = u.lambda.value[a];
      for (int b = a; b < 3; ++b) {
        host(0,Z4c::I_Z4C_GXX+tensor[a][b],k,j,i) = u.metric.g[a][b];
        host(0,Z4c::I_Z4C_AXX+tensor[a][b],k,j,i) = u.a.k[a][b];
      }
    }
  }
  Kokkos::deep_copy(data,host);
  if (analytic_reconstruction) {
    Kokkos::deep_copy(jets,jh);
    patch.SetReconstruction(data,jets);
  }
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CARTESIAN_TRUMPET_HPP_
