// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_WORMHOLE_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_WORMHOLE_HPP_

#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "z4c/hyperboloidal/layer_wormhole.hpp"

namespace z4c {
namespace hyperboloidal {

// Initialize geometric Schwarzschild wormhole data in the live Cartesian arrays.
// The patch's immutable Minkowski reference remains the gauge/source target.
// SetReconstruction uses analytic INITIAL jets plus finite differences of the
// evolving deviation from that fixed initial profile. It does not subtract a
// black-hole RHS, supply an analytic evolution, or make these data stationary.
inline void InitializeCartesianWormhole(CartesianConformalPatch& patch,
                                        const DvceArray5D<Real>& data, Real mass,
                                        bool analytic_reconstruction = true) {
  const LayerWormhole<Real> wormhole(patch.reference, mass);
  const auto g = patch.grid;
  for (int d = 0; d < 3; ++d) {
    if (g.n[d] % 2 != 0) {
      throw std::invalid_argument(
          "layer wormhole requires even cell counts excluding r=0");
    }
  }
  patch.InitializeReference(data);
  const auto nodes =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), patch.active);
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
  Kokkos::View<Z4cJet<Real>*> jets("Cartesian layer wormhole jets", nodes.extent(0));
  auto jh = Kokkos::create_mirror_view(jets);
  for (size_t point = 0; point < nodes.extent(0); ++point) {
    const int s = nodes(point), i = s % g.n[0], j = s / g.n[0] % g.n[1];
    const int k = s / (g.n[0] * g.n[1]);
    const Real x = g.first[0] + i * g.h[0], y = g.first[1] + j * g.h[1];
    const Real z = g.first[2] + k * g.h[2];
    const Real r = std::sqrt(x * x + y * y + z * z);
    if (!(r > 0 && r < patch.reference.scri_radius)) {
      throw std::invalid_argument(
          "layer wormhole grid must exclude exact puncture and scri");
    }
    auto p = patch.reference.At(x, y, z);
    p.state = wormhole.At(x, y, z);
    jh(point) = p.state;
    for (int f = 0; f < Z4c::nz4c; ++f) {
      host(0, f, k, j, i) = ReferenceComponent(f, p);
    }
  }
  Kokkos::deep_copy(data, host);
  if (analytic_reconstruction) {
    Kokkos::deep_copy(jets, jh);
    patch.SetReconstruction(data, jets);
  }
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CARTESIAN_WORMHOLE_HPP_
