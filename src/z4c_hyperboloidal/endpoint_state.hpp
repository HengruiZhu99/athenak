// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ENDPOINT_STATE_HPP_
#define Z4C_HYPERBOLOIDAL_ENDPOINT_STATE_HPP_

#include "z4c_hyperboloidal/point_state.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"

namespace z4c_hyperboloidal {
template <class SurfaceKernel>
KOKKOS_INLINE_FUNCTION mapped::Jet SampleEndpointJet(
    int p, int f, const BlockPointReader &reader, const Real inv_h[3],
    const LogicalShift &shift, const SurfaceKernel &surface, Real &transport) {
  const auto jet = surface.EndpointResidualJet(p, f, reader.state);
  // Radial transport already belongs to the endpoint reconstruction. Only
  // the ordinary angular centered-to-biased corrections are added here.
  for (int a = 1; a < 3; ++a) {
    const Real b = shift.value[a];
    const int sign = b >= 0 ? 1 : -1;
    const AxialSamples samples(reader, f, a);
    const Real far = OffsetValue(reader, f, a, 3 * sign);
    const Real biased = samples.Biased(reader(f), far, inv_h[a], sign);
    transport += b * (biased - samples.First(inv_h[a]));
  }
  return jet;
}

template <class SurfaceKernel>
KOKKOS_INLINE_FUNCTION PointState
EndpointState(int p, const BlockPointReader &reader, const mapped::Point &map,
              const Real inv_h[3], const CartesianChartJets &reference,
              const HyperbolicShiftParameters<Real> &parameters,
              const SurfaceKernel &surface, const mapped::Jet *cached_weight = nullptr,
              IndependentFields independent = {}) {
  PointState out;
  Real beta[3]{};
  for (int a = 0; a < 3; ++a) beta[a] = reader(22 + a) + reference[22 + a].f - map.x[a];
  const auto shift = LogicalShift::Project(map, beta, false);
  for (int f = 0; f < Fields::count; ++f) {
    if (!independent.Has(f)) continue;
    out.jets[f] =
        SampleEndpointJet(p, f, reader, inv_h, shift, surface, out.transport[f]);
  }
  mapped::Jet lift[3];
  for (int a = 0; a < 3; ++a) lift[a] = surface.NormalCorrection(p, map, a);
  RestorePointState(out, reader, map, inv_h, reference, parameters, lift, cached_weight,
                    independent);
  return out;
}
}  // namespace z4c_hyperboloidal
#endif
