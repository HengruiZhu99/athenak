// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ENDPOINT_STATE_HPP_
#define Z4C_HYPERBOLOIDAL_ENDPOINT_STATE_HPP_

#include "z4c_hyperboloidal/point_state.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"

namespace z4c_hyperboloidal {
template<class SurfaceKernel>
KOKKOS_INLINE_FUNCTION PointState EndpointState(int p,const BlockPointReader &reader,
    const mapped::Point &map,const Real inv_h[3],const CartesianChartJets &reference,
    const HyperbolicShiftParameters<Real> &parameters,const SurfaceKernel &surface) {
  PointState out;
  Real beta[3]{};
  for (int a=0;a<3;++a) beta[a]=reader(22+a)+reference[22+a].f-map.x[a];
  for (int f=0;f<Fields::count;++f) if (Fields::Independent(f)) {
    out.jets[f]=surface.EndpointResidualJet(p,f,reader.state);
    // Radial transport already belongs to the endpoint reconstruction. Only
    // the ordinary angular centered-to-biased corrections are added here.
    for (int a=1;a<3;++a) {
      Real b=0;for (int d=0;d<3;++d) b+=map.J[a][d]*beta[d];
      const int sign=b>=0?1:-1;
      const Real center=reader(f);
      const Real vm=OffsetValue(reader,f,a,-sign)-center;
      const Real v1=OffsetValue(reader,f,a,sign)-center;
      const Real v2=OffsetValue(reader,f,a,2*sign)-center;
      const Real v3=OffsetValue(reader,f,a,3*sign)-center;
      const Real biased=sign*inv_h[a]*(-3*vm+18*v1-6*v2+v3)/12;
      out.transport[f]+=b*(biased-FirstDerivative(reader,f,a,inv_h[a]));
    }
  }
  mapped::Jet lift[3];
  for (int a=0;a<3;++a) lift[a]=surface.NormalCorrection(p,map,a);
  RestorePointState(out,reader,map,inv_h,reference,parameters,lift);
  return out;
}
}  // namespace z4c_hyperboloidal
#endif
