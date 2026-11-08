// Ported from scri-ko01 src/z4c/hyperboloidal/stationary_gauge_reference.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_STATIONARY_GAUGE_REFERENCE_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_STATIONARY_GAUGE_REFERENCE_HPP_
#include "z4c_hyperboloidal/hyperbolic_shift.hpp"
#include "z4c_hyperboloidal/native_wormhole.hpp"

namespace z4c_hyperboloidal {
// Host setup only. Native inversion/quadrature is cached by exact radius and
// never performed inside a device/RK point kernel. It supplies prescribed gauge
// sources, not analytic replacements for evolved geometry or derivatives.
class StationaryGaugeReference {
  HyperbolicShiftParameters<Real> parameters_;
  NativeWormhole native_;
 public:
  explicit StationaryGaugeReference(const HyperbolicShiftParameters<Real>&p)
      :parameters_(p),native_(1./3.,2e-13,p.reference_start_M,p.reference_end_M,
                             NativeWormhole::TransitionProfile::polynomial_c6,p.reference_compact_end) {
    p.Validate();
    if (!(p.profile.flat_radius>native_.IsotropicScale()/6))
      throw std::invalid_argument("stationary lapse blend must lie outside the horizon");
  }
  StationaryGaugeSource<Real> At(Real r) {
    StationaryGaugeSource<Real> out{};
    const auto&p=parameters_;
    if (!(r>=0&&r<=1))throw std::invalid_argument("invalid stationary reference radius");
    if (r<=p.profile.flat_radius)return out;
    if (r==1){out.alpha=1;return out;} // exact value/first-normal limits of sources
    const auto v=native_.Point(r);const Real old=(1-r)*(1+r)/2;
    const Real R=v.areal,S=-v.trace/3+old*v.ar/2,SR=-1.5*old*v.ar/R;
    const Real a=std::sqrt(1-2/(3*R)+R*R*S*S);
    const auto o=p.profile.Radial(r);const Real A=(1+r*r)/2;
    out.alpha=o.value*a;out.shift_ratio=1-S;
    if (!(out.alpha>0&&std::isfinite(out.alpha)))throw std::runtime_error("invalid stationary reference lapse");
    if (r<=p.inner_begin)return out; // exactly unchanged inner telegrapher RHS
    const auto c=ShiftCoefficients(r,out.alpha,Real(1),p);
    const Real ap=o.first*a+o.value*(1/(3*R)+R*R*S*S+R*R*R*S*SR)/r;
    const Real da=out.alpha-A;
    const Real lapse_regular=r*r-S*r*ap;
    const Real lapse_pole=-c.F*(v.trace+3)-2*c.xi*out.alpha*da
        +16*std::pow(o.value,4)*r*r*da/A+r*r*(A-S*out.alpha);
    const Real shift_regular=S*S+S*R*a*SR-1;
    const Real W=1-c.inner;
    out.lapse_offset=-W*(lapse_regular+lapse_pole/o.value);
    out.lapse_quadratic=W*(v.trace+3)/o.value;
    // F(alpha)-F(alpha_*)=(alpha-alpha_*)(alpha+alpha_*+4 Omega²).
    // This consistently replaces K+3 by K-K_* in the slicing source.
    out.lapse_linear=W*2*c.xi*da/o.value
        +out.lapse_quadratic*(2*out.alpha+4*o.value*o.value);
    out.shift_offset=W*(-shift_regular+p.restoring*c.collar*(1-S)/o.value);
    for(Real q:{out.lapse_offset,out.lapse_linear,out.lapse_quadratic,out.shift_offset})
      if(!std::isfinite(q))throw std::runtime_error("invalid prescribed gauge source");
    return out;
  }
};
} // namespace z4c_hyperboloidal
#endif
