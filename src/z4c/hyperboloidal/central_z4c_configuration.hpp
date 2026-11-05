// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CENTRAL_Z4C_CONFIGURATION_HPP_
#define Z4C_HYPERBOLOIDAL_CENTRAL_Z4C_CONFIGURATION_HPP_
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
namespace z4c::hyperboloidal {
// Reference areal radius on the outer branch of the initial wormhole map.
// It is not a proper-distance radius or a binary orbital separation.
struct CentralZ4cConfiguration {
  double inner_z4c_radius_M=20;
  double slicing_end_ratio=2;
  double omega_fraction=.25, gauge_begin_fraction=.35, gauge_end_fraction=.65;
  double restoration_end_fraction=.85;
  double plateau=.42;
  struct Resolved {
    HyperbolicShiftParameters<double> gauge;
    double compact_radius, isotropic_scale, slicing_end;
    bool FitsCartesianBox(double half_width, double coordinate_buffer=0) const {
      return std::isfinite(half_width) && coordinate_buffer>=0 &&
             compact_radius+coordinate_buffer<=half_width;
    }
  };
  Resolved Resolve() const {
    for(double v:{inner_z4c_radius_M,slicing_end_ratio,omega_fraction,
        gauge_begin_fraction,gauge_end_fraction,restoration_end_fraction,plateau})
      if(!std::isfinite(v))throw std::invalid_argument("nonfinite central configuration");
    if(!(inner_z4c_radius_M>2 && slicing_end_ratio>1 && 0<omega_fraction &&
        omega_fraction<gauge_begin_fraction && gauge_begin_fraction<gauge_end_fraction &&
        gauge_end_fraction<restoration_end_fraction && restoration_end_fraction<1))
      throw std::invalid_argument("invalid central configuration ordering");
    NativeWormhole map(1./3.,2e-13,inner_z4c_radius_M,
        slicing_end_ratio*inner_z4c_radius_M,NativeWormhole::TransitionProfile::polynomial_c6);
    Resolved out{};
    out.compact_radius=map.InnerRadius();out.isotropic_scale=map.IsotropicScale();
    out.slicing_end=map.OuterTransitionRadius();
    auto&p=out.gauge;const double r=out.compact_radius,L=1-r;
    p.profile={true,r,r+omega_fraction*L,plateau};
    p.inner_begin=r+gauge_begin_fraction*L;p.inner_end=r+gauge_end_fraction*L;
    p.coefficient_begin=p.inner_begin;p.coefficient_end=p.inner_end;
    p.restoration_begin=p.inner_begin;p.restoration_end=r+restoration_end_fraction*L;
    p.inner_gamma=out.isotropic_scale*out.isotropic_scale;
    p.inner_chi_guard=p.inner_gamma*InnerTelegrapher::chi_guard;
    p.cartesian_chi_guard=true;
    p.reference_start_M=inner_z4c_radius_M;
    p.reference_end_M=slicing_end_ratio*inner_z4c_radius_M;
    p.stationary_outer_reference=true;p.Validate();
    if(!(r>0 && r<out.slicing_end && out.slicing_end<1))
      throw std::invalid_argument("nonmonotone central initial map");
    return out;
  }
};
} // namespace z4c::hyperboloidal
#endif
