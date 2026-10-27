// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_PUNCTURE_POSITION_HPP_
#define Z4C_HYPERBOLOIDAL_PUNCTURE_POSITION_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "athena.hpp"

namespace z4c { namespace hyperboloidal {

// Cartesian kinematics only. No regularity power, parity condition, metric
// constraint or beta boundary condition belongs to a puncture position.
// stage==-1 denotes an accepted position; nonnegative stages are RK states.
struct PuncturePosition {
  std::array<Real,3> position{},velocity{};
  Real time=0;
  std::uint64_t revision=0;
  int stage=-1;

  struct Stamp {
    std::array<Real,3> position,velocity;
    Real time;
    std::uint64_t revision;
    int stage;
  };
  struct RadialJet {
    Real value=0,time_derivative=0;
    std::array<Real,3> gradient{};
    std::array<std::array<Real,3>,3> hessian{};
  };

  void Validate() const {
    if(!std::isfinite(time)||stage< -1)
      throw std::invalid_argument("invalid puncture time/stage");
    for(int a=0;a<3;++a)
      if(!std::isfinite(position[a])||!std::isfinite(velocity[a]))
        throw std::invalid_argument("nonfinite puncture kinematics");
  }
  void SetKinematics(const std::array<Real,3> &new_position,
      const std::array<Real,3> &new_velocity,Real new_time,int new_stage=-1) {
    PuncturePosition next=*this;
    next.position=new_position;next.velocity=new_velocity;
    next.time=new_time;next.stage=new_stage;next.Validate();
    if(revision==std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("puncture revision exhausted");
    next.revision=revision+1;*this=next;
  }
  void SetVelocityFromShift(const Real beta_at_puncture[3]) {
    std::array<Real,3> next;
    for(int a=0;a<3;++a)next[a]=-beta_at_puncture[a];
    SetKinematics(position,next,time,stage);
  }
  Stamp Capture() const {
    Validate();return {position,velocity,time,revision,stage};
  }
  bool Matches(const Stamp &stamp) const {
    // Public fields make simple manufactured configurations convenient. Compare
    // their values too, so a direct edit cannot silently retain a cached plan.
    Validate();return revision==stamp.revision&&stage==stamp.stage&&time==stamp.time
      &&position==stamp.position&&velocity==stamp.velocity;
  }
  std::array<Real,3> Relative(const Real x[3]) const {
    Validate();std::array<Real,3> y;
    for(int a=0;a<3;++a){
      if(!std::isfinite(x[a]))throw std::invalid_argument("nonfinite puncture query");
      y[a]=x[a]-position[a];
      if(!std::isfinite(y[a]))throw std::overflow_error("puncture coordinate overflow");
    }
    return y;
  }
  Real RadiusSquared(const Real x[3]) const {
    const auto y=Relative(x);Real r2=0;
    for(Real value:y)r2+=value*value;
    if(!std::isfinite(r2))throw std::overflow_error("puncture radius overflow");
    return r2;
  }
  bool AtEndpoint(const Real x[3]) const {return RadiusSquared(x)==0;}
  bool InNeighborhood(const Real x[3],Real radius) const {
    if(!(radius>0)||!std::isfinite(radius))
      throw std::invalid_argument("invalid puncture neighborhood radius");
    const Real r2=radius*radius;
    if(!std::isfinite(r2))throw std::overflow_error("puncture neighborhood radius overflow");
    return RadiusSquared(x)<=r2;
  }
  RadialJet SquaredRadiusJet(const Real x[3]) const {
    const auto y=Relative(x);RadialJet out;out.value=RadiusSquared(x);
    for(int a=0;a<3;++a){out.gradient[a]=2*y[a];out.hessian[a][a]=2;
      out.time_derivative-=2*y[a]*velocity[a];}
    if(!std::isfinite(out.time_derivative))
      throw std::overflow_error("puncture squared-radius time derivative overflow");
    return out;
  }
  RadialJet RadialPowerJet(const Real x[3],Real power) const {
    if(!std::isfinite(power))throw std::invalid_argument("nonfinite chart power");
    const auto y=Relative(x);const Real r2=RadiusSquared(x);
    if(!(r2>0))throw std::domain_error("radial-power chart requires explicit endpoint data");
    RadialJet out;
    if(power==0){out.value=1;return out;}
    const Real r=std::sqrt(r2),first=power*std::pow(r,power-1);
    const Real second=power*std::pow(r,power-2);
    out.value=std::pow(r,power);
    for(int a=0;a<3;++a){const Real na=y[a]/r;out.gradient[a]=first*na;
      out.time_derivative-=out.gradient[a]*velocity[a];
      for(int b=0;b<3;++b)out.hessian[a][b]=second*((a==b?1.:0)+(power-2)*na*y[b]/r);}
    if(!(out.value>0)||!std::isfinite(out.value)||!std::isfinite(out.time_derivative))
      throw std::overflow_error("radial-power factor outside floating-point range");
    for(int a=0;a<3;++a){if(!std::isfinite(out.gradient[a]))
        throw std::overflow_error("radial-power gradient outside floating-point range");
      for(int b=0;b<3;++b)if(!std::isfinite(out.hessian[a][b]))
        throw std::overflow_error("radial-power Hessian outside floating-point range");}
    return out;
  }
  // F=f(x-X(t))*u, at FIXED Cartesian x, for a caller-selected fixed factor.
  // This is a time conversion for an evolved chart, not an extra source in an
  // instantaneous spatial elliptic solve. No value is supplied at rho=0.
  Real FixedChartRate(const Real x[3],Real power,Real physical_rate,
      Real coefficient) const {
    const auto factor=RadialPowerJet(x,power);
    if(!(factor.value>0)||!std::isfinite(physical_rate)||!std::isfinite(coefficient))
      throw std::domain_error("noninvertible/nonfinite radial chart rate");
    const Real rate=(physical_rate-factor.time_derivative*coefficient)/factor.value;
    if(!std::isfinite(rate))throw std::overflow_error("radial chart rate overflow");
    return rate;
  }
  Real ComovingRate(Real fixed_grid_rate,const Real spatial_gradient[3]) const {
    Validate();Real rate=fixed_grid_rate;
    for(int a=0;a<3;++a){if(!std::isfinite(spatial_gradient[a]))
        throw std::invalid_argument("nonfinite moving-coordinate gradient");
      rate+=velocity[a]*spatial_gradient[a];}
    if(!std::isfinite(rate))throw std::overflow_error("moving-coordinate rate overflow");
    return rate;
  }

  // The actual CompactObjectTracker has GetPos(int) and SetPos(Real*), but no
  // velocity accessor. Supply beta from the converged COMPOSITE stage solve.
  // Sampling native Cauchy data or invoking its final-stage Euler evolution is
  // deliberately not hidden inside this adapter.
  template<typename Tracker>
  static PuncturePosition FromTracker(const Tracker &tracker,
      const Real beta_at_puncture[3],Real sample_time,int sample_stage=-1) {
    PuncturePosition out;std::array<Real,3> x,v;
    for(int a=0;a<3;++a){x[a]=tracker.GetPos(a);v[a]=-beta_at_puncture[a];}
    out.SetKinematics(x,v,sample_time,sample_stage);return out;
  }
  template<typename Tracker> void PublishAccepted(Tracker &tracker) const {
    Validate();if(stage!=-1)throw std::logic_error("cannot publish an intermediate puncture stage");
    Real x[3]={position[0],position[1],position[2]};tracker.SetPos(x);
  }
};

}}  // namespace z4c::hyperboloidal
#endif
