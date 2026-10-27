// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SPHERICAL_MATCHED_EVOLUTION_HPP_
#define Z4C_HYPERBOLOIDAL_SPHERICAL_MATCHED_EVOLUTION_HPP_
#include "z4c/hyperboloidal/composite_evolution.hpp"
namespace z4c { namespace hyperboloidal {

// Explicit host research configuration. These are the gauge/time-update controls
// of the a9214bccf spherical duration run, not a claim of spatial equivalence:
// Cartesian/shell derivatives, interface transfers and dissipation differ.
struct SphericalMatchedControls {
  Real lapse_attenuation=1e-4, physical_lapse_floor=1e-10;
  bool total_lapse_update=true, scri_regularity_traces=true;
  void Validate() const {
    if(!std::isfinite(lapse_attenuation)||lapse_attenuation<0||lapse_attenuation>1
       ||!std::isfinite(physical_lapse_floor)||physical_lapse_floor<0)
      throw std::invalid_argument("invalid matched spherical lapse controls");
  }
};
struct MatchedFloorDiagnostics {
  size_t initial_clamps=0, evolution_clamps=0, nonpositive_clamps=0;
  Real maximum_initial_correction=0, maximum_evolution_correction=0,
       maximum_activation_radius=0;
};

// Research RK4 only. Native Driver/register/restart integration is deliberately
// separate: a legacy CompositeCheckpoint must never be used to resume this state
// without these controls and both compensation arrays.
class SphericalMatchedEvolution : public CompositeTensorEvolution {
 public:
  const SphericalMatchedControls controls;
  V compensation, boundary_compensation;
  MatchedFloorDiagnostics floor;
  SphericalMatchedEvolution(cubed_sphere::CompositeMesh mesh,Real tolerance,
                            SphericalMatchedControls options={})
      : CompositeTensorEvolution(std::move(mesh),true,tolerance,4,true,true,0,7,
                                 true,2,.24,false,true,true,false,.12,0,1,false),
        controls(options), compensation(state.size(),0),
        boundary_compensation(scri->values.size(),0) {
    controls.Validate();
    telegraph_attenuation=controls.lapse_attenuation;
    spherical_scri_traces=controls.scri_regularity_traces;
  }
  bool StoreAlpha(V& values,size_t p,Real alpha,bool initial=false) {
    if(!std::isfinite(alpha))throw std::runtime_error("nonfinite matched total lapse");
    const auto&x=shift.grid.geometry[p].x;
    const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2],omega=(1-r2)/2;
    if(!(omega>0&&r2>0))throw std::runtime_error("matched active lapse must be inside scri and off puncture");
    const Real lower=controls.physical_lapse_floor*omega;
    const bool clipped=alpha<lower;
    if(clipped){
      if(initial){++floor.initial_clamps;floor.maximum_initial_correction=
          std::max(floor.maximum_initial_correction,lower-alpha);}
      else {++floor.evolution_clamps;floor.maximum_evolution_correction=
          std::max(floor.maximum_evolution_correction,lower-alpha);}
      floor.nonpositive_clamps+=alpha<=0;
      floor.maximum_activation_radius=std::max(floor.maximum_activation_radius,std::sqrt(r2));
      alpha=lower;
    }
    if(!(alpha>0))throw std::runtime_error("nonpositive matched lapse without active floor");
    values[p*fields+18]=EncodePositive(18,alpha,x);
    // The bound is represented in the same log chart as the state.
    if(lower>0)values[p*fields+18]=std::max(values[p*fields+18],EncodePositive(18,lower,x));
    return clipped;
  }
  void InitializeWormhole(const std::string&path,Real amplitude=0,bool angular=false) {
    CompositeTensorEvolution::InitializeWormhole(path,amplitude,angular);
    for(size_t p=0;p<shift.grid.active.size();++p)
      StoreAlpha(state,p,DecodePositive(18,state[p*fields+18],shift.grid.geometry[p].x),true);
  }
  V MatchedRates(const V& values,Real dt) {
    V rate=RHS(values,dt);
    if(controls.total_lapse_update)for(size_t p=0;p<jets.size();++p)
      rate[p*fields+18]*=jets[p].alpha.value;
    return rate;
  }
  V MakeStage(const V& start,const V& rate,Real factor) {
    V next(start.size());
    for(size_t j=0;j<start.size();++j)next[j]=start[j]+factor*rate[j];
    for(size_t p=0;p<shift.grid.active.size();++p){
      const auto&x=shift.grid.geometry[p].x;
      const Real alpha=controls.total_lapse_update?
        DecodePositive(18,start[p*fields+18],x)+factor*rate[p*fields+18]:
        DecodePositive(18,next[p*fields+18],x);
      StoreAlpha(next,p,alpha);
    }
    return next;
  }
  static void Compensated(V& values,V& remainder,const V& a,const V& b,
                          const V& c,const V& d,Real dt) {
    for(size_t j=0;j<values.size();++j){
      const Real increment=dt*(a[j]+2*b[j]+2*c[j]+d[j])/6-remainder[j];
      const Real next=values[j]+increment;
      remainder[j]=(next-values[j])-increment;values[j]=next;
    }
  }
  void Step(Real dt) {
    if(!(dt>0)||!std::isfinite(dt))throw std::invalid_argument("matched step must be positive finite");
    const V begin=state,edge=scri->values;
    const V k1=MatchedRates(begin,dt),b1=boundary_rhs;
    for(size_t j=0;j<edge.size();++j)scri->values[j]=edge[j]+dt*b1[j]/2;
    const V y2=MakeStage(begin,k1,dt/2),k2=MatchedRates(y2,dt),b2=boundary_rhs;
    for(size_t j=0;j<edge.size();++j)scri->values[j]=edge[j]+dt*b2[j]/2;
    const V y3=MakeStage(begin,k2,dt/2),k3=MatchedRates(y3,dt),b3=boundary_rhs;
    for(size_t j=0;j<edge.size();++j)scri->values[j]=edge[j]+dt*b3[j];
    const V y4=MakeStage(begin,k3,dt),k4=MatchedRates(y4,dt),b4=boundary_rhs;
    scri->values=edge;Compensated(scri->values,boundary_compensation,b1,b2,b3,b4,dt);
    for(size_t j=0;j<state.size();++j){
      const Real increment=dt*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6-compensation[j];
      if(controls.total_lapse_update&&j%fields==18){
        const size_t p=j/fields;const auto&x=shift.grid.geometry[p].x;
        const Real old=DecodePositive(18,begin[j],x);
        const bool clipped=StoreAlpha(state,p,old+increment);
        compensation[j]=clipped?0:(DecodePositive(18,state[j],x)-old)-increment;
      }else{
        const Real next=begin[j]+increment;
        compensation[j]=(next-begin[j])-increment;state[j]=next;
      }
    }
    if(!controls.total_lapse_update)for(size_t p=0;p<shift.grid.active.size();++p)
      if(StoreAlpha(state,p,DecodePositive(18,state[p*fields+18],shift.grid.geometry[p].x)))
        compensation[p*fields+18]=0;
  }
};
}} // namespace z4c::hyperboloidal
#endif
