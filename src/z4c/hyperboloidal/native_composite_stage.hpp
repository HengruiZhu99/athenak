// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_COMPOSITE_STAGE_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_COMPOSITE_STAGE_HPP_

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "athena.hpp"
#include "driver/driver.hpp"

namespace z4c { namespace hyperboloidal {

// Exact coefficients used by native CopyU/ExpRKUpdate. No production table is
// duplicated here: FromDriver reads the actual live Driver arrays. Capture is
// also available to standalone component tests and restart readers.
struct NativeCompositeRK {
  std::string integrator;
  int stages=0;
  std::array<Real,4> gam0{},gam1{},beta{},delta{};
  static NativeCompositeRK Capture(const std::string &name,int explicit_stages,
      int implicit_stages,const Real *g0,const Real *g1,const Real *b,const Real *d) {
    if(implicit_stages!=0||!((name=="rk3"&&explicit_stages==3)||(name=="rk4"&&explicit_stages==4)))
      throw std::invalid_argument("composite stage supports native explicit RK3/RK4 only");
    NativeCompositeRK out;out.integrator=name;out.stages=explicit_stages;
    for(int s=0;s<explicit_stages;++s){out.gam0[s]=g0[s];out.gam1[s]=g1[s];out.beta[s]=b[s];
      // Native RK3 leaves delta uninitialized and never uses it. Do not read it.
      if(name=="rk4")out.delta[s]=d[s];}
    out.Validate();return out;
  }
  static NativeCompositeRK FromDriver(const Driver &driver) {
    return Capture(driver.integrator,driver.nexp_stages,driver.nimp_stages,
        driver.gam0,driver.gam1,driver.beta,driver.delta);
  }
  void Validate() const {
    if(!((integrator=="rk3"&&stages==3)||(integrator=="rk4"&&stages==4)))
      throw std::invalid_argument("composite restart RK method");
    for(int s=0;s<4;++s)if(!std::isfinite(gam0[s])||!std::isfinite(gam1[s])||
        !std::isfinite(beta[s])||!std::isfinite(delta[s]))
      throw std::invalid_argument("nonfinite composite RK coefficients");
  }
  bool operator==(const NativeCompositeRK &other) const {
    return integrator==other.integrator&&stages==other.stages&&gam0==other.gam0&&
        gam1==other.gam1&&beta==other.beta&&delta==other.delta;
  }
};

// Coefficient RK storage only. It neither defines a chart/puncture power nor
// performs physics, interpolation, a solve or native task dispatch. The three
// groups may contain arbitrary volume, scri and extra endpoint coefficients.
// Algebraic beta has a separate warm-start vector and NEVER enters an RK sum.
class NativeCompositeStageState {
 public:
  using Vector=std::vector<Real>;
  struct Values {Vector volume,scri,endpoints;};
  struct Metadata {
    std::string chart;
    Vector chart_parameters;
    std::uint64_t mesh_revision=0,chart_revision=0,puncture_revision=0;
  };
  struct Epoch {
    std::uint64_t cycle=0,geometry=0;
    int stage=0;
    bool operator==(const Epoch &b) const {return cycle==b.cycle&&geometry==b.geometry&&stage==b.stage;}
  };
  struct Snapshot {
    std::uint32_t version=1;
    Values values,registers;
    Metadata metadata;
    NativeCompositeRK rk;
    Vector shift_warm_start;
    std::uint64_t cycle=0,completed_cycles=0,geometry=0;
    int next_stage=1,open_stage=0;
    bool active=false,stage_open=false,has_rk=false;
    Real time=0,dt=0,abscissa=0,register_abscissa=0;
  };

  NativeCompositeStageState(Values values,size_t shift_size)
      :NativeCompositeStageState(std::move(values),shift_size,Metadata{}) {}
  NativeCompositeStageState(Values values,size_t shift_size,Metadata metadata)
      :values_(std::move(values)),registers_(values_),metadata_(std::move(metadata)),shift_(shift_size,0) {
    ValidateValues(values_);ValidateMetadata(metadata_);
  }
  const Values &Coefficients() const {return values_;}
  const Metadata &GeometryMetadata() const {return metadata_;}
  const Vector &ShiftWarmStart() const {return shift_;}
  std::uint64_t CompletedCycles() const {return completed_cycles_;}
  bool Active() const {return active_;}
  bool StageOpen() const {return stage_open_;}
  int NextStage() const {return next_stage_;}
  Real Time() const {return time_+(active_?abscissa_*dt_:Real(0));}
  Real Abscissa() const {return active_?abscissa_:Real(0);}
  Epoch CurrentEpoch() const {return {cycle_,geometry_,stage_open_?open_stage_:0};}

  // A moved puncture, changed chart, regrid or changed coefficient geometry
  // invalidates any solved beta, even if the volume coefficients are unchanged.
  void SetGeometryMetadata(Metadata metadata) {
    ValidateMetadata(metadata);
    // This class cannot transform a chart's meaning. An owner changing chart
    // parameters must supply transformed coefficients at an accepted boundary.
    if(metadata.chart!=metadata_.chart||metadata.chart_parameters!=metadata_.chart_parameters)
      throw std::logic_error("replace accepted coefficients when changing chart representation");
    if(active_&&metadata.mesh_revision!=metadata_.mesh_revision)
      throw std::logic_error("composite regrid requires an accepted cycle boundary");
    metadata_=std::move(metadata);InvalidateGeometry();
  }
  void InvalidateGeometry() {
    if(geometry_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("composite geometry epoch");
    ++geometry_;shift_valid_=false;
  }
  // Accepted-state replacement permits regridding and a different endpoint
  // layout. Caller supplies its explicit transferred warm start and metadata.
  void ReplaceAcceptedState(Values values,Vector warm_start,Metadata metadata) {
    if(active_)throw std::logic_error("cannot replace composite state during RK cycle");
    ValidateValues(values);ValidateFinite(warm_start);ValidateMetadata(metadata);
    values_=std::move(values);registers_=values_;shift_=std::move(warm_start);
    metadata_=std::move(metadata);InvalidateGeometry();
  }
  void BeginCycle(const NativeCompositeRK &rk,Real dt,Real accepted_time) {
    if(active_)throw std::logic_error("composite RK cycle already active");
    rk.Validate();
    if(!(dt>0)||!std::isfinite(dt)||!std::isfinite(accepted_time))
      throw std::invalid_argument("composite RK time/step");
    if(cycle_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("composite cycle epoch");
    rk_=rk;has_rk_=true;dt_=dt;time_=accepted_time;abscissa_=register_abscissa_=0;
    active_=true;stage_open_=false;next_stage_=1;open_stage_=0;++cycle_;InvalidateGeometry();
  }
  Epoch BeginStage(int stage) {
    if(!active_||stage_open_||stage!=next_stage_||stage<1||stage>rk_.stages)
      throw std::logic_error("composite RK stage order");
    Values updated=registers_;
    Real updated_c=register_abscissa_;
    if(stage==1){updated=values_;updated_c=abscissa_;}
    else if(rk_.integrator=="rk4"){
      Add(updated,values_,rk_.delta[stage-1]);updated_c+=rk_.delta[stage-1]*abscissa_;
    }
    ValidateValues(updated);registers_=std::move(updated);register_abscissa_=updated_c;
    stage_open_=true;open_stage_=stage;shift_valid_=false;return CurrentEpoch();
  }
  void InstallShift(const Epoch &epoch,Vector beta) {
    RequireEpoch(epoch);
    if(beta.size()!=shift_.size())throw std::invalid_argument("composite stage beta shape");
    ValidateFinite(beta);shift_=std::move(beta);shift_epoch_=epoch;shift_valid_=true;
  }
  const Vector &RequireShift(const Epoch &epoch) const {
    RequireEpoch(epoch);
    if(!shift_valid_||!(shift_epoch_==epoch))throw std::logic_error("composite tensor RHS needs current geometry/stage shift");
    return shift_;
  }
  const Values &TensorRHSState(const Epoch &epoch) const {RequireShift(epoch);return values_;}
  void AdvanceStage(const Epoch &epoch,const Values &rate) {
    RequireShift(epoch);
    if(!active_||!stage_open_||epoch.stage!=next_stage_)
      throw std::logic_error("composite RK stage is not ready to advance");
    RequireShape(values_,rate);ValidateValues(rate);const int s=open_stage_-1;
    Values next=values_;Combine(next,registers_,rate,rk_.gam0[s],rk_.gam1[s],rk_.beta[s]*dt_);
    ValidateValues(next);
    const Real next_c=rk_.gam0[s]*abscissa_+rk_.gam1[s]*register_abscissa_+rk_.beta[s];
    if(!std::isfinite(next_c))throw std::runtime_error("nonfinite composite stage time");
    values_=std::move(next);abscissa_=next_c;stage_open_=false;open_stage_=0;++next_stage_;InvalidateGeometry();
    if(next_stage_>rk_.stages){active_=false;next_stage_=1;++completed_cycles_;time_+=dt_;abscissa_=0;}
  }
  Snapshot Save() const {
    return {1,values_,registers_,metadata_,rk_,shift_,cycle_,completed_cycles_,geometry_,
        next_stage_,open_stage_,active_,stage_open_,has_rk_,time_,dt_,abscissa_,register_abscissa_};
  }
  // Serialization belongs to the native checkpoint owner. Restore preserves
  // coefficient/register bits, but never trusts a previously converged solve:
  // hierarchy/geometry rebuild requires a new shift for the restored epoch.
  static NativeCompositeStageState Restore(const Snapshot &s,const NativeCompositeRK &expected_rk) {
    if(s.version!=1)throw std::invalid_argument("composite stage snapshot version");
    expected_rk.Validate();
    if(s.has_rk){if(!(s.rk==expected_rk))throw std::invalid_argument("composite restart Driver coefficients changed");s.rk.Validate();}
    else if(s.active||s.cycle!=0||s.completed_cycles!=0||s.dt!=0)
      throw std::invalid_argument("composite restart has no RK binding");
    RequireShape(s.values,s.registers);ValidateValues(s.values);ValidateValues(s.registers);
    ValidateFinite(s.shift_warm_start);ValidateMetadata(s.metadata);
    if(!std::isfinite(s.time)||(s.has_rk?!(s.dt>0):s.dt!=0)||!std::isfinite(s.dt)||!std::isfinite(s.abscissa)||
        !std::isfinite(s.register_abscissa)||s.geometry==std::numeric_limits<std::uint64_t>::max()||
        (s.active&&s.completed_cycles==std::numeric_limits<std::uint64_t>::max())||
        (s.active?s.cycle!=s.completed_cycles+1:s.cycle!=s.completed_cycles)||
        (s.stage_open&&!s.active)||s.next_stage<1||s.next_stage>expected_rk.stages||
        (s.stage_open?s.open_stage!=s.next_stage:s.open_stage!=0)||
        (!s.active&&(s.next_stage!=1||s.abscissa!=0)))
      throw std::invalid_argument("invalid composite stage snapshot phase");
    NativeCompositeStageState out(s.values,s.shift_warm_start.size(),s.metadata);
    out.registers_=s.registers;out.shift_=s.shift_warm_start;out.rk_=expected_rk;out.has_rk_=s.has_rk;
    out.cycle_=s.cycle;out.completed_cycles_=s.completed_cycles;out.geometry_=s.geometry;
    out.next_stage_=s.next_stage;out.open_stage_=s.open_stage;out.active_=s.active;out.stage_open_=s.stage_open;
    out.time_=s.time;out.dt_=s.dt;out.abscissa_=s.abscissa;out.register_abscissa_=s.register_abscissa;
    out.InvalidateGeometry();return out;
  }
 private:
  Values values_,registers_;
  Metadata metadata_;
  NativeCompositeRK rk_;
  Vector shift_;
  Epoch shift_epoch_;
  std::uint64_t cycle_=0,completed_cycles_=0,geometry_=0;
  int next_stage_=1,open_stage_=0;
  bool active_=false,stage_open_=false,has_rk_=false,shift_valid_=false;
  Real time_=0,dt_=0,abscissa_=0,register_abscissa_=0;
  void RequireEpoch(const Epoch &epoch) const {
    if(!(epoch==CurrentEpoch()))throw std::logic_error("stale composite geometry/stage epoch");
  }
  static void ValidateFinite(const Vector &v) {for(Real x:v)if(!std::isfinite(x))throw std::invalid_argument("nonfinite composite coefficient");}
  static void ValidateValues(const Values &v){ValidateFinite(v.volume);ValidateFinite(v.scri);ValidateFinite(v.endpoints);}
  static void ValidateMetadata(const Metadata &m){ValidateFinite(m.chart_parameters);}
  static void RequireShape(const Values &a,const Values &b){
    if(a.volume.size()!=b.volume.size()||a.scri.size()!=b.scri.size()||a.endpoints.size()!=b.endpoints.size())
      throw std::invalid_argument("composite RK coefficient group shape");
  }
  static void Add(Values &a,const Values &b,Real factor){
    RequireShape(a,b);for(auto pair:{std::pair<Vector*,const Vector*>{&a.volume,&b.volume},{&a.scri,&b.scri},{&a.endpoints,&b.endpoints}})
      for(size_t i=0;i<pair.first->size();++i)(*pair.first)[i]+=factor*(*pair.second)[i];
  }
  static void Combine(Values &a,const Values &b,const Values &rate,Real g0,Real g1,Real step){
    RequireShape(a,b);RequireShape(a,rate);
    auto group=[&](Vector &v,const Vector &u,const Vector &f){for(size_t i=0;i<v.size();++i)v[i]=g0*v[i]+g1*u[i]+step*f[i];};
    group(a.volume,b.volume,rate.volume);group(a.scri,b.scri,rate.scri);group(a.endpoints,b.endpoints,rate.endpoints);
  }
};
}}  // namespace z4c::hyperboloidal
#endif
