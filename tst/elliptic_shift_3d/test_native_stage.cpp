// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <type_traits>
#include "z4c/hyperboloidal/native_composite_stage.hpp"
namespace h=z4c::hyperboloidal;
using Stage=h::NativeCompositeStageState;
using RK=h::NativeCompositeRK;
using Vector=Stage::Vector;
static_assert(std::is_same<decltype(RK::FromDriver(std::declval<const Driver&>())),RK>::value,
    "live native Driver accessor contract");

void Check(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
template<class F> void Reject(F action,const char *message){bool rejected=false;
  try{action();}catch(const std::exception &){rejected=true;}Check(rejected,message);}
Real Number(std::string expression){expression.erase(std::remove_if(expression.begin(),expression.end(),
  [](unsigned char c){return std::isspace(c);}),expression.end());
  const auto slash=expression.find('/');
  auto term=[](const std::string &s){size_t used=0;Real x=std::stod(s,&used);if(used!=s.size())throw std::runtime_error("native RK coefficient expression");return x;};
  return slash==std::string::npos?term(expression):term(expression.substr(0,slash))/term(expression.substr(slash+1));
}
// Parse the CURRENT native driver.cpp assignments instead of silently duplicating
// production coefficients in a test table. No Driver constructor or native task
// execution is claimed by this standalone component test.
RK ReadNative(const std::filesystem::path &file,const std::string &name){
  std::ifstream in(file);Check(bool(in),"cannot read actual native driver coefficient source");
  const std::string text((std::istreambuf_iterator<char>(in)),{});
  const auto first=text.find("integrator == \""+name+"\"");Check(first!=std::string::npos,"native RK method missing");
  const auto last=text.find("} else if (integrator",first+1);const auto block=text.substr(first,last-first);
  Real g0[4]{},g1[4]{},b[4]{},d[4];for(auto &x:d)x=std::numeric_limits<Real>::quiet_NaN();
  bool seen[4][4]{};const std::regex assignment(R"((gam0|gam1|beta|delta)\[([0-3])\]\s*=\s*([^;]+);)");
  for(std::sregex_iterator it(block.begin(),block.end(),assignment),end;it!=end;++it){
    const int field=(*it)[1]=="gam0"?0:((*it)[1]=="gam1"?1:((*it)[1]=="beta"?2:3));
    const int stage=std::stoi((*it)[2]);Real *values=field==0?g0:(field==1?g1:(field==2?b:d));
    values[stage]=Number((*it)[3]);seen[field][stage]=true;
  }
  const int stages=name=="rk3"?3:4;
  for(int field=0;field<(name=="rk4"?4:3);++field)for(int s=0;s<stages;++s)Check(seen[field][s],"incomplete native RK coefficient source");
  return RK::Capture(name,stages,0,g0,g1,b,d);
}
Vector Flatten(const Stage::Values &q){Vector out=q.volume;out.insert(out.end(),q.scri.begin(),q.scri.end());out.insert(out.end(),q.endpoints.begin(),q.endpoints.end());return out;}
Stage::Values Split(const Vector &q){Check(q.size()==12,"test ODE layout");return {{q.begin(),q.begin()+4},{q.begin()+4,q.begin()+7},{q.begin()+7,q.end()}};}
bool Same(const Stage::Values &a,const Stage::Values &b){return a.volume==b.volume&&a.scri==b.scri&&a.endpoints==b.endpoints;}
Vector ODE(const Vector &q,Real t){Vector out(q.size());for(size_t i=0;i<q.size();++i)
  out[i]=-.4*q[i]+.13*q[(i+1)%q.size()]-.07*q[(i+q.size()-1)%q.size()]+.02*(i+1)*t;return out;}
Stage::Metadata Meta(){Stage::Metadata m;m.chart="configurable-fractional-chart";m.chart_parameters={1.3,.67,-.03};m.mesh_revision=9;m.chart_revision=4;m.puncture_revision=2;return m;}
Stage Initial(){Vector q(12);for(size_t i=0;i<q.size();++i)q[i]=.17*(i+1)-.6;return Stage(Split(q),3,Meta());}
Vector Beta(int stage){return {.21*stage,-.3*stage,.17*(stage+2)};}

struct Result {Real recurrence=0,restart=0,beta_contamination=0,abscissa=0;};
Result Audit(const RK &rk){Result report;auto state=Initial();auto initial=state.Save();
  auto initial_restored=Stage::Restore(initial,rk);Check(Same(initial.values,initial_restored.Coefficients()),"initial accepted snapshot");
  const Real dt=.025,start=.7;state.BeginCycle(rk,dt,start);
  Vector reference=Flatten(initial.values),backup=reference;Real c=0,backup_c=0;
  const Real expected4[4]={0,1.193743905974738,.4314013217808047,.9999999999999964},expected3[3]={0,1,.5};
  Stage::Snapshot midway;
  Vector beta_register={-.9,.8,.4};
  for(int s=1;s<=rk.stages;++s){
    Reject([&]{state.BeginStage(s+1);},"out of order stage was accepted");
    const auto epoch=state.BeginStage(s);
    report.abscissa=std::max(report.abscissa,std::abs(state.Abscissa()-(rk.integrator=="rk4"?expected4[s-1]:expected3[s-1])));
    Check(std::abs(state.Time()-(start+c*dt))<2e-15,"native stage evaluation time");
    Reject([&]{state.TensorRHSState(epoch);},"tensor RHS used unsolved stage beta");
    if(s==1){backup=reference;backup_c=c;}
    else if(rk.integrator=="rk4"){for(size_t i=0;i<backup.size();++i)backup[i]+=rk.delta[s-1]*reference[i];backup_c+=rk.delta[s-1]*c;}
    const auto beta=Beta(s);state.InstallShift(epoch,beta);Check(state.RequireShift(epoch)==beta,"installed algebraic beta changed");
    const auto rate=ODE(reference,start+c*dt);
    if(s==2)midway=state.Save();
    // Demonstrate that ordinary native field mixing WOULD corrupt this supplied
    // stage beta, while the separate warm start remains exactly unchanged.
    for(int v=0;v<3;++v){const Real mixed=rk.gam0[s-1]*beta[v]+rk.gam1[s-1]*beta_register[v];
      report.beta_contamination=std::max(report.beta_contamination,std::abs(mixed-beta[v]));}
    state.AdvanceStage(epoch,Split(rate));Check(state.ShiftWarmStart()==beta,"beta entered a coefficient RK mixture");
    Reject([&]{state.RequireShift(state.CurrentEpoch());},"updated geometry retained valid old shift");
    Reject([&]{state.RequireShift(epoch);},"old geometry epoch was accepted");
    for(size_t i=0;i<reference.size();++i)reference[i]=rk.gam0[s-1]*reference[i]+rk.gam1[s-1]*backup[i]+rk.beta[s-1]*dt*rate[i];
    c=rk.gam0[s-1]*c+rk.gam1[s-1]*backup_c+rk.beta[s-1];
    const auto actual=Flatten(state.Coefficients());for(size_t i=0;i<actual.size();++i)report.recurrence=std::max(report.recurrence,std::abs(actual[i]-reference[i]));
  }
  Check(!state.Active()&&state.CompletedCycles()==1&&state.Time()==start+dt,"accepted cycle phase/time");
  auto restarted=Stage::Restore(midway,rk);
  Check(Same(restarted.Coefficients(),midway.values)&&Same(restarted.Save().registers,midway.registers),"restart changed coefficient/register bits");
  Check(restarted.ShiftWarmStart()==midway.shift_warm_start,"restart lost algebraic warm start");
  Reject([&]{restarted.RequireShift(restarted.CurrentEpoch());},"restart trusted a pre-rebuild solved shift");
  for(int s=2;s<=rk.stages;++s){const auto epoch=s==2?restarted.CurrentEpoch():restarted.BeginStage(s);
    restarted.InstallShift(epoch,Beta(s));const auto rate=ODE(Flatten(restarted.TensorRHSState(epoch)),restarted.Time());restarted.AdvanceStage(epoch,Split(rate));}
  Check(Same(restarted.Coefficients(),state.Coefficients())&&Same(restarted.Save().registers,state.Save().registers),"mid-stage continuation is not bitwise identical");
  auto accepted=Stage::Restore(state.Save(),rk);Check(Same(accepted.Coefficients(),state.Coefficients()),"accepted restart changed state");
  auto wrong=rk;wrong.beta[0]+=1e-8;Reject([&]{Stage::Restore(midway,wrong);},"restart accepted changed Driver coefficients");
  auto corrupt=midway;corrupt.open_stage=1;Reject([&]{Stage::Restore(corrupt,rk);},"restart accepted inconsistent phase");
  corrupt=midway;corrupt.registers.endpoints.pop_back();Reject([&]{Stage::Restore(corrupt,rk);},"restart accepted missing endpoint register");
  // Geometry metadata can carry changing regularity and moving punctures. The
  // state machine assumes no universal radial exponent and pins no beta value.
  auto geometry=Initial();geometry.BeginCycle(rk,dt,start);const auto old=geometry.BeginStage(1);geometry.InstallShift(old,Beta(1));
  auto metadata=geometry.GeometryMetadata();++metadata.puncture_revision;
  geometry.SetGeometryMetadata(metadata);Reject([&]{geometry.TensorRHSState(old);},"puncture/chart change did not invalidate beta");
  const auto fresh=geometry.CurrentEpoch();Reject([&]{geometry.TensorRHSState(fresh);},"new geometry reused stale beta");geometry.InstallShift(fresh,{.7,-.2,.1});
  Check(geometry.RequireShift(fresh)[0]==.7,"nonzero translating beta was pinned");
  auto changed=metadata;changed.chart="changed-regularity";changed.chart_parameters={.731,2.19};
  Reject([&]{geometry.SetGeometryMetadata(changed);},"changed chart silently reinterpreted existing RK coefficients");
  accepted.ReplaceAcceptedState({{.1,.2},{},{.3,.4,.5,.6,.7}},Vector(6,.9),changed);
  Check(accepted.Coefficients().endpoints.size()==5&&accepted.Coefficients().scri.empty(),"arbitrary endpoint/regrid layout was restricted");
  return report;
}

int main(int argc,char **argv){try{
  const auto source=argc>1?std::filesystem::path(argv[1]):std::filesystem::path(__FILE__).parent_path().parent_path().parent_path()/"src/driver/driver.cpp";
  const auto rk3=ReadNative(source,"rk3"),rk4=ReadNative(source,"rk4");const auto a=Audit(rk3),b=Audit(rk4);
  Check(std::max(a.recurrence,b.recurrence)<3e-15,"native coefficient recurrence error");
  Check(std::max(a.abscissa,b.abscissa)<2e-14,"native stage abscissae changed");
  Check(a.beta_contamination>.1&&b.beta_contamination>.1,"test did not exercise stale-beta mixing hazard");
  std::cout<<std::setprecision(17)<<"{\"methods\":[\"rk3\",\"rk4\"],\"driver_coefficients_parsed_from_source\":true,\"recurrence_max\":"<<std::max(a.recurrence,b.recurrence)
    <<",\"abscissa_max\":"<<std::max(a.abscissa,b.abscissa)<<",\"counterfactual_native_beta_mixing_error\":"<<b.beta_contamination
    <<",\"midstage_restart_bitwise\":true,\"geometry_epoch_checks\":true,\"generic_endpoint_and_chart_metadata\":true}\n";
  return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";return 1;}}
