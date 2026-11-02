// Optional additive scri-rate closure checks; no time evolution.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/hyperbolic_scri_device.hpp"
#include "z4c/hyperboloidal/algebraic_projection.hpp"

namespace h=z4c::hyperboloidal;
using Model=h::HyperbolicComposite;
using Device=h::HyperbolicScriDevice;
using V=std::vector<Real>;
static void Require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
static bool Free(int f){return f<18||(f>=19&&f<=21);}
static Device::Buffer Upload(const V&values){
  Device::Buffer result("closure test upload",values.size());auto host=Kokkos::create_mirror_view(result);
  for(size_t i=0;i<values.size();++i)host(i)=values[i];Kokkos::deep_copy(result,host);return result;
}
static V Download(Device::ConstBuffer values){
  auto host=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),values);V result(values.extent(0));
  for(size_t i=0;i<result.size();++i)result[i]=host(i);return result;
}
static Real Compare(const V&expected,const V&actual,int first=0){
  Real error=0;for(size_t i=0;i<actual.size();++i){Require(std::isfinite(actual[i]),"nonfinite closure output");
    const Real e=expected[31*first+i];error=std::max(error,std::abs(e-actual[i])/(1+std::abs(e)));}return error;
}
static void Seed(Model&model,bool perturb){
  auto fill=[&](const Real*x,Real*v){
    for(int f=0;f<31;++f)v[f]=(f==0?1.:0.)+(perturb?Real(.0001)*(1+f%4)*
      std::sin((2.1+.23*f)*x[0]+(.7+.11*f)*x[1]-(1.7+.17*f)*x[2]):0.);
    Require(h::ProjectAlgebraicNativeState(v).valid,"invalid seeded algebraic geometry");
  };
  for(size_t p=0;p<model.grid.active.size();++p)fill(model.grid.geometry[p].x,&model.state[31*p]);
  for(size_t p=0;p<model.scri.rows.size();++p)fill(model.scri.rows[p].geometry.x,&model.scri.values[31*p]);
  model.BoundaryValues();
}
// Independent differentiated null relation. Adding Q to geometric rates must
// change D by this amount even though alpha and beta are algebraic boundary data.
static Real CheckTraceResponse(const Model&model,const V&projected,const V&q,
                               const V&before,const V&after){
  Real error=0,change=0;
  for(size_t p=0;p<model.scri.rows.size();++p){
    const auto*x=model.scri.rows[p].geometry.x;const auto*v=&projected[31*p];h::MetricJet<Real> metric{};
    for(int t=0;t<6;++t){const int i=Model::ti[t],j=Model::tj[t];metric.g[i][j]=metric.g[j][i]=v[1+t]+(i==j);}
    const auto geometry=h::Geometry(metric);Real inverse_normal[3]{},normal_metric=0,metric_rate=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)inverse_normal[i]+=geometry.inverse[i][j]*x[j];
    for(int i=0;i<3;++i)normal_metric+=x[i]*inverse_normal[i];
    for(int t=0;t<6;++t){const int i=Model::ti[t],j=Model::tj[t];
      metric_rate-=(i==j?1:2)*inverse_normal[i]*q[31*p+1+t]*inverse_normal[j];}
    const Real speed_rate=(q[31*p]*normal_metric+v[0]*metric_rate)/(2*std::sqrt(v[0]*normal_metric));
    const Real mu=h::ShiftCoefficients(Real(1),Real(1),v[0]).mu;
    for(int a=0;a<3;++a){const Real expected=-x[a]*speed_rate/mu;
      error=std::max(error,std::abs(after[31*p+25+a]-before[31*p+25+a]-expected));
      change=std::max(change,std::abs(expected));}
    Require(Model::NullResidual(x,&after[31*p])<3e-14,"additive rate broke algebraic nullness");
  }
  Require(change>1e-9,"fixture does not exercise moving-normal dependent D");return error;
}
// Independent absolute zero-restoration closure oracle, using actual geometric
// rates (including Q/penalties) and nonzero physical shift gradients.
static void CheckZeroTrace(const Model&model,const V&projected,const V&rate,const V&trace){
  if constexpr(!h::zero_shift_restoration)return;
  const auto padded=model.ActivePadded(model.state);Real error=0,null_rate=0,old_limit=0,geometry_rate=0;
  for(size_t p=0;p<model.scri.rows.size();++p){
    const auto*x=model.scri.rows[p].geometry.x;
    const auto sample=model.scri.Sample(p,padded,projected,31);
    const auto u=Model::Jet(x,sample.value,sample.d,sample.dd);const auto g=h::Geometry(u.metric);
    Real invn[3]{},q=0,qdot=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)invn[i]+=g.inverse[i][j]*x[j];
    for(int i=0;i<3;++i)q+=x[i]*invn[i];
    for(int t=0;t<6;++t){int i=Model::ti[t],j=Model::tj[t];qdot-=(i==j?1:2)*invn[i]*rate[31*p+1+t]*invn[j];}
    const Real speed=std::sqrt(u.chi.value*q),sdot=(rate[31*p]*q+u.chi.value*qdot)/(2*speed);
    Real beta_normal=0,beta_normal_rate=0;
    for(int i=0;i<3;++i){Real adv=0,old=0;
      for(int j=0;j<3;++j){adv+=u.beta.value[j]*u.beta.d[j][i];old+=4*x[j]*sample.d[22+i][j];}
      const Real beta_rate=adv-x[i]+1.5*u.chi.value*trace[31*p+25+i];
      error=std::max(error,std::abs(beta_rate+sdot*x[i]));old_limit=std::max(old_limit,std::abs(old));
      beta_normal+=x[i]*u.beta.value[i];beta_normal_rate+=x[i]*beta_rate;
    }
    null_rate=std::max(null_rate,std::abs(2*beta_normal*beta_normal_rate-rate[31*p]*q-u.chi.value*qdot));
    geometry_rate=std::max(geometry_rate,std::abs(sdot));
  }
  Require(error<3e-11&&null_rate<3e-11,"zero-restoration absolute/differentiated null compatibility failed");
  Require(old_limit>1e-8&&geometry_rate>1e-8,"zero-restoration fixture is trivial");
  std::cout<<"zero_absolute_trace_error="<<error<<" differentiated_null_error="<<null_rate
           <<" removed_limit_nonzero="<<old_limit<<" geometry_speed_rate="<<geometry_rate<<'\n';
}
static void DeviceChecks(Model&model,const V&q,const V*cpu_expected=nullptr,const V*cpu_trace=nullptr){
  const int total=model.scri.rows.size(),split=total/2+7;const Real dt=.0001;
  const auto projected=model.scri.values;const auto active=Upload(model.state),surface=Upload(projected);
  Device full(model,0,total),left(model,0,split),right(model,split,total-split),empty(model,total,0);
  Require(full.Evaluate(active,surface,dt).bad_rows==0,"baseline device closure failed");
  const auto baseline=Download(full.LocalRHS()),baseline_trace=Download(full.LocalSurface());
  Require(full.Evaluate(active,surface,dt,Device::ConstBuffer{}).bad_rows==0&&
          baseline==Download(full.LocalRHS())&&baseline_trace==Download(full.LocalSurface()),
          "absent additive input changed baseline bits");
  V zeros(q.size(),0);Require(full.Evaluate(active,surface,dt,Upload(zeros)).bad_rows==0&&
          baseline==Download(full.LocalRHS())&&baseline_trace==Download(full.LocalSurface()),
          "zero additive input changed baseline values");
  auto expected=baseline;for(size_t i=0;i<q.size();++i)if(Free(i%31))expected[i]+=q[i];
  const auto rate=Upload(q);const auto summary=full.Evaluate(active,surface,dt,rate);
  Require(summary.bad_rows==0&&summary.compatibility<2e-12,"additive native closure failed");
  const auto actual=Download(full.LocalRHS()),trace=Download(full.LocalSurface());
  CheckZeroTrace(model,projected,actual,trace);
  const Real additive_error=Compare(expected,actual),trace_error=CheckTraceResponse(model,projected,q,baseline_trace,trace);
  Require(additive_error<2e-13&&trace_error<3e-12,"additive rate or dependent-D response incorrect");
  const V qleft(q.begin(),q.begin()+31*split),qright(q.begin()+31*split,q.end());
  Require(left.Evaluate(active,surface,dt,Upload(qleft)).bad_rows==0&&
          right.Evaluate(active,surface,dt,Upload(qright)).bad_rows==0&&
          empty.Evaluate(active,surface,dt).bad_rows==0,"partitioned additive evaluation failed");
  Require(Compare(actual,Download(left.LocalRHS()))<2e-13&&Compare(actual,Download(right.LocalRHS()),split)<2e-13&&
          Compare(trace,Download(left.LocalSurface()))<2e-13&&Compare(trace,Download(right.LocalSurface()),split)<2e-13,
          "partitioned additive closure differs");
  const Real cpu_rate_error=cpu_expected?Compare(*cpu_expected,actual):0;
  const Real cpu_trace_error=cpu_trace?Compare(*cpu_trace,trace):0;
  if(cpu_expected)Require(cpu_rate_error<2e-10&&cpu_trace_error<2e-10,
                         "CPU/device angular closure parity failed");
  bool rejected=false;try{full.Evaluate(active,surface,dt,Upload(V(q.size()+1,0)));}catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"wrong additive shape accepted");
  rejected=false;try{full.Evaluate(active,surface,dt,full.LocalRHS());}catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"writable additive alias accepted");
  auto invalid=q;invalid[31*7+25]=1;auto bad=full.Evaluate(active,surface,dt,Upload(invalid));
  Require(bad.bad_rows==1&&bad.first_bad==7&&bad.reason==Device::invalid_input,"independent D filtering accepted");
  invalid=q;invalid[31*2]=std::numeric_limits<Real>::quiet_NaN();bad=full.Evaluate(active,surface,dt,Upload(invalid));
  Require(bad.bad_rows==1&&bad.first_bad==2&&bad.reason==Device::invalid_input,"nonfinite additive rate accepted");
  std::cout<<"closure_case="<<(cpu_expected?"plan":"synthetic")
           <<" additive_rate_error="<<additive_error<<" dependent_D_response_error="<<trace_error
           <<" compatibility="<<summary.compatibility<<" cpu_rate_error="<<cpu_rate_error
           <<" cpu_trace_error="<<cpu_trace_error<<" partition_parity=1 zero_off=1 rejection_checks=1\n";
}
static void PlanChecks(){
  const auto mesh=cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3);
  Model implicit_off(mesh,false,true),explicit_off(mesh,false,true,0),enabled(mesh,false,true,.3);
  Require(!implicit_off.angular_plan&&!explicit_off.angular_plan&&enabled.angular_plan,
          "angular plan construction does not respect zero-off setting");
  const auto shared=enabled.angular_plan;
  Require(shared==enabled.angular_plan,"copy did not retain immutable shared angular plan");
  const Real dt=.0001;
  for(bool perturb:{false,true}){
    Seed(implicit_off,perturb);Seed(explicit_off,perturb);Seed(enabled,perturb);
    const auto projected=enabled.scri.values;
    const auto q=enabled.angular_plan->Apply(projected,enabled.angular_epsilon);
    const auto off=implicit_off.BoundaryRates(implicit_off.ActivePadded(implicit_off.state),dt);
    const auto off_trace=implicit_off.scri.values;
    const auto off_explicit=explicit_off.BoundaryRates(explicit_off.ActivePadded(explicit_off.state),dt);
    Require(off==off_explicit&&off_trace==explicit_off.scri.values,"explicit epsilon zero changed CPU baseline");
    const auto actual=enabled.BoundaryRates(enabled.ActivePadded(enabled.state),dt),trace=enabled.scri.values;
    Require(enabled.angular_cfl>0&&enabled.angular_cfl<=2.7&&enabled.angular_cfl_max>=enabled.angular_cfl,
            "angular CFL not exposed correctly");
    auto expected=off;for(size_t i=0;i<q.size();++i){
      if(Free(i%31))expected[i]+=q[i];else Require(q[i]==0,"plan filters a dependent field");}
    Require(Compare(expected,actual)<2e-13,"CPU angular rates differ from supplied free-field Q");
    if(!perturb){
      for(Real value:q)Require(value==0,"constant Minkowski free fields received dissipation");
      Require(off==actual&&off_trace==trace,"Minkowski closure changed with angular dissipation");
    }else{
      Require(CheckTraceResponse(enabled,projected,q,off_trace,trace)<3e-12,"CPU Q omitted moving-normal D response");
      enabled.scri.values=projected;DeviceChecks(enabled,q,&actual,&trace);
    }
    std::cout<<"plan_perturb="<<perturb<<" host_rate_error="<<Compare(expected,actual)
             <<" angular_cfl="<<enabled.angular_cfl<<" null="<<enabled.null_max
             <<" compatibility="<<enabled.compatibility_max<<'\n';
  }
  bool rejected=false;
  try{enabled.CheckAngularCFL(2.8/(enabled.angular_epsilon*6*enabled.angular_plan->SpectralUpperBound()));}
  catch(const std::runtime_error&){rejected=true;}
  Require(rejected&&enabled.angular_cfl>2.7&&enabled.angular_cfl_max>=enabled.angular_cfl,"isolated angular CFL guard absent");
  rejected=false;try{Model invalid(mesh,false,false,.3);}catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"dissipation accepted without cross-face jets");
  rejected=false;try{Model invalid(mesh,false,true,-1);}catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"negative angular strength accepted");
  rejected=false;try{Model invalid(mesh,false,true,std::numeric_limits<Real>::quiet_NaN());}
  catch(const std::invalid_argument&){rejected=true;}
  Require(rejected,"nonfinite angular strength accepted");
  std::cout<<"cfl_rejection=1 cross_face_guard=1 epsilon_validation=1 shared_plan=1\n";
}
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  const char*local_rank=std::getenv("MPI_LOCALRANKID");
  const std::string rank=local_rank?local_rank:"0";
  const std::string report=argc==2?std::string(argv[1])+".rank"+rank+".json":"";
  if(!report.empty()&&std::ifstream(report).good()){Kokkos::finalize();return 2;}
  try{
    std::cout<<std::setprecision(17)<<"backend="<<Kokkos::DefaultExecutionSpace::name()<<'\n';
    Model model(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3),false,true);
    Seed(model,true);V synthetic(model.scri.values.size(),0);
    for(size_t p=0;p<model.scri.rows.size();++p){const auto*x=model.scri.rows[p].geometry.x;
      for(int f=0;f<31;++f)if(Free(f))synthetic[31*p+f]=.003*(1+f%3)*std::sin((3.1+.19*f)*x[0]-1.7*x[1]+2.3*x[2]);}
    DeviceChecks(model,synthetic);
    PlanChecks();
    std::cout<<"Optional additive scri closure checks passed\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  if(!report.empty()){
    std::ofstream out(report);out<<"{\"passed\":"<<(result==0?"true":"false")
      <<",\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"rank\":"<<rank
      <<",\"zero_shift_restoration\":"<<(h::zero_shift_restoration?"true":"false")<<"}\n";
    if(!out)result=1;
  }
  Kokkos::finalize();return result;
}
