// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "mesh/cubed_sphere/map.hpp"
#include "z4c/hyperboloidal/positive_field_chart.hpp"
#include "z4c/hyperboloidal/wormhole_chart.hpp"
#include "z4c/hyperboloidal/composite_state.hpp"

namespace h=z4c::hyperboloidal;
using J=cubed_sphere::Jet;
using Mode=h::PositiveFieldMode;
using Status=h::PositiveFieldStatus;

J Exp(const J &q){J out(std::exp(q.f));for(int a=0;a<3;++a){out.d[a]=out.f*q.d[a];
  for(int b=0;b<3;++b)out.h[a][b]=out.f*(q.h[a][b]+q.d[a]*q.d[b]);}return out;}
h::ScalarJet<Real> Scalar(const J &q){h::ScalarJet<Real> out{};out.value=q.f;
  for(int a=0;a<3;++a){out.d[a]=q.d[a];for(int b=0;b<3;++b)out.dd[a][b]=q.h[a][b];}return out;}
void Require(Status status,const char *where){if(status!=Status::valid)
  throw std::runtime_error(std::string(where)+": "+h::PositiveFieldStatusName(status));}
const char *Name(Mode mode){return mode==Mode::direct?"direct":(mode==Mode::log_reference?"log_reference":"square");}

// The same general AD algebra independently supplies three spatial directions
// or a time direction. Its exponent changes with time and its angular factor
// is not a regular-center polynomial. No geometric power is put in the chart.
std::pair<J,J> Manufactured(const J x[3],const J &time,Real power){
  const Real center[3]={.03,-.04,.01},velocity[3]={.02,-.01,.03};
  J y[3],rho2(0);for(int i=0;i<3;++i){y[i]=x[i]-(J(center[i])+J(velocity[i])*time);rho2=rho2+y[i]*y[i];}
  const J angular=J(.07)*y[0]*y[1]/rho2+J(.04)*y[2]*y[2]/rho2;
  const J exponent=J(power)-J(.13)*(time-J(.17));
  const J field=Exp(exponent*cubed_sphere::Log(rho2)*J(.5)+angular+J(.03)*x[0]+J(.02)*time);
  const J reference=Exp(J(.01)*time)*(J(1)+J(.3)*Exp(J(-25)*rho2));
  return {field,reference};
}
J Stored(Mode mode,const J &field,const J &reference){
  if(mode==Mode::direct)return field;
  if(mode==Mode::square)return field*field;
  return cubed_sphere::Log(field)-cubed_sphere::Log(reference);
}
Real JetError(const h::ScalarJet<Real> &a,const h::ScalarJet<Real> &b,Real scale,Real rho){
  Real error=std::abs(a.value-b.value)/scale;
  for(int i=0;i<3;++i){error=std::max(error,std::abs(a.d[i]-b.d[i])*rho/scale);
    for(int j=0;j<3;++j)error=std::max(error,std::abs(a.dd[i][j]-b.dd[i][j])*rho*rho/scale);}
  return error;
}
h::OmegaJet<Real> Omega(const Real x[3],const h::Z4cJet<Real> &u){h::OmegaJet<Real> o{};
  o.omega=(1-x[0]*x[0]-x[1]*x[1]-x[2]*x[2])/2;
  for(int i=0;i<3;++i){o.gradient[i]=-x[i];o.hessian[i][i]=-1;}
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;}
h::Z4cJet<Real> OffConstraint(const Real point[3],const h::ScalarJet<Real> &chi,const h::ScalarJet<Real> &alpha){
  h::Z4cJet<Real> u{};u.chi=chi;u.alpha=alpha;J x[3];for(int a=0;a<3;++a)x[a]=J::Variable(point[a],a);
  const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  const J g[6]={J(1)+J(.04)*x[0],J(.015)+J(.01)*x[2],J(.007)*x[0]*x[1],
                J(1)-J(.03)*x[1],J(-.012)+J(.005)*x[0],J(1)+J(.02)*x[2]};
  for(int t=0;t<6;++t){const int i=ti[t],j=tj[t];const J a=J(.002*(t+1))+J(.003)*x[t%3];
    u.metric.g[i][j]=u.metric.g[j][i]=g[t].f;u.a.k[i][j]=u.a.k[j][i]=a.f;
    for(int d=0;d<3;++d){u.metric.dg[d][i][j]=u.metric.dg[d][j][i]=g[t].d[d];u.a.dk[d][i][j]=u.a.dk[d][j][i]=a.d[d];
      for(int e=0;e<3;++e)u.metric.ddg[d][e][i][j]=u.metric.ddg[d][e][j][i]=g[t].h[d][e];}}
  u.trace=Scalar(J(-1)+J(.02)*x[0]);u.theta=Scalar(J(.004)+J(.003)*x[1]);
  for(int v=0;v<3;++v){const J lambda=J(.02*(v+1))+J(.01)*x[v];
    const J beta=J(.01*(v+1))-x[v]+J(.01)*x[(v+1)%3]*x[(v+2)%3];
    u.lambda.value[v]=lambda.f;u.beta.value[v]=beta.f;
    for(int a=0;a<3;++a){u.lambda.d[a][v]=lambda.d[a];u.beta.d[a][v]=beta.d[a];
      for(int b=0;b<3;++b){u.lambda.dd[a][b][v]=lambda.h[a][b];u.beta.dd[a][b][v]=beta.h[a][b];}}}
  return u;
}
void Compare(Real a,Real b,Real &worst){if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite equivalence comparison");
  worst=std::max(worst,std::abs(a-b)/std::max({Real(1),std::abs(a),std::abs(b)}));}
void CompareRHS(const h::Z4cRHS<Real> &a,const h::Z4cRHS<Real> &b,Real &worst){
  Compare(a.chi,b.chi,worst);Compare(a.trace,b.trace,worst);Compare(a.theta,b.theta,worst);
  for(int i=0;i<3;++i){Compare(a.lambda[i],b.lambda[i],worst);
    for(int j=0;j<3;++j){Compare(a.metric[i][j],b.metric[i][j],worst);Compare(a.a[i][j],b.a[i][j],worst);}}}

void CheckComposite(const Real point[3],const h::Z4cJet<Real> &physical,Real rho,Mode mode,
                    Real &mapping_error,Real &tensor_error,Real &lapse_error){
  std::vector<Real> values(22,0);std::vector<std::array<Real,3>> d(22);
  std::vector<std::array<std::array<Real,3>,3>> dd(22);
  auto put=[&](int f,const h::ScalarJet<Real> &q){values[f]=q.value;
    for(int a=0;a<3;++a){d[f][a]=q.d[a];for(int b=0;b<3;++b)dd[f][a][b]=q.dd[a][b];}};
  for(int t=0;t<6;++t){const int i=h::kCompositeTensorI[t],j=h::kCompositeTensorJ[t];h::ScalarJet<Real> g{},a{};
    g.value=physical.metric.g[i][j]-(i==j?1:0);a.value=physical.a.k[i][j];
    for(int k=0;k<3;++k){g.d[k]=physical.metric.dg[k][i][j];a.d[k]=physical.a.dk[k][i][j];
      for(int l=0;l<3;++l)g.dd[k][l]=physical.metric.ddg[k][l][i][j];}
    put(1+t,g);put(7+t,a);
  }
  auto trace=physical.trace;trace.value+=3;put(13,trace);put(14,physical.theta);
  for(int v=0;v<3;++v){h::ScalarJet<Real> lambda{};lambda.value=physical.lambda.value[v];
    for(int a=0;a<3;++a){lambda.d[a]=physical.lambda.d[a][v];for(int b=0;b<3;++b)lambda.dd[a][b]=physical.lambda.dd[a][b][v];}put(15+v,lambda);}
  auto reference=h::UnitPositiveFieldReference<Real>();reference.value=(1+point[0]*point[0]+point[1]*point[1]+point[2]*point[2])/2;
  for(int a=0;a<3;++a){reference.d[a]=point[a];reference.dd[a][a]=1;}
  const h::PositiveFieldChart<Real> chi_chart(mode),alpha_chart(mode,reference);h::ScalarJet<Real> stored{};
  Require(chi_chart.Encode(physical.chi,stored),"pack actual composite chi");put(0,stored);
  Require(alpha_chart.Encode(physical.alpha,stored),"pack actual composite alpha");put(18,stored);
  const int storage=mode==Mode::direct?1:(mode==Mode::log_reference?2:3);
  // Deliberately pass a nonzero legacy isotropic-reference selector. It must
  // not apply the legacy W weight to any signed field in these new modes.
  const auto actual=h::UnpackComposite(point,values,d,dd,true,2,storage);
  auto expected=physical;expected.beta={};for(int v=0;v<3;++v){expected.beta.value[v]=-point[v];expected.beta.d[v][v]=-1;}
  mapping_error=std::max(mapping_error,JetError(actual.chi,expected.chi,expected.chi.value,rho));
  mapping_error=std::max(mapping_error,JetError(actual.alpha,expected.alpha,expected.alpha.value,rho));
  Compare(actual.trace.value,expected.trace.value,mapping_error);Compare(actual.theta.value,expected.theta.value,mapping_error);
  for(int i=0;i<3;++i){Compare(actual.lambda.value[i],expected.lambda.value[i],mapping_error);Compare(actual.beta.value[i],expected.beta.value[i],mapping_error);
    Compare(actual.trace.d[i],expected.trace.d[i],mapping_error);Compare(actual.theta.d[i],expected.theta.d[i],mapping_error);
    for(int j=0;j<3;++j){Compare(actual.metric.g[i][j],expected.metric.g[i][j],mapping_error);Compare(actual.a.k[i][j],expected.a.k[i][j],mapping_error);
      Compare(actual.trace.dd[i][j],expected.trace.dd[i][j],mapping_error);Compare(actual.theta.dd[i][j],expected.theta.dd[i][j],mapping_error);
      Compare(actual.lambda.d[j][i],expected.lambda.d[j][i],mapping_error);Compare(actual.beta.d[j][i],expected.beta.d[j][i],mapping_error);
      for(int a=0;a<3;++a){Compare(actual.metric.dg[a][i][j],expected.metric.dg[a][i][j],mapping_error);Compare(actual.a.dk[a][i][j],expected.a.dk[a][i][j],mapping_error);
        Compare(actual.lambda.dd[j][a][i],expected.lambda.dd[j][a][i],mapping_error);Compare(actual.beta.dd[j][a][i],expected.beta.dd[j][a][i],mapping_error);
        for(int b=0;b<3;++b)Compare(actual.metric.ddg[a][b][i][j],expected.metric.ddg[a][b][i][j],mapping_error);}}}
  const auto p0=h::ConformalRHS(expected,Omega(point,expected),4.,-.99),p1=h::ConformalRHS(actual,Omega(point,actual),4.,-.99);
  if(!p0.valid||!p1.valid)throw std::runtime_error("invalid actual composite tensor jet");
  CompareRHS(p0.regular,p1.regular,tensor_error);CompareRHS(p0.pole,p1.pole,tensor_error);
  Compare(h::WormholeLapseRate(point,expected),h::WormholeLapseRate(point,actual),lapse_error);
}

int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;
try{
  Real jet_error=0,rate_error=0,tensor_error=0,lapse_error=0,omitted_reference_time_error=0,unpack_error=0;int cases=0;
  constexpr Real time=.17;const Real center[3]={.03+.02*time,-.04-.01*time,.01+.03*time};
  for(Real power:{2.,4.,1.09,1.5})for(Real rho:{1e-6,1e-3,.03,.2})
  for(const auto direction:{std::array<Real,3>{1,2,3},std::array<Real,3>{-2,1,2}}){
    const Real length=std::sqrt(direction[0]*direction[0]+direction[1]*direction[1]+direction[2]*direction[2]);
    Real point[3];J space[3],fixed[3];for(int a=0;a<3;++a){point[a]=center[a]+rho*direction[a]/length;space[a]=J::Variable(point[a],a);fixed[a]=J(point[a]);}
    const auto base=Manufactured(space,J(time),power),timed=Manufactured(fixed,J::Variable(time,0),power);
    const auto other=Manufactured(space,J(time),power==4.?1.09:4.);
    const auto other_time=Manufactured(fixed,J::Variable(time,0),power==4.?1.09:4.);
    const auto physical=Scalar(base.first),reference=Scalar(base.second);
    const auto original=OffConstraint(point,physical,Scalar(other.first));const auto oparts=h::ConformalRHS(original,Omega(point,original),4.,-.99);
    if(!oparts.valid)throw std::runtime_error("invalid original off-constraint tensor jet");
    for(Mode mode:{Mode::direct,Mode::log_reference,Mode::square}){
      h::PositiveFieldChart<Real> chart(mode,reference,timed.second.d[0]);h::ScalarJet<Real> encoded{},reconstructed{};
      Require(chart.Encode(physical,encoded),"encode analytic field");Require(chart.Reconstruct(encoded,reconstructed),"reconstruct analytic field");
      const auto expected=Scalar(Stored(mode,base.first,base.second));
      const Real stored_scale=mode==Mode::log_reference?std::max(Real(1),std::abs(expected.value)):expected.value;
      jet_error=std::max(jet_error,JetError(encoded,expected,stored_scale,rho));
      jet_error=std::max(jet_error,JetError(reconstructed,physical,physical.value,rho));
      Real stored_rate=0,physical_rate=0;Require(chart.EncodeRate(physical.value,timed.first.d[0],stored_rate),"encode analytic time rate");
      Require(chart.ReconstructRate(encoded.value,stored_rate,physical_rate),"reconstruct analytic time rate");
      const Real expected_rate=Stored(mode,timed.first,timed.second).d[0];
      Compare(stored_rate,expected_rate,rate_error);
      rate_error=std::max(rate_error,std::abs(physical_rate-timed.first.d[0])*rho/physical.value);
      auto reconstructed_state=original;reconstructed_state.chi=reconstructed;
      h::PositiveFieldChart<Real> alpha_chart(mode,Scalar(other.second),other_time.second.d[0]);
      h::ScalarJet<Real> alpha_encoded{};Require(alpha_chart.Encode(Scalar(other.first),alpha_encoded),"encode lapse");
      Require(alpha_chart.Reconstruct(alpha_encoded,reconstructed_state.alpha),"reconstruct lapse");
      const auto parts=h::ConformalRHS(reconstructed_state,Omega(point,reconstructed_state),4.,-.99);
      if(!parts.valid)throw std::runtime_error("invalid reconstructed off-constraint tensor jet");
      CompareRHS(parts.regular,oparts.regular,tensor_error);CompareRHS(parts.pole,oparts.pole,tensor_error);
      h::Z4cRHS<Real> rhs0{},rhs1{};
      if(!h::AssembleInterior(oparts,Omega(point,original).omega,rhs0)||!h::AssembleInterior(parts,Omega(point,reconstructed_state).omega,rhs1))
        throw std::runtime_error("invalid assembled off-constraint RHS");
      CompareRHS(rhs0,rhs1,tensor_error);
      Compare(h::WormholeLapseRate(point,original),h::WormholeLapseRate(point,reconstructed_state),lapse_error);
      CheckComposite(point,original,rho,mode,unpack_error,tensor_error,lapse_error);
      // No beta value, gradient or Hessian has been reinterpreted by the chart.
      for(int v=0;v<3;++v){if(original.beta.value[v]!=reconstructed_state.beta.value[v])throw std::runtime_error("beta changed");
        for(int a=0;a<3;++a){if(original.beta.d[a][v]!=reconstructed_state.beta.d[a][v])throw std::runtime_error("beta derivative changed");
          for(int b=0;b<3;++b)if(original.beta.dd[a][b][v]!=reconstructed_state.beta.dd[a][b][v])throw std::runtime_error("beta Hessian changed");}}
      if(mode==Mode::log_reference){auto wrong=chart;wrong.reference_time=0;
        Require(wrong.ReconstructRate(encoded.value,stored_rate,physical_rate),"intentional missing-reference-time control");
        omitted_reference_time_error=std::max(omitted_reference_time_error,std::abs(physical_rate-timed.first.d[0])/physical.value);}
      ++cases;
    }
  }
  if(jet_error>3e-11||unpack_error>3e-11||rate_error>3e-10||tensor_error>3e-10||lapse_error>3e-10||omitted_reference_time_error<1e-4)
    throw std::runtime_error("positive chart representation equivalence failed");

  int expected_range_failures=0,zero_endpoint_errors=0;std::string conditioning;
  for(Real value:{1e-120,1e-160,1e-200,1e200})for(Mode mode:{Mode::direct,Mode::log_reference,Mode::square}){
    h::ScalarJet<Real> field{};field.value=value;field.d[0]=2*value;field.dd[0][0]=3*value;
    h::PositiveFieldChart<Real> chart(mode);h::ScalarJet<Real> stored{},back{};
    const auto status=chart.Encode(field,stored);
    const bool should_fail=mode==Mode::square&&value!=1e-120;
    if((status!=Status::valid)!=should_fail)throw std::runtime_error("unexpected extreme-range control status");
    Real jacobian=0,relative=0;
    if(status==Status::valid){Require(chart.Reconstruct(stored,back),"small-field reconstruction");Require(chart.ReconstructionJacobian(stored.value,jacobian),"conditioning Jacobian");relative=std::abs(back.value/value-1);
      if(relative>2e-12)throw std::runtime_error("small-field reconstruction lost accuracy");}
    else{if(status!=Status::underflow&&status!=Status::overflow&&status!=Status::subnormal_value)throw std::runtime_error("range failure was hidden");++expected_range_failures;}
    if(!conditioning.empty())conditioning+=",";
    std::ostringstream row;row<<std::setprecision(17)<<"{\"physical_value\":"<<value<<",\"mode\":\""<<Name(mode)<<"\",\"status\":\""<<h::PositiveFieldStatusName(status)<<"\",\"relative_value_error\":";
    if(status==Status::valid)row<<relative<<",\"absolute_reconstruction_Jacobian\":"<<jacobian;else row<<"null,\"absolute_reconstruction_Jacobian\":null";
    row<<"}";conditioning+=row.str();
  }
  for(Mode mode:{Mode::log_reference,Mode::square}){
    h::PositiveFieldChart<Real> chart(mode);h::ScalarJet<Real> zero{},out{};out.value=123;
    if(chart.Encode(zero,out)!=Status::endpoint_data_required||out.value!=123)throw std::runtime_error("zero encoding silently changed output");
    if(mode==Mode::log_reference)zero.value=-std::numeric_limits<Real>::infinity();
    if(chart.Reconstruct(zero,out)!=Status::endpoint_data_required||out.value!=123)throw std::runtime_error("zero derivative limit invented");
    Real rate=456;if(chart.ReconstructRate(zero.value,1.,rate)!=Status::endpoint_data_required||rate!=456)throw std::runtime_error("zero rate limit invented");++zero_endpoint_errors;
  }
  h::PositiveFieldChart<Real> log_chart(Mode::log_reference);h::ScalarJet<Real> extreme{},out{};
  extreme.value=-1000;if(log_chart.Reconstruct(extreme,out)!=Status::underflow)throw std::runtime_error("log underflow not reported");++expected_range_failures;
  extreme.value=1000;if(log_chart.Reconstruct(extreme,out)!=Status::overflow)throw std::runtime_error("log overflow not reported");++expected_range_failures;
  auto invalid=log_chart;invalid.reference.value=0;if(invalid.Encode(extreme,out)!=Status::invalid_reference)throw std::runtime_error("invalid reference accepted");
  h::PositiveFieldChart<Real> direct;h::ScalarJet<Real> endpoint{};endpoint.d[0]=.2;
  Require(direct.Reconstruct(endpoint,out),"explicit direct endpoint jet");if(out.d[0]!=.2)throw std::runtime_error("direct endpoint jet was changed");
  endpoint.value=-1;if(direct.Encode(endpoint,out)!=Status::negative_value)throw std::runtime_error("negative physical field was clipped or accepted");
  int unpack_endpoint_errors=0;std::vector<Real> zero_values(22,0);std::vector<std::array<Real,3>> zero_d(22);
  std::vector<std::array<std::array<Real,3>,3>> zero_dd(22);const Real point[3]={0,0,0};
  for(int storage:{2,3}){zero_values[0]=storage==2?-std::numeric_limits<Real>::infinity():0;zero_values[18]=storage==2?0:1;
    try{h::UnpackComposite(point,zero_values,zero_d,zero_dd,true,0,storage);throw std::runtime_error("composite endpoint was inferred");}
    catch(const std::runtime_error &e){if(std::string(e.what()).find("endpoint_data_required")==std::string::npos)throw;++unpack_endpoint_errors;}}
  bool invalid_mode=false;try{h::UnpackComposite(point,zero_values,zero_d,zero_dd,true,0,4);}catch(const std::invalid_argument &){invalid_mode=true;}
  if(!invalid_mode)throw std::runtime_error("invalid composite positive-storage mode accepted");
  std::cout<<std::setprecision(17)<<"{\"cases\":"<<cases<<",\"jet_error\":"<<jet_error<<",\"rate_error\":"<<rate_error
    <<",\"actual_UnpackComposite_error\":"<<unpack_error<<",\"actual_UnpackComposite_endpoint_errors\":"<<unpack_endpoint_errors
    <<",\"full_tensor_RHS_error\":"<<tensor_error<<",\"selected_Minkowski_lapse_error\":"<<lapse_error
    <<",\"omitted_reference_time_control_error\":"<<omitted_reference_time_error
    <<",\"beta_untouched\":true,\"endpoint_limit_errors_reported\":"<<zero_endpoint_errors
    <<",\"expected_range_failures_retained\":"<<expected_range_failures<<",\"conditioning\":["<<conditioning<<"]}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";result=1;}
Kokkos::finalize();return result;}
