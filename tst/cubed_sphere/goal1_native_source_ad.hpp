// Actual nonlinear q-source value and first spatial derivatives, private AD.
#pragma once
#include "goal1_native_ad.hpp"
#include "goal1_typed_support.hpp"
#include "goal1_typed_momentum.hpp"
#include "goal1_holonomic_chart.hpp"
#include "goal1_reference_connection_ad.hpp"
namespace cubed_sphere {namespace native_source_ad {
namespace h=z4c::hyperboloidal;namespace hc=holonomic_chart;
struct Result{bool valid=false;Real value[10]{},gradient[3][10]{};};
inline h::LayerPoint<Jet> ReferenceFirst(const h::LayerPoint<Real>&p,const Real third[3][3][3]){
 auto r=typed_support::Constant<Jet>(p);
 auto scalar=[&](const h::ScalarJet<Real>&s){h::ScalarJet<Jet>a{};a.value=Jet(s.value);for(int d=0;d<3;++d){a.value.d[d]=s.d[d];a.d[d]=Jet(s.d[d]);for(int e=0;e<3;++e){a.d[d].d[e]=s.dd[e][d];a.dd[d][e]=Jet(s.dd[d][e]);}}return a;};
 r.state.chi=scalar(p.state.chi);r.state.alpha=scalar(p.state.alpha);r.state.trace=scalar(p.state.trace);r.state.theta=scalar(p.state.theta);
 for(int i=0;i<3;++i){r.state.beta.value[i]=Jet(p.state.beta.value[i]);r.state.lambda.value[i]=Jet(p.state.lambda.value[i]);for(int d=0;d<3;++d){r.state.beta.value[i].d[d]=p.state.beta.d[d][i];r.state.lambda.value[i].d[d]=p.state.lambda.d[d][i];for(int e=0;e<3;++e)r.state.beta.d[d][i].d[e]=p.state.beta.dd[e][d][i];}for(int j=0;j<3;++j){r.state.a.k[i][j]=Jet(p.state.a.k[i][j]);r.state.metric.g[i][j]=Jet(p.state.metric.g[i][j]);for(int d=0;d<3;++d){r.state.a.k[i][j].d[d]=p.state.a.dk[d][i][j];r.state.metric.g[i][j].d[d]=p.state.metric.dg[d][i][j];for(int e=0;e<3;++e)r.state.metric.dg[d][i][j].d[e]=p.state.metric.ddg[e][d][i][j];}}}
 r.alpha=r.state.alpha.value;r.k_physical=r.state.trace.value;r.omega=Jet(p.omega);
 for(int d=0;d<3;++d){r.omega.d[d]=p.domega[d];r.beta[d]=r.state.beta.value[d];r.dalpha[d]=r.state.alpha.d[d];r.domega[d]=Jet(p.domega[d]);for(int e=0;e<3;++e){r.domega[d].d[e]=p.omega_hessian[e][d];r.omega_hessian[d][e]=Jet(p.omega_hessian[d][e]);for(int f=0;f<3;++f)r.omega_hessian[d][e].d[f]=third[f][d][e];}}
 return r;
}
inline Result Source(const hc::Config&q,const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&p,const Real x[3],Real unused_variation=0){
 Result result;hc::Config q0;if(!hc::Encode(p.state,q0))return result;auto basecfg=q;for(int n=0;n<10;++n)for(int d=0;d<3;++d){basecfg[n].d[d]=q0[n].d[d];for(int e=0;e<3;++e)basecfg[n].h[d][e]=q0[n].h[d][e];}auto base=p.state;h::Z4cJet<Real>live{};if(!hc::Decode(basecfg,base)||!hc::Decode(q,live))return result;
 auto connection=reference_connection_ad::Build(descriptor,p,x);if(!connection.connection.valid)return result;auto reference=ReferenceFirst(p,connection.omega_third);auto u=typed_support::Map<Jet>(base,[](int,Real v){return Jet(v);});
 u.trace.value=reference.state.trace.value;u.theta.value=reference.state.theta.value;for(int i=0;i<3;++i){u.lambda.value[i]=reference.state.lambda.value[i];for(int j=0;j<3;++j)u.a.k[i][j]=reference.state.a.k[i][j];}
 for(int d=0;d<3;++d){u.chi.value.d[d]=q[0].d[d];u.alpha.value.d[d]=q[6].d[d];for(int i=0;i<3;++i){u.beta.value[i].d[d]=q[7+i].d[d];for(int j=0;j<3;++j)u.metric.g[i][j].d[d]=live.metric.dg[d][i][j];}
  for(int i=0;i<3;++i){u.chi.d[i].d[d]=q0[0].h[d][i];u.alpha.d[i].d[d]=q0[6].h[d][i];for(int j=0;j<3;++j)u.beta.d[i][j].d[d]=q0[7+j].h[d][i];
   auto mixed=q;for(int n=0;n<10;++n){for(int a=0;a<3;++a){mixed[n].d[a]=a==0?q[n].d[d]:(a==1?q0[n].d[i]:0);for(int b=0;b<3;++b)mixed[n].h[a][b]=0;}mixed[n].h[0][1]=mixed[n].h[1][0]=q0[n].h[d][i];}h::Z4cJet<Real>m{};if(!hc::Decode(mixed,m))return result;for(int a=0;a<3;++a)for(int b=0;b<3;++b)u.metric.dg[i][a][b].d[d]=m.metric.ddg[0][1][a][b];
  }
 }
 // Dependency audit only: these outer variations would need higher physical
 // jets for p outputs. They do not enter q output derivatives; never expose a
 // p derivative computed from these placeholders.
 if(unused_variation!=0)for(int d=0;d<3;++d){u.trace.d[d].d[d]=unused_variation;u.theta.d[d].d[d]=-unused_variation;for(int i=0;i<3;++i){u.lambda.d[d][i].d[d]=unused_variation*(i+1);for(int j=0;j<3;++j){u.a.dk[d][i][j].d[d]=unused_variation;for(int k=0;k<3;++k)u.metric.ddg[d][k][i][j].d[d]=unused_variation*(j+1);}}for(int i=0;i<3;++i){u.chi.dd[d][i].d[d]=unused_variation;u.alpha.dd[d][i].d[d]=-unused_variation;for(int j=0;j<3;++j)u.beta.dd[d][i][j].d[d]=unused_variation;}}
 auto native=typed_support::Core(u,reference,connection.connection);h::Z4cRHS<Jet>f{};h::GaugeRHS<Jet>g{};f.chi=native[0];f.trace=native[7];f.theta=native[17];g.alpha=native[18];const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};for(int i=0;i<6;++i)f.metric[a[i]][b[i]]=f.metric[b[i]][a[i]]=native[1+i];for(int i=0;i<3;++i)g.beta[i]=native[19+i];std::array<Jet,10>velocity;if(!typed_momentum::Velocity(u,f,g,velocity))return result;
 result.valid=true;for(int n=0;n<10;++n){result.value[n]=velocity[n].f;result.valid=result.valid&&std::isfinite(result.value[n]);for(int d=0;d<3;++d){result.gradient[d][n]=velocity[n].d[d];result.valid=result.valid&&std::isfinite(result.gradient[d][n]);}}return result;
}
}}
