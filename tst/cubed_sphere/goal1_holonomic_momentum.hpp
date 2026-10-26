// Exact finite-Omega nonlinear configuration momentum map, same native gauge.
#pragma once
#include "goal1_holonomic_chart.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
namespace cubed_sphere {namespace holonomic_momentum {
namespace h=z4c::hyperboloidal;namespace hc=holonomic_chart;
inline bool MetricVelocity(const h::Z4cJet<Real>&u,const Real m[3][3],Real v[5]){
 h::Z4cRHS<Real>r{};h::GaugeRHS<Real>g{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)r.metric[i][j]=m[i][j];std::array<Real,10>q;if(!hc::Velocity(u,r,g,q))return false;for(int i=0;i<5;++i)v[i]=q[1+i];return true;
}
inline bool MetricFromVelocity(const h::Z4cJet<Real>&u,const Real v[5],Real m[3][3]){
 hc::Config q;if(!hc::Encode(u,q))return false;for(auto&a:q)for(int d=0;d<3;++d){a.d[d]=0;for(int e=0;e<3;++e)a.h[d][e]=0;}for(int i=0;i<5;++i)q[1+i].d[0]=v[i];h::Z4cJet<Real>decoded{};if(!hc::Decode(q,decoded))return false;for(int i=0;i<3;++i)for(int j=0;j<3;++j)m[i][j]=decoded.metric.dg[0][i][j];return true;
}
inline bool Momentum(const h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real omega,std::array<Real,10>&pi){
 if(!(omega>0)||!(u.alpha.value>0)||!(u.chi.value>0))return false;const auto g=h::Geometry(u.metric);if(!g.valid)return false;Real div=0;for(int i=0;i<3;++i)div+=u.beta.d[i][i]-ref.beta.d[i][i];const Real a=u.alpha.value,chi=u.chi.value;
 pi[0]=-Real(2./3.)*chi*div+Real(2./3.)*a*chi*((u.trace.value-ref.trace.value)+2*(u.theta.value-ref.theta.value))/omega;
 Real m[3][3]{},v[5];for(int i=0;i<3;++i)for(int j=0;j<3;++j){m[i][j]=-2*a*(u.a.k[i][j]-ref.a.k[i][j])-Real(2./3.)*u.metric.g[i][j]*div;for(int k=0;k<3;++k)m[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-ref.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-ref.beta.d[i][k]);}
 if(!MetricVelocity(u,m,v))return false;for(int i=0;i<5;++i)pi[1+i]=v[i];pi[6]=-a*a*(u.trace.value-ref.trace.value)/omega;
 for(int i=0;i<3;++i){pi[7+i]=a*a*chi*(u.lambda.value[i]-ref.lambda.value[i]);for(int j=0;j<3;++j)pi[7+i]+=g.inverse[i][j]*(a*a*(u.chi.d[j]-ref.chi.d[j])/2-a*chi*(u.alpha.d[j]-ref.alpha.d[j]));}return true;
}
inline bool Reconstruct(h::Z4cJet<Real>&u,const h::Z4cJet<Real>&ref,Real omega,const std::array<Real,10>&pi){
 if(!(omega>0)||!(u.alpha.value>0)||!(u.chi.value>0))return false;const auto g=h::Geometry(u.metric);if(!g.valid)return false;const Real a=u.alpha.value,chi=u.chi.value;Real div=0;for(int i=0;i<3;++i)div+=u.beta.d[i][i]-ref.beta.d[i][i];const Real dp=-omega*pi[6]/(a*a);u.trace.value=ref.trace.value+dp;u.theta.value=ref.theta.value+(3*omega*(pi[0]+Real(2./3.)*chi*div)/(2*a*chi)-dp)/2;
 Real source[3][3]{},s[5],v[5],A[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){source[i][j]=2*a*ref.a.k[i][j]-Real(2./3.)*u.metric.g[i][j]*div;for(int k=0;k<3;++k)source[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-ref.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-ref.beta.d[i][k]);}
 if(!MetricVelocity(u,source,s))return false;for(int i=0;i<5;++i)v[i]=(s[i]-pi[1+i])/(2*a);if(!MetricFromVelocity(u,v,A))return false;for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=A[i][j];
 for(int i=0;i<3;++i){Real gradient=0;for(int j=0;j<3;++j)gradient+=g.inverse[i][j]*(a*a*(u.chi.d[j]-ref.chi.d[j])/2-a*chi*(u.alpha.d[j]-ref.alpha.d[j]));u.lambda.value[i]=ref.lambda.value[i]+(pi[7+i]-gradient)/(a*a*chi);}return true;
}
inline bool NativeVelocity(const h::Z4cJet<Real>&u,const h::LayerPoint<Real>&ref,const Real x[3],std::array<Real,10>&velocity){
 h::Z4cRHS<Real>f{};h::GaugeRHS<Real>g{};if(!h::AssembleInterior(h::ConformalRHS(u,h::CartesianOmega(u,ref),10/u.alpha.value,0.),ref.omega,f)||!rwm::Assemble(rwm::Gauge(ref,u,rwm::ReferenceConnection(ref,x)),ref.omega,g))return false;return hc::Velocity(u,f,g,velocity);
}
// F(q,x) is the actual native q RHS evaluated at zero configuration-gradient
// deviation and reference momenta. It is a change-of-variables source, not an
// RHS counterterm or an imposed manufactured forcing.
inline bool Source(const h::Z4cJet<Real>&u,const h::LayerPoint<Real>&ref,const Real x[3],std::array<Real,10>&source){
 hc::Config q,q0;if(!hc::Encode(u,q)||!hc::Encode(ref.state,q0))return false;for(int n=0;n<10;++n){for(int d=0;d<3;++d){q[n].d[d]=q0[n].d[d];for(int e=0;e<3;++e)q[n].h[d][e]=q0[n].h[d][e];}}
 auto zero=ref.state;if(!hc::Decode(q,zero))return false;return NativeVelocity(zero,ref,x,source);
}
}}
