// Source-inclusive native configuration velocity momentum; private candidate.
#pragma once
#include "goal1_typed_support.hpp"
#include "goal1_typed_momentum.hpp"
namespace cubed_sphere {namespace velocity_momentum {
namespace h=z4c::hyperboloidal;
template<class T>struct Parts{bool valid=false;std::array<T,10>regular{},pole{};};
template<class T>Parts<T> ConfigurationParts(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const rwm::Connection<T>&c){
 Parts<T>out;h::OmegaJet<T>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
 auto f=h::ConformalRHS(u,o,T(10)/u.alpha.value,T(0));auto g=rwm::Gauge(p,u,c);if(!f.valid||!g.valid)return out;
 out.valid=typed_momentum::Velocity(u,f.regular,g.regular,out.regular)&&typed_momentum::Velocity(u,f.pole,g.pole,out.pole);return out;
}
// Reference configuration-coordinate gradients, evaluated through the live
// holonomic Jacobian. Higher base jets do not enter exposed q source parts.
template<class T>Parts<T> SourceParts(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const rwm::Connection<T>&c){
 auto base=p.state;base.chi.value=u.chi.value;base.alpha.value=u.alpha.value;
 for(int i=0;i<3;++i){base.beta.value[i]=u.beta.value[i];for(int j=0;j<3;++j)base.metric.g[i][j]=u.metric.g[i][j];}
 for(int d=0;d<3;++d){T v[5],m[3][3];if(!typed_momentum::MetricVelocity(p.state,p.state.metric.dg[d],v)||!typed_momentum::MetricFromVelocity(u,v,m))return {};for(int i=0;i<3;++i)for(int j=0;j<3;++j)base.metric.dg[d][i][j]=m[i][j];}
 return ConfigurationParts(base,p,c);
}
template<class T>bool GradientDeviation(const h::Z4cJet<T>&u,const h::Z4cJet<T>&ref,int d,std::array<T,10>&g){
 g[0]=u.chi.d[d]-ref.chi.d[d];g[6]=u.alpha.d[d]-ref.alpha.d[d];for(int i=0;i<3;++i)g[7+i]=u.beta.d[d][i]-ref.beta.d[d][i];T v[5],v0[5];if(!typed_momentum::MetricVelocity(u,u.metric.dg[d],v)||!typed_momentum::MetricVelocity(ref,ref.metric.dg[d],v0))return false;for(int n=0;n<5;++n)g[1+n]=v[n]-v0[n];return true;
}
template<class T>Parts<T> MomentumParts(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const rwm::Connection<T>&c){
 auto out=ConfigurationParts(u,p,c);if(!out.valid)return out;
 for(int d=0;d<3;++d){std::array<T,10>g;if(!GradientDeviation(u,p.state,d,g))return {};for(int n=0;n<10;++n)out.regular[n]-=u.beta.value[d]*g[n];}return out;
}
template<class T>bool AssembleInterior(const Parts<T>&parts,T omega,std::array<T,10>&value){if(!parts.valid||!(omega>0))return false;for(int n=0;n<10;++n)value[n]=parts.regular[n]+parts.pole[n]/omega;return true;}
// beta_quotient must be F_beta_pole/Omega in the interior and its compatible
// derived limit at scri. It is never freely supplied incoming radiation.
template<class T>bool ReconstructWithQuotient(h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const Parts<T>&F,const std::array<T,10>&W,const std::array<T,3>&beta_quotient){
 if(!F.valid||!(u.alpha.value>0)||!(u.chi.value>0))return false;const auto geometry=h::Geometry(u.metric);if(!geometry.valid)return false;const auto a=u.alpha.value,chi=u.chi.value;std::array<T,10>V;for(int n=0;n<10;++n)V[n]=W[n]-F.regular[n];T div=0;for(int d=0;d<3;++d)div+=u.beta.d[d][d]-p.state.beta.d[d][d];
 const T dp=(F.pole[6]-p.omega*V[6])/(a*a);u.trace.value=p.state.trace.value+dp;u.theta.value=p.state.theta.value+(T(3)*(p.omega*(V[0]+T(2./3.)*chi*div)-F.pole[0])/(2*a*chi)-dp)/2;
 T strain[3][3]{},s[5],v[5],A[3][3];for(int i=0;i<3;++i)for(int j=0;j<3;++j){strain[i][j]=2*a*p.state.a.k[i][j]-T(2./3.)*u.metric.g[i][j]*div;for(int k=0;k<3;++k)strain[i][j]+=u.metric.g[k][i]*(u.beta.d[j][k]-p.state.beta.d[j][k])+u.metric.g[k][j]*(u.beta.d[i][k]-p.state.beta.d[i][k]);}
 if(!typed_momentum::MetricVelocity(u,strain,s))return false;for(int n=0;n<5;++n)v[n]=(s[n]-V[1+n])/(2*a);if(!typed_momentum::MetricFromVelocity(u,v,A))return false;for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=A[i][j];
 for(int i=0;i<3;++i){T gradient=0;for(int j=0;j<3;++j)gradient+=geometry.inverse[i][j]*(a*a*(u.chi.d[j]-p.state.chi.d[j])/2-a*chi*(u.alpha.d[j]-p.state.alpha.d[j]));u.lambda.value[i]=p.state.lambda.value[i]+(V[7+i]-beta_quotient[i]-gradient)/(a*a*chi);}return true;
}
template<class T>bool ReconstructInterior(h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const rwm::Connection<T>&c,const std::array<T,10>&W){if(!(p.omega>0))return false;auto F=SourceParts(u,p,c);std::array<T,3>Q;for(int i=0;i<3;++i)Q[i]=F.pole[7+i]/p.omega;return ReconstructWithQuotient(u,p,F,W,Q);}
}}
