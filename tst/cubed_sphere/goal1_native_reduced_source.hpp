// Finite-Omega nonlinear first-order reduction sources, same native RWM RHS.
#pragma once
#include "goal1_native_ad.hpp"
#include "goal1_native_source_ad.hpp"
namespace cubed_sphere {namespace native_reduced_source {
namespace h=z4c::hyperboloidal;namespace hc=holonomic_chart;
using Vector=std::array<Real,10>;
using Gradient=std::array<Vector,3>;
using Hessian=std::array<Gradient,3>; // [spatial derivative][gradient component]
struct Evaluation {
 bool valid=false;h::Z4cJet<Real>native{};std::array<Real,22>rhs{};
 Vector qdot{},Pdot{};Gradient gdot{};
 Real det_time=0,A_trace_time=0;
};
inline bool Configuration(const h::LayerPoint<Real>&ref,const Vector&qbar,const Gradient&g,const Hessian&Dg,hc::Config&q){
 if(!hc::Encode(ref.state,q))return false;for(int n=0;n<10;++n){q[n].f+=qbar[n];for(int i=0;i<3;++i){q[n].d[i]+=g[i][n];for(int j=0;j<3;++j)q[n].h[i][j]+=.5*(Dg[i][j][n]+Dg[j][i][n]);}}return true;
}
inline Evaluation Evaluate(const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&ref,const Real x[3],const Vector&qbar,const Gradient&g,const Vector&P,const Hessian&Dg,const Gradient&DP,Real kappa=1){
 Evaluation out;hc::Config q;if(!Configuration(ref,qbar,g,Dg,q))return out;auto u=ref.state;if(!hc::Decode(q,u))return out;auto conn=reference_connection_ad::Build(descriptor,ref,x);if(!conn.connection.valid)return out;auto reference=native_source_ad::ReferenceFirst(ref,conn.omega_third);auto inverse=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});
 for(int d=0;d<3;++d){inverse.chi.value.d[d]=u.chi.d[d];inverse.alpha.value.d[d]=u.alpha.d[d];for(int i=0;i<3;++i){inverse.beta.value[i].d[d]=u.beta.d[d][i];for(int j=0;j<3;++j)inverse.metric.g[i][j].d[d]=u.metric.dg[d][i][j];}}
 hc::Config q0;if(!hc::Encode(ref.state,q0))return out;
 for(int d=0;d<3;++d)for(int i=0;i<3;++i){inverse.chi.d[i].d[d]=q0[0].h[d][i]+Dg[d][i][0];inverse.alpha.d[i].d[d]=q0[6].h[d][i]+Dg[d][i][6];for(int j=0;j<3;++j)inverse.beta.d[i][j].d[d]=q0[7+j].h[d][i]+Dg[d][i][7+j];}
 Jet omega(ref.omega);std::array<Jet,10>pi;for(int d=0;d<3;++d)omega.d[d]=ref.domega[d];for(int n=0;n<10;++n){pi[n]=Jet(P[n]-kappa*qbar[n]);for(int d=0;d<3;++d)pi[n].d[d]=DP[d][n]-kappa*g[d][n];}
 if(!typed_momentum::Reconstruct(inverse,reference.state,omega,pi))return out;
 u.trace.value=inverse.trace.value.f;u.theta.value=inverse.theta.value.f;
 for(int d=0;d<3;++d){u.trace.d[d]=inverse.trace.value.d[d];u.theta.d[d]=inverse.theta.value.d[d];for(int i=0;i<3;++i){u.lambda.value[i]=inverse.lambda.value[i].f;u.lambda.d[d][i]=inverse.lambda.value[i].d[d];for(int j=0;j<3;++j){u.a.k[i][j]=inverse.a.k[i][j].f;u.a.dk[d][i][j]=inverse.a.k[i][j].d[d];}}}
 auto F=native_source_ad::Source(q,descriptor,ref,x);if(!F.valid)return out;
 for(int i=0;i<3;++i)for(int n=0;n<10;++n){out.gdot[i][n]=DP[i][n]-kappa*g[i][n]+F.gradient[i][n];for(int j=0;j<3;++j)out.gdot[i][n]+=u.beta.value[j]*Dg[j][i][n]+u.beta.d[i][j]*g[j][n];}
 out.rhs=typed_support::Core(u,ref,rwm::ReferenceConnection(ref,x));h::Z4cRHS<Real>f{};h::GaugeRHS<Real>gauge{};const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};f.chi=out.rhs[0];f.trace=out.rhs[7];f.theta=out.rhs[17];gauge.alpha=out.rhs[18];for(int i=0;i<6;++i){f.metric[a[i]][b[i]]=f.metric[b[i]][a[i]]=out.rhs[1+i];f.a[a[i]][b[i]]=f.a[b[i]][a[i]]=out.rhs[8+i];}for(int i=0;i<3;++i){f.lambda[i]=out.rhs[14+i];gauge.beta[i]=out.rhs[19+i];}
 if(!typed_momentum::Velocity(u,f,gauge,out.qdot))return out;
 auto time=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});time.chi.value.d[0]=f.chi;time.alpha.value.d[0]=gauge.alpha;time.trace.value.d[0]=f.trace;time.theta.value.d[0]=f.theta;
 for(int i=0;i<3;++i){time.beta.value[i].d[0]=gauge.beta[i];time.lambda.value[i].d[0]=f.lambda[i];for(int j=0;j<3;++j){time.metric.g[i][j].d[0]=f.metric[i][j];time.a.k[i][j].d[0]=f.a[i][j];}}
 for(int i=0;i<3;++i){time.chi.d[i].d[0]=out.gdot[i][0];time.alpha.d[i].d[0]=out.gdot[i][6];for(int j=0;j<3;++j)time.beta.d[i][j].d[0]=out.gdot[i][7+j];}
 auto fixed=typed_support::Constant<Jet>(ref);std::array<Jet,10>PiTime;if(!typed_momentum::Momentum(time,fixed.state,Jet(ref.omega),PiTime))return out;for(int n=0;n<10;++n)out.Pdot[n]=PiTime[n].d[0]+kappa*out.qdot[n];
 auto geo=h::Geometry(u.metric);Real tr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){tr+=geo.inverse[i][j]*f.metric[i][j];out.A_trace_time+=geo.inverse[i][j]*f.a[i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)out.A_trace_time-=geo.inverse[i][k]*f.metric[k][l]*geo.inverse[l][j]*u.a.k[i][j];}out.det_time=geo.determinant*tr;out.native=u;out.valid=true;return out;
}
struct Sources {bool valid=false;Vector q{},P{};Gradient g{};};
inline Sources Source(const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&ref,const Real x[3],const Vector&qbar,const Gradient&g,const Vector&P,Real kappa=1){Hessian Dg{};Gradient DP{};auto e=Evaluate(descriptor,ref,x,qbar,g,P,Dg,DP,kappa);Sources s;s.valid=e.valid;s.q=e.qdot;s.P=e.Pdot;s.g=e.gdot;return s;}
}}
