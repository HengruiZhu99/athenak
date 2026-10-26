// Actual nonlinear finite-Omega velocity-reduction source; private candidate.
#pragma once
#include "goal1_velocity_momentum.hpp"
#include "goal1_native_reduced_source.hpp"
namespace cubed_sphere {namespace velocity_reduced_source {
namespace h=z4c::hyperboloidal;namespace hc=holonomic_chart;namespace old=native_reduced_source;
using Vector=old::Vector;using Gradient=old::Gradient;using Hessian=old::Hessian;using Evaluation=old::Evaluation;
inline h::Z4cJet<Jet> Time(const h::Z4cJet<Real>&u,const std::array<Real,22>&rhs,const Gradient&gdot,Real unused=0){
 auto time=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};time.chi.value.d[0]=rhs[0];time.alpha.value.d[0]=rhs[18];time.trace.value.d[0]=rhs[7];time.theta.value.d[0]=rhs[17];for(int n=0;n<6;++n){time.metric.g[a[n]][b[n]].d[0]=time.metric.g[b[n]][a[n]].d[0]=rhs[1+n];time.a.k[a[n]][b[n]].d[0]=time.a.k[b[n]][a[n]].d[0]=rhs[8+n];}for(int i=0;i<3;++i){time.beta.value[i].d[0]=rhs[19+i];time.lambda.value[i].d[0]=rhs[14+i];time.chi.d[i].d[0]=gdot[i][0];time.alpha.d[i].d[0]=gdot[i][6];for(int j=0;j<3;++j){time.beta.d[i][j].d[0]=gdot[i][7+j];for(int k=0;k<3;++k)time.metric.dg[i][j][k].d[0]=unused*(i+j+k+1);}}return time;
}
inline Evaluation Evaluate(const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&ref,const Real x[3],const Vector&qbar,const Gradient&g,const Vector&P,const Hessian&Dg,const Gradient&DP,Real kappa=1){
 Evaluation out;if(!(ref.omega>0))return out;hc::Config q,q0;if(!old::Configuration(ref,qbar,g,Dg,q)||!hc::Encode(ref.state,q0))return out;auto u=ref.state;if(!hc::Decode(q,u))return out;auto conn=reference_connection_ad::Build(descriptor,ref,x);if(!conn.connection.valid)return out;auto reference=native_source_ad::ReferenceFirst(ref,conn.omega_third);auto inverse=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});
 for(int d=0;d<3;++d){inverse.chi.value.d[d]=u.chi.d[d];inverse.alpha.value.d[d]=u.alpha.d[d];for(int i=0;i<3;++i){inverse.beta.value[i].d[d]=u.beta.d[d][i];for(int j=0;j<3;++j)inverse.metric.g[i][j].d[d]=u.metric.dg[d][i][j];}}
 for(int d=0;d<3;++d)for(int i=0;i<3;++i){inverse.chi.d[i].d[d]=q0[0].h[d][i]+Dg[d][i][0];inverse.alpha.d[i].d[d]=q0[6].h[d][i]+Dg[d][i][6];for(int j=0;j<3;++j)inverse.beta.d[i][j].d[d]=q0[7+j].h[d][i]+Dg[d][i][7+j];}
 std::array<Jet,10>W;for(int n=0;n<10;++n){W[n]=Jet(P[n]-kappa*qbar[n]);for(int d=0;d<3;++d)W[n].d[d]=DP[d][n]-kappa*g[d][n];}if(!velocity_momentum::ReconstructInterior(inverse,reference,conn.connection,W))return out;
 u.trace.value=inverse.trace.value.f;u.theta.value=inverse.theta.value.f;for(int d=0;d<3;++d){u.trace.d[d]=inverse.trace.value.d[d];u.theta.d[d]=inverse.theta.value.d[d];for(int i=0;i<3;++i){u.lambda.value[i]=inverse.lambda.value[i].f;u.lambda.d[d][i]=inverse.lambda.value[i].d[d];for(int j=0;j<3;++j){u.a.k[i][j]=inverse.a.k[i][j].f;u.a.dk[d][i][j]=inverse.a.k[i][j].d[d];}}}
 for(int i=0;i<3;++i)for(int n=0;n<10;++n){out.gdot[i][n]=DP[i][n]-kappa*g[i][n];for(int j=0;j<3;++j)out.gdot[i][n]+=u.beta.value[j]*Dg[j][i][n]+u.beta.d[i][j]*g[j][n];}
 out.rhs=typed_support::Core(u,ref,rwm::ReferenceConnection(ref,x));h::Z4cRHS<Real>f{};h::GaugeRHS<Real>gauge{};const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};f.chi=out.rhs[0];gauge.alpha=out.rhs[18];for(int n=0;n<6;++n)f.metric[a[n]][b[n]]=f.metric[b[n]][a[n]]=out.rhs[1+n];for(int i=0;i<3;++i)gauge.beta[i]=out.rhs[19+i];if(!typed_momentum::Velocity(u,f,gauge,out.qdot))return out;
 auto time=Time(u,out.rhs,out.gdot);auto fixed=typed_support::Constant<Jet>(ref);auto c=typed_support::Constant<Jet>(rwm::ReferenceConnection(ref,x));std::array<Jet,10>Wtime;if(!velocity_momentum::AssembleInterior(velocity_momentum::MomentumParts(time,fixed,c),Jet(ref.omega),Wtime))return out;for(int n=0;n<10;++n)out.Pdot[n]=Wtime[n].d[0]+kappa*out.qdot[n];
 auto geometry=h::Geometry(u.metric);Real trace=0;
 // Algebraic diagnostics use the actual metric/A rows explicitly.
 Real fm[3][3]{},fa[3][3]{};for(int n=0;n<6;++n){fm[a[n]][b[n]]=fm[b[n]][a[n]]=out.rhs[1+n];fa[a[n]][b[n]]=fa[b[n]][a[n]]=out.rhs[8+n];}trace=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){trace+=geometry.inverse[i][j]*fm[i][j];out.A_trace_time+=geometry.inverse[i][j]*fa[i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)out.A_trace_time-=geometry.inverse[i][k]*fm[k][l]*geometry.inverse[l][j]*u.a.k[i][j];}out.det_time=geometry.determinant*trace;out.native=u;out.valid=true;return out;
}
inline old::Sources Source(const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&ref,const Real x[3],const Vector&qbar,const Gradient&g,const Vector&P,Real kappa=1){Hessian Dg{};Gradient DP{};auto e=Evaluate(descriptor,ref,x,qbar,g,P,Dg,DP,kappa);old::Sources s;s.valid=e.valid;s.q=e.qdot;s.P=e.Pdot;s.g=e.gdot;return s;}
}}
