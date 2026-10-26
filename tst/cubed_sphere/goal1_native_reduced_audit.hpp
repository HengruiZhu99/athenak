// Independent unforced native time/spatial chain audit on compatible jets.
#pragma once
#include "goal1_native_reduced_source.hpp"
namespace cubed_sphere {namespace native_reduced_audit {
namespace h=z4c::hyperboloidal;
struct Result {bool valid=false;std::array<Real,10>Pdot{};Real q_gradient_scaled=0,q_value_scaled=0;};
inline Result Direct(const h::LayerReference<Real>&descriptor,const h::LayerPoint<Real>&p,const Real x[3],const native_reduced_source::Evaluation&e,Real kappa=1){
 Result out;const auto&u=e.native;auto conn=reference_connection_ad::Build(descriptor,p,x);if(!conn.connection.valid)return out;auto ref=native_source_ad::ReferenceFirst(p,conn.omega_third);auto spatial=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});
 for(int d=0;d<3;++d){spatial.chi.value.d[d]=u.chi.d[d];spatial.alpha.value.d[d]=u.alpha.d[d];spatial.trace.value.d[d]=u.trace.d[d];spatial.theta.value.d[d]=u.theta.d[d];for(int i=0;i<3;++i){spatial.beta.value[i].d[d]=u.beta.d[d][i];spatial.lambda.value[i].d[d]=u.lambda.d[d][i];for(int j=0;j<3;++j){spatial.metric.g[i][j].d[d]=u.metric.dg[d][i][j];spatial.a.k[i][j].d[d]=u.a.dk[d][i][j];}}
  for(int i=0;i<3;++i){spatial.chi.d[i].d[d]=u.chi.dd[d][i];spatial.alpha.d[i].d[d]=u.alpha.dd[d][i];for(int j=0;j<3;++j){spatial.beta.d[i][j].d[d]=u.beta.dd[d][i][j];for(int k=0;k<3;++k)spatial.metric.dg[i][j][k].d[d]=u.metric.ddg[d][i][j][k];}}
 }
 auto native=typed_support::Core(spatial,ref,conn.connection);h::Z4cRHS<Jet>f{};h::GaugeRHS<Jet>g{};const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};f.chi=native[0];g.alpha=native[18];for(int n=0;n<6;++n)f.metric[a[n]][b[n]]=f.metric[b[n]][a[n]]=native[1+n];for(int i=0;i<3;++i)g.beta[i]=native[19+i];std::array<Jet,10>qdot;if(!typed_momentum::Velocity(spatial,f,g,qdot))return out;
 for(int n=0;n<10;++n){out.q_value_scaled=std::max(out.q_value_scaled,std::abs(qdot[n].f-e.qdot[n])/std::max(Real(1),std::abs(e.qdot[n])));for(int d=0;d<3;++d)out.q_gradient_scaled=std::max(out.q_gradient_scaled,std::abs(qdot[n].d[d]-e.gdot[d][n])/std::max({Real(1),std::abs(qdot[n].d[d]),std::abs(e.gdot[d][n])}));}
 auto time=typed_support::Map<Jet>(u,[](int,Real v){return Jet(v);});time.chi.value.d[0]=e.rhs[0];time.alpha.value.d[0]=e.rhs[18];time.trace.value.d[0]=e.rhs[7];time.theta.value.d[0]=e.rhs[17];for(int n=0;n<6;++n){time.metric.g[a[n]][b[n]].d[0]=time.metric.g[b[n]][a[n]].d[0]=e.rhs[1+n];time.a.k[a[n]][b[n]].d[0]=time.a.k[b[n]][a[n]].d[0]=e.rhs[8+n];}for(int i=0;i<3;++i){time.lambda.value[i].d[0]=e.rhs[14+i];time.beta.value[i].d[0]=e.rhs[19+i];time.chi.d[i].d[0]=qdot[0].d[i];time.alpha.d[i].d[0]=qdot[6].d[i];for(int j=0;j<3;++j)time.beta.d[i][j].d[0]=qdot[7+j].d[i];}
 auto fixed=typed_support::Constant<Jet>(p);std::array<Jet,10>pi;if(!typed_momentum::Momentum(time,fixed.state,Jet(p.omega),pi))return out;for(int n=0;n<10;++n)out.Pdot[n]=pi[n].d[0]+kappa*qdot[n].f;out.valid=true;return out;
}
}}
