#ifndef PRIVATE_FINITE_RB_CONFIGURATION_ROWS_HPP_
#define PRIVATE_FINITE_RB_CONFIGURATION_ROWS_HPP_
#include "generic_gauge.hpp"
using Spatial=S1<D>;
using SpatialState=hyp::Z4cJet<Spatial>;
constexpr std::array<int,11> configuration_raw_indices{0,1,2,3,4,5,6,18,19,20,21};

SpatialState SpatialStateOf(const Jet&u,double unused=0) {
  SpatialState v{};
  auto scalar=[&](auto&a,const auto&b){a.value=Spatial(b.value);for(int i=0;i<3;++i){
    a.value.d[i]=b.d[i];a.d[i]=Spatial(b.d[i]);for(int j=0;j<3;++j){
      a.d[i].d[j]=b.dd[i][j];a.dd[i][j]=Spatial(b.dd[i][j]);
      for(int k=0;k<3;++k)a.dd[i][j].d[k]=D(unused,unused);}}};
  scalar(v.chi,u.chi);scalar(v.alpha,u.alpha);scalar(v.trace,u.trace);scalar(v.theta,u.theta);
  for(int i=0;i<3;++i){
    v.beta.value[i]=Spatial(u.beta.value[i]);v.lambda.value[i]=Spatial(u.lambda.value[i]);
    for(int d=0;d<3;++d){
      v.beta.value[i].d[d]=u.beta.d[d][i];v.lambda.value[i].d[d]=u.lambda.d[d][i];
      v.beta.d[d][i]=Spatial(u.beta.d[d][i]);v.lambda.d[d][i]=Spatial(u.lambda.d[d][i]);
      for(int e=0;e<3;++e){v.beta.d[d][i].d[e]=u.beta.dd[d][e][i];v.lambda.d[d][i].d[e]=u.lambda.dd[d][e][i];
        v.beta.dd[d][e][i]=Spatial(u.beta.dd[d][e][i]);v.lambda.dd[d][e][i]=Spatial(u.lambda.dd[d][e][i]);
        for(int k=0;k<3;++k){v.beta.dd[d][e][i].d[k]=D(unused,unused);v.lambda.dd[d][e][i].d[k]=D(unused,unused);}}
    }
    for(int j=0;j<3;++j){
      v.metric.g[i][j]=Spatial(u.metric.g[i][j]);v.a.k[i][j]=Spatial(u.a.k[i][j]);
      for(int d=0;d<3;++d){v.metric.g[i][j].d[d]=u.metric.dg[d][i][j];v.a.k[i][j].d[d]=u.a.dk[d][i][j];
        v.metric.dg[d][i][j]=Spatial(u.metric.dg[d][i][j]);v.a.dk[d][i][j]=Spatial(u.a.dk[d][i][j]);
        for(int e=0;e<3;++e){v.metric.dg[d][i][j].d[e]=u.metric.ddg[d][e][i][j];v.a.dk[d][i][j].d[e]=D(unused,unused);
          v.metric.ddg[d][e][i][j]=Spatial(u.metric.ddg[d][e][i][j]);
          for(int k=0;k<3;++k)v.metric.ddg[d][e][i][j].d[k]=D(unused,unused);}}
    }
  }
  return v;
}
hyp::LayerPoint<Spatial> SpatialReference(const hyp::LayerPoint<double>&p,const X3&x,double unused=0) {
  hyp::LayerPoint<Spatial> q{};q.state=SpatialStateOf(Lift(p.state),unused);
  q.alpha=q.state.alpha.value;q.radius=Spatial(p.radius);q.omega=Spatial(p.omega);
  q.k_physical=q.state.trace.value;q.k_bar=Spatial(p.k_bar);q.w_omega=Spatial(p.w_omega);
  for(int i=0;i<3;++i){
    q.radius.d[i]=D(p.radius>0?x[i]/p.radius:0);q.omega.d[i]=D(p.domega[i]);
    // These derivatives are not consumed in physical-P/source-off rows.
    q.k_bar.d[i]=D(unused,unused);q.w_omega.d[i]=D(unused,unused);
    q.beta[i]=q.state.beta.value[i];q.dalpha[i]=q.state.alpha.d[i];q.domega[i]=Spatial(p.domega[i]);
    for(int j=0;j<3;++j){q.domega[i].d[j]=D(p.omega_hessian[i][j]);q.omega_hessian[i][j]=Spatial(p.omega_hessian[i][j]);
      for(int k=0;k<3;++k)q.omega_hessian[i][j].d[k]=D(unused,unused);}
  }
  return q;
}
// Only the unchanged first-order configuration rows are evaluated. A/trace/
// Theta/Lambda evolution and their unknown third spatial jets are not used.
std::array<Spatial,22> ConfigurationSpatial(const hyp::LayerPoint<double>&p,
                                           const Jet&u,const X3&x,double unused=0) {
  const auto v=SpatialStateOf(u,unused);const auto q=SpatialReference(p,x,unused);
  const auto gauge=GenericGauge(q,v,.5,true);std::array<Spatial,22> f{};
  Spatial div=0,w=0;for(int d=0;d<3;++d){div+=v.beta.d[d][d];w-=v.beta.value[d]*q.domega[d]/v.alpha.value;}
  f[0]=-Spatial(2./3.)*v.chi.value*div
    +Spatial(2./3.)*v.alpha.value*v.chi.value*(v.trace.value+Spatial(2)*v.theta.value-Spatial(3)*w)/q.omega;
  for(int d=0;d<3;++d)f[0]+=v.beta.value[d]*v.chi.d[d];
  for(int t=0;t<6;++t){const int i=ti[t],j=tj[t];
    auto z=-Spatial(2)*v.alpha.value*v.a.k[i][j]-Spatial(2./3.)*v.metric.g[i][j]*div;
    for(int d=0;d<3;++d)z+=v.beta.value[d]*v.metric.dg[d][i][j]
      +v.metric.g[d][j]*v.beta.d[i][d]+v.metric.g[i][d]*v.beta.d[j][d];
    f[1+t]=z;
  }
  f[18]=gauge.alpha;for(int i=0;i<3;++i)f[19+i]=gauge.beta[i];
  return f;
}
#endif
