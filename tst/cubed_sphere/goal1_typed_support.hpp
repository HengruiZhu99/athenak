#pragma once
#include "goal1_native_ad.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "mesh/cubed_sphere/reference_wave_map.hpp"
namespace cubed_sphere {namespace typed_support {
namespace h=z4c::hyperboloidal;
template<class R,class T,class F>h::Z4cJet<R> Map(const h::Z4cJet<T>&u,F fn){
 h::Z4cJet<R>r{};int serial=0;
 auto scalar=[&](const h::ScalarJet<T>&s){h::ScalarJet<R>a{};a.value=fn(serial++,s.value);for(int d=0;d<3;++d)a.d[d]=fn(serial++,s.d[d]);for(int d=0;d<3;++d)for(int e=d;e<3;++e)a.dd[d][e]=a.dd[e][d]=fn(serial++,s.dd[d][e]);return a;};
 r.chi=scalar(u.chi);r.alpha=scalar(u.alpha);r.trace=scalar(u.trace);r.theta=scalar(u.theta);
 for(int i=0;i<3;++i){h::ScalarJet<T>s{},t{};s.value=u.beta.value[i];t.value=u.lambda.value[i];for(int d=0;d<3;++d){s.d[d]=u.beta.d[d][i];t.d[d]=u.lambda.d[d][i];for(int e=0;e<3;++e){s.dd[d][e]=u.beta.dd[d][e][i];t.dd[d][e]=u.lambda.dd[d][e][i];}}auto b=scalar(s),l=scalar(t);r.beta.value[i]=b.value;r.lambda.value[i]=l.value;for(int d=0;d<3;++d){r.beta.d[d][i]=b.d[d];r.lambda.d[d][i]=l.d[d];for(int e=0;e<3;++e){r.beta.dd[d][e][i]=b.dd[d][e];r.lambda.dd[d][e][i]=l.dd[d][e];}}}
 for(int i=0;i<3;++i)for(int j=i;j<3;++j){h::ScalarJet<T>s{};s.value=u.metric.g[i][j];for(int d=0;d<3;++d){s.d[d]=u.metric.dg[d][i][j];for(int e=0;e<3;++e)s.dd[d][e]=u.metric.ddg[d][e][i][j];}auto g=scalar(s);r.metric.g[i][j]=r.metric.g[j][i]=g.value;for(int d=0;d<3;++d){r.metric.dg[d][i][j]=r.metric.dg[d][j][i]=g.d[d];for(int e=0;e<3;++e)r.metric.ddg[d][e][i][j]=r.metric.ddg[d][e][j][i]=g.dd[d][e];}r.a.k[i][j]=r.a.k[j][i]=fn(serial++,u.a.k[i][j]);for(int d=0;d<3;++d)r.a.dk[d][i][j]=r.a.dk[d][j][i]=fn(serial++,u.a.dk[d][i][j]);}
 return r;
}
template<class T>h::LayerPoint<T> Constant(const h::LayerPoint<Real>&p){
 h::LayerPoint<T>r{};r.omega=p.omega;r.alpha=p.alpha;r.k_bar=p.k_bar;r.k_physical=p.k_physical;r.hessian_omega=p.hessian_omega;r.hessian_alpha=p.hessian_alpha;r.einstein_nn=p.einstein_nn;r.einstein_stress_diagonal=p.einstein_stress_diagonal;
 r.radius=p.radius;r.L=p.L;r.b=p.b;r.w_omega=p.w_omega;r.n_residue=p.n_residue;r.outgoing=p.outgoing;r.ingoing=p.ingoing;
 for(int i=0;i<3;++i){r.beta[i]=p.beta[i];r.domega[i]=p.domega[i];r.dalpha[i]=p.dalpha[i];r.einstein_momentum[i]=p.einstein_momentum[i];for(int j=0;j<3;++j)r.omega_hessian[i][j]=p.omega_hessian[i][j];}r.state=Map<T>(p.state,[](int,Real v){return T(v);});return r;
}
template<class T>rwm::Connection<T> Constant(const rwm::Connection<Real>&p){rwm::Connection<T>r{};r.valid=p.valid;for(int i=0;i<4;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)r.scaled[i][j][k]=p.scaled[i][j][k];return r;}
template<class T>std::array<T,22> Core(const h::Z4cJet<T>&u,const h::LayerPoint<T>&p,const rwm::Connection<T>&connection){
 h::OmegaJet<T>o{};o.omega=p.omega;for(int i=0;i<3;++i){o.gradient[i]=p.domega[i];for(int j=0;j<3;++j)o.hessian[i][j]=p.omega_hessian[i][j];}h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);h::Z4cRHS<T>f{};h::GaugeRHS<T>g{};
 if(!h::AssembleInterior(h::ConformalRHS(u,o,T(10)/u.alpha.value,T(0)),p.omega,f)||!rwm::Assemble(rwm::Gauge(p,u,connection),p.omega,g))throw std::runtime_error("typed actual native RHS invalid");
 std::array<T,22>r{};const int a[6]={0,0,0,1,1,2},b[6]={0,1,2,1,2,2};r[0]=f.chi;r[7]=f.trace;r[17]=f.theta;r[18]=g.alpha;for(int i=0;i<6;++i){r[1+i]=f.metric[a[i]][b[i]];r[8+i]=f.a[a[i]][b[i]];}for(int i=0;i<3;++i){r[14+i]=f.lambda[i];r[19+i]=g.beta[i];}return r;
}
}}
