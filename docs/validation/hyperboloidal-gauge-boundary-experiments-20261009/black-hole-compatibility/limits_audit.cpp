#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "detached_wormhole.hpp"
namespace hyp=z4c::hyperboloidal;
void Require(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
bool FiniteConsumed(const hyp::Z4cJet<double>&u) {
 bool ok=true;
 for(double v:{u.alpha.value,u.chi.value,u.trace.value,u.theta.value})ok=ok&&std::isfinite(v);
 for(int d=0;d<3;++d) {
  for(double v:{u.alpha.d[d],u.chi.d[d],u.trace.d[d],u.theta.d[d],u.beta.value[d],u.lambda.value[d]})ok=ok&&std::isfinite(v);
  for(int e=0;e<3;++e) {
   for(double v:{u.alpha.dd[d][e],u.chi.dd[d][e],u.beta.d[d][e],u.lambda.d[d][e],u.metric.g[d][e],u.a.k[d][e]})ok=ok&&std::isfinite(v);
   for(int j=0;j<3;++j) {
    ok=ok&&std::isfinite(u.beta.dd[d][e][j])&&std::isfinite(u.metric.dg[d][e][j])&&std::isfinite(u.a.dk[d][e][j]);
    for(int k=0;k<3;++k)ok=ok&&std::isfinite(u.metric.ddg[d][e][j][k]);
   }
  }
 }
 return ok;
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);
 try {
  double H=0,M=0,scri=0;int points=0,configs=0,invalid=0;
  struct Config {double S,a,m;hyp::LayerParameters compact,height;};
  for(auto c:{Config{1,.5,.5,{true,.05,.95},{true,.3,.95}},
              Config{1,1,.5,{true,.05,.95},{true,.3,.8}},
              Config{2,3.5,.4,{true,.1,1.9},{true,.5,1.7}},
              Config{1,.5,.1,{true,.05,.95},{true,.08,.95}}}) {
   hyp::LayerReference<double> ref(c.S,c.a,c.compact);
   hyp::DetachedWormhole<double> bh(ref,c.m,c.height);++configs;
   std::vector<double> rs{1e-5*c.S,.01*c.S,c.compact.r0,std::nextafter(c.compact.r0,c.S),c.height.r0,std::nextafter(c.height.r0,c.S),std::nextafter(c.height.r1,0.),c.height.r1,std::nextafter(c.compact.r1,0.),c.compact.r1,c.S*(1-1e-12),c.S*(1-1e-15),c.S};
   for(int i=1;i<80;++i)rs.push_back(c.S*i/80.);
   for(double g:{750.,1000.,1400.,1490.,1500.,1550.,1600.}) {
    const double v=2/(g+2+std::hypot(g,2.));rs.push_back(c.height.r0+(c.height.r1-c.height.r0)*v);
   }
   for(double r:rs)for(auto n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}}) {
    const auto p=ref.At(r*n[0],r*n[1],r*n[2]);const auto u=bh.At(r*n[0],r*n[1],r*n[2]);++points;
    Require(FiniteConsumed(u)&&u.alpha.value>0&&u.chi.value>0,"nonfinite consumed limit");
    const auto cons=hyp::EvolvedConstraints(u,hyp::CartesianOmega(u,p));
    Require(cons.valid,"invalid limiting constraints");H=std::max(H,std::abs(cons.hamiltonian));M=std::max(M,std::sqrt(cons.momentum_conformal_norm2));
   }
   const auto u=bh.At(c.S,0,0);
   for(double v:{u.alpha.value-c.S/c.a,u.chi.value-1,u.beta.value[0]+c.S/c.a,u.trace.value+3/c.a,u.a.k[0][0]+4*c.m/(3*c.a*c.S),u.a.k[1][1]-2*c.m/(3*c.a*c.S),u.metric.g[0][0]-1,u.metric.g[1][1]-1,u.lambda.value[0]})scri=std::max(scri,std::abs(v));
   const auto origin=bh.At(0,0,0);Require(FiniteConsumed(origin)&&origin.alpha.value==0&&origin.chi.value==0,"origin branch finite");
  }
  Require(H<1e-7&&M<1e-7&&scri<1e-12,"generic limiting constraints");
  hyp::LayerReference<double> ref(1,.5,{true,.05,.95});
  auto reject=[&](double m,hyp::LayerParameters hp) {try{hyp::DetachedWormhole<double> bad(ref,m,hp);}catch(std::invalid_argument const&){++invalid;return;}throw std::runtime_error("invalid constructor accepted");};
  reject(0,{true,.3,.95});reject(-1,{true,.3,.95});reject(INFINITY,{true,.3,.95});reject(NAN,{true,.3,.95});reject(.5,{false,.3,.95});reject(.5,{true,.1,.95});
  std::cout<<std::setprecision(17)<<"PASS configs="<<configs<<" points="<<points<<" invalidRejected="<<invalid<<" H="<<H<<" M="<<M<<" scriLimits="<<scri<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
