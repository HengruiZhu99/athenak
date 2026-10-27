// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/wormhole_chart.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;
try{
  std::cout<<std::setprecision(17)<<"{\"cases\":[";
  int count=0;
  for(Real r:{1e-8,1e-5,.001,.03,.25,.499999,.5,.75,.99}){
    const Real x[3]={r/3,2*r/3,2*r/3},center[3]={0,0,0},v[3]={.03,-.02,.01};
    const h::WormholeChart chart(x,center,v);
    h::Z4cJet<Real> u{};u.alpha.value=chart.alpha.f*1.13;u.trace.value=chart.weight.f*.17;
    for(int i=0;i<3;++i){u.alpha.d[i]=1.13*chart.alpha.d[i]+u.alpha.value*.12*x[i];u.beta.value[i]=-.83*x[i];}
    if(count++)std::cout<<",";
    std::cout<<"{\"r\":"<<r<<",\"alpha\":"<<u.alpha.value<<",\"trace\":"<<u.trace.value
      <<",\"alpha_r\":";
    Real ar=0;for(int i=0;i<3;++i)ar+=x[i]/r*u.alpha.d[i];
    std::cout<<ar<<",\"beta\":"<<-.83*r<<",\"rate\":"<<h::WormholeLapseRate(x,u)<<",\"refs\":[";
    for(int f=0;f<3;++f){if(f)std::cout<<",";const auto &j=f==0?chart.chi:(f==1?chart.weight:chart.alpha);
      std::cout<<"["<<j.f;for(int a=0;a<3;++a)std::cout<<","<<j.d[a];
      for(int a=0;a<3;++a)for(int b=0;b<3;++b)std::cout<<","<<j.h[a][b];std::cout<<"]";}
    std::cout<<"],\"ref_t\":["<<chart.chi_t<<","<<chart.weight_t<<","<<chart.alpha_t<<"],\"isotropic_refs\":[";
    const auto matched=h::WormholeChart::Isotropic(x);
    for(int f=0;f<2;++f){if(f)std::cout<<",";const auto &j=f==0?matched.chi:matched.alpha;
      std::cout<<"["<<j.f;for(int a=0;a<3;++a)std::cout<<","<<j.d[a];
      for(int a=0;a<3;++a)for(int b=0;b<3;++b)std::cout<<","<<j.h[a][b];std::cout<<"]";}
    std::cout<<"],\"asymptotic_refs\":[";
    const auto smooth=h::WormholeChart::Asymptotic(x);
    for(int f=0;f<2;++f){if(f)std::cout<<",";const auto &j=f==0?smooth.chi:smooth.alpha;
      std::cout<<"["<<j.f;for(int a=0;a<3;++a)std::cout<<","<<j.d[a];
      for(int a=0;a<3;++a)for(int b=0;b<3;++b)std::cout<<","<<j.h[a][b];std::cout<<"]";}
    std::cout<<"]}";
  }
  // Moving-reference time derivative, without changing the physical tensor.
  const Real x[3]={.17,-.11,.09},c[3]={.02,.03,-.01},v[3]={.03,-.02,.01},eps=1e-5;
  Real cp[3],cm[3];for(int a=0;a<3;++a){cp[a]=c[a]+eps*v[a];cm[a]=c[a]-eps*v[a];}
  h::WormholeChart a(x,c,v),p(x,cp,v),m(x,cm,v);Real err=0;
  err=std::max(err,std::abs((p.chi.f-m.chi.f)/(2*eps)-a.chi_t));
  err=std::max(err,std::abs((p.weight.f-m.weight.f)/(2*eps)-a.weight_t));
  err=std::max(err,std::abs((p.alpha.f-m.alpha.f)/(2*eps)-a.alpha_t));
  if(err>2e-10)throw std::runtime_error("moving puncture reference chain rule");
  std::cout<<"],\"moving_reference_time_error\":"<<err<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";result=1;}
Kokkos::finalize();return result;}
