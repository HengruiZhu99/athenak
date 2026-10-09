#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "detached_wormhole.hpp"
namespace hyp=z4c::hyperboloidal;
void Require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
double RhsNorm(hyp::Z4cRHS<double> const& r) {
 double norm=std::max({std::abs(r.chi),std::abs(r.trace),std::abs(r.theta)});
 for(int i=0;i<3;++i){norm=std::max(norm,std::abs(r.lambda[i]));for(int j=0;j<3;++j)norm=std::max({norm,std::abs(r.a[i][j]),std::abs(r.metric[i][j])});}
 return norm;
}
std::array<double,21> Fields(hyp::LayerPoint<double> const&p) {
 auto const&u=p.state;
 return {u.alpha.value,u.trace.value,u.beta.value[0],u.beta.value[1],u.beta.value[2],
 u.a.k[0][0],u.a.k[0][1],u.a.k[0][2],u.a.k[1][1],u.a.k[1][2],u.a.k[2][2],
 u.chi.value,u.metric.g[0][0],u.metric.g[0][1],u.metric.g[0][2],u.metric.g[1][1],u.metric.g[1][2],u.metric.g[2][2],u.lambda.value[0],u.lambda.value[1],u.lambda.value[2]};
}
std::array<double,21> Derivatives(hyp::LayerPoint<double> const&p,int d) {
 auto const&u=p.state;
 return {u.alpha.d[d],u.trace.d[d],u.beta.d[d][0],u.beta.d[d][1],u.beta.d[d][2],
 u.a.dk[d][0][0],u.a.dk[d][0][1],u.a.dk[d][0][2],u.a.dk[d][1][1],u.a.dk[d][1][2],u.a.dk[d][2][2],
 u.chi.d[d],u.metric.dg[d][0][0],u.metric.dg[d][0][1],u.metric.dg[d][0][2],u.metric.dg[d][1][1],u.metric.dg[d][1][2],u.metric.dg[d][2][2],u.lambda.d[d][0],u.lambda.d[d][1],u.lambda.d[d][2]};
}
std::array<double,21> Fields(hyp::Z4cJet<double> const&u) {
 hyp::LayerPoint<double> p{};p.state=u;return Fields(p);
}
std::array<double,21> Derivatives(hyp::Z4cJet<double> const&u,int d) {
 hyp::LayerPoint<double> p{};p.state=u;return Derivatives(p,d);
}
void DifferenceAudit(hyp::DetachedWormhole<double> const& ref,double& first,double& second) {
 const std::array<int,4> off{-2,-1,1,2};const std::array<int,4> coeff{1,-8,8,-1};
 for(double r:{.02,.05,.15,.2494075325183059,.30,.4,.65,.90,.95}) {
  std::array<double,3> x{r*.36,r*-.48,r*.8};hyp::LayerPoint<double> p{};p.state=ref.At(x[0],x[1],x[2]);
  const double h=1e-5;
  for(int d=0;d<3;++d) {
   std::array<double,21> fd{};auto exact=Derivatives(p,d);
   for(int q=0;q<4;++q){auto y=x;y[d]+=off[q]*h;auto f=Fields(ref.At(y[0],y[1],y[2]));for(int j=0;j<21;++j)fd[j]+=coeff[q]*f[j];}
   for(int j=0;j<21;++j)first=std::max(first,std::abs(fd[j]/(12*h)-exact[j])/(1+std::abs(exact[j])));
   for(int e=0;e<3;++e) {
    std::array<double,21> fdd{};
    for(int q=0;q<4;++q)for(int k=0;k<4;++k){auto y=x;y[d]+=off[q]*h;y[e]+=off[k]*h;auto f=Fields(ref.At(y[0],y[1],y[2]));for(int j:{0,2,3,4,11,12,13,14,15,16,17})fdd[j]+=double(coeff[q]*coeff[k])*f[j];}
    const double want[21]={p.state.alpha.dd[d][e],0,p.state.beta.dd[d][e][0],p.state.beta.dd[d][e][1],p.state.beta.dd[d][e][2],0,0,0,0,0,0,p.state.chi.dd[d][e],p.state.metric.ddg[d][e][0][0],p.state.metric.ddg[d][e][0][1],p.state.metric.ddg[d][e][0][2],p.state.metric.ddg[d][e][1][1],p.state.metric.ddg[d][e][1][2],p.state.metric.ddg[d][e][2][2],0,0,0};
    for(int j:{0,2,3,4,11,12,13,14,15,16,17})second=std::max(second,std::abs(fdd[j]/(144*h*h)-want[j])/(1+std::abs(want[j])));
   }
  }
 }
}

int main(int argc,char**argv){Kokkos::initialize(argc,argv);
 try {
  hyp::LayerReference<double> ref(1,.5,{true,.05,.95});
  hyp::DetachedWormhole<double> bh(ref,.5,{true,.3,.95});
  double first=0,second=0;DifferenceAudit(bh,first,second);
  Require(first<1e-7&&second<1e-3,"detached BH consumed Cartesian jets failed");
  double mass0=0;
  hyp::DetachedWormhole<double> small(ref,1e-9,{true,.05,.95});
  for(double r:{.01,.04,.1,.2494075325183059,.3,.5,.75,.95,.9999}) {
   auto u=small.At(r*.36,r*-.48,r*.8);auto p=ref.At(r*.36,r*-.48,r*.8);
   auto f=Fields(p);hyp::LayerPoint<double> tmp{};tmp.state=u;auto b=Fields(tmp);
   for(int i=0;i<21;++i)mass0=std::max(mass0,std::abs(f[i]-b[i]));
  }
  Require(mass0<1e-6,"mass0 reference limit failed");
  auto p=bh.At(.2494075325183059*.36,.2494075325183059*-.48,.2494075325183059*.8);
  auto left=bh.At((.2494075325183059-1e-7)*.36,(.2494075325183059-1e-7)*-.48,(.2494075325183059-1e-7)*.8);
  auto right=bh.At((.2494075325183059+1e-7)*.36,(.2494075325183059+1e-7)*-.48,(.2494075325183059+1e-7)*.8);
  Require(p.alpha.value>0&&left.alpha.value>0&&right.alpha.value>0,"throat lapse positivity");
  std::cout<<std::setprecision(17)<<"PASS full21-field Cartesian consumed jets first="<<first<<" second="<<second<<" mass0="<<mass0<<" throatAlpha="<<p.alpha.value<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
