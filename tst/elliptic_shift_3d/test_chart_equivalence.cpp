// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/composite_state.hpp"
#include "z4c/hyperboloidal/elliptic_shift.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
namespace h=z4c::hyperboloidal;
using J=cubed_sphere::Jet;
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;
try{Real worst=0,source_error=0,boundary_error=0;int cases=0;
for(Real r:{.03,.17,.25,.6,.93,1.}){
 const Real x[3]={r/3,2*r/3,2*r/3};J xx[3];for(int a=0;a<3;++a)xx[a]=J::Variable(x[a],a);
 const J chi=J(.8)+xx[0]*xx[1]*J(.03),alpha=J(.9)+xx[2]*J(.02);
 std::vector<J> physical(22);physical[0]=chi-J(1);physical[18]=alpha-(J(1)+xx[0]*xx[0]+xx[1]*xx[1]+xx[2]*xx[2])*J(.5);
 physical[1]=xx[0]*J(.02);physical[2]=xx[2]*J(.01);physical[4]=xx[1]*J(-.03);physical[6]=xx[0]*xx[1]*J(.02);
 physical[13]=J(.07)+xx[2]*J(.01);physical[14]=xx[1]*J(.003);
 for(int t=0;t<6;++t)physical[7+t]=J(.002*(t+1))+xx[t%3]*J(.001);
 for(int v=0;v<3;++v)physical[15+v]=J(.01*(v+1))+xx[v]*J(.002);
 auto unpack=[&](const std::vector<J> &v,bool chart,int reference){std::vector<Real> values(22);
  std::vector<std::array<Real,3>> d(22);std::vector<std::array<std::array<Real,3>,3>> dd(22);
  for(int f=0;f<22;++f){values[f]=v[f].f;for(int a=0;a<3;++a){d[f][a]=v[f].d[a];for(int b=0;b<3;++b)dd[f][a][b]=v[f].h[a][b];}}
  return h::UnpackComposite(x,values,d,dd,chart,reference);};
 auto original=unpack(physical,false,0);
 h::OmegaJet<Real> omega{};omega.omega=(1-r*r)/2;for(int a=0;a<3;++a){omega.gradient[a]=-x[a];omega.hessian[a][a]=-1;}
 h::SetStationaryOmegaNormal(original.alpha.value,original.beta.value,original.alpha.d,original.beta.d,omega);
 const auto rhs=h::ConformalRHS(original,omega,4.,-.99);
 Real source[3];h::EllipticShiftSource(original,h::Geometry(original.metric),.2,source);
 const h::CMCReference<Real> minkowski{1,1};
 const auto gauge=h::UnfactoredReferenceGauge(minkowski,minkowski.At(x[0],x[1],x[2]),original,h::GaugeParameters<Real>{1,0,2,0},true,true);
 for(int reference=0;reference<3;++reference){const auto factors=h::WormholeChart::At(x,reference);auto coeff=physical;
  coeff[0]=cubed_sphere::Log(chi/factors.chi);coeff[18]=cubed_sphere::Log(alpha/factors.alpha);
  coeff[13]=(physical[13]-J(3))/factors.weight+J(3);coeff[14]=physical[14]/factors.weight;
  for(int t=0;t<6;++t)coeff[7+t]=physical[7+t]/factors.weight;
  const auto u=unpack(coeff,true,reference);const auto parts=h::ConformalRHS(u,omega,4.,-.99);
  auto compare=[&](Real a,Real b){if(!std::isfinite(a+b))throw std::runtime_error("nonfinite chart equivalence comparison");
    worst=std::max(worst,std::abs(a-b)/std::max({Real(1),std::abs(a),std::abs(b)}));};
  compare(u.chi.value,original.chi.value);compare(u.alpha.value,original.alpha.value);
  for(int a=0;a<3;++a){compare(u.chi.d[a],original.chi.d[a]);compare(u.alpha.d[a],original.alpha.d[a]);
   for(int b=0;b<3;++b){compare(u.chi.dd[a][b],original.chi.dd[a][b]);compare(u.alpha.dd[a][b],original.alpha.dd[a][b]);}}
  for(const auto pair:{std::pair{parts.regular,rhs.regular},std::pair{parts.pole,rhs.pole}}){
   compare(pair.first.chi,pair.second.chi);compare(pair.first.trace,pair.second.trace);compare(pair.first.theta,pair.second.theta);
   for(int a=0;a<3;++a){compare(pair.first.lambda[a],pair.second.lambda[a]);for(int b=0;b<3;++b){compare(pair.first.a[a][b],pair.second.a[a][b]);compare(pair.first.metric[a][b],pair.second.metric[a][b]);}}}
  Real now[3];h::EllipticShiftSource(u,h::Geometry(u.metric),.2,now);
  for(int a=0;a<3;++a)source_error=std::max(source_error,std::abs(now[a]-source[a]));
  const auto g=h::UnfactoredReferenceGauge(minkowski,minkowski.At(x[0],x[1],x[2]),u,h::GaugeParameters<Real>{1,0,2,0},true,true);
  compare(g.regular.alpha,gauge.regular.alpha);compare(g.pole.alpha,gauge.pole.alpha);
  if(r<1)compare(h::WormholeLapseRate(x,u),h::WormholeLapseRate(x,original));
  Real before[3],after[3];h::FixedAngleScriShift(h::Geometry(original.metric),original.alpha.value,original.chi.value,x,before);
  h::FixedAngleScriShift(h::Geometry(u.metric),u.alpha.value,u.chi.value,x,after);
  for(int a=0;a<3;++a)boundary_error=std::max(boundary_error,std::abs(after[a]-before[a]));++cases;
 }
}
if(worst>3e-10||source_error>3e-10||boundary_error>1e-13)throw std::runtime_error("coefficient chart changed physical equations");
std::cout<<std::setprecision(17)<<"{\"cases\":"<<cases<<",\"full_tensor_gauge_relative_error\":"<<worst<<",\"original_shift_source_error\":"<<source_error<<",\"null_boundary_error\":"<<boundary_error<<"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<"\n";result=1;}
Kokkos::finalize();return result;}
