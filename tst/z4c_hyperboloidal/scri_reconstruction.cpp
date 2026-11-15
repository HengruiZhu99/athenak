// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.

#include <iostream>
#include <random>
#include <iomanip>
#include "z4c_hyperboloidal/scri_reconstruction.hpp"
#include "z4c_hyperboloidal/endpoint.hpp"
#include <stdexcept>
namespace nh=z4c_hyperboloidal;
int main(){
 nh::HyperbolicShiftParameters<Real> par; par.outer_lapse_damping=.25;
 nh::StationaryGaugeSource<Real> source{};source.alpha=1;
 std::mt19937 rng(347);std::uniform_real_distribution<double> rand(-.03,.03);
 double worst=0, solved=0, inv_err=0;
 for(int sample=0;sample<200;++sample){
  nh::mapped::Point map{};double len=0;for(int a=0;a<3;++a){map.x[a]=rand(rng);len+=map.x[a]*map.x[a];}len=sqrt(len);for(auto &x:map.x)x/=len;
  nh::CartesianChartJets j{};j[0].f=1;
  for(int f=0;f<28;++f){if(f!=18)j[f].f+=rand(rng);for(int a=0;a<3;++a)j[f].d[a]=rand(rng);}
  auto u=nh::SphericalPolicy::Physical(map.x,j);auto g=nh::Geometry(u.metric);double Q=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)Q+=g.inverse[a][b]*map.x[a]*map.x[b];
  double s=sqrt(j[0].f*Q);for(int a=0;a<3;++a)j[22+a].f=(1-s)*map.x[a];
  auto evaluate=[&](const auto& jets){return nh::SphericalPolicy::Boundary(map,.02,jets,par,source,.5);};
  auto original=evaluate(j);double z=0,Bn=0;for(int a=0;a<3;++a){z-=map.x[a]*original.rate[22+a];Bn+=map.x[a]*original.defect[a];}
  double A=2*(j[13].f+.25),d=4-s,k=-s*(j[13].f-3+2*j[14].f)/3;
  double I=original.clock+Bn/d-(A/k+1/d)*z;
  auto trial=j;double ell=rand(rng),q[3],qn=0;for(int a=0;a<3;++a){q[a]=rand(rng);qn+=map.x[a]*q[a];}
  auto correct=[&](auto& jets,double l,const double* v){for(int a=0;a<3;++a){jets[18].d[a]+=l*map.x[a];for(int b=0;b<3;++b)jets[22+b].d[a]+=v[b]*map.x[a];}};
  correct(trial,ell,q);auto changed=evaluate(trial);double zz=0,BBn=0;
  for(int a=0;a<3;++a){zz-=map.x[a]*changed.rate[22+a];BBn+=map.x[a]*changed.defect[a];worst=std::max(worst,std::abs(changed.defect[a]-original.defect[a]-k*map.x[a]*ell-d*q[a]));}
  worst=std::max(worst,std::abs(changed.clock-original.clock-A*ell+qn));worst=std::max(worst,std::abs(zz-z-k*ell));
  inv_err=std::max(inv_err,std::abs(changed.clock+BBn/d-(A/k+1/d)*zz-I));
  const auto fix=nh::ReconstructRegularScri(map.x,nh::SphericalPolicy::Physical(map.x,j),g,original,par);
  if(!fix.valid)throw std::runtime_error("regularity reconstruction invalid");
  ell=fix.lapse;for(int a=0;a<3;++a)q[a]=fix.shift[a];
  inv_err=std::max(inv_err,std::abs(fix.invariant-I));
  trial=j;correct(trial,ell,q);changed=evaluate(trial);zz=0;for(int a=0;a<3;++a){zz-=map.x[a]*changed.rate[22+a];solved=std::max(solved,std::abs(changed.defect[a]));}
  solved=std::max(solved,std::abs(zz));solved=std::max(solved,std::abs(changed.clock-I));
 }
 std::cout<<std::setprecision(16)<<"200 perturbed boundary jets: affine error="<<worst<<", invariant error="<<inv_err<<", closure error="<<solved<<"\n";
 nh::mapped::Point map{};map.x[0]=1;nh::CartesianChartJets j{};j[0].f=1;double eps=.01;j[25].f=eps;
 auto rate=[&](){return nh::SphericalPolicy::Boundary(map,.02,j,par,source,.5);};
 auto initial=rate();j[18].d[0]=-.6*eps;j[22].d[0]=-.3*eps;auto fixed=rate();
 std::cout<<"clock-zero counterexample: clock="<<fixed.clock<<", shift="<<fixed.defect[0]<<", s_t="<<-fixed.rate[22]<<", p_t="<<fixed.rate[13]<<"\n";
 // Explicit off-manifold test: B cancels the relaxation of a preexisting s error.
 j={};j[0].f=1.0001;j[22].f=1-sqrt(j[0].f);
 auto raw=rate();auto u=nh::SphericalPolicy::Physical(map.x,j);auto g=nh::Geometry(u.metric);
 auto fix=nh::ReconstructRegularScri(map.x,u,g,raw,par);
 if(!fix.valid || std::abs(raw.speed_t)<1e-5)throw std::runtime_error("missing off-manifold relaxation");
 j[18].d[0]+=fix.lapse;for(int a=0;a<3;++a)j[22+a].d[0]+=fix.shift[a];
 const auto frozen=rate();
 if(std::abs(frozen.speed_t)>1e-12 || j[0].f!=1.0001)
  throw std::runtime_error("B must preserve, not remove, the existing chiQ error");
 std::cout<<"off-manifold: original s_t="<<raw.speed_t<<", B s_t="<<frozen.speed_t<<", chiQ-1="<<j[0].f-1<<"\n";
 u.trace.value=-2*u.theta.value;
 if(nh::ReconstructRegularScri(map.x,u,g,raw,par).valid)throw std::runtime_error("unguarded k=0");
 u.trace.value=-3;auto singular=par;singular.restoring=sqrt(j[0].f);
 if(nh::ReconstructRegularScri(map.x,u,g,raw,singular).valid)throw std::runtime_error("unguarded d=0");
 // Full normal lift includes both radial Hessian and angular cross derivatives.
 for(int a=0;a<3;++a)map.J[a][a]=1;
 nh::mapped::Jet amplitude(.2);amplitude.d[1]=.3;amplitude.d[2]=-.4;
 const auto lift=nh::NormalLift(map,amplitude,.01,.5);
 double sum=0;for(int m=0;m<6;++m)sum+=1/(m+.5);
 if(std::abs(lift.f)>1e-14 || std::abs(lift.d[0]-.2)>1e-13 ||
    std::abs(lift.h[0][0]-.2*2/.01*sum)>1e-10 ||
    std::abs(lift.h[0][1]-.3)>1e-13 || std::abs(lift.h[0][2]+.4)>1e-13)
  throw std::runtime_error("incomplete normal lift Hessian");
 std::cout<<"PASS singular guards and full lift Hessian/angular cross terms\n";
 // C cancels the regular geometric clock obstruction in the local compatible example.
 j={};j[0].f=1;j[25].f=eps;par.scri_treatment=nh::ScriTreatment::clock_source;
 auto c0=rate();u=nh::SphericalPolicy::Physical(map.x,j);g=nh::Geometry(u.metric);
 fix=nh::ReconstructRegularScri(map.x,u,g,c0,par);
 j[18].d[0]+=fix.lapse;for(int a=0;a<3;++a)j[22+a].d[0]+=fix.shift[a];
 const auto c1=rate();
 if(std::abs(c1.clock)>1e-12 || std::abs(c1.clock_original-eps/2)>1e-12 ||
    std::abs(c1.speed_t)>1e-12 || std::abs(c1.clock_source+eps/2)>1e-12)
  throw std::runtime_error("C local clock cancellation mismatch");
 // The prescribed subtraction makes the extra source vanish on its reference.
 for(double radius:{.97,.98,.985,.995,1.}) {
  double x[3]={radius,0,0};auto omega=par.profile.At(x);
  nh::CartesianChartJets ref{};ref[0].f=1.01;
  ref[0].d[0]=.02;auto reference_u=nh::SphericalPolicy::Physical(x,ref);double D[3]={.02,0,0};
  auto prescribed=source;
  prescribed.scri_obstruction_reference=nh::EvaluateScriGeometry(x,reference_u,D,omega,par).obstruction;
  if(nh::ScriClockSource(x,reference_u,D,omega,par,prescribed)!=0)
    throw std::runtime_error("C reference subtraction changed reference RHS");
 }
 std::cout<<"PASS C finite source, local clock cancellation and prescribed subtraction (hyperbolicity gate separate)\n";
 return worst<1e-10 && inv_err<1e-10 && solved<1e-10 && std::abs(fixed.rate[22]-.6*eps)<1e-12 ? 0:1;
}
