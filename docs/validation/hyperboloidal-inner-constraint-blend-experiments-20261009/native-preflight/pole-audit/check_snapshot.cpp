#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
namespace h=z4c::hyperboloidal;

void Manual(h::Z4cRHS<Real>&pole,const h::Z4cJet<Real>&u,
            const h::OmegaJet<Real>&o,Real r) {
  const Real c=1-h::SmoothCutoff(r,Real(.45),Real(.85)).value;
  if(c==0)return;
  const auto g=h::Geometry(u.metric);
  pole.theta-=c*u.alpha.value*u.theta.value*
      (u.trace.value+2*u.theta.value-3*o.normal);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    pole.a[i][j]-=2*c*u.alpha.value*u.a.k[i][j]*u.theta.value;
    pole.lambda[i]+=(-2*c*u.theta.value*g.inverse[i][j]*u.alpha.d[j]
        +2*c*u.alpha.value*u.theta.value*g.inverse[i][j]*o.gradient[j]/o.omega);
  }
}
Real Difference(const h::Z4cRHS<Real>&a,const h::Z4cRHS<Real>&b) {
  Real value=std::max({std::abs(a.chi-b.chi),std::abs(a.trace-b.trace),std::abs(a.theta-b.theta)});
  for(int i=0;i<3;++i){value=std::max(value,std::abs(a.lambda[i]-b.lambda[i]));
    for(int j=0;j<3;++j)value=std::max({value,std::abs(a.a[i][j]-b.a[i][j]),std::abs(a.metric[i][j]-b.metric[i][j])});}
  return value;
}
int main(int argc,char**argv){
  Kokkos::ScopeGuard guard(argc,argv);
  if(argc!=2)return 2;
  std::ifstream file(argv[1],std::ios::binary);
  h::SphericalGhostGrid grid{};grid.radius=1;
  file.read(reinterpret_cast<char*>(grid.n),sizeof(grid.n));
  file.read(reinterpret_cast<char*>(grid.first),sizeof(grid.first));
  file.read(reinterpret_cast<char*>(grid.h),sizeof(grid.h));
  if(!file||grid.n[0]!=30||grid.n[1]!=30||grid.n[2]!=30)return 3;
  h::LayerGaugeParameters gauge;gauge.physical_trace_lapse=true;
  gauge.preferred_source=false;gauge.scri_lapse_damping=2;
  h::CartesianConformalPatch patch(grid,.5,2,{true,.05,.95},gauge,true);
  patch.kappa1=10;
  auto data=patch.Allocate("restart snapshot");auto host=Kokkos::create_mirror_view(data);
  for(int f=0;f<25;++f)for(int k=0;k<30;++k)for(int j=0;j<30;++j)for(int i=0;i<30;++i)
    file.read(reinterpret_cast<char*>(&host(0,f,k,j,i)),sizeof(Real));
  if(!file||file.peek()!=std::char_traits<char>::eof())return 4;
  Kokkos::deep_copy(data,host);
  const auto actual=patch.Diagnose(data);
  const auto q=h::BindCartesianFields(patch.deviations);
  const auto active=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.active);
  const Real idx[3]={1/grid.h[0],1/grid.h[1],1/grid.h[2]};
  Real synthetic_error=0,missing_double_control=0;
  for(Real r:{.3,.5,.65,.75,.825,.84,.9}){
    const auto p=patch.reference.At(r,Real(0),Real(0));auto u=p.state;
    u.theta.value=.03;u.alpha.value*=1.01;u.alpha.d[0]+=.02;
    u.a.k[0][0]+=.02;u.a.k[1][1]-=.01;u.a.k[2][2]-=.01;
    const auto o=h::CartesianOmega(u,p);
    h::Z4cRHS<Real> manual{},helper{},omitted{};
    Manual(manual,u,o,r);h::ResearchAddBlendedPole(u,o,r,gauge,helper);
    synthetic_error=std::max(synthetic_error,Difference(manual,helper));
    const auto parts=h::BulkC1Additions(u,o,r,gauge);
    for(int i=0;i<3;++i)omitted.lambda[i]=parts.double_pole.lambda[i]/o.omega;
    missing_double_control=std::max(missing_double_control,Difference(omitted,h::Z4cRHS<Real>{}));
  }
  Real manual_delta=0,helper_error=0,c0_delta=0,addition_max=0;
  int shell=0,overlap=0;
  for(size_t point=0;point<active.extent(0);++point){
    const int s=active(point),i=s%30,j=s/30%30,k=s/900;
    const auto p=patch.reference.At(grid.first[0]+i*grid.h[0],grid.first[1]+j*grid.h[1],grid.first[2]+k*grid.h[2]);
    if(p.radius<=1-2*grid.h[0])continue;
    ++shell;if(p.radius<.85)++overlap;
    auto u=h::LoadMeshJet<3>(q,idx,0,k,j,i);h::AddReferenceJet(u,p,patch.reference);
    const auto o=h::CartesianOmega(u,p);
    const auto base=h::ConformalRHS(u,o,10/u.alpha.value,Real(0)).pole;
    auto manual=base,helper=base;Manual(manual,u,o,p.radius);
    h::ResearchAddBlendedPole(u,o,p.radius,gauge,helper);
    helper_error=std::max(helper_error,Difference(manual,helper));
    addition_max=std::max(addition_max,Difference(manual,base));
    h::Z4cJet<Real> reference{};h::AddReferenceJet(reference,p,patch.reference);
    const auto o0=h::CartesianOmega(reference,p);
    const auto pole0=h::ConformalRHS(reference,o0,Real(10),Real(0)).pole;
    auto manual0=pole0;Manual(manual0,reference,o0,p.radius);
    manual_delta=std::max(manual_delta,Difference(manual,manual0));
    c0_delta=std::max(c0_delta,Difference(base,pole0));
  }
  std::cout<<std::setprecision(17)<<"{\"H\":"<<actual.h_l2<<",\"M\":"<<actual.m_l2
    <<",\"Z\":"<<actual.z_l2<<",\"Theta\":"<<actual.theta_l2
    <<",\"actual_pole_deviation\":"<<actual.shell_max_pole_deviation
    <<",\"independent_pole_deviation\":"<<manual_delta<<",\"C0_pole_deviation\":"<<c0_delta
    <<",\"manual_vs_helper_max\":"<<helper_error<<",\"blend_addition_max\":"<<addition_max
    <<",\"synthetic_manual_vs_helper_max\":"<<synthetic_error
    <<",\"synthetic_missing_double_pole_control\":"<<missing_double_control
    <<",\"shell_cells\":"<<shell<<",\"shell_overlap_support_cells\":"<<overlap<<"}\n";
  return helper_error<1e-12&&synthetic_error<1e-12&&missing_double_control>.01
      &&std::abs(manual_delta-actual.shell_max_pole_deviation)<1e-12?0:1;
}
