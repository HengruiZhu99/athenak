#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "z4c/hyperboloidal/cartesian_wormhole.hpp"
int main(int argc,char**argv){
 Kokkos::ScopeGuard guard(argc,argv);std::cout<<std::setprecision(17);
 for(int n:{24,36,48}){
  z4c::hyperboloidal::SphericalGhostGrid g;g.radius=1;for(int d=0;d<3;++d){g.n[d]=n+6;g.h[d]=2.1/n;g.first[d]=-.5*(n+5)*g.h[d];}
  z4c::hyperboloidal::LayerParameters lp;lp.enabled=true;lp.r0=.05;lp.r1=.95;
  z4c::hyperboloidal::LayerGaugeParameters lg;lg.physical_trace_lapse=true;lg.preferred_source=false;
  z4c::hyperboloidal::CartesianConformalPatch patch(g,.5,2,lp,lg,true);
  auto q=patch.Allocate("reference check"),saved=patch.Allocate("saved reference"),rhs=patch.Allocate("reference rhs");
  patch.InitializeReference(q);Kokkos::deep_copy(saved,q);
  for(int repeat=0;repeat<5;++repeat)patch.Prepare(q);
  double reference_change=0;for(size_t s=0;s<q.size();++s)reference_change=std::max(reference_change,std::abs(q.data()[s]-saved.data()[s]));
  patch.RHS(q,rhs);double maximum=0;for(size_t s=0;s<rhs.size();++s)maximum=std::max(maximum,std::abs(rhs.data()[s]));
  const auto diagnostic=patch.Diagnose(q);
  if(reference_change>1e-12||maximum>1e-11||diagnostic.h_l2>1e-10||diagnostic.m_l2>1e-10||diagnostic.z_l2>1e-10)throw std::runtime_error("reference audit failed");
  std::cout<<"{\"n\":"<<n<<",\"reference_prepare_max_change\":"<<reference_change<<",\"reference_rhs_max\":"<<maximum<<",\"H\":"<<diagnostic.h_l2<<",\"M\":"<<diagnostic.m_l2<<",\"Z\":"<<diagnostic.z_l2<<"}\n";
 }
 z4c::hyperboloidal::SphericalGhostGrid g;g.radius=1;for(int d=0;d<3;++d){g.n[d]=30;g.h[d]=2.1/24;g.first[d]=-.5*29*g.h[d];}
 z4c::hyperboloidal::LayerParameters lp;lp.enabled=true;lp.r0=.05;lp.r1=.95;
 z4c::hyperboloidal::LayerGaugeParameters lg;lg.physical_trace_lapse=true;lg.preferred_source=false;
 z4c::hyperboloidal::CartesianConformalPatch patch(g,.5,2,lp,lg,true);auto q=patch.Allocate("reconstruction check");
 z4c::hyperboloidal::InitializeCartesianWormhole(patch,q,.05);const auto before=patch.Diagnose(q);for(int r=0;r<5;++r)patch.Prepare(q);const auto after=patch.Diagnose(q);
 if(std::max({after.h_l2,after.m_l2,after.z_l2})>1e-8)throw std::runtime_error("initial reconstruction audit failed");
 std::cout<<"{\"kind\":\"analytic_initial_reconstruction\",\"mass\":0.05,\"initial_H\":"<<before.h_l2<<",\"repeated_H\":"<<after.h_l2<<",\"M\":"<<after.m_l2<<",\"Z\":"<<after.z_l2<<"}\n";
}
