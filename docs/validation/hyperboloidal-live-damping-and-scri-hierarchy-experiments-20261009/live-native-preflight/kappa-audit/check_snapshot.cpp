#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "live_damping_profile.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::ScopeGuard guard(argc,argv);if(argc!=2)return 2;
 std::ifstream file(argv[1],std::ios::binary);h::SphericalGhostGrid grid{};grid.radius=1;
 file.read(reinterpret_cast<char*>(grid.n),sizeof(grid.n));
 file.read(reinterpret_cast<char*>(grid.first),sizeof(grid.first));
 file.read(reinterpret_cast<char*>(grid.h),sizeof(grid.h));
 if(!file||grid.n[0]!=30||grid.n[1]!=30||grid.n[2]!=30)return 3;
 const int cells=grid.n[0]*grid.n[1]*grid.n[2];std::vector<Real>fields(25*cells);
 file.read(reinterpret_cast<char*>(fields.data()),fields.size()*sizeof(Real));
 if(!file||file.peek()!=std::char_traits<char>::eof())return 4;
 h::LayerGaugeParameters gauge;gauge.physical_trace_lapse=true;
 gauge.preferred_source=false;gauge.scri_lapse_damping=2;
 h::CartesianConformalPatch patch(grid,.5,2,{true,.05,.95},gauge,true);patch.kappa1=10;
 auto active=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.active);
 Real lo=std::numeric_limits<Real>::infinity(),hi=-lo,elo=lo,ehi=-lo;
 Real helper_error=0,sigma_error=0,outer_constant_error=0;
 int positive_k2=0,nonpositive_eff=0,constant_cells=0;
 for(size_t point=0;point<active.extent(0);++point){
  const int s=active(point),i=s%30,j=s/30%30,k=s/900;
  const auto p=patch.reference.At(grid.first[0]+i*grid.h[0],grid.first[1]+j*grid.h[1],grid.first[2]+k*grid.h[2]);
  auto u=p.state;u.alpha.value=fields[18*cells+s];
  Real dot=0;for(int a=0;a<3;++a){u.beta.value[a]=fields[(19+a)*cells+s];dot+=u.beta.value[a]*p.domega[a];}
  const auto o=h::CartesianOmega(u,p);const Real V=h::SmoothCutoff(p.radius,Real(.15),Real(.3)).value;
  const Real k2=h::ResearchLiveKappa2Profile(u,o,p.radius,Real(10));
  const Real manual=V*(o.omega-1+2*dot/10),eff=10*(1+k2);
  const Real sigma=(-2*u.alpha.value*o.normal-eff)/o.omega;
  const Real predicted=(1-V)*(2*dot-10)/o.omega-V*10;
  if(!std::isfinite(k2)||!std::isfinite(sigma))return 5;
  lo=std::min(lo,k2);hi=std::max(hi,k2);elo=std::min(elo,eff);ehi=std::max(ehi,eff);
  positive_k2+=k2>0;nonpositive_eff+=eff<=0;
  helper_error=std::max(helper_error,std::abs(k2-manual));sigma_error=std::max(sigma_error,std::abs(sigma-predicted));
  if(V==1){++constant_cells;outer_constant_error=std::max(outer_constant_error,std::abs(sigma+10));}
 }
 std::cout<<std::setprecision(17)<<"{\"active_cells\":"<<active.extent(0)
  <<",\"kappa2_min\":"<<lo<<",\"kappa2_max\":"<<hi
  <<",\"effective_kappa_min\":"<<elo<<",\"effective_kappa_max\":"<<ehi
  <<",\"positive_kappa2_cells\":"<<positive_k2<<",\"nonpositive_effective_cells\":"<<nonpositive_eff
  <<",\"manual_helper_max_error\":"<<helper_error<<",\"sigma_identity_max_error\":"<<sigma_error
  <<",\"V1_cells\":"<<constant_cells<<",\"V1_sigma_plus10_max_error\":"<<outer_constant_error<<"}\n";
 return helper_error<1e-12&&sigma_error<1e-10&&outer_constant_error<1e-10?0:1;
}
