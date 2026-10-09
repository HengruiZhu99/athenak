#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "q_null_feedback.hpp"
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
 h::LayerGaugeParameters gauge;gauge.physical_trace_lapse=false;
 gauge.preferred_source=true;gauge.scri_lapse_damping=2;
 h::CartesianConformalPatch patch(grid,.5,2,{true,.05,.95},gauge,true);patch.kappa1=10;
 auto active=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),patch.active);
 Real ndmax=0,feedbackmax=0,nullerr=0,poleerr=0,assemblyerr=0,qnummax=0;
 int outer=0;
 for(size_t point=0;point<active.extent(0);++point){
  const int s=active(point),i=s%30,j=s/30%30,k=s/900;
  const auto p=patch.reference.At(grid.first[0]+i*grid.h[0],grid.first[1]+j*grid.h[1],grid.first[2]+k*grid.h[2]);
  auto u=p.state;u.chi.value=fields[Z4c::I_Z4C_CHI*cells+s];
  u.alpha.value=fields[Z4c::I_Z4C_ALPHA*cells+s];u.trace.value=fields[Z4c::I_Z4C_KHAT*cells+s];
  const int ij[6][2]={{0,0},{0,1},{0,2},{1,1},{1,2},{2,2}};
  for(int c=0;c<6;++c)u.metric.g[ij[c][0]][ij[c][1]]=u.metric.g[ij[c][1]][ij[c][0]]=fields[(Z4c::I_Z4C_GXX+c)*cells+s];
  Real B=0,Bh=0,norm=0;
  for(int d=0;d<3;++d){u.beta.value[d]=fields[(Z4c::I_Z4C_BETAX+d)*cells+s];B+=u.beta.value[d]*p.domega[d];Bh+=p.beta[d]*p.domega[d];norm+=p.domega[d]*p.domega[d];}
  const auto g=h::Geometry(u.metric),gh=h::Geometry(p.state.metric);
  Real G=0,Gh=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b){G+=u.chi.value*g.inverse[a][b]*p.domega[a]*p.domega[b];Gh+=p.state.chi.value*gh.inverse[a][b]*p.domega[a]*p.domega[b];}
  const Real nd=(G-B*B/(u.alpha.value*u.alpha.value))-(Gh-Bh*Bh/(p.alpha*p.alpha));
  const Real factored=qnf::NullDifference(p,u);nullerr=std::max(nullerr,std::abs(nd-factored));
  const auto five=qnf::Gauge(p,u,gauge,{.85,.95,5,true});const auto zero=qnf::Gauge(p,u,gauge,{.85,.95,0,true});
  if(!five.valid||!zero.valid)return 5;
  h::GaugeRHS<Real>ff{},f0{};if(!qnf::Assemble(five,p.omega,ff)||!qnf::Assemble(zero,p.omega,f0))return 6;
  const Real V=h::SmoothCutoff(p.radius,Real(.85),Real(.95)).value;
  for(int d=0;d<3;++d){const Real manual=norm>0?V*5*u.alpha.value*u.alpha.value*p.domega[d]*nd/norm:0;
   const Real actual=five.pole.beta[d]-zero.pole.beta[d];poleerr=std::max(poleerr,std::abs(actual-manual));
   assemblyerr=std::max(assemblyerr,std::abs((ff.beta[d]-f0.beta[d])-actual/p.omega));feedbackmax=std::max(feedbackmax,std::abs(actual/p.omega));}
  if(p.radius>=.85){++outer;ndmax=std::max(ndmax,std::abs(factored));const Real D=u.alpha.value*Bh/p.alpha-B;
   qnummax=std::max(qnummax,std::abs(u.trace.value-p.k_physical-3*D/u.alpha.value));}
 }
 std::cout<<std::setprecision(17)<<"{\"active_cells\":"<<active.extent(0)<<",\"outer_cells\":"<<outer
  <<",\"null_difference_max\":"<<ndmax<<",\"Q_deviation_numerator_max\":"<<qnummax
  <<",\"feedback_beta_rhs_max\":"<<feedbackmax<<",\"manual_null_error\":"<<nullerr
  <<",\"manual_beta_pole_error\":"<<poleerr<<",\"single_pole_assembly_error\":"<<assemblyerr<<"}\n";
 return nullerr<1e-10&&poleerr<1e-10&&assemblyerr<1e-10?0:1;
}
