#include <fstream>
#include <iomanip>
#include <iostream>
#include "athena.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int status=0;
 try{
  if(argc!=2)throw std::invalid_argument("usage: initial_continuum analytic_jet_file");
  std::ifstream in(argv[1]);size_t count=0;if(!(in>>count))throw std::runtime_error("missing jets");
  std::cout<<std::setprecision(17)<<"r,H,Mr,M,H_gauge_changed,Mr_gauge_changed\n";
  for(size_t k=0;k<count;++k){
   double r,q[8],d[8],dd[8];in>>r;for(auto&v:q)in>>v;for(auto&v:d)in>>v;for(auto&v:dd)in>>v;
   if(!in||r<=0||r>=1)throw std::runtime_error("invalid analytic jet");
   auto u=h::SphericalJet(r,q,d,dd,h::CMCReference<double>{1,1});
   h::OmegaJet<double> o{};o.omega=(1-r*r)/2;o.gradient[0]=-r;
   for(int i=0;i<3;++i)o.hessian[i][i]=-1;
   const auto c=h::EvolvedConstraints(u,o);
   u.alpha.value=.25;u.beta.value[0]=42.;o.normal=7.;
   const auto altered=h::EvolvedConstraints(u,o);
   if(!c.valid||!altered.valid||c.hamiltonian!=altered.hamiltonian||c.momentum[0]!=altered.momentum[0])
    throw std::runtime_error("constraint validity/gauge independence failed");
   std::cout<<r<<','<<c.hamiltonian<<','<<c.momentum[0]<<','<<std::sqrt(c.momentum_conformal_norm2)<<','<<altered.hamiltonian<<','<<altered.momentum[0]<<'\n';
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 Kokkos::finalize();return status;
}
