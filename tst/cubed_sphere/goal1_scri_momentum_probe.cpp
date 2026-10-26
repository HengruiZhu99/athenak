#include "goal1_typed_support.hpp"
#include "goal1_holonomic_momentum.hpp"
#include <iostream>
#include <iomanip>
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);int index,status=0;Real eps,x[3];
 try{while(std::cin>>index>>eps>>x[0]>>x[1]>>x[2]){
  namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;
  h::Z4cJet<Real>empty{};auto u=cs::typed_support::Map<Real>(empty,[](int,Real){Real v;if(!(std::cin>>v))throw std::runtime_error("incomplete high precision fixture");return v;});
  h::LayerReference<Real>descriptor(1.,.5,{true,.05,.95});auto p=descriptor.At(x[0],x[1],x[2]);
  auto f=cs::typed_support::Core(u,p,rwm::ReferenceConnection(p,x));std::array<Real,10>pi{},source{},velocity{};
  if(!cs::holonomic_momentum::Momentum(u,p.state,p.omega,pi)||!cs::holonomic_momentum::Source(u,p,x,source)||!cs::holonomic_momentum::NativeVelocity(u,p,x,velocity))throw std::runtime_error("chart witness invalid");
  Real rhs=0;for(Real v:f)rhs=std::max(rhs,std::abs(v));
  cs::holonomic_chart::Config q,q0;cs::holonomic_chart::Encode(u,q);cs::holonomic_chart::Encode(p.state,q0);Real decomposition=0;
  for(int n=0;n<10;++n){Real expect=pi[n]+source[n];for(int i=0;i<3;++i)expect+=u.beta.value[i]*(q[n].d[i]-q0[n].d[i]);decomposition=std::max(decomposition,std::abs(expect-velocity[n]));}
  std::cout<<"{\"case\":"<<index<<",\"epsilon\":"<<eps<<",\"Omega\":"<<p.omega<<",\"native_rhs_max\":"<<rhs<<",\"conditioned_rhs\":"<<rhs*p.omega*p.omega<<",\"q_decomposition_absolute\":"<<decomposition<<",\"Pi_alpha\":"<<pi[6]<<",\"F_alpha\":"<<source[6]<<",\"alpha_rhs\":"<<f[18]<<",\"Omega_Pi_alpha\":"<<p.omega*pi[6]<<",\"trace\":"<<u.trace.value<<",\"native_rhs\":[";for(int n=0;n<22;++n)std::cout<<(n?",":"")<<f[n];std::cout<<"]}"<<std::endl;
 }}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}
 Kokkos::finalize();return status;
}
