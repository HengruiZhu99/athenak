#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "athena.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int result=0;
 try{
  h::NativeWormhole data(1./3.,2e-14);
  // A fixed integration grid decouples the norm quadrature from the FD step.
  // Zero uses actual production radial cell centres. No solution interpolation.
  const int quadrature=argc>1?std::stoi(argv[1]):0;
  if(quadrature<0)throw std::invalid_argument("negative quadrature resolution");
  std::cout<<std::setprecision(17)<<"n,h,H_rms,Mr_rms,M_rms,H_max,H_radius,M_max,M_radius\n";
  for(int n:{16,24,32,48,64,96,128,192,256}){
   std::ofstream profile;
   if(argc>2){profile.open(std::string(argv[2])+"-n"+std::to_string(n)+".csv");
    if(!profile)throw std::runtime_error("cannot open radial profile");
    profile<<std::setprecision(17)<<"r,H,Mr,M\n";}
   const double step=.3/n;double H2=0,Mr2=0,M2=0,V=0,Hmax=0,Mmax=0,Hr=0,Mr=0;
   const int samples=quadrature?quadrature:n;
   const double integration_step=.3/samples;
   for(int i=0;i<samples;++i){
    const double r=.45+(i+.5)*integration_step;double f[5][8]{},q[8]{},d[8]{},dd[8]{};
    for(int a=0;a<5;++a){const double x=r+(a-2)*step;const auto v=data.Point(x);
     f[a][0]=v.chi-1;f[a][2]=v.ar;f[a][3]=v.trace+3;f[a][6]=v.lapse-(1+x*x)/2;}
    for(int j=0;j<8;++j){
     q[j]=f[2][j];d[j]=((f[0][j]-q[j])-8*(f[1][j]-q[j])+8*(f[3][j]-q[j])-(f[4][j]-q[j]))/(12*step);
     dd[j]=(-(f[0][j]-q[j])+16*(f[1][j]-q[j])+16*(f[3][j]-q[j])-(f[4][j]-q[j]))/(12*step*step);
    }
    const auto u=h::SphericalJet(r,q,d,dd,h::CMCReference<double>{1,1});h::OmegaJet<double>o{};
    o.omega=(1-r*r)/2;o.gradient[0]=-r;for(int a=0;a<3;++a)o.hessian[a][a]=-1;
    const auto c=h::EvolvedConstraints(u,o);if(!c.valid)throw std::runtime_error("invalid radial constraint sample");
    if(profile.is_open())profile<<r<<','<<c.hamiltonian<<','<<c.momentum[0]<<','<<std::sqrt(c.momentum_conformal_norm2)<<'\n';
    const double w=r*r*integration_step;V+=w;H2+=w*c.hamiltonian*c.hamiltonian;Mr2+=w*c.momentum[0]*c.momentum[0];M2+=w*c.momentum_conformal_norm2;
    if(std::abs(c.hamiltonian)>Hmax){Hmax=std::abs(c.hamiltonian);Hr=r;}
    if(std::sqrt(c.momentum_conformal_norm2)>Mmax){Mmax=std::sqrt(c.momentum_conformal_norm2);Mr=r;}
   }
   std::cout<<n<<','<<step<<','<<std::sqrt(H2/V)<<','<<std::sqrt(Mr2/V)<<','<<std::sqrt(M2/V)<<','<<Hmax<<','<<Hr<<','<<Mmax<<','<<Mr<<'\n';
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
 Kokkos::finalize();return result;
}
