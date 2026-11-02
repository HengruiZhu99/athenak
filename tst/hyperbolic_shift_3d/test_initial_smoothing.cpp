// Bounded t=0 radial stencil study. No evolution, no ghost interpolation.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>
#include "athena.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
 Kokkos::initialize(argc,argv);int status=0;
 try{
  if(argc!=5&&argc!=6)throw std::invalid_argument("usage: initial_smoothing R0/M R1/M norm_samples_per_band tolerance [bump|poly6]");
  const double begin=std::stod(argv[1]),end=std::stod(argv[2]),tol=std::stod(argv[4]);
  const int samples=std::stoi(argv[3]);if(samples<0)throw std::invalid_argument("negative sample count");
  const std::string profile=argc==6?argv[5]:"bump";
  if(profile!="bump"&&profile!="poly6")throw std::invalid_argument("unknown transition profile");
  h::NativeWormhole data(1./3.,tol,begin,end,profile=="bump"?h::NativeWormhole::TransitionProfile::bump:h::NativeWormhole::TransitionProfile::polynomial_c6);
  std::cerr<<std::setprecision(17)<<"R0/M="<<begin<<" R1/M="<<end<<" compact transition="<<data.InnerRadius()<<","<<data.OuterTransitionRadius()<<" horizon="<<data.IsotropicScale()/6<<'\n';
  std::cout<<std::setprecision(17)<<"n,H_rms,M_rms,H_max,H_radius,M_max,M_radius,H_gauge_difference,M_gauge_difference\n";
  // Fixed norm region remains unchanged; wide profiles can extend below it.
  // FD step follows each actual shell; angular tensor derivatives are analytic.
  for(int n:{16,24,32,48,64,96}){
   double H2=0,M2=0,V=0,Hmax=0,Mmax=0,Hr=0,Mr=0,Hg=0,Mg=0;
   const double lower[]={.16,.45,.75},upper[]={.45,.75,1};
   for(int band=0;band<3;++band){
    const double lo=std::max(.3,lower[band]),hi=std::min(.95,upper[band]);
    const int count=samples?samples:n;
    const double step=(upper[band]-lower[band])/n;
    const double ds=samples?(hi-lo)/samples:step;
    for(int i=0;i<count;++i){
     const double r=(samples?lo:lower[band])+(i+.5)*ds;if(r<lo||r>hi)continue;
     double f[5][8]{},q[8]{},d[8]{},dd[8]{};
     for(int a=0;a<5;++a){const double x=r+(a-2)*step;const auto v=data.Point(x);
      f[a][0]=v.chi-1;f[a][2]=v.ar;f[a][3]=v.trace+3;f[a][6]=v.lapse-(1+x*x)/2;}
     for(int j=0;j<8;++j){q[j]=f[2][j];
      d[j]=((f[0][j]-q[j])-8*(f[1][j]-q[j])+8*(f[3][j]-q[j])-(f[4][j]-q[j]))/(12*step);
      dd[j]=(-(f[0][j]-q[j])+16*(f[1][j]-q[j])+16*(f[3][j]-q[j])-(f[4][j]-q[j]))/(12*step*step);}
     auto u=h::SphericalJet(r,q,d,dd,h::CMCReference<double>{1,1});h::OmegaJet<double>o{};
     o.omega=(1-r*r)/2;o.gradient[0]=-r;for(int a=0;a<3;++a)o.hessian[a][a]=-1;
     const auto c=h::EvolvedConstraints(u,o);if(!c.valid)throw std::runtime_error("invalid radial sample");
     u.alpha.value=.25;u.beta.value[0]=42.;o.normal=7.;const auto changed=h::EvolvedConstraints(u,o);
     if(!changed.valid)throw std::runtime_error("invalid gauge-only control");
     Hg=std::max(Hg,std::abs(c.hamiltonian-changed.hamiltonian));Mg=std::max(Mg,std::abs(c.momentum_conformal_norm2-changed.momentum_conformal_norm2));
     const double w=r*r*ds,m=std::sqrt(c.momentum_conformal_norm2);V+=w;H2+=w*c.hamiltonian*c.hamiltonian;M2+=w*m*m;
     if(std::abs(c.hamiltonian)>Hmax){Hmax=std::abs(c.hamiltonian);Hr=r;}if(m>Mmax){Mmax=m;Mr=r;}
    }
   }
   std::cout<<n<<','<<std::sqrt(H2/V)<<','<<std::sqrt(M2/V)<<','<<Hmax<<','<<Hr<<','<<Mmax<<','<<Mr<<','<<Hg<<','<<Mg<<std::endl;
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
 Kokkos::finalize();return status;
}
