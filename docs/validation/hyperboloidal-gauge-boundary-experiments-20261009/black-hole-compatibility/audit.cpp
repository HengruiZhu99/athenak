#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "detached_wormhole.hpp"
namespace hyp=z4c::hyperboloidal;
void Require(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);
 try {
  hyp::LayerReference<double> ref(1,.5,{true,.05,.95});
  hyp::DetachedWormhole<double> bh(ref,.5,{true,.3,.95});
  double lo=.05,hi=.3;
  for(int i=0;i<90;++i){double r=(lo+hi)/2;if(r/ref.At(r,0.,0.).omega<.25)lo=r;else hi=r;}
  const double throat=(lo+hi)/2;
  std::vector<double> radii{1e-4,.01,.05,std::nextafter(.05,1.),.1,.2,throat-1e-8,throat,throat+1e-8,.3,std::nextafter(.3,1.),.3000001,.4,.5,.7,.9,std::nextafter(.95,0.),.95,.99,1-1e-12,1-1e-15,1.};
  for(int i=1;i<250;++i)radii.push_back(i/250.);
  double H=0,M=0,mass_error=0,null_error=0,min_alpha=1,lambda_difference=0,det=0,tf=0;
  int points=0;
  for(double r:radii)for(auto n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}}) {
   auto p=ref.At(r*n[0],r*n[1],r*n[2]);auto u=bh.At(r*n[0],r*n[1],r*n[2]);
   auto o=hyp::CartesianOmega(u,p);auto c=hyp::EvolvedConstraints(u,o);++points;
   Require(c.valid&&u.alpha.value>0&&u.chi.value>0,"invalid live initial data");
   H=std::max(H,std::abs(c.hamiltonian));M=std::max(M,std::sqrt(c.momentum_conformal_norm2));
   min_alpha=std::min(min_alpha,u.alpha.value);det=std::max(det,std::abs(c.z4.determinant_residual));tf=std::max(tf,std::abs(c.z4.tracefree_residual));
   for(int i=0;i<3;++i)lambda_difference=std::max(lambda_difference,std::abs(u.lambda.value[i]-p.state.lambda.value[i]));
   if(p.radius<=.3) {Require(u.trace.value==0,"Cauchy core K nonzero");for(int i=0;i<3;++i){Require(u.beta.value[i]==0,"Cauchy core shift nonzero");for(int j=0;j<3;++j)Require(u.a.k[i][j]==0,"Cauchy core A nonzero");}}
   if(n[0]==1&&p.omega>1e-8) {
    const double om=p.omega,op=p.domega[0],rphys=r/om;
    const double m=.25/rphys,psi=1+m,rho=psi*psi*rphys;
    const double rphys_d=p.L/(om*om),rho_d=psi*(1-m)*rphys_d;
    const double E=u.metric.g[0][0]/u.chi.value/(om*om);
    const double kt=om*u.a.k[1][1]/u.metric.g[1][1]+u.trace.value/3;
    const double measured=rho/2*(1-rho_d*rho_d/E+rho*rho*kt*kt);
    mass_error=std::max(mass_error,std::abs(measured-.5));
   }
   if(p.radius>=.95){null_error=std::max(null_error,std::abs(c.null_residual-(p.omega*p.omega)*p.n_residue/std::pow(1+.25*p.omega/p.radius,4)));}
  }
  Require(H<1e-7&&M<1e-7,"constraints failed");Require(mass_error<1e-8,"independent mass failed");Require(det<1e-12&&tf<1e-12,"algebraic constraints failed");Require(null_error<1e-12,"outer null norm failed");Require(lambda_difference>.01,"detached metric accidentally reused actual reference");
  bool rejected=false;try{hyp::DetachedWormhole<double>bad(ref,.5,{true,.1,.95});}catch(std::invalid_argument const&){rejected=true;}Require(rejected,"throat safety not enforced");
  const auto origin=bh.At(0,0,0);Require(origin.alpha.value==0&&origin.chi.value==0,"origin limit branch");
  std::cout<<std::setprecision(17)<<"PASS points="<<points<<" throat="<<throat<<" heightCoreR="<<.3/ref.At(.3,0.,0.).omega<<" H="<<H<<" M="<<M<<" massError="<<mass_error<<" outerNullError="<<null_error<<" minAlpha="<<min_alpha<<" det="<<det<<" tracefree="<<tf<<" LambdaDifference="<<lambda_difference<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
