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
double Norm(const hyp::Z4cRHS<double>&s) {
 double v=std::max({std::abs(s.chi),std::abs(s.trace),std::abs(s.theta)});
 for(int i=0;i<3;++i){v=std::max(v,std::abs(s.lambda[i]));for(int j=0;j<3;++j)v=std::max({v,std::abs(s.metric[i][j]),std::abs(s.a[i][j])});}
 return v;
}
hyp::Z4cJet<double> StaticJet(const hyp::DetachedWormhole<double>&bh,double r,std::array<double,3>n) {
 auto u=bh.At(r*n[0],r*n[1],r*n[2]);const auto p=bh.geometry().At(r,0.,0.);
 const hyp::Radial2<double> radius{r,1,0},half{bh.mass()/2,0,0};
 const hyp::Radial2<double> omega{p.omega,p.domega[0],p.omega_hessian[0][0]};
 const auto lapse=(radius-half*omega)/(radius+half*omega)*hyp::Radial2<double>{p.alpha,p.dalpha[0],p.state.alpha.dd[0][0]};
 u.alpha.value=lapse.value;
 for(int i=0;i<3;++i){u.alpha.d[i]=lapse.d*n[i];for(int j=0;j<3;++j)u.alpha.dd[i][j]=(lapse.dd-lapse.d/r)*n[i]*n[j]+(i==j?lapse.d/r:0);}
 return u;
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);
 try {
  hyp::LayerReference<double> ref(1,.5,{true,.05,.95});
  hyp::DetachedWormhole<double> bh(ref,.5,{true,.3,.95});
  double vacuum=0,mink=0,positive_initial_dynamic=0,small_omega_pole=0,small_omega_rhs=0;
  int points=0;
  hyp::LayerGaugeParameters gauge;gauge.physical_trace_lapse=true;gauge.preferred_source=false;
  double mink_gauge=0;
  std::vector<double> radii{.2501,.26,.3,std::nextafter(.3,1.),.3000001,.4,.5,.7,.9,.95,.99,.9999};
  for(int i=0;i<130;++i)radii.push_back(.251+i*(.9999-.251)/129);
  for(double r:radii)for(auto n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}}) {
   auto p=ref.At(r*n[0],r*n[1],r*n[2]);auto u=StaticJet(bh,r,n);
   Require(u.alpha.value>0,"static exterior lapse nonpositive");
   hyp::Z4cRHS<double> rhs{};
   Require(hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),5/u.alpha.value,0.),p.omega,rhs),"static RHS invalid");
   vacuum=std::max(vacuum,Norm(rhs));++points;
   const auto init=bh.At(r*n[0],r*n[1],r*n[2]);
   Require(hyp::AssembleInterior(hyp::ConformalRHS(init,hyp::CartesianOmega(init,p),5/init.alpha.value,0.),p.omega,rhs),"initialized RHS invalid");
   positive_initial_dynamic=std::max(positive_initial_dynamic,Norm(rhs));
  }
  for(double r:{0.,.01,.05,.1,.25,.3,.4,.6,.9,.95,.9999}) {
   const auto p=bh.geometry().At(.36*r,-.48*r,.8*r);hyp::Z4cRHS<double> rhs{};hyp::GaugeRHS<double> gr{};
   const auto c=hyp::EvolvedConstraints(p.state,hyp::CartesianOmega(p.state,p));
   Require(c.valid&&std::abs(c.hamiltonian)<1e-9&&c.momentum_conformal_norm2<1e-18,"counterfactual Mink constraints");
   Require(hyp::AssembleInterior(hyp::ConformalRHS(p.state,hyp::CartesianOmega(p.state,p),5/p.alpha,0.),p.omega,rhs),"counterfactual Mink RHS");
   mink=std::max(mink,Norm(rhs));
   Require(hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,p.state,gauge),p.omega,gr),"counterfactual Mink gauge");
   mink_gauge=std::max(mink_gauge,std::abs(gr.alpha));for(int i=0;i<3;++i)mink_gauge=std::max(mink_gauge,std::abs(gr.beta[i]));
  }
  for(double r:{1-1e-8,1-1e-12,1-1e-15,1.}) {
   const auto p=ref.At(r,0.,0.);const auto u=bh.At(r,0.,0.);auto parts=hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),5/u.alpha.value,0.);
   Require(parts.valid&&std::isfinite(Norm(parts.regular))&&std::isfinite(Norm(parts.pole)),"small Omega parts invalid");
   small_omega_pole=std::max(small_omega_pole,Norm(parts.pole));
   if(p.omega>0){hyp::Z4cRHS<double> rhs{};Require(hyp::AssembleInterior(parts,p.omega,rhs),"small Omega assembly invalid");small_omega_rhs=std::max(small_omega_rhs,Norm(rhs));}
   else Require(Norm(parts.pole)<1e-12,"exact scri pole nonzero");
  }
  Require(vacuum<1e-7&&mink<1e-7&&mink_gauge<1e-10,"stationary geometry failed");
  Require(positive_initial_dynamic>.01,"precollapsed lapse incorrectly asserted stationary");
  const double M=bh.mass(),R=M/2,psi=2,rho=psi*psi*R;
  Require(rho==2*M&&4*M_PI*rho*rho==16*M_PI*M*M,"throat area invariant");
  std::cout<<std::setprecision(17)<<"PASS exterior points="<<points<<" staticVacuumRHS="<<vacuum<<" detachedMinkRHS="<<mink<<" detachedMinkGauge="<<mink_gauge<<" precollapsedRHS="<<positive_initial_dynamic<<" smallOmegaPole="<<small_omega_pole<<" smallOmegaAssembled="<<small_omega_rhs<<" throatAreal="<<rho<<" throatArea="<<4*M_PI*rho*rho<<" SchwarzschildKretschmann="<<48*M*M/std::pow(rho,6)<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
