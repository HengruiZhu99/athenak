#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "native_injection.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace hyp=z4c::hyperboloidal;
void Require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
double RhsNorm(hyp::Z4cRHS<double> const& r) {
 double norm=std::max({std::abs(r.chi),std::abs(r.trace),std::abs(r.theta)});
 for(int i=0;i<3;++i){norm=std::max(norm,std::abs(r.lambda[i]));for(int j=0;j<3;++j)norm=std::max({norm,std::abs(r.a[i][j]),std::abs(r.metric[i][j])});}
 return norm;
}
std::array<double,14> Fields(hyp::LayerPoint<double> const&p) {
 auto const&u=p.state;
 return {u.alpha.value,u.trace.value,u.beta.value[0],u.beta.value[1],u.beta.value[2],
 u.a.k[0][0],u.a.k[0][1],u.a.k[0][2],u.a.k[1][1],u.a.k[1][2],u.a.k[2][2],
 u.chi.value,u.metric.g[0][0],u.lambda.value[0]};
}
std::array<double,14> Derivatives(hyp::LayerPoint<double> const&p,int d) {
 auto const&u=p.state;
 return {u.alpha.d[d],u.trace.d[d],u.beta.d[d][0],u.beta.d[d][1],u.beta.d[d][2],
 u.a.dk[d][0][0],u.a.dk[d][0][1],u.a.dk[d][0][2],u.a.dk[d][1][1],u.a.dk[d][1][2],u.a.dk[d][2][2],
 u.chi.d[d],u.metric.dg[d][0][0],u.lambda.d[d][0]};
}
void DifferenceAudit(hyp::LayerReference<double> const& ref,double& first,double& second) {
 const std::array<int,4> off{-2,-1,1,2};const std::array<int,4> coeff{1,-8,8,-1};
 for(double frac:{.18,.45,.8}) {
  double r=ref.layer.r0+(ref.layer.r1-ref.layer.r0)*frac;
  std::array<double,3> x{r*.36,r*-.48,r*.8};auto p=ref.At(x[0],x[1],x[2]);
  const double h=(ref.layer.r1-ref.layer.r0)*2e-5;
  for(int d=0;d<3;++d) {
   std::array<double,14> fd{};auto exact=Derivatives(p,d);
   for(int q=0;q<4;++q){auto y=x;y[d]+=off[q]*h;auto f=Fields(ref.At(y[0],y[1],y[2]));for(int j=0;j<14;++j)fd[j]+=coeff[q]*f[j];}
   for(int j=0;j<14;++j)first=std::max(first,std::abs(fd[j]/(12*h)-exact[j])/(1+std::abs(exact[j])));
   for(int e=0;e<3;++e) {
    std::array<double,14> fdd{};
    for(int q=0;q<4;++q)for(int k=0;k<4;++k){auto y=x;y[d]+=off[q]*h;y[e]+=off[k]*h;auto f=Fields(ref.At(y[0],y[1],y[2]));for(int j:{0,2,3,4,11,12,13})fdd[j]+=double(coeff[q]*coeff[k])*f[j];}
    const double want[14]={p.state.alpha.dd[d][e],0,p.state.beta.dd[d][e][0],p.state.beta.dd[d][e][1],p.state.beta.dd[d][e][2],0,0,0,0,0,0,p.state.chi.dd[d][e],p.state.metric.ddg[d][e][0][0],p.state.lambda.dd[d][e][0]};
    for(int j:{0,2,3,4,11,12,13})second=std::max(second,std::abs(fdd[j]/(144*h*h)-want[j])/(1+std::abs(want[j])));
   }
  }
 }
}
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 try {
  const std::vector<std::array<double,4>> configs{{1,.5,.05,.95}};
  int points=0,outer_equal=0;double hmax=0,mmax=0,rmax=0,gmax=0,first=0,second=0,null_residual=0,scri_roundoff=0;
  double speed=0;
  for(auto cfg:configs) {
   hyp::LayerReference<double> candidate(cfg[0],cfg[1],{true,cfg[2],cfg[3]});candidate.Validate();
   hyp::OriginalLayerReference<double> old(cfg[0],cfg[1],candidate.layer);
   std::vector<double> sample{0.,cfg[2],std::nextafter(cfg[2],cfg[3]),std::nextafter(cfg[3],cfg[2]),cfg[3],.9*cfg[0],(1.-1e-12)*cfg[0],(1.-1e-15)*cfg[0]};
   for(int q=1;q<100;++q)sample.push_back(cfg[2]+(cfg[3]-cfg[2])*q/100.);
   double config_speed=0;
   for(double r:sample) for(auto n:{std::array<double,3>{1.,0.,0.},std::array<double,3>{.36,-.48,.8}}) {
    auto p=candidate.At(r*n[0],r*n[1],r*n[2]);auto&u=p.state;++points;
    auto baseline=old.At(r*n[0],r*n[1],r*n[2]);
    if(p.radius<=cfg[2]||p.radius>=cfg[3]) {
     Require(std::memcmp(&p,&baseline,sizeof(p))==0,"endpoint/outer branch changed");++outer_equal;
    }
    Require(u.chi.value==1,"chi not identically one");
    for(int i=0;i<3;++i){Require(u.lambda.value[i]==0,"Lambda not zero");for(int j=0;j<3;++j)Require(u.metric.g[i][j]==(i==j?1.:0.),"metric not flat");}
    auto o=hyp::CartesianOmega(u,p);auto c=hyp::EvolvedConstraints(u,o);
    Require(c.valid,"invalid constraints");
    hmax=std::max(hmax,std::abs(c.hamiltonian));mmax=std::max(mmax,std::sqrt(c.momentum_conformal_norm2));
    hyp::Z4cRHS<double> rhs{};
    Require(hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),o.omega,rhs),"invalid geometric RHS");
    Require(std::isfinite(RhsNorm(rhs)),"nonfinite geometric RHS");
    if(p.omega>=1e-4)rmax=std::max(rmax,RhsNorm(rhs));else{
     hyp::Z4cRHS<double> legacy{};Require(hyp::AssembleInterior(hyp::ConformalRHS(baseline.state,hyp::CartesianOmega(baseline.state,baseline),10/baseline.alpha,0.),baseline.omega,legacy),"invalid legacy RHS");
     Require(std::memcmp(&rhs,&legacy,sizeof(rhs))==0,"outer roundoff differs from legacy");scri_roundoff=std::max(scri_roundoff,RhsNorm(rhs));
    }
    for(bool physical:{false,true}) {
     hyp::LayerGaugeParameters gauge;gauge.r0=(cfg[2]+cfg[3])/2;gauge.r1=(cfg[3]+cfg[0])/2;gauge.physical_trace_lapse=physical;gauge.preferred_source=!physical;
     hyp::GaugeRHS<double> gr{};Require(hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,gauge),p.omega,gr),"invalid gauge RHS");
     gmax=std::max(gmax,std::abs(gr.alpha));for(int i=0;i<3;++i)gmax=std::max(gmax,std::abs(gr.beta[i]));
    }
    null_residual=std::max(null_residual,std::abs(p.alpha*p.alpha-p.b*p.b-p.omega*p.omega)/(1+p.L*p.L));
    config_speed=std::max(config_speed,p.outgoing);
   }
   DifferenceAudit(candidate,first,second);speed=std::max(speed,config_speed);
   std::cout<<"configuration S="<<cfg[0]<<" a="<<cfg[1]<<" r0="<<cfg[2]<<" r1="<<cfg[3]<<" maxOutgoing="<<config_speed<<"\n";
  }
  Require(hmax<1e-8&&mmax<1e-8,"analytic constraints failed");
  Require(rmax<1e-7&&gmax<1e-8,"regular-region fixedpoint failed");
  Require(first<1e-7&&second<1e-3,"Cartesian finite-difference jets failed");
  Require(null_residual<1e-13,"null/flat metric identity failed");
  std::cout<<std::setprecision(17)<<"PASS points="<<points<<" outer/core byte-equal="<<outer_equal<<" H="<<hmax<<" M="<<mmax<<" geometric(Omega>=1e-4)="<<rmax<<" gauge="<<gmax<<" firstFD="<<first<<" secondFD="<<second<<" nullIdentity="<<null_residual<<" legacySmallOmegaAmplification="<<scri_roundoff<<" maxOutgoing="<<speed<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
