#include "native_injection.hpp"
#undef InteriorLayerGauge
#undef AssembleGaugeInterior
#include "spatial_norm_control.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main unused_versioned_main
#include "/Users/hz0693/research/hyperboloidal/tst/hyperboloidal/test_physical_gauge.cpp"
#undef main
#pragma clang diagnostic pop
int main(){double refmax=0,sourceerror=0,boxerror=0;
  const auto g=Gauge(1.5);
  for(double a:{.5})for(bool wide:{false,true}){
    const hyp::LayerReference<double> ref(1.,a,{true,wide?.05:.2,wide?.95:.8});
    for(double r:{.1,.4,.6,.75,.85,.9,.95,.98,.999999}){
      const auto p=ref.At(.36*r,-.48*r,.8*r);const spatial_norm::Parameters par{1/a,1.5/(a*a),1/(3*a)};
      hyp::GaugeRHS<double> rg{};if(!hyp::ResearchNativeSpatialAssemble(hyp::ResearchNativeSpatialGauge(p,p.state,g),p.omega,rg))throw std::runtime_error("invalid shift reference");
      refmax=std::max(refmax,std::abs(rg.alpha));for(double b:rg.beta)refmax=std::max(refmax,std::abs(b));
      if(r==.999999)continue;
      auto u=p.state;u.alpha.value*=1.1;u.chi.value*=.94;u.trace.value+=.04;u.theta.value=.006;
      u.beta.value[0]+=.03;u.beta.value[1]-=.02;u.beta.value[2]+=.015;u.lambda.value[0]+=.002;u.lambda.value[1]-=.005;
      u.alpha.d[0]+=.017;u.chi.d[1]-=.013;u.metric.g[0][0]*=1.04;u.metric.g[1][2]+=.003;u.metric.g[2][1]=u.metric.g[1][2];
      hyp::Z4cRHS<double> geom{};hyp::GaugeRHS<double> base{},shift{};
      if(!hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),10/u.alpha.value,0.),p.omega,geom)||
         !hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,g),p.omega,base)||
         !hyp::ResearchNativeSpatialAssemble(hyp::ResearchNativeSpatialGauge(p,u,g),p.omega,shift))throw std::runtime_error("invalid shift source input");
      double gb[4]{},gs[4]{};IndependentFourGamma(u,geom,base,gb);IndependentFourGamma(u,geom,shift,gs);
      const double W=hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight;
      const double falpha=-(par.xi-g.scri_lapse_damping)*W*(u.alpha.value+p.alpha)*(u.alpha.value-p.alpha);
      const double expected0=-falpha/(u.alpha.value*u.alpha.value*u.alpha.value*p.omega);
      sourceerror=std::max(sourceerror,std::abs(gs[0]-gb[0]-expected0));double difference=0,expectedbox=0;
      double q=0;for(double d:p.domega)q+=d*d;
      const double relative_norm=W==0?0:spatial_norm::NormDeviation(p,u);
      for(int i=0;i<3;++i){const double n=q==0?0:-p.domega[i]/std::sqrt(q);
        const double expected=par.eta*W*((u.beta.value[i]-p.beta[i])+par.C*n*relative_norm)/(u.alpha.value*u.alpha.value*p.omega)-u.beta.value[i]*expected0;
        sourceerror=std::max(sourceerror,std::abs(gs[1+i]-gb[1+i]-expected));difference-=(gs[1+i]-gb[1+i])*p.domega[i];expectedbox-=expected*p.domega[i];}
      boxerror=std::max(boxerror,std::abs(difference-expectedbox));
    }
  }
  Close(refmax,0,2e-10,"spatial norm fixed point");Close(sourceerror,0,2e-10,"spatial norm GH source change");Close(boxerror,0,2e-10,"spatial norm Box source change");
  std::cout<<"PASS spatial norm fixedpoint/Gamma4/Box identity "<<refmax<<' '<<sourceerror<<' '<<boxerror<<'\n';
}
