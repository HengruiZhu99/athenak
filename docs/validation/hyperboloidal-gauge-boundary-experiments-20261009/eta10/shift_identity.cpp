#include "shift_injection.hpp"
#undef InteriorLayerGauge
#undef AssembleGaugeInterior
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main unused_versioned_main
#include "/Users/hz0693/research/hyperboloidal/tst/hyperboloidal/test_physical_gauge.cpp"
#undef main
#pragma clang diagnostic pop
int main(){double refmax=0,sourceerror=0,boxerror=0;
  const auto g=Gauge(1.5);
  for(double a:{.5,.75,1.,2.})for(bool wide:{false,true}){
    const hyp::LayerReference<double> ref(1.,a,{true,wide?.05:.2,wide?.95:.8});
    for(double r:{.1,.4,.6,.75,.85,.9,.95,.98,.999999}){
      const auto p=ref.At(.36*r,-.48*r,.8*r);
      hyp::GaugeRHS<double> rg{};if(!hyp::ResearchAssembleOuterShift(hyp::ResearchOuterShiftGauge(p,p.state,g),p.omega,rg))throw std::runtime_error("invalid shift reference");
      refmax=std::max(refmax,std::abs(rg.alpha));for(double b:rg.beta)refmax=std::max(refmax,std::abs(b));
      if(r==.999999)continue;
      auto u=p.state;u.alpha.value*=1.1;u.chi.value*=.94;u.trace.value+=.04;u.theta.value=.006;
      u.beta.value[0]+=.03;u.beta.value[1]-=.02;u.beta.value[2]+=.015;u.lambda.value[0]+=.002;u.lambda.value[1]-=.005;
      u.alpha.d[0]+=.017;u.chi.d[1]-=.013;
      hyp::Z4cRHS<double> geom{};hyp::GaugeRHS<double> base{},shift{};
      if(!hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),10/u.alpha.value,0.),p.omega,geom)||
         !hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,g),p.omega,base)||
         !hyp::ResearchAssembleOuterShift(hyp::ResearchOuterShiftGauge(p,u,g),p.omega,shift))throw std::runtime_error("invalid shift source input");
      double gb[4]{},gs[4]{};IndependentFourGamma(u,geom,base,gb);IndependentFourGamma(u,geom,shift,gs);
      sourceerror=std::max(sourceerror,std::abs(gs[0]-gb[0]));double difference=0,expectedbox=0;
      const double W=hyp::LayerCoefficients(p.radius,u.alpha.value,g).weight;
      for(int i=0;i<3;++i){const double expected=10*W*(u.beta.value[i]-p.beta[i])/(u.alpha.value*u.alpha.value*p.omega);
        sourceerror=std::max(sourceerror,std::abs(gs[1+i]-gb[1+i]-expected));difference-=(gs[1+i]-gb[1+i])*p.domega[i];expectedbox-=expected*p.domega[i];}
      boxerror=std::max(boxerror,std::abs(difference-expectedbox));
    }
  }
  Close(refmax,0,2e-10,"shift fixed point");Close(sourceerror,0,2e-10,"shift GH source change");Close(boxerror,0,2e-10,"shift Box source change");
  std::cout<<"PASS shift fixedpoint/Gamma4/Box identity "<<refmax<<' '<<sourceerror<<' '<<boxerror<<'\n';
  const hyp::LayerReference<double> ref(1.,1.,{true,.05,.95});constexpr double dq=.01;
  for(int k=3;k<=8;++k){const auto p=ref.At(1-std::pow(10.,-k),0.,0.);auto u=p.state;u.trace.value+=p.omega*dq;for(int i=0;i<3;++i)u.trace.d[i]+=p.domega[i]*dq;
    const auto o=hyp::CartesianOmega(u,p);hyp::Z4cRHS<double> geom{};hyp::GaugeRHS<double> gauge{};
    if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),p.omega,geom)||!hyp::ResearchAssembleOuterShift(hyp::ResearchOuterShiftGauge(p,u,g),p.omega,gauge))throw std::runtime_error("shift counterexample invalid");
    double n=geom.trace+3*o.normal*gauge.alpha/u.alpha.value;for(int i=0;i<3;++i)n+=3*p.domega[i]*gauge.beta[i]/u.alpha.value;
    Close(n,2*dq,10*std::pow(10.,-k)+2e-7,"shift finite Q unclosed");std::cout<<"counterexample Omega="<<p.omega<<" OmegaQdot="<<n<<" ThetaDot="<<geom.theta<<'\n';
  }
}
