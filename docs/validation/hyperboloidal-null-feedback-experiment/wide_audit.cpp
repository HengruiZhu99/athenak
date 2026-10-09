#include "candidate_injection.hpp"
#undef InteriorLayerGauge
#undef AssembleGaugeInterior
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#define main unused_versioned_gauge_main
#include "/Users/hz0693/research/hyperboloidal/tst/hyperboloidal/test_physical_gauge.cpp"
#undef main
#pragma clang diagnostic pop
#include "null_feedback.hpp"

void CandidateReferenceAudit(){
  double maximum=0;
  for(double a:{.5,.75,1.,2.})for(bool layer:{false,true}){
    const hyp::LayerReference<double> ref(1.,a,{layer,.05,.95});
    const auto g=Gauge(1.5);double local=0;
    for(double r:{.1,.4,.6,.84,.85,.86,.89,.9,.93,.949,.95,.97,.99,.999,.9999,.999999})
      for(const auto& n:{std::array<double,3>{1,0,0},std::array<double,3>{.36,-.48,.8}}){
        const auto p=ref.At(r*n[0],r*n[1],r*n[2]);
        const auto rhs=research::Assemble(research::Gauge(p,p.state,g),p.omega);
        local=std::max(local,std::abs(rhs.alpha));
        for(double beta:rhs.beta)local=std::max(local,std::abs(beta));
      }
    maximum=std::max(maximum,local);
    std::cout<<"reference a="<<a<<" layer="<<layer<<" max="<<local<<'\n';
  }
  Close(maximum,0,2e-10,"scratch feedback reference fixed point");
}
void CandidateIdentityAudit(){
  const auto g=Gauge(1.5);double identityerror=0,boxerror=0;
  for(double a:{.5,.75,1.,2.}){
    const hyp::LayerReference<double> ref(1.,a,{true,.05,.95});
    for(double r:{.84,.85,.86,.9,.93,.95,.99,.999}){
      const auto p=ref.At(.36*r,-.48*r,.8*r);auto u=p.state;
      u.alpha.value*=1.1;u.chi.value*=.94;
      u.beta.value[0]+=.03;u.beta.value[1]-=.02;u.beta.value[2]+=.015;
      // Retain the nonflat reference metric and its determinant-compatible jets.
      
      u.trace.value+=.04;u.theta.value=.006;
      u.lambda.value[0]+=.002;u.lambda.value[1]-=.005;
      u.alpha.d[0]+=.017;u.chi.d[1]-=.013;
      const auto o=hyp::CartesianOmega(u,p),oreference=hyp::CartesianOmega(p.state,p);
      hyp::Z4cRHS<double> geom{};
      if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),p.omega,geom))
        throw std::runtime_error("invalid geometric identity input");
      const auto candidate=research::Assemble(research::Gauge(p,u,g),p.omega);
      const auto base=research::Assemble(hyp::InteriorLayerGauge(p,u,g),p.omega);
      double gamma[4]{},gammabase[4]{};
      IndependentFourGamma(u,geom,candidate,gamma);IndependentFourGamma(u,geom,base,gammabase);
      const auto gt=hyp::Geometry(u.metric),gr=hyp::Geometry(p.state.metric);
      double dtalpha=candidate.alpha,G=0,GR=0,hessian4=0,boxbase=0,boxactual=0,zcontraction=0;
      for(int i=0;i<3;++i)dtalpha-=u.beta.value[i]*u.alpha.d[i];
      const double f0=-dtalpha/std::pow(u.alpha.value,3)
                      -(u.trace.value-3*o.normal)/(u.alpha.value*p.omega);
      identityerror=std::max(identityerror,std::abs(gamma[0]+2*u.theta.value/(u.alpha.value*p.omega)-f0));
      for(int i=0;i<3;++i){
        double dtbeta=candidate.beta[i];for(int j=0;j<3;++j)dtbeta-=u.beta.value[j]*u.beta.d[j][i];
        double source=u.chi.value*u.lambda.value[i]-dtbeta/(u.alpha.value*u.alpha.value)-u.beta.value[i]*f0;
        for(int j=0;j<3;++j){
          source+=u.chi.value*gt.inverse[i][j]*(u.chi.d[j]/(2*u.chi.value)-u.alpha.d[j]/u.alpha.value);
          hessian4+=(u.chi.value*gt.inverse[i][j]-u.beta.value[i]*u.beta.value[j]/(u.alpha.value*u.alpha.value))*p.omega_hessian[i][j];
          G+=u.chi.value*gt.inverse[i][j]*p.domega[i]*p.domega[j];
          GR+=p.state.chi.value*gr.inverse[i][j]*p.domega[i]*p.domega[j];
        }
        const double zi=.5*u.chi.value*(u.lambda.value[i]-gt.contracted[i])-u.beta.value[i]*u.theta.value/(u.alpha.value*p.omega);
        identityerror=std::max(identityerror,std::abs(gamma[i+1]+2*zi-source));
        zcontraction+=zi*p.domega[i];
        boxactual-=gamma[i+1]*p.domega[i];boxbase-=gammabase[i+1]*p.domega[i];
      }
      boxactual+=hessian4;boxbase+=hessian4;
      const double nullDifference=(G-o.normal*o.normal)-(GR-oreference.normal*oreference.normal);
      const double fulltarget=p.omega*p.w_omega+2*zcontraction+5*nullDifference/p.omega;
      const double v=hyp::SmoothCutoff(p.radius,.85,.95).value;
      boxerror=std::max(boxerror,std::abs(boxactual-((1-v)*boxbase+v*fulltarget)));
    }
  }
  Close(identityerror,0,5e-11,"scratch 4D source identity");
  Close(boxerror,0,5e-11,"scratch blended off-constraint Box identity");
  std::cout<<"source identity max="<<identityerror<<" Box extension max="<<boxerror<<'\n';
}
void FullPole(){
  const auto g=Gauge(1.5);std::cout<<std::setprecision(17)<<'[';bool first=true;
  for(double a:{.5,.75,1.,2.}){
    const hyp::LayerReference<double> ref(1.,a,{true,.05,.95});
    const auto p=ref.At(1.,0.,0.);double m[20][20]{};constexpr double eps=1e-6;
    for(int c=0;c<20;++c){double v[2][20];for(int sign=0;sign<2;++sign){
      auto u=p.state;Perturb(c,sign?eps:-eps,u);Pole(p,u,1.5,10.,true,v[sign]);
      const auto candidate=research::Gauge(p,u,g);for(int i=0;i<3;++i)v[sign][4+i]=candidate.pole.beta[i];
    }for(int r=0;r<20;++r)m[r][c]=(v[1][r]-v[0][r])/(2*eps);}
    if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"M\":[";
    for(int r=0;r<20;++r){if(r)std::cout<<',';std::cout<<'[';for(int c=0;c<20;++c){if(c)std::cout<<',';std::cout<<m[r][c];}std::cout<<']';}std::cout<<"]}";
  }std::cout<<"]\n";
}

void RegularityCounterexample(){
  const hyp::LayerReference<double> ref(1.,1.,{true,.05,.95});
  const auto g=Gauge(1.5);constexpr double dq=.01;
  for(int k=2;k<=8;++k){
    const double r=1-std::pow(10.,-k);const auto p=ref.At(r,0.,0.);auto u=p.state;
    u.trace.value+=p.omega*dq;
    for(int i=0;i<3;++i)u.trace.d[i]+=p.domega[i]*dq;
    const auto o=hyp::CartesianOmega(u,p);hyp::Z4cRHS<double> geom{};
    if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,10/u.alpha.value,0.),p.omega,geom))
      throw std::runtime_error("counterexample invalid");
    const auto gauge=research::Assemble(research::Gauge(p,u,g),p.omega);
    double numerator=geom.trace+3*o.normal*gauge.alpha/u.alpha.value;
    for(int i=0;i<3;++i)numerator+=3*p.domega[i]*gauge.beta[i]/u.alpha.value;
    Close(numerator,2*dq,10*std::pow(10.,-k)+2e-7,"unclosed Q counterexample");
    std::cout<<"unclosed Omega="<<p.omega<<" OmegaQdot="<<numerator<<" ThetaDot="<<geom.theta<<'\n';
  }
}

int main(int argc,char**argv){if(argc==2){FullPole();return 0;}CandidateReferenceAudit();CandidateIdentityAudit();RegularityCounterexample();}
