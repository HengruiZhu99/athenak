#define main frozen_beta_only_main
#include "frozen_kernel.cpp"
#undef main
#include "spatial_norm.hpp"

int main(int argc,char**argv){Kokkos::initialize(argc,argv);
 try {
  hyp::LayerReference<double> ref(1,.5,{true,.05,.95});
  hyp::DetachedWormhole<double> bh(ref,.5,{true,.3,.95});
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=2;
  std::cout<<std::setprecision(17);
  double agreement=0,H=0,M=0,maxPole=0,first=0,second=0,minAlpha=INFINITY;
  int rows=0;
  for(double eta:{5.,6.,10.})for(double db2:{0.,(100-7*eta)/(128-8*eta)}) {
   const FreeGauge free{0,0,0,db2};const double C=2*(1-4/eta);
   for(double r:{.95,.99,.9999,1-1e-6,1-1e-8,1-1e-12,1-1e-15,1.}) {
    const auto p=ref.At(r,0.,0.);const auto u=WithFreeGauge(bh.At(r,0.,0.),p,free);
    const auto O=hyp::CartesianOmega(u,p);const auto cons=hyp::EvolvedConstraints(u,O);
    Require(cons.valid&&u.alpha.value>0,"modified gauge invalid ADM data");
    H=std::max(H,std::abs(cons.hamiltonian));M=std::max(M,std::sqrt(cons.momentum_conformal_norm2));
    auto f=FactoredRates(u,p,g,0,free);const double m=.5*p.omega/(2*r),psi=1+m;
    const double dbdiv=.5/(2*.5)*(4+3*m+m*m)/std::pow(psi,3)+db2*p.omega*hyp::SmoothCutoff(r,.97,.99).value;
    const double chdiv=-.5/(2*r)*(4+6*m+4*m*m+m*m*m)/std::pow(psi,4);
    f.gauge.beta[0]-=eta*(dbdiv+C*chdiv);
    const double ndot=NullRate(u,p,f.geometry,f.gauge);
    const double gdot=(f.geometry.chi-u.chi.value*f.geometry.metric[0][0])*p.domega[0]*p.domega[0];
    const double shdot=f.gauge.beta[0]+C*gdot/(p.domega[0]*p.domega[0]);
    const auto geo=hyp::ConformalRHS(u,O,10/u.alpha.value,0.);
    const auto gauge=hyp::SpatialNormGauge(p,u,g,1.,.5,eta);
    Require(geo.valid&&gauge.valid,"invalid actual source");
    double raw=NAN,adot=NAN,bdot=NAN,pdot=NAN,thetadot=NAN,Ardot=NAN,Atdot=NAN,ldot=NAN;
    if(p.omega>0) {
     hyp::Z4cRHS<double> gr{};hyp::GaugeRHS<double> ga{};
     Require(hyp::AssembleInterior(geo,p.omega,gr)&&hyp::AssembleSpatialNorm(gauge,p.omega,ga),"invalid raw kernel assembly");
     raw=NullRate(u,p,gr,ga);adot=ga.alpha;bdot=ga.beta[0];pdot=gr.trace;thetadot=gr.theta;Ardot=gr.a[0][0];Atdot=gr.a[1][1];ldot=gr.lambda[0];
     if(p.omega>=1e-4)agreement=std::max({agreement,std::abs(raw-ndot),std::abs(adot-f.gauge.alpha),std::abs(bdot-f.gauge.beta[0])});
     if(r==1-1e-6){Require(std::abs(ldot-32*db2/3)<2e-3,"Lambda leading rate not matched");Require(std::abs(pdot)+std::abs(thetadot)+std::abs(Ardot)+std::abs(Atdot)<2e-3,"physical trace/Theta/shear leading rates not zero");}
    }else {
     Require(std::abs(f.gauge.alpha)+std::abs(f.gauge.beta[0])+std::abs(f.geometry.chi)+std::abs(f.geometry.metric[0][0])+std::abs(ndot)+std::abs(shdot)<1e-11,"zero-jet boundary source preservation failed");
     Require(std::abs(f.traceQ+2)<1e-12,"finite Qtrace initial consistency");
     maxPole=std::max({maxPole,std::abs(gauge.pole.alpha),std::abs(gauge.pole.beta[0]),std::abs(geo.pole.chi),std::abs(geo.pole.trace),std::abs(geo.pole.theta),std::abs(geo.pole.lambda[0]),std::abs(geo.pole.a[0][0]),std::abs(geo.pole.a[1][1])});
     const auto unmodified=bh.At(r,0.,0.);const auto base=hyp::ConformalRHS(unmodified,hyp::CartesianOmega(unmodified,p),10/unmodified.alpha.value,0.);
     Require(std::abs(geo.regular.lambda[0]-base.regular.lambda[0]-32*db2/3)<1e-11,"exact Lambda Hessian increment failed");
     // One-sided time-jet derivative from the complete metric RHS, independent
     // of the Lambda equation. At scri g=I and gdot=0, GammaDot^r=dgdot_rr/dr.
     auto timejet=[&](double rr){const auto pp=ref.At(rr,0.,0.);const auto uu=WithFreeGauge(bh.At(rr,0.,0.),pp,free);return FactoredRates(uu,pp,g,0,free).geometry;};
     const double h=1e-5;const auto f0=timejet(1),f1=timejet(1-h),f2=timejet(1-2*h);
     const double drg=(3*f0.metric[0][0]-4*f1.metric[0][0]+f2.metric[0][0])/(2*h);
     const double dtg=(3*f0.metric[1][1]-4*f1.metric[1][1]+f2.metric[1][1])/(2*h);
     const double dc=(3*f0.chi-4*f1.chi+f2.chi)/(2*h);
     Require(std::abs(drg-32*db2/3)<1e-6&&std::abs(dtg+16*db2/3)<1e-6&&std::abs(dc-8*db2/3)<1e-6,"actual geometric first time jets failed");
     Require(std::abs((drg-dc)+(dtg-dc))<1e-6,"physical connection derivative shear not zero");
    }
    auto out=[](double v){std::ostringstream s;s<<std::setprecision(17);if(std::isfinite(v))s<<v;else s<<"null";return s.str();};
    std::cout<<"{\"eta\":"<<eta<<",\"C\":"<<C<<",\"delta_b2\":"<<db2<<",\"r\":"<<r<<",\"Omega\":"<<p.omega<<",\"N_raw\":"<<f.Nraw<<",\"Ndot_factored\":"<<ndot<<",\"Ndot_raw\":"<<out(raw)<<",\"alpha_dot_factored\":"<<f.gauge.alpha<<",\"beta_dot_factored\":"<<f.gauge.beta[0]<<",\"Gdot_factored\":"<<gdot<<",\"shift_manifold_dot\":"<<shdot<<",\"Qtrace\":"<<f.traceQ<<",\"Pdot_raw\":"<<out(pdot)<<",\"ThetaDot_raw\":"<<out(thetadot)<<",\"Ardot_raw\":"<<out(Ardot)<<",\"Atdot_raw\":"<<out(Atdot)<<",\"LambdaDot_raw\":"<<out(ldot)<<"}\n";++rows;
   }
  }
  const FreeGauge free{0,0,0,29./40};
  auto at=[&](std::array<double,3>x){const double r=std::hypot(std::hypot(x[0],x[1]),x[2]);auto p=ref.At(x[0],x[1],x[2]);std::array<double,3>n{x[0]/r,x[1]/r,x[2]/r};return WithFreeGauge(bh.At(x[0],x[1],x[2]),p,free,n);};
  for(int i=0;i<=100;++i){const double r=.96+.04*i/100;const auto u=at({r,0,0});Require(u.alpha.value>0,"corrected lapse not positive");minAlpha=std::min(minAlpha,u.alpha.value);}
  for(double r:{.97,.975,.98,.985,.99,.9999}) {
   const std::array<double,3>x{.36*r,-.48*r,.8*r};const auto u=at(x);const double h=1e-6;
   const auto p=ref.At(x[0],x[1],x[2]);const auto gp=hyp::SpatialNormGauge(p,u,g,1.,.5,6.);
   Require(gp.valid,"oblique spatial-norm source invalid");
   for(int d=0;d<3;++d){auto xp=x,xm=x;xp[d]+=h;xm[d]-=h;const auto up=at(xp),um=at(xm);
    for(int i=0;i<3;++i)first=std::max(first,std::abs((up.beta.value[i]-um.beta.value[i])/(2*h)-u.beta.d[d][i])/(1+std::abs(u.beta.d[d][i])));
    for(int e=0;e<3;++e){auto pp=x,pm=x,mp=x,mm=x;pp[d]+=h;pp[e]+=h;pm[d]+=h;pm[e]-=h;mp[d]-=h;mp[e]+=h;mm[d]-=h;mm[e]-=h;
     const auto upp=at(pp),upm=at(pm),ump=at(mp),umm=at(mm);
     for(int i=0;i<3;++i)second=std::max(second,std::abs((upp.beta.value[i]-upm.beta.value[i]-ump.beta.value[i]+umm.beta.value[i])/(4*h*h)-u.beta.dd[d][e][i])/(1+std::abs(u.beta.dd[d][e][i])));
    }
   }
  }
  for(double r:{0.,.1,.4}){const auto p=ref.At(r,0.,0.);const auto u=p.state;const auto original=hyp::InteriorLayerGauge(p,u,g);const auto candidate=hyp::SpatialNormGauge(p,u,g,1.,.5,6.);Require(candidate.valid&&candidate.regular.alpha==original.regular.alpha&&candidate.pole.alpha==original.pole.alpha,"Cauchy gauge changed");for(int i=0;i<3;++i)Require(candidate.regular.beta[i]==original.regular.beta[i]&&candidate.pole.beta[i]==original.pole.beta[i],"core shift changed");}
  for(int j=0;j<=100;++j){const double r=j/100.;const auto p=ref.At(.36*r,-.48*r,.8*r);const auto s=hyp::SpatialNormGauge(p,p.state,g,1.,.5,6.);Require(s.valid&&std::abs(s.regular.alpha)+std::abs(s.pole.alpha)<1e-14,"reference lapse source changed");for(int i=0;i<3;++i)Require(std::abs(s.regular.beta[i])+std::abs(s.pole.beta[i])<1e-14,"reference shift source changed");}
  Require(agreement<1e-8&&H<1e-10&&M<1e-10&&maxPole<1e-11&&first<1e-5&&second<1e-3,"spatial-norm kernel/constraints/jet audit failed");
  std::cerr<<std::setprecision(17)<<"PASS rows="<<rows<<" referencePoints=101 agreement(Omega>=1e-4)="<<agreement<<" H="<<H<<" M="<<M<<" scriPole="<<maxPole<<" betaFirst="<<first<<" betaSecond="<<second<<" minAlpha="<<minAlpha<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
