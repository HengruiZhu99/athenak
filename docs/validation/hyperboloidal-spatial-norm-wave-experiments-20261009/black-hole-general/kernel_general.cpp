#define main frozen_beta_only_main
#include "frozen_kernel.cpp"
#undef main
#include "spatial_norm.hpp"
#include "general_beta2.hpp"
struct Parameters {const char*name;double S,a,M,rho,nu,etaR;};
constexpr Parameters cases[]{
 {"original",1,.5,.5,1.5,1.5,1},{"light",1,.5,.2,1,1.5,1},
 {"a075",1,.75,.35,1.5,1.5,1},{"rho25",1,1,.5,2.5,1.5,1},
 {"scale_default",2,1,1,1.5,1.5,1},{"large_a",2,2,.3,2,1.5,1},
 {"small_S",.8,.8,.16,1.25,1.5,1},{"scale_covariant",2,1,1,1.5,.75,.5}};
double Correction(Parameters v){return outer_beta2::Coefficient(v.S,v.a,v.M,v.rho,v.nu,v.etaR);}
hyp::Z4cJet<double> AddShift(hyp::Z4cJet<double>u,const hyp::LayerPoint<double>&p,double D,double S,std::array<double,3>n={1,0,0}){
 const double r=p.radius;const auto sw=hyp::SmoothCutoff(r,.97*S,.99*S);
 double op=0,opp=0;for(int i=0;i<3;++i){op+=p.domega[i]*n[i];for(int j=0;j<3;++j)opp+=p.omega_hessian[i][j]*n[i]*n[j];}
 const hyp::Radial2<double>O{p.omega,op,opp},W{sw.value,sw.d,sw.dd};const auto db=W*O*O*hyp::Radial2<double>{D,0,0};
 hyp::Z4cJet<double>axis{};axis.beta.value[0]=db.value;axis.beta.d[0][0]=db.d;axis.beta.dd[0][0][0]=db.dd;
 const auto delta=hyp::CartesianRadialJet(axis,r,n.data());
 for(int i=0;i<3;++i){u.beta.value[i]+=delta.beta.value[i];for(int j=0;j<3;++j){u.beta.d[j][i]+=delta.beta.d[j][i];for(int k=0;k<3;++k)u.beta.dd[j][k][i]+=delta.beta.dd[j][k][i];}}
 return u;
}
Factored GeneralRates(const hyp::Z4cJet<double>&u,const hyp::LayerPoint<double>&p,const hyp::LayerGaugeParameters&g,Parameters v,double D){
 const double r=p.radius,a=v.a,M=v.M,O=p.omega,op=p.domega[0],L=p.alpha,m=M*O/(2*r),psi=1+m,alpha=u.alpha.value,chi=u.chi.value;
 const double eta=v.rho*v.S/(a*a),C=v.S/a*(1-1/v.rho),W=hyp::SmoothCutoff(r,.97*v.S,.99*v.S).value;
 const double dadiv=-M*L/(r*psi),dbdiv=M/(2*a)*(4+3*m+m*m)/std::pow(psi,3)+W*D*O;
 const double pd=M/(2*a*r)*(8-m-6*m*m-3*m*m*m)/((1-m)*std::pow(psi,3));
 const double chdiv=-M/(2*r)*(4+6*m+4*m*m+m*m*m)/std::pow(psi,4);
 const double what=-p.beta[0]*op/p.alpha,dwn_div=-(dbdiv*op+what*dadiv)/alpha,Q=-3/(a*L)+pd-3*dwn_div;
 const auto parts=hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),10/v.S/alpha,0.);
 Factored out{parts.regular,{},0,Q};out.geometry.chi+=2*alpha*chi*Q/3;
 out.gauge=hyp::InteriorLayerGauge(p,u,g).regular;
 out.gauge.alpha-=alpha*alpha*pd+(alpha+p.alpha)*dadiv/a+(alpha*dbdiv+p.beta[0]*dadiv)*op;
 out.gauge.beta[0]-=eta*(dbdiv+C*chdiv);
 out.Nraw=O*(O*op*op/(L*L)+chdiv*op*op-dwn_div*(2*what+O*dwn_div));
 return out;
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);try{
 std::cout<<std::setprecision(17);double agreement=0,H=0,Mom=0,pole=0,jet1=0,jet2=0,connectionError=0;int rows=0;
 for(const auto v:cases){
  hyp::LayerReference<double>ref(v.S,v.a,{true,.05*v.S,.95*v.S});ref.Validate();
  hyp::DetachedWormhole<double>bh(ref,v.M,{true,.3*v.S,.95*v.S});
  hyp::LayerGaugeParameters g;g.r0=.45*v.S;g.r1=.85*v.S;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=1/v.a;g.lapse_outer=v.nu;g.shift_outer=v.etaR;g.Validate(v.S);
  const double eta=v.rho*v.S/(v.a*v.a),C=v.S/v.a*(1-1/v.rho),D=Correction(v);
  for(double dB:{0.,D})for(double fraction:{.95,.99,.9999,1-1e-6,1-1e-10,1-1e-15,1.}){
   const double r=fraction*v.S;const auto p=ref.At(r,0.,0.);const auto u=AddShift(bh.At(r,0.,0.),p,dB,v.S);
   const auto o=hyp::CartesianOmega(u,p);const auto cons=hyp::EvolvedConstraints(u,o);Require(cons.valid&&u.alpha.value>0,"general BH data invalid");
   H=std::max(H,std::abs(cons.hamiltonian));Mom=std::max(Mom,std::sqrt(cons.momentum_conformal_norm2));
   const auto fact=GeneralRates(u,p,g,v,dB);const double ndot=NullRate(u,p,fact.geometry,fact.gauge);
   const auto geo=hyp::ConformalRHS(u,o,10/v.S/u.alpha.value,0.);const auto ga=hyp::SpatialNormGauge(p,u,g,v.S,v.a,eta);
   Require(geo.valid&&ga.valid,"general spatialnorm source invalid");double raw=NAN,ldot=NAN;
   if(p.omega>0){hyp::Z4cRHS<double>f{};hyp::GaugeRHS<double>b{};Require(hyp::AssembleInterior(geo,p.omega,f)&&hyp::AssembleSpatialNorm(ga,p.omega,b),"general RHS assembly invalid");raw=NullRate(u,p,f,b);ldot=f.lambda[0];
    if(p.omega>=1e-4)agreement=std::max({agreement,std::abs(raw-ndot),std::abs(b.alpha-fact.gauge.alpha),std::abs(b.beta[0]-fact.gauge.beta[0])});
    if(fraction==1-1e-6)Require(std::abs(ldot-8*dB/(3*v.a*v.a))<.01/(v.S*v.S),"general Lambda leading limit failed");
   }else {
    Require(std::abs(fact.gauge.alpha)+std::abs(fact.gauge.beta[0])+std::abs(fact.geometry.chi)+std::abs(fact.geometry.metric[0][0])+std::abs(ndot)<1e-10,"general scri leading rates not zero");
    Require(std::abs(fact.traceQ-(-3/v.S+v.M/(v.a*v.S)))<1e-11,"general finite trace Q mismatch");
    pole=std::max({pole,std::abs(ga.pole.alpha),std::abs(ga.pole.beta[0]),std::abs(geo.pole.chi),std::abs(geo.pole.trace),std::abs(geo.pole.theta),std::abs(geo.pole.lambda[0]),std::abs(geo.pole.a[0][0]),std::abs(geo.pole.a[1][1])});
    const auto base=bh.At(r,0.,0.);const auto original=hyp::ConformalRHS(base,hyp::CartesianOmega(base,p),10/v.S/base.alpha.value,0.);
    Require(std::abs(geo.regular.lambda[0]-original.regular.lambda[0]-8*dB/(3*v.a*v.a))<1e-10,"general exact Lambda Hessian increment mismatch");
    auto timejet=[&](double rr){const auto pp=ref.At(rr,0.,0.);const auto uu=AddShift(bh.At(rr,0.,0.),pp,dB,v.S);return GeneralRates(uu,pp,g,v,dB).geometry;};
    const double h=1e-5*v.S;const auto f0=timejet(v.S),f1=timejet(v.S-h),f2=timejet(v.S-2*h);
    const double drg=(3*f0.metric[0][0]-4*f1.metric[0][0]+f2.metric[0][0])/(2*h),dtg=(3*f0.metric[1][1]-4*f1.metric[1][1]+f2.metric[1][1])/(2*h),dc=(3*f0.chi-4*f1.chi+f2.chi)/(2*h);
    connectionError=std::max({connectionError,std::abs(drg-8*dB/(3*v.a*v.a)),std::abs(dtg+4*dB/(3*v.a*v.a)),std::abs(dc-2*dB/(3*v.a*v.a)),std::abs(drg+dtg-2*dc)});
   }
   auto number=[](double x){std::ostringstream s;s<<std::setprecision(17);if(std::isfinite(x))s<<x;else s<<"null";return s.str();};
   std::cout<<"{\"name\":\""<<v.name<<"\",\"S\":"<<v.S<<",\"a\":"<<v.a<<",\"M\":"<<v.M<<",\"rho\":"<<v.rho<<",\"delta_b2\":"<<dB<<",\"Omega\":"<<p.omega<<",\"N_raw\":"<<fact.Nraw<<",\"Ndot_factored\":"<<ndot<<",\"Ndot_raw\":"<<number(raw)<<",\"alpha_dot\":"<<fact.gauge.alpha<<",\"beta_dot\":"<<fact.gauge.beta[0]<<",\"chi_dot\":"<<fact.geometry.chi<<",\"g_rr_dot\":"<<fact.geometry.metric[0][0]<<",\"Qtrace\":"<<fact.traceQ<<",\"LambdaDot_raw\":"<<number(ldot)<<"}\n";++rows;
  }
  auto at=[&](std::array<double,3>x){const double r=std::hypot(std::hypot(x[0],x[1]),x[2]);auto p=ref.At(x[0],x[1],x[2]);return AddShift(bh.At(x[0],x[1],x[2]),p,D,v.S,{x[0]/r,x[1]/r,x[2]/r});};
  for(double rf:{.975,.985,.9999}){std::array<double,3>x{.36*rf*v.S,-.48*rf*v.S,.8*rf*v.S};const auto u=at(x);const double h=1e-6*v.S;
   for(int d=0;d<3;++d){auto xp=x,xm=x;xp[d]+=h;xm[d]-=h;const auto up=at(xp),um=at(xm);for(int i=0;i<3;++i)jet1=std::max(jet1,std::abs((up.beta.value[i]-um.beta.value[i])/(2*h)-u.beta.d[d][i])/(1+std::abs(u.beta.d[d][i])));
    for(int e=0;e<3;++e){auto pp=x,pm=x,mp=x,mm=x;pp[d]+=h;pp[e]+=h;pm[d]+=h;pm[e]-=h;mp[d]-=h;mp[e]+=h;mm[d]-=h;mm[e]-=h;const auto upp=at(pp),upm=at(pm),ump=at(mp),umm=at(mm);for(int i=0;i<3;++i)jet2=std::max(jet2,std::abs((upp.beta.value[i]-upm.beta.value[i]-ump.beta.value[i]+umm.beta.value[i])/(4*h*h)-u.beta.dd[d][e][i])/(1+std::abs(u.beta.dd[d][e][i])));}
   }
  }
 }
 Require(agreement<1e-8&&H<1e-10&&Mom<1e-10&&pole<1e-10&&jet1<1e-5&&jet2<1e-3&&connectionError<1e-6,"general kernel constraints/factoring/jet/time-connection audit failed");
 std::cerr<<std::setprecision(17)<<"PASS configs=8 rows="<<rows<<" agreement="<<agreement<<" H="<<H<<" M="<<Mom<<" scriPole="<<pole<<" betaFirst="<<jet1<<" betaSecond="<<jet2<<" connectionError="<<connectionError<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}Kokkos::finalize();return 0;}
