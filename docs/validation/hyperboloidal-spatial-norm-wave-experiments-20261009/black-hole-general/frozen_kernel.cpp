#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "../detached_wormhole.hpp"
#include "preferred_snapshot.hpp"
namespace hyp=z4c::hyperboloidal;
void Require(bool ok,const char*s){if(!ok)throw std::runtime_error(s);}
struct FreeGauge {double da1=0,db1=0,da2=0,db2=0;};
hyp::Z4cJet<double> WithFreeGauge(hyp::Z4cJet<double>u,const hyp::LayerPoint<double>&p,FreeGauge d,std::array<double,3>n={1,0,0}) {
 if(p.radius<=.97)return u;
 const double r=p.radius;const auto sw=hyp::SmoothCutoff(r,.97,.99);
 double op=0,opp=0;for(int i=0;i<3;++i){op+=p.domega[i]*n[i];for(int j=0;j<3;++j)opp+=p.omega_hessian[i][j]*n[i]*n[j];}
 const hyp::Radial2<double> O{p.omega,op,opp},W{sw.value,sw.d,sw.dd};
 const auto da=W*O*(hyp::Radial2<double>{d.da1,0,0}+O*hyp::Radial2<double>{d.da2,0,0});
 const auto db=W*O*(hyp::Radial2<double>{d.db1,0,0}+O*hyp::Radial2<double>{d.db2,0,0});
 hyp::Z4cJet<double>axis{};axis.alpha.value=da.value;axis.alpha.d[0]=da.d;axis.alpha.dd[0][0]=da.dd;
 axis.beta.value[0]=db.value;axis.beta.d[0][0]=db.d;axis.beta.dd[0][0][0]=db.dd;
 const auto added=hyp::CartesianRadialJet(axis,r,n.data());
 u.alpha.value+=added.alpha.value;
 for(int i=0;i<3;++i){u.alpha.d[i]+=added.alpha.d[i];for(int j=0;j<3;++j)u.alpha.dd[i][j]+=added.alpha.dd[i][j];}
 for(int i=0;i<3;++i){u.beta.value[i]+=added.beta.value[i];for(int j=0;j<3;++j){u.beta.d[j][i]+=added.beta.d[j][i];for(int k=0;k<3;++k)u.beta.dd[j][k][i]+=added.beta.dd[j][k][i];}}
 return u;
}
double NullRate(const hyp::Z4cJet<double>&u,const hyp::LayerPoint<double>&p,const hyp::Z4cRHS<double>&f,const hyp::GaugeRHS<double>&g) {
 const auto gt=hyp::Geometry(u.metric);const auto O=hyp::CartesianOmega(u,p);
 double v[3]{},spatial=0;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)v[i]+=gt.inverse[i][j]*p.domega[j];
 for(int i=0;i<3;++i){spatial+=f.chi*v[i]*p.domega[i];for(int j=0;j<3;++j)spatial-=u.chi.value*v[i]*f.metric[i][j]*v[j];}
 double shift=0;for(int i=0;i<3;++i)shift+=g.beta[i]*p.domega[i];
 return spatial+2*O.normal/u.alpha.value*(shift+O.normal*g.alpha);
}
struct Factored {hyp::Z4cRHS<double>geometry;hyp::GaugeRHS<double>gauge;double Nraw,traceQ;};
Factored FactoredRates(const hyp::Z4cJet<double>&u,const hyp::LayerPoint<double>&p,const hyp::LayerGaugeParameters&g,int mode,FreeGauge d,double eta=10) {
 const double r=p.radius,a=.5,M=.5,O=p.omega,op=p.domega[0],L=p.alpha;
 const double m=M*O/(2*r),psi=1+m,alpha=u.alpha.value,beta=u.beta.value[0],chi=u.chi.value;
 const auto sw=hyp::SmoothCutoff(r,.97,.99);
 const double dadiv=-M*L/(r*psi)+sw.value*(d.da1+d.da2*O);
 const double dbdiv=M/(2*a)*(4+3*m+m*m)/(psi*psi*psi)+sw.value*(d.db1+d.db2*O);
 const double pd=M/(2*a*r)*(8-m-6*m*m-3*m*m*m)/((1-m)*psi*psi*psi);
 const double chdiv=-M/(2*r)*(4+6*m+4*m*m+m*m*m)/std::pow(psi,4);
 const double what=-p.beta[0]*op/p.alpha,dwn_div=-(dbdiv*op+what*dadiv)/alpha;
 const double kdiv=-3/(a*L)+pd-3*dwn_div;
 const auto o=hyp::CartesianOmega(u,p);const auto parts=hyp::ConformalRHS(u,o,10/alpha,0.);
 Factored out{parts.regular,{},{},kdiv};
 out.geometry.chi+=2*alpha*chi*kdiv/3;
 const auto gaugeparts=hyp::InteriorLayerGauge(p,u,g);out.gauge=gaugeparts.regular;
 out.gauge.alpha-=alpha*alpha*pd+g.scri_lapse_damping*(alpha+p.alpha)*dadiv+(alpha*dbdiv+p.beta[0]*dadiv)*op;
 const auto c=hyp::LayerCoefficients(r,alpha,g);
 if(mode==3)out.gauge.beta[0]-=eta*dbdiv;
 const double deltaNull_div=chdiv*op*op-dwn_div*(2*what+O*dwn_div);
 out.Nraw=O*(O*op*op/(L*L)+deltaNull_div);
 if(mode==1||mode==2) {
  const double f0refdiv=3/(a*L*L);
  const double f0pole_div=f0refdiv+(3*dwn_div-f0refdiv*O*dadiv)/alpha+((alpha*dbdiv+p.beta[0]*dadiv)*op+g.scri_lapse_damping*(alpha+p.alpha)*dadiv)/(alpha*alpha*alpha);
  const double f0reg=(p.beta[0]*p.dalpha[0]+alpha*c.nu*std::log1p(O*dadiv/p.alpha))/(alpha*alpha*alpha);
  const double source=(p.beta[0]*p.state.beta.d[0][0]+c.eta*O*dbdiv)/(alpha*alpha)-beta*f0reg-chi*p.dalpha[0]/p.alpha;
  const double hess4=(chi-beta*beta/(alpha*alpha))*p.omega_hessian[0][0]+2*chi*op/r;
  const double deltaReg=hess4-O*p.w_omega-op*source;
  out.gauge.beta[0]-=alpha*alpha/op*deltaReg+alpha*alpha*beta*f0pole_div;
  if(mode==2)out.gauge.beta[0]+=5*alpha*alpha/op*deltaNull_div;
 }
 return out;
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);
 try {
  hyp::LayerReference<double> ref(1,.5,{true,.05,.95});hyp::DetachedWormhole<double> bh(ref,.5,{true,.3,.95});
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;
  const char*names[]={"baseline","preferred","feedback","eta10"};
  std::cout<<std::setprecision(17);double agreement=0,H=0,mom=0;
  int rows=0;
  for(auto free:{FreeGauge{},FreeGauge{-3.5,3.5,0,0},FreeGauge{-3.5,3.5,35.9375,0}}) {
   for(double r:{.95,.99,.9999,1-1e-6,1-1e-8,1-1e-12,1-1e-15,1.}) {
    const auto p=ref.At(r,0.,0.);const auto u=WithFreeGauge(bh.At(r,0.,0.),p,free);
    const auto cons=hyp::EvolvedConstraints(u,hyp::CartesianOmega(u,p));
    Require(cons.valid&&u.alpha.value>0,"modified gauge invalid ADM data");H=std::max(H,std::abs(cons.hamiltonian));mom=std::max(mom,std::sqrt(cons.momentum_conformal_norm2));
    for(int mode=0;mode<4;++mode) {
     const auto f=FactoredRates(u,p,g,mode,free);const double factored=NullRate(u,p,f.geometry,f.gauge);
     double raw=NAN;
     if(p.omega>0) {
      hyp::Z4cRHS<double> rhs{};Require(hyp::AssembleInterior(hyp::ConformalRHS(u,hyp::CartesianOmega(u,p),10/u.alpha.value,0.),p.omega,rhs),"raw geometry invalid");
      hyp::GaugeRHSParts<double> gp;
      if(mode==1||mode==2)gp=research::Gauge(p,u,g,{.85,.95,mode==2?5.:0.});
      else {gp=hyp::InteriorLayerGauge(p,u,g);if(mode==3)gp.pole.beta[0]-=10*(u.beta.value[0]-p.beta[0]);}
      const auto gr=research::Assemble(gp,p.omega);raw=NullRate(u,p,rhs,gr);
      if(p.omega>=1e-4)agreement=std::max(agreement,std::abs(raw-factored));
     }
     if(r==1.) {
      const double a1=-1+free.da1,b1=2+free.db1;
      const double expected=mode==0?16*(2*b1-1.5*a1-4):(mode==3?16*((2-2.5)*b1-1.5*a1-4):mode==1?64*(a1+b1-1):-96*(a1+b1-1));
      Require(std::abs(factored-expected)<1e-11,"first-jet limiting gate failed");
      Require(std::abs(f.traceQ+2)<1e-12,"trace Q finite first-jet consistency");
     }
     std::ostringstream rawtext;rawtext<<std::setprecision(17);if(std::isfinite(raw))rawtext<<raw;else rawtext<<"null";
     std::cout<<"{\"mode\":\""<<names[mode]<<"\",\"da1\":"<<free.da1<<",\"db1\":"<<free.db1<<",\"da2\":"<<free.da2<<",\"r\":"<<r<<",\"Omega\":"<<p.omega<<",\"N_raw\":"<<f.Nraw<<",\"Ndot_factored\":"<<factored<<",\"Ndot_raw\":"<<rawtext.str()<<",\"alpha_dot\":"<<f.gauge.alpha<<",\"beta_dot\":"<<f.gauge.beta[0]<<",\"chi_dot\":"<<f.geometry.chi<<",\"g_rr_dot\":"<<f.geometry.metric[0][0]<<",\"g_tt_dot\":"<<f.geometry.metric[1][1]<<",\"Qtrace\":"<<f.traceQ<<"}\n";++rows;
    }
   }
  }
  g.scri_lapse_damping=2;
  for(auto free:{FreeGauge{},FreeGauge{0,0,0,.75}})for(double r:{.99,.9999,1-1e-8,1.}) {
   const auto p=ref.At(r,0.,0.);const auto u=WithFreeGauge(bh.At(r,0.,0.),p,free);const auto f=FactoredRates(u,p,g,3,free,4);
   const double ndot=NullRate(u,p,f.geometry,f.gauge);
   if(r==1.)Require(std::abs(f.gauge.alpha)+std::abs(f.gauge.beta[0])+std::abs(f.geometry.chi)+std::abs(f.geometry.metric[0][0])+std::abs(ndot)<1e-12,"zero-jet pair failed");
   std::cout<<"{\"mode\":\"xi2_eta4\",\"db2\":"<<free.db2<<",\"r\":"<<r<<",\"Omega\":"<<p.omega<<",\"Ndot_factored\":"<<ndot<<",\"alpha_dot\":"<<f.gauge.alpha<<",\"beta_dot\":"<<f.gauge.beta[0]<<",\"chi_dot\":"<<f.geometry.chi<<",\"g_rr_dot\":"<<f.geometry.metric[0][0]<<",\"Qtrace\":"<<f.traceQ<<"}\n";++rows;
  }
  double first=0,second=0,minAlpha=INFINITY;
  const FreeGauge corrected{-3.5,3.5,35.9375,0};
  auto at=[&](std::array<double,3>x){double r=std::hypot(std::hypot(x[0],x[1]),x[2]);auto p=ref.At(x[0],x[1],x[2]);std::array<double,3>n{x[0]/r,x[1]/r,x[2]/r};return WithFreeGauge(bh.At(x[0],x[1],x[2]),p,corrected,n);};
  for(int i=0;i<=100;++i){double r=.96+.04*i/100;auto p=ref.At(r,0.,0.);auto u=WithFreeGauge(bh.At(r,0.,0.),p,corrected);Require(u.alpha.value>0,"free gauge cutoff lapse not positive");minAlpha=std::min(minAlpha,u.alpha.value);}
  for(double r:{.97,.975,.98,.985,.99,.9999}) {
   const std::array<double,3>x{.36*r,-.48*r,.8*r};const auto u=at(x);const double h=1e-6;
   for(int d=0;d<3;++d) {
    auto xp=x,xm=x;xp[d]+=h;xm[d]-=h;const auto up=at(xp),um=at(xm);
    first=std::max(first,std::abs((up.alpha.value-um.alpha.value)/(2*h)-u.alpha.d[d])/(1+std::abs(u.alpha.d[d])));
    for(int i=0;i<3;++i)first=std::max(first,std::abs((up.beta.value[i]-um.beta.value[i])/(2*h)-u.beta.d[d][i])/(1+std::abs(u.beta.d[d][i])));
    for(int e=0;e<3;++e) {
     auto ypp=x,ypm=x,ymp=x,ymm=x;ypp[d]+=h;ypp[e]+=h;ypm[d]+=h;ypm[e]-=h;ymp[d]-=h;ymp[e]+=h;ymm[d]-=h;ymm[e]-=h;
     const auto pp=at(ypp),pm=at(ypm),mp=at(ymp),mm=at(ymm);
     second=std::max(second,std::abs((pp.alpha.value-pm.alpha.value-mp.alpha.value+mm.alpha.value)/(4*h*h)-u.alpha.dd[d][e])/(1+std::abs(u.alpha.dd[d][e])));
     for(int i=0;i<3;++i)second=std::max(second,std::abs((pp.beta.value[i]-pm.beta.value[i]-mp.beta.value[i]+mm.beta.value[i])/(4*h*h)-u.beta.dd[d][e][i])/(1+std::abs(u.beta.dd[d][e][i])));
    }
   }
  }
  Require(agreement<1e-8&&H<1e-10&&mom<1e-10&&first<1e-5&&second<1e-3,"kernel/factoring/unchanged ADM/free gauge jets audit failed");
  std::cerr<<std::setprecision(17)<<"PASS rows="<<rows<<" raw/factored agreement(Omega>=1e-4)="<<agreement<<" fixedADM_H="<<H<<" fixedADM_M="<<mom<<" freeGauge_first="<<first<<" freeGauge_second="<<second<<" minAlpha="<<minAlpha<<"\n";
 }catch(std::exception const&e){std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize();return 0;
}
