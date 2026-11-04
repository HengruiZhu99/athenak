#include <iomanip>
#include <iostream>
#include <fstream>
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/native_boosted_wormhole.hpp"
namespace h=z4c::hyperboloidal;
static void Require(bool yes,const char*why){if(!yes)throw std::runtime_error(why);}
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);Kokkos::initialize(argc,argv);int code=0;
 try{
  Require(argc==2,"supply stationary reference config");
  hyperbolic_reproduction::RefinedConfiguration cfg;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  auto p=cfg.gauge;Require(p.stationary_outer_reference,"reference option required");
  h::StationaryGaugeReference reference(p);h::NativeWormhole base(1./3.,2e-13,p.reference_start_M,p.reference_end_M,h::NativeWormhole::TransitionProfile::polynomial_c6);
  Real balance=0,perturbation=0,inner_error=0,geometry_error=0,max_lapse_response=-1e99;int checks=0;
  // Independent derivatives of the reference, not the production source formula.
  auto alpha_beta=[&](Real r){const auto v=base.Point(r);const Real old=(1-r*r)/2,S=-v.trace/3+old*v.ar/2;
    return std::array<Real,2>{p.profile.Radial(r).value*std::sqrt(1-2/(3*v.areal)+v.areal*v.areal*S*S),-S*r};};
  for(Real r:{.63,.7,.78,.8,.85,.9,.90875,.93,.97,.99}){
   auto q=reference.At(r);const Real step=1e-5,x[3]={r,0,0};auto f=alpha_beta(r);
   auto a=alpha_beta(r-2*step),b=alpha_beta(r-step),c=alpha_beta(r+step),d=alpha_beta(r+2*step);
   const Real ap=(a[0]-8*b[0]+8*c[0]-d[0])/(12*step),bp=(a[1]-8*b[1]+8*c[1]-d[1])/(12*step);
   const auto bg=base.Point(r);h::Z4cJet<Real> u{};u.chi.value=bg.chi;u.alpha.value=f[0];u.alpha.d[0]=ap;u.trace.value=bg.trace;
   u.beta.value[0]=f[1];u.beta.d[0][0]=bp;u.beta.d[1][1]=u.beta.d[2][2]=f[1]/r;
   for(int i=0;i<3;++i)u.metric.g[i][i]=1;
   h::VectorJet<Real>B{};Real D[3]{};
   auto original=h::HyperbolicGauge(x,u,B,D,p),changed=original;h::ApplyStationaryGaugeSource(x,u.alpha.value,q,changed);
   if(r>=p.inner_end)balance=std::max({balance,std::abs(changed.lapse_regular+changed.lapse_pole/p.profile.Radial(r).value),std::abs(changed.shift_regular[0]+changed.shift_pole[0]/p.profile.Radial(r).value)});
   if(r>=p.inner_end){Real rates[2]{};
    for(int sign:{-1,1}){auto v=u;v.alpha.value+=sign*1e-6;auto g=h::HyperbolicGauge(x,v,B,D,p);h::ApplyStationaryGaugeSource(x,v.alpha.value,q,g);rates[(sign+1)/2]=g.lapse_regular+g.lapse_pole/p.profile.Radial(r).value;}
    const Real response=(rates[1]-rates[0])/2e-6;max_lapse_response=std::max(max_lapse_response,response);Require(response<0,"positive frozen-geometry lapse response");
   }
   // For arbitrary perturbed fields the added source is quadratic in lapse and
   // independent of all jets, metric, D, B, shift and its spatial derivatives.
   // The lapse dependence is quadratic, including the reference K target.
   for(Real eps:{-.02,.03}){
    auto v=u;v.alpha.value+=eps;v.alpha.d[0]+=.13;v.beta.value[1]=.02;v.beta.d[2][0]=-.07;v.trace.value+=.01;v.lambda.value[0]=.003;
    D[1]=.17;B.value[0]=.11;B.d[0][1]=-.19;
    auto old=h::HyperbolicGauge(x,v,B,D,p),now=old;h::ApplyStationaryGaugeSource(x,v.alpha.value,q,now);
    perturbation=std::max(perturbation,std::abs((now.lapse_regular-old.lapse_regular)-(q.lapse_offset+(q.lapse_linear+q.lapse_quadratic*(v.alpha.value-q.alpha))*(v.alpha.value-q.alpha))));
    for(int i=0;i<3;++i){Require(now.shift_pole[i]==old.shift_pole[i]&&now.flux_rate[i]==old.flux_rate[i],"source altered pole or B principal terms");perturbation=std::max(perturbation,std::abs(now.shift_regular[i]-old.shift_regular[i]-q.shift_offset*x[i]));}
    Require(now.lapse_pole==old.lapse_pole,"source altered lapse pole storage");++checks;
   }
  }
  for(Real r:{0.,.01,.2,.4,.5,.62}){auto q=reference.At(r);inner_error=std::max({inner_error,std::abs(q.lapse_offset),std::abs(q.lapse_linear),std::abs(q.lapse_quadratic),std::abs(q.shift_offset)});}
  // Reference configuration changes only lapse and shift, including boosted
  // geometry; retained core initial gauge is exactly unchanged.
  auto oldcfg=cfg;oldcfg.gauge.stationary_outer_reference=false;oldcfg.stationary_reference.reset();
  for(Real boost:{0.,.1}){h::NativeBoostedWormhole native(boost,1./3.,true,p.reference_start_M,p.reference_end_M);
   for(Real r:{.01,.15,.399,.4,.45,.55,.62,.75,.9,.98})for(auto n:{std::array<Real,3>{1,0,0},std::array<Real,3>{.6,0,.8}}){Real x[3]={r*n[0],r*n[1],r*n[2]};auto a=native.Fields(x),b=a;oldcfg.InitialFields(x,a);cfg.InitialFields(x,b);
    Require(b[18]+(1+r*r)/2>0,"nonpositive reference initial lapse");
    for(int f=0;f<31;++f)if((f!=18&&(f<22||f>24))||r<=p.profile.flat_radius)geometry_error=std::max(geometry_error,std::abs(a[f]-b[f]));
   }
  }
  auto limit=reference.At(1);Require(limit.alpha==1&&limit.shift_ratio==0&&limit.lapse_offset==0&&limit.lapse_linear==0&&limit.lapse_quadratic==0&&limit.shift_offset==0,"nonzero exact scri correction");
  std::cout<<std::setprecision(17)<<"balance="<<balance<<" perturbation="<<perturbation<<" inner="<<inner_error<<" geometry="<<geometry_error<<" max_lapse_response="<<max_lapse_response<<" perturbations="<<checks<<'\n';
  Require(balance<3e-7&&perturbation<2e-13&&inner_error==0&&geometry_error==0,"stationary source or geometry oracle failed");
  Real previous_linear=0,previous_offset=0;
  for(int k=4;k<=12;++k){const Real eps=std::ldexp(1.,-k),r=1-eps;auto q=reference.At(r);std::cout<<"scri epsilon="<<eps<<" offset="<<q.lapse_offset<<" linear="<<q.lapse_linear<<" shift="<<q.shift_offset<<'\n';Require(q.shift_offset==0&&q.lapse_quadratic==0,"nonzero CMC outer shift or K reference source");
    Require(std::abs(q.lapse_offset)<.002*eps*eps+3e-12 && std::abs(q.lapse_linear)<.6*eps*eps,"nonvanishing scri value/normal limit");
    if(k>4&&k<=7){Require(std::abs(previous_linear/q.lapse_linear)>3.8 && std::abs(previous_linear/q.lapse_linear)<4.5,"perturbed lapse correction fails quadratic limit");Require(std::abs(previous_offset/q.lapse_offset)>12,"reference lapse offset fails higher-order limit");}
    previous_linear=q.lapse_linear;previous_offset=q.lapse_offset;}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
