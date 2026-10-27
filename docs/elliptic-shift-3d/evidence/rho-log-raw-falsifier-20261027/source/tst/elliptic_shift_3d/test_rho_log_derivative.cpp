// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Reuse the unchanged independent analytic fields/FD controls from the prior
// full-odd falsifier. Its renamed main is never invoked by this executable.
#define main full_odd_reference_entry_not_invoked
#include "test_odd_radial_log_derivative.cpp"
#undef main
#include "z4c/hyperboloidal/rho_log_derivative.hpp"
#include <type_traits>

using Single=h::RhoLogDerivative;
// A labeled falsifier only: preserve the helper's strict defaults, and expose
// raw accuracy/noise where its new functional guards reject. Never adopt this
// object or use it as a silent fallback in an evolution-stage sampler.
class RawSingle:public Single {public:
  bool Build(const Point&q,const std::vector<Point>&s,Real h,const h::PuncturePosition&p){Single::Options o;o.maximum_gradient_l1=1e9;o.maximum_hessian_l1=1e9;return Single::Build(q,s,h,p,o);}
};
struct Audit {Real value=0,gradient=0,hessian=0,enrichment=0;};
template<class Fit>Audit FunctionalAudit(const Fit &plan,int count,Real dx,const h::PuncturePosition &p){Audit out;Real gradient[3]{},hessian[3][3]{};
  for(int i=0;i<count;++i){std::vector<Real> unit(count,0);unit[i]=1;typename Fit::Result r;if(!plan.Evaluate(unit,p,r))throw std::runtime_error("functional impulse rejected");
    out.value+=std::abs(r.fitted_value);out.enrichment+=std::abs(r.enrichment_coefficient);for(int a=0;a<3;++a){gradient[a]+=dx*std::abs(r.gradient[a]);for(int b=0;b<3;++b)hessian[a][b]+=dx*dx*std::abs(r.hessian[a][b]);}}
  for(int a=0;a<3;++a){out.gradient=std::max(out.gradient,gradient[a]);for(int b=0;b<3;++b)out.hessian=std::max(out.hessian,hessian[a][b]);}return out;}
template<class R>Real Error(const R &r,const Jet &j,Real dx){Real e=0;for(int a=0;a<3;++a){e=std::max(e,dx*std::abs(r.gradient[a]-j.d[a]));for(int b=0;b<3;++b)e=std::max(e,dx*dx*std::abs(r.hessian[a][b]-j.h[a][b]));}return e;}
Jet NewField(Point x,const h::PuncturePosition &p,Real power,const std::string &family,Real exponent){
  if(family=="wormhole")return Field(x,p,power,1,exponent);
  Jet q=Field(x,p,power,0,exponent),y[3];for(int a=0;a<3;++a)y[a]=Jet::Variable(x[a]-p.position[a],a);const Jet r2=y[0]*y[0]+y[1]*y[1]+y[2]*y[2];
  if(family=="exact_rho_polynomial")q=q-Jet(.7)*Power(r2,.5);
  if(family=="radial_fractional")q=q+Jet(.1)*Power(r2,.5*exponent);
  if(family=="angular_xy")q=q+Jet(.1)*y[0]*y[1]*Power(r2,.5*(exponent-2));
  if(family=="angular_xz")q=q+Jet(.1)*y[0]*y[2]*Power(r2,.5*(exponent-2));
  if(family=="angular_zz")q=q+Jet(.1)*(Jet(3)*y[2]*y[2]-r2)*Power(r2,.5*(exponent-2));
  return q;
}
template<class Fit>void Cases(const char *name,const Fit &fit,const Audit &audit,const std::vector<Point>&points,Point query,Real dx,const h::PuncturePosition &p,int translated,int octant){
  const std::vector<std::string> families={"exact_log_polynomial","exact_rho_polynomial","radial_fractional","angular_xy","angular_xz","angular_zz","wormhole"};
  for(const auto &family:families){if(family=="wormhole"&&translated)continue;const bool exact=family.find("exact_")==0;
    for(Real exponent:exact||family=="wormhole"?std::vector<Real>{0}:std::vector<Real>{.3,1.1,2.4})
    for(Real power:exact?std::vector<Real>{4,2,1.09,.5,2+.4*std::sin(.2),2+.4*std::sin(.8)}:std::vector<Real>{family=="wormhole"?4.:2.}){
      std::vector<Real> values;for(const auto &x:points)values.push_back(NewField(x,p,power,family,exponent).f);typename Fit::Result r,perturbed;
      if(!fit.Evaluate(values,p,r))throw std::runtime_error("fresh manufactured evaluation rejected");const auto analytic=NewField(query,p,power,family,exponent);const Real error=Error(r,analytic,dx);
      if((family=="exact_log_polynomial"||(family=="exact_rho_polynomial"&&std::string(name)!="base"))&&error>2e-8)throw std::runtime_error("declared exact-space reproduction failed");
      auto stale=p;stale.time+=.01;if(fit.Evaluate(values,stale,perturbed))throw std::runtime_error("stale stage accepted");
      const auto saved=values;for(size_t i=0;i<values.size();++i)values[i]+=1e-12*(std::sin(1.7*i)+std::cos(.9*i))/2;
      if(!fit.Evaluate(values,p,perturbed))throw std::runtime_error("finite noisy evaluation rejected");Real noise=0;
      for(int a=0;a<3;++a){noise=std::max(noise,dx*std::abs(r.gradient[a]-perturbed.gradient[a]));for(int b=0;b<3;++b)noise=std::max(noise,dx*dx*std::abs(r.hessian[a][b]-perturbed.hessian[a][b]));}
      if(noise>std::max(audit.gradient,audit.hessian)*1e-12+2e-13)throw std::runtime_error("functional L1 deterministic-noise bound failed");
      std::cout<<"{\"phase\":\"mms\",\"basis\":\""<<name<<"\",\"family\":\""<<family<<"\",\"translated\":"<<translated<<",\"octant\":"<<octant<<",\"h\":"<<dx<<",\"power\":"<<power<<",\"fractional_exponent\":"<<exponent<<",\"scaled_derivative_error\":"<<error<<",\"physical_H_error\":"<<std::abs(Hamiltonian(query,analytic.f,r)-Hamiltonian(query,analytic.f,Exact(analytic)))<<",\"scaled_noisy_derivative_change\":"<<noise<<",\"scaled_noise_bound\":"<<std::max(audit.gradient,audit.hessian)*1e-12<<"}\n";
    }
  }
}
template<class Fit>bool BuildReport(const char *name,Fit &plan,Point query,const std::vector<Point>&points,Real dx,const h::PuncturePosition &p,int translated,int octant,Audit &audit,const char *phase="build"){
  const bool ready=plan.Build(query,points,dx,p);const auto &d=plan.diagnostics();if(ready)audit=FunctionalAudit(plan,points.size(),dx,p);
  if constexpr(std::is_same_v<Fit,Single>){if(!ready){audit.value=d.value_weight_l1;audit.gradient=d.scaled_gradient_weight_l1;audit.hessian=d.scaled_hessian_weight_l1;}
    else if(std::abs(audit.gradient-d.scaled_gradient_weight_l1)>1e-9||std::abs(audit.hessian-d.scaled_hessian_weight_l1)>1e-9)throw std::runtime_error("reported functional L1 differs from impulse audit");}
  std::cout<<"{\"phase\":\""<<phase<<"\",\"basis\":\""<<name<<"\",\"translated\":"<<translated<<",\"octant\":"<<octant<<",\"h\":"<<dx<<",\"rho_over_h\":"<<std::sqrt(p.RadiusSquared(query.data()))/dx<<",\"status\":\""<<d.status<<"\",\"rank\":"<<d.rank<<",\"condition\":"<<d.condition<<",\"basis_reproduction\":"<<d.reproduction<<",\"maximum_raw_functional_l1\":"<<d.maximum_weight_l1<<",\"value_l1\":"<<audit.value<<",\"scaled_gradient_l1\":"<<audit.gradient<<",\"scaled_hessian_l1\":"<<audit.hessian<<",\"coefficient_l1\":"<<audit.enrichment<<",\"ready\":"<<ready<<"}\n";return ready;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  std::cout<<std::setprecision(17);int new_ready=0,new_rejected=0;h::PuncturePosition p;p.velocity={-.17,.03,.09};p.time=.2;p.stage=2;p.revision=7;
  for(int translated=0;translated<2;++translated){p.position=translated?Point{.037,.013,-.021}:Point{};
    for(Real dx:{.04,.02,.01,.005})for(int octant=0;octant<8;++octant){Point query;for(int a=0;a<3;++a)query[a]=dx*(std::floor(p.position[a]/dx-.5)+((octant>>a)&1)+.5);auto points=Samples(query,dx,3,true);
      Baseline base;Single single;RawSingle raw;Plan odd;Audit a,b,c,d;
      if(BuildReport("base",base,query,points,dx,p,translated,octant,a))Cases("base",base,a,points,query,dx,p,translated,octant);
      if(BuildReport("rho",single,query,points,dx,p,translated,octant,b)){++new_ready;Cases("rho",single,b,points,query,dx,p,translated,octant);}else ++new_rejected;
      if(BuildReport("rho_raw",raw,query,points,dx,p,translated,octant,d))Cases("rho_raw",raw,d,points,query,dx,p,translated,octant);
      if(BuildReport("fullodd",odd,query,points,dx,p,translated,octant,c))Cases("fullodd",odd,c,points,query,dx,p,translated,octant);
    }
  }
  for(Real ratio:{.03,.1,.3,.87,1.66,2.598,4.56,6.54,8.0,12.5,16.5}){const Real dx=.02;Point query{.01,.01,.01};p.position={query[0]-ratio*dx/std::sqrt(1.58),query[1]-.7*ratio*dx/std::sqrt(1.58),query[2]+.3*ratio*dx/std::sqrt(1.58)};auto points=Samples(query,dx,3,true);Baseline base;Single single;Plan odd;Audit a,b,c;
    BuildReport("base",base,query,points,dx,p,1,0,a,"conditioning");BuildReport("rho",single,query,points,dx,p,1,0,b,"conditioning");RawSingle raw;Audit d;BuildReport("rho_raw",raw,query,points,dx,p,1,0,d,"conditioning");BuildReport("fullodd",odd,query,points,dx,p,1,0,c,"conditioning");}
  p.position=Point{};Single rejected;auto points=Samples(Point{.01,.01,.01},.02,3,true);
  if(rejected.Build(Point{},points,.02,p))throw std::runtime_error("puncture query accepted");std::cout<<"{\"phase\":\"rejection\",\"case\":\"puncture_query\",\"status\":\""<<rejected.diagnostics().status<<"\"}\n";
  points[0]=Point{};if(rejected.Build(Point{.01,.01,.01},points,.02,p))throw std::runtime_error("puncture donor accepted");std::cout<<"{\"phase\":\"rejection\",\"case\":\"puncture_donor\",\"status\":\""<<rejected.diagnostics().status<<"\"}\n";
  if(rejected.Build(Point{.01,.01,.01},Samples(Point{.01,.01,.01},.02,2,false),.02,p))throw std::runtime_error("rank-deficient five cube accepted");std::cout<<"{\"phase\":\"rejection\",\"case\":\"five_cubed\",\"status\":\""<<rejected.diagnostics().status<<"\"}\n";
  std::cout<<"{\"phase\":\"summary\",\"near_core_rho_ready\":"<<new_ready<<",\"near_core_rho_rejected\":"<<new_rejected<<",\"value_l1_guard\":16,\"gradient_l1_guard\":32,\"hessian_l1_guard\":64,\"raw_comparison_relaxes_only_gradient_hessian_guards\":true,\"noise_amplitude\":1e-12,\"production_adopted\":false}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
