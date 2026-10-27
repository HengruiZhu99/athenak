// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include "mesh/cubed_sphere/map.hpp"
#include "z4c/hyperboloidal/enriched_log_derivative.hpp"
#include "z4c/hyperboloidal/odd_radial_log_derivative.hpp"
namespace h=z4c::hyperboloidal;
using Plan=h::OddRadialLogDerivative;using Baseline=h::EnrichedLogDerivative;using Point=Plan::Point;using Jet=cubed_sphere::Jet;
using cubed_sphere::Log;using cubed_sphere::Power;
std::vector<Point> Samples(Point x,Real dx,int extent,bool nearest){
  std::vector<Point> points;for(int i=-extent;i<=extent;++i)for(int j=-extent;j<=extent;++j)
    for(int k=-extent;k<=extent;++k)points.push_back({x[0]+i*dx,x[1]+j*dx,x[2]+k*dx});
  if(nearest){std::stable_sort(points.begin(),points.end(),[&](const Point&a,const Point&b){
    Real ra=0,rb=0;for(int d=0;d<3;++d){ra+=std::pow((a[d]-x[d])/dx,2);rb+=std::pow((b[d]-x[d])/dx,2);}
    // Rounded integer distances give deterministic Cartesian tie breaking.
    return std::make_tuple(std::lround(ra),a)<std::make_tuple(std::lround(rb),b);});points.resize(125);}
  return points;
}
Jet Field(Point x,const h::PuncturePosition &p,Real power,int kind,Real angular=1.1){
  Jet y[3];for(int a=0;a<3;++a)y[a]=Jet::Variable(x[a]-p.position[a],a);
  Jet r2=y[0]*y[0]+y[1]*y[1]+y[2]*y[2];
  Jet smooth=Jet(.17)+Jet(.3)*y[0]+Jet(.2)*y[0]*y[1]-Jet(.11)*y[1]*y[2]
    +Jet(.15)*y[0]*y[0]*y[2]+Jet(.3)*y[1]*y[1]*y[1]*y[1]
    +Jet(.07)*y[0]*y[0]*y[0]*y[0]*y[0];
  if(kind==1){Jet omega=Jet(.5)*(Jet(1)-r2);const Real k=.37754694621653295,a=k/6;
    return Jet(std::log(k*k))+Jet(2)*Log(r2)-Jet(2)*Log(omega)-Jet(4)*Log(Power(r2,.5)+Jet(a));}
  Jet q=Jet(.5*power)*Log(r2)+smooth;
  if(kind==2)q=q+Jet(.1)*y[0]*y[1]*Power(r2,.5*(angular-2));
  if(kind==3)q=q+Jet(-.7)*Power(r2,.5)+Jet(.4)*Power(r2,1.5)-Jet(.2)*Power(r2,2.5);
  if(kind==4)q=q+Jet(.1)*Power(r2,.5*angular);
  return q;
}
Plan::Result FD(Point x,Real dx,const h::PuncturePosition &p,Real power,int kind,Real angular){
  Plan::Result out;out.fitted_value=Field(x,p,power,kind,angular).f;
  const Real c[5]={1./12,-2./3,0,2./3,-1./12};
  const Real c2[5]={-1./12,4./3,-2.5,4./3,-1./12};
  for(int a=0;a<3;++a)for(int s=-2;s<=2;++s){Point z=x;z[a]+=s*dx;
    Real v=Field(z,p,power,kind,angular).f;out.gradient[a]+=c[s+2]*v/dx;out.hessian[a][a]+=c2[s+2]*v/(dx*dx);}
  for(int a=0;a<3;++a)for(int b=a+1;b<3;++b){Real d=0;
    for(int s=-2;s<=2;++s)for(int t=-2;t<=2;++t){Point z=x;z[a]+=s*dx;z[b]+=t*dx;
      d+=c[s+2]*c[t+2]*Field(z,p,power,kind,angular).f/(dx*dx);}
    out.hessian[a][b]=out.hessian[b][a]=d;}
  return out;
}
template<typename Result>Real Hamiltonian(Point x,Real q,const Result &jet){
  Real r2=0,dot=0,grad2=0,lap=0;for(int a=0;a<3;++a){r2+=x[a]*x[a];
    dot+=x[a]*jet.gradient[a];grad2+=jet.gradient[a]*jet.gradient[a];lap+=jet.hessian[a][a];}
  const Real omega=(1-r2)/2;
  return std::exp(q)*(-6+2*omega*dot+omega*omega*(2*lap-.5*grad2));
}
Plan::Result Exact(const Jet &j){Plan::Result r;r.fitted_value=j.f;
  for(int a=0;a<3;++a){r.gradient[a]=j.d[a];for(int b=0;b<3;++b)r.hessian[a][b]=j.h[a][b];}return r;}
Real ScaledError(const Plan::Result&a,const Jet &b,Real dx){Real e=0;
  for(int i=0;i<3;++i){e=std::max(e,std::abs(a.gradient[i]-b.d[i])*dx);
    for(int j=0;j<3;++j)e=std::max(e,std::abs(a.hessian[i][j]-b.h[i][j])*dx*dx);}return e;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
  std::ostream *out=&std::cout;std::ofstream file;if(argc>1){file.open(argv[1]);if(!file)throw std::runtime_error("output file");out=&file;}
  *out<<std::setprecision(17);std::vector<std::string> rejected;
  h::PuncturePosition p;Point x{.01,.01,.01};Plan probe;
  if(probe.Build(x,Samples(x,.02,2,false),.02,p))throw std::runtime_error("5^3 degree5 rank deficiency not rejected");
  rejected.push_back("five_cubed:"+probe.diagnostics().status);
  if(probe.Build(Point{},Samples(x,.02,3,true),.02,p))throw std::runtime_error("endpoint accepted");
  rejected.push_back("query_endpoint:"+probe.diagnostics().status);
  auto collocated=Samples(x,.02,3,true);collocated[0]=Point{};
  if(probe.Build(x,collocated,.02,p))throw std::runtime_error("sample endpoint accepted");
  rejected.push_back("sample_endpoint:"+probe.diagnostics().status);
  Point far{.8,.1,-.1};if(probe.Build(far,Samples(far,.001,3,true),.001,p))
    throw std::runtime_error("far log degeneracy unexpectedly accepted");
  rejected.push_back("far_log:"+probe.diagnostics().status);
  *out<<"{\n\"qualification\":\"host-only derivative prototype; original physical fields and equations unchanged; no evolution validation\",\n\"rejections\":[";
  for(std::size_t i=0;i<rejected.size();++i)*out<<(i?",":"")<<'"'<<rejected[i]<<'"';
  *out<<"],\n\"cases\":[\n";bool first=true;int builds=0,checks=0;Real maxerror=0,maxcondition=0,maxreproduction=0,maxorthogonality=0;
  Real maximum_weight_l1=0,baseline_maximum_weight_l1=0,maximum_odd_noise=0,maximum_log_noise=0;
  const Real spacings[]={.04,.02,.01,.005};
  const Real powers[]={4,2,1.09,.5,2+.4*std::sin(.2),2+.4*std::sin(.8)};
  for(int translated=0;translated<2;++translated){p.position=translated?Point{.037,.013,-.021}:Point{};
    for(Real dx:spacings)for(int octant=0;octant<8;++octant){
      Point query;for(int a=0;a<3;++a)query[a]=dx*(std::floor(p.position[a]/dx-.5)+((octant>>a)&1)+.5);
      auto points=Samples(query,dx,3,true);Plan plan;Baseline baseline;builds+=2;
      if(!plan.Build(query,points,dx,p))throw std::runtime_error("odd-radial near-core plan rejected: "+plan.diagnostics().status);
      if(!baseline.Build(query,points,dx,p))throw std::runtime_error("baseline near-core plan rejected");
      maxcondition=std::max(maxcondition,plan.diagnostics().condition);
      maximum_weight_l1=std::max(maximum_weight_l1,plan.diagnostics().maximum_weight_l1);
      baseline_maximum_weight_l1=std::max(baseline_maximum_weight_l1,baseline.diagnostics().maximum_weight_l1);
      {std::vector<Real> data;for(auto z:points)data.push_back(Field(z,p,2,3).f);
        Plan::Result exact,noisy;Baseline::Result oldexact,oldnoisy;
        plan.Evaluate(data,p,exact);baseline.Evaluate(data,p,oldexact);
        for(size_t i=0;i<data.size();++i)data[i]+=1.e-12*(std::sin(1.7*i)+std::cos(.9*i))/2;
        plan.Evaluate(data,p,noisy);baseline.Evaluate(data,p,oldnoisy);
        for(int a=0;a<3;++a){maximum_odd_noise=std::max(maximum_odd_noise,dx*std::abs(noisy.gradient[a]-exact.gradient[a]));
          maximum_log_noise=std::max(maximum_log_noise,dx*std::abs(oldnoisy.gradient[a]-oldexact.gradient[a]));
          for(int b=0;b<3;++b){maximum_odd_noise=std::max(maximum_odd_noise,dx*dx*std::abs(noisy.hessian[a][b]-exact.hessian[a][b]));
            maximum_log_noise=std::max(maximum_log_noise,dx*dx*std::abs(oldnoisy.hessian[a][b]-oldexact.hessian[a][b]));}}}
      maxreproduction=std::max(maxreproduction,plan.diagnostics().reproduction);
      maxorthogonality=std::max(maxorthogonality,plan.diagnostics().orthogonality);
      for(Real power:powers){std::vector<Real> values;for(auto z:points)values.push_back(Field(z,p,power,3).f);
        Plan::Result fit;if(!plan.Evaluate(values,p,fit))throw std::runtime_error("fresh evaluation rejected");
        const auto analytic=Field(query,p,power,3);Real error=ScaledError(fit,analytic,dx);
        maxerror=std::max(maxerror,error);++checks;
        if(error>2.e-8||std::abs(fit.enrichment_coefficient-power)>2.e-8)
          throw std::runtime_error("logpower + oddradial + degree5 polynomial reproduction failed");
        auto stale=p;stale.time+=1;if(plan.Evaluate(values,stale,fit))throw std::runtime_error("stale stage accepted");
        plan.Evaluate(values,p,fit);const auto fd=FD(query,dx,p,power,3,0);Baseline::Result oldfit;
        if(!baseline.Evaluate(values,p,oldfit))throw std::runtime_error("baseline evaluation");
        if(!first)*out<<",\n";first=false;
        *out<<"{\"family\":\"power_odd_polynomial\",\"translated\":"<<translated<<",\"octant\":"<<octant
          <<",\"h\":"<<dx<<",\"p\":"<<power<<",\"scaled_derivative_error\":"<<error
          <<",\"H_enriched_error\":"<<std::abs(Hamiltonian(query,analytic.f,fit)-Hamiltonian(query,analytic.f,Exact(analytic)))
          <<",\"H_log_only_error\":"<<std::abs(Hamiltonian(query,analytic.f,oldfit)-Hamiltonian(query,analytic.f,Exact(analytic)))
          <<",\"H_fd_error\":"<<std::abs(Hamiltonian(query,analytic.f,fd)-Hamiltonian(query,analytic.f,Exact(analytic)))<<"}";
      }
      // The true isotropic branch is centered at the origin only. Its remainder
      // contains all radial powers; fixed odd enrichment is a finite approximation.
      if(!translated){std::vector<Real> values;for(auto z:points)values.push_back(Field(z,p,4,1).f);
        Plan::Result fit;plan.Evaluate(values,p,fit);auto analytic=Field(query,p,4,1);auto fd=FD(query,dx,p,4,1,0);Baseline::Result oldfit;
        if(!baseline.Evaluate(values,p,oldfit))throw std::runtime_error("wormhole baseline evaluation");
        *out<<",\n{\"family\":\"isotropic_wormhole_inner\",\"octant\":"<<octant<<",\"h\":"<<dx
          <<",\"H_exact\":"<<Hamiltonian(query,analytic.f,Exact(analytic))
          <<",\"H_enriched_error\":"<<std::abs(Hamiltonian(query,analytic.f,fit)-Hamiltonian(query,analytic.f,Exact(analytic)))
          <<",\"H_log_only_error\":"<<std::abs(Hamiltonian(query,analytic.f,oldfit)-Hamiltonian(query,analytic.f,Exact(analytic)))
          <<",\"H_fd_error\":"<<std::abs(Hamiltonian(query,analytic.f,fd)-Hamiltonian(query,analytic.f,Exact(analytic)))<<"}";
      }
      if(!translated&&octant==7)for(int kind:{2,4})for(Real angular:{.3,1.1,2.4}){
        std::vector<Real> values;for(auto z:points)values.push_back(Field(z,p,2,kind,angular).f);
        Plan::Result fit;if(!plan.Evaluate(values,p,fit))throw std::runtime_error("fractional evaluation");
        auto analytic=Field(query,p,2,kind,angular);auto fd=FD(query,dx,p,2,kind,angular);Baseline::Result oldfit;
        if(!baseline.Evaluate(values,p,oldfit))throw std::runtime_error("fractional baseline evaluation");
        *out<<",\n{\"family\":\""<<(kind==2?"angular_fractional":"radial_fractional")<<"\",\"h\":"<<dx<<",\"s\":"<<angular
          <<",\"scaled_derivative_error\":"<<ScaledError(fit,analytic,dx)
          <<",\"H_enriched_error\":"<<std::abs(Hamiltonian(query,analytic.f,fit)-Hamiltonian(query,analytic.f,Exact(analytic)))
          <<",\"H_log_only_error\":"<<std::abs(Hamiltonian(query,analytic.f,oldfit)-Hamiltonian(query,analytic.f,Exact(analytic)))
          <<",\"H_fd_error\":"<<std::abs(Hamiltonian(query,analytic.f,fd)-Hamiltonian(query,analytic.f,Exact(analytic)))<<"}";
      }
    }
  }
  *out<<"\n],\n\"radius_conditioning\":[";
  p.position=Point{};bool firstprobe=true;int conditioning_plans=0;
  for(Real radial:{.5,1.5,2.5,4.5,6.5,8.5,12.5,16.5}){
    const Real dx=.02;Point query{radial*dx,.5*dx,.5*dx};auto points=Samples(query,dx,3,true);
    Plan plan;Baseline baseline;const bool valid=plan.Build(query,points,dx,p),oldvalid=baseline.Build(query,points,dx,p);conditioning_plans+=2;
    if(!firstprobe)*out<<',';firstprobe=false;
    *out<<"{\"rho_over_h\":"<<std::sqrt(p.RadiusSquared(query.data()))/dx
      <<",\"odd_status\":\""<<plan.diagnostics().status<<"\",\"odd_condition\":"<<plan.diagnostics().condition
      <<",\"odd_rank\":"<<plan.diagnostics().rank<<",\"odd_weight_l1\":"<<plan.diagnostics().maximum_weight_l1
      <<",\"log_status\":\""<<baseline.diagnostics().status<<"\",\"log_condition\":"<<baseline.diagnostics().condition
      <<",\"log_weight_l1\":"<<baseline.diagnostics().maximum_weight_l1<<"}";
  }
  *out<<"],\n\"summary\":{\"plans\":"<<builds<<",\"conditioning_plans\":"<<conditioning_plans<<",\"exact_family_checks\":"<<checks<<",\"maximum_scaled_derivative_error\":"<<maxerror
    <<",\"maximum_condition\":"<<maxcondition<<",\"maximum_reproduction_error\":"<<maxreproduction
    <<",\"maximum_Q_orthogonality_error\":"<<maxorthogonality
    <<",\"noise_input_amplitude\":1e-12,\"maximum_weight_l1\":"<<maximum_weight_l1<<",\"baseline_maximum_weight_l1\":"<<baseline_maximum_weight_l1
    <<",\"maximum_scaled_derivative_noise\":"<<maximum_odd_noise
    <<",\"baseline_scaled_derivative_noise\":"<<maximum_log_noise<<",\"implementation_checks_passed\":true}}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
