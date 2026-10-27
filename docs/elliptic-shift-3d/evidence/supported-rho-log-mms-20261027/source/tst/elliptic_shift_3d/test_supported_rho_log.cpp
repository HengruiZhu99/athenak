// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#define RHO_LOG_MMS_NO_MAIN
#include "test_rho_log_derivative.cpp"
#include "z4c/hyperboloidal/supported_rho_log_derivative.hpp"
#include <map>
using Supported=h::SupportedRhoLogDerivative;
std::vector<Point> Catalog(Point query,Real dx){auto points=Samples(query,dx,4,false);std::stable_sort(points.begin(),points.end(),[&](const Point&a,const Point&b){Real da=0,db=0;for(int k=0;k<3;++k){da+=std::pow((a[k]-query[k])/dx,2);db+=std::pow((b[k]-query[k])/dx,2);}return std::make_tuple(std::lround(da),a)<std::make_tuple(std::lround(db),b);});points.resize(256);return points;}
void Attempts(const Supported&fit,Real dx,const h::PuncturePosition&p,Point query,int translated,int octant,const char *phase){for(const auto &attempt:fit.attempts()){const auto &d=attempt.diagnostic;
  std::cout<<"{\"phase\":\""<<phase<<"\",\"translated\":"<<translated<<",\"octant\":"<<octant<<",\"h\":"<<dx<<",\"rho_over_h\":"<<std::sqrt(p.RadiusSquared(query.data()))/dx<<",\"donors\":"<<attempt.donors<<",\"status\":\""<<d.status<<"\",\"rank\":"<<d.rank<<",\"condition\":"<<d.condition<<",\"basis_reproduction_including_rho\":"<<d.reproduction<<",\"value_l1\":"<<d.value_weight_l1<<",\"gradient_l1\":"<<d.scaled_gradient_weight_l1<<",\"hessian_l1\":"<<d.scaled_hessian_weight_l1<<"}\n";}}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
  std::cout<<std::setprecision(17);h::PuncturePosition p;p.velocity={-.17,.03,.09};p.time=.2;p.stage=2;p.revision=7;size_t ready=0,rejected=0;std::map<size_t,size_t>counts;
  for(int translated=0;translated<2;++translated){p.position=translated?Point{.037,.013,-.021}:Point{};for(Real dx:{.04,.02,.01,.005})for(int octant=0;octant<8;++octant){Point query;for(int a=0;a<3;++a)query[a]=dx*(std::floor(p.position[a]/dx-.5)+((octant>>a)&1)+.5);const auto catalog=Catalog(query,dx);std::vector<Point>original(catalog.begin(),catalog.begin()+125);Baseline base;Audit ba;
      if(!BuildReport("base125",base,query,original,dx,p,translated,octant,ba))throw std::runtime_error("reference base rejected");Cases("base125",base,ba,original,query,dx,p,translated,octant);
      Supported fit;const bool valid=fit.Build(query,catalog,dx,p);Attempts(fit,dx,p,query,translated,octant,"support_attempt");
      if(valid){++ready;++counts[fit.selected_count()];std::vector<Point>points(catalog.begin(),catalog.begin()+fit.selected_count());const auto audit=FunctionalAudit(fit,points.size(),dx,p);if(audit.value>16+1e-10||audit.gradient>32+1e-10||audit.hessian>64+1e-10)throw std::runtime_error("independent support-cap audit failed");
        Cases("supported_rho",fit,audit,points,query,dx,p,translated,octant);Baseline matched;Audit ma;if(!BuildReport("base_matched_support",matched,query,points,dx,p,translated,octant,ma))throw std::runtime_error("matched support base rejected");Cases("base_matched_support",matched,ma,points,query,dx,p,translated,octant);
      }else{++rejected;std::cout<<"{\"phase\":\"unsupported\",\"translated\":"<<translated<<",\"octant\":"<<octant<<",\"h\":"<<dx<<",\"status\":\""<<fit.diagnostics().status<<"\"}\n";}
    }}
  for(Real ratio:{.03,.1,.3,.87,1.66,2.598,4.56,6.54,8.0,12.5,16.5}){const Real dx=.02;Point query{.01,.01,.01};p.position={query[0]-ratio*dx/std::sqrt(1.58),query[1]-.7*ratio*dx/std::sqrt(1.58),query[2]+.3*ratio*dx/std::sqrt(1.58)};Supported fit;fit.Build(query,Catalog(query,dx),dx,p);Attempts(fit,dx,p,query,1,0,"conditioning_attempt");}
  std::cout<<"{\"phase\":\"summary\",\"ready\":"<<ready<<",\"unsupported\":"<<rejected<<",\"support_counts\":{";bool first=true;for(const auto &x:counts){if(!first)std::cout<<',';first=false;std::cout<<'"'<<x.first<<"\":"<<x.second;}std::cout<<"},\"basis_columns\":58,\"caps_unchanged_16_32_64\":true,\"full_rho_reproduction_required\":true,\"no_blend_or_lower_degree_fallback\":true,\"production_adopted\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
