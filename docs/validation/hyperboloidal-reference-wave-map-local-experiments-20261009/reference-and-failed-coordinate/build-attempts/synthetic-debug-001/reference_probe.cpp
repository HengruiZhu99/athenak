#include <Kokkos_Core.hpp>
#include "z4c/hyperboloidal/layer_reference.hpp"
#include "complete_reference.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
namespace hyp=z4c::hyperboloidal;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
 hyp::LayerReference<double> old(1.,.5,{true,.05,.95});double r;std::cout<<std::setprecision(17);
 while(std::cin>>r){const auto p=CompleteReference(r);const auto d=RadialDerivatives(p);const auto o=old.At(r,0.,0.);double discrepancy=0,absmax=0;
  auto compare=[&](double x,double y){if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("nonfinite reference");absmax=std::max(absmax,std::abs(x-y));discrepancy=std::max(discrepancy,std::abs(x-y)/std::max({1.,std::abs(x),std::abs(y)}));};
  for(int k=0;k<=2;++k){auto scalar=[&](const auto&u){return k==0?u.value:k==1?u.d[0]:u.dd[0][0];};
   compare(d.at("alpha")[k],scalar(o.state.alpha));compare(d.at("chi")[k],scalar(o.state.chi));
   compare(d.at("beta")[k],k==0?o.state.beta.value[0]:k==1?o.state.beta.d[0][0]:o.state.beta.dd[0][0][0]);
   compare(d.at("g_radial")[k],k==0?o.state.metric.g[0][0]:k==1?o.state.metric.dg[0][0][0]:o.state.metric.ddg[0][0][0][0]);
   compare(d.at("omega")[k],k==0?o.omega:k==1?o.domega[0]:o.omega_hessian[0][0]);
  }
  for(int k=0;k<=1;++k){compare(d.at("A_radial")[k],k==0?o.state.a.k[0][0]:o.state.a.dk[0][0][0]);compare(d.at("A_tangent")[k],k==0?o.state.a.k[1][1]:o.state.a.dk[0][1][1]);compare(d.at("P")[k],k==0?o.state.trace.value:o.state.trace.d[0]);compare(d.at("lambda")[k],k==0?o.state.lambda.value[0]:o.state.lambda.d[0][0]);}
  std::cout<<r<<' '<<discrepancy<<' '<<absmax;
  for(const auto&entry:d)for(int k=0;k<(entry.first=="lambda"?3:(entry.first=="omega"||entry.first=="weight"||entry.first=="complement"?5:4));++k){if(!std::isfinite(entry.second[k]))throw std::runtime_error("nonfinite complete derivative");std::cout<<' '<<entry.second[k];}
  std::cout<<'\n';
 }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
