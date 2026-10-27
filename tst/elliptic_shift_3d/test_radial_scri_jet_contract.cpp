// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "mesh/cubed_sphere/anchored_radial_scri_probe.hpp"
namespace c=cubed_sphere;
int main(){try{using P=c::RadialScriJetProbe;Real chain=0,constant=0,polynomial=0,noise_ratio=0;int rejected=0;const Real R=1.7;
 for(int mode=0;mode<3;++mode)for(int count:{6,7,9}){std::vector<P::Node> nodes;for(int i=0;i<count-1;++i)nodes.push_back({R-.31+.31*(i+.5)/(count-1),i});nodes.push_back({R,-1});auto row=P::Weights(nodes,R,P::Basis(mode));
  // Complete coordinate-polynomial space, including nonzero linear/quadratic.
  for(int k=0;k<count;++k){long double d=0,dd=0;for(int i=0;i<count-1;++i){auto t=P::Coordinate(nodes[i].r,R,P::Basis(mode));Real v=k?std::pow(t,k):0;d+=(long double)row.first[i]*v;dd+=(long double)row.second[i]*v;}Real a=mode==0?1:mode==1?-1:-1/R,b=mode==0?0:mode==1?-1/R:1/(R*R);polynomial=std::max({polynomial,Real(std::abs(d-(k==1?a:0))),Real(std::abs(dd-(k==1?b:k==2?2*a*a:0)))});if(!k)constant=std::max({constant,Real(std::abs(d)),Real(std::abs(dd))});}
  const Real eps=1e-14;long double perturb=0,active_l1=0;for(int i=0;i<count-1;++i){perturb+=(long double)row.first[i]*(row.first[i]>=0?eps:-eps);active_l1+=std::abs(row.first[i]);}noise_ratio=std::max(noise_ratio,Real(std::abs(perturb)/(eps*active_l1)));
 }
 // Overdetermined fixed-degree6 fit retains all six free coefficients.
 for(int mode:{0,2})for(int count:{9,13,15}){std::vector<P::Node> nodes;for(int i=0;i<count-1;++i)nodes.push_back({R-.3+.3*(i+.5)/(count-1),i});nodes.push_back({R,-1});auto row=c::AnchoredRadialScriProbe::FitWeights(nodes,R,P::Basis(mode),6);for(int k=1;k<=6;++k){long double d=0,dd=0;for(int i=0;i<count-1;++i){auto t=P::Coordinate(nodes[i].r,R,P::Basis(mode));d+=(long double)row.first[i]*std::pow(t,k);dd+=(long double)row.second[i]*std::pow(t,k);}Real a=mode==0?1:-1/R,b=mode==0?0:1/(R*R);polynomial=std::max({polynomial,Real(std::abs(d-(k==1?a:0))),Real(std::abs(dd-(k==1?b:k==2?2*a*a:0)))});}}
 // Analytic full chain for arbitrary radial function exp(t), not a fitted power.
 for(Real r:{.3,.8,1.7}){Real t=-std::log(r/R),ft=std::exp(t),ftt=ft;chain=std::max({chain,std::abs(-ft/r-(-R/(r*r))),std::abs((ftt+ft)/(r*r)-2*R/(r*r*r))});}
 for(auto bad:std::vector<std::vector<P::Node>>{{{1,0},{1,1},{1,-1}},{{0,0},{.5,1},{1,-1}},{{.5,0},{.8,1}}}){try{P::Weights(bad,1,P::Basis::Radius);}catch(const std::exception&){++rejected;}}
 std::cout<<std::setprecision(17)<<"{\"coordinate_polynomial_physical_derivative_error\":"<<polynomial<<",\"constant_anchored_derivative\":"<<constant<<",\"log_chain_error\":"<<chain<<",\"worst_case_data_noise_bound_ratio\":"<<noise_ratio<<",\"rejected_bad_node_sets\":"<<rejected<<",\"physical_r_linear_quadratic_not_assumed_exact_in_log_basis\":true}\n";if(polynomial>1e-9||constant||chain>1e-12||noise_ratio>1+1e-12||rejected!=3)throw std::runtime_error("radial coordinate contract failed");return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
