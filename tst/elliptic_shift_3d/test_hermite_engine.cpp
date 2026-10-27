// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/composite_evolution.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool yes,const char *message){if(!yes)throw std::runtime_error(message);}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  auto mesh=cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2);
  auto build=[&](bool scri,int scheme){return std::make_unique<h::CompositeTensorEvolution>(mesh,true,1e-11,4,scri,false,0,7,true,0,.24,false,true,false,false,.12,0,scheme);};
  for(auto args:{std::pair<bool,int>{true,2},{false,1}}){bool caught=false;
    try{build(args.first,args.second);}catch(const std::invalid_argument&){caught=true;}
    Require(caught,"unsupported Hermite engine configuration accepted");}
  auto engine=build(true,1);engine->Initialize(0,false);
  const auto saved=engine->state,surface=engine->scri->values;
  auto rate=engine->RHS(engine->state,.0001);Real maximum=0;
  for(Real value:rate)maximum=std::max(maximum,std::abs(value));
  for(Real value:engine->boundary_rhs)maximum=std::max(maximum,std::abs(value));
  Require(maximum<1e-10&&engine->boundary_pole<1e-10,"Hermite engine violates Minkowski equilibrium");
  Require(saved==engine->state&&surface==engine->scri->values,"Hermite RHS projected state");
  Require(engine->boundary_epoch==1,"first RHS did not prepare one stage epoch");
  // Reuse the engine with genuinely new angular data; the immutable field and
  // beta preparations must be recreated for each RHS, not retained across RK.
  engine->Initialize(.001,true);const auto pulse=engine->state;
  rate=engine->RHS(engine->state,.0001);Real response=0;
  for(Real value:rate)response=std::max(response,std::abs(value));
  Require(response>1e-6&&engine->state==pulse&&engine->boundary_epoch==2,"new stage data not used or projected");
  std::cout<<"{\"minkowski_rate_max\":"<<maximum<<",\"pulse_response_max\":"<<response
    <<",\"stage_epochs\":"<<engine->boundary_epoch<<",\"active_values_unchanged\":true,\"status\":\"passed\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
