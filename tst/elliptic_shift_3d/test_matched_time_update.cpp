// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/spherical_matched_evolution.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 h::SphericalMatchedEvolution e(cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2,2),1e-11);
 e.Initialize(0,false);const auto original=e.state,edge=e.scri->values;
 auto rate=e.state;std::fill(rate.begin(),rate.end(),0);
 const size_t p=0,j=18;const auto&x=e.shift.grid.geometry[p].x;
 const double alpha=e.DecodePositive(18,e.state[j],x),increment=.015;
 rate[j]=increment;const auto stage=e.MakeStage(e.state,rate,1);
 const double error=std::abs(e.DecodePositive(18,stage[j],x)-(alpha+increment));
 if(error>2e-16||stage[0]!=e.state[0]||e.scri->values!=edge)throw std::runtime_error("total stage contract");
 auto clamped=e.state;e.StoreAlpha(clamped,0,-1e-12);
 const double r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
 const double floor_error=std::abs(e.DecodePositive(18,clamped[j],x)/(1-r2)*2-1e-10);
 if(floor_error>1e-24||e.floor.nonpositive_clamps!=1||e.floor.evolution_clamps!=1)
   throw std::runtime_error("visible physical floor contract");
 h::SphericalMatchedControls options;options.physical_lapse_floor=0;
 h::SphericalMatchedEvolution nofloor(cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2,2),1e-11,options);
 nofloor.Initialize(0,false);bool rejected=false;
 try{nofloor.StoreAlpha(nofloor.state,0,-1e-12);}catch(const std::runtime_error&){rejected=true;}
 if(!rejected)throw std::runtime_error("disabled floor must reject nonpositive lapse");
 e.floor={};e.Step(1e-4);e.Step(1e-4);
 double drift=0,clock=0;
 for(size_t k=0;k<original.size();++k)drift=std::max(drift,std::abs(e.state[k]-original[k]));
 for(size_t k=0;k<edge.size()/e.fields;++k)clock=std::max(clock,std::abs(e.scri->values[k*e.fields+18]-edge[k*e.fields+18]));
 std::cout<<std::setprecision(17)<<"{\"total_stage_error\":"<<error<<",\"floor_error\":"<<floor_error
 <<",\"disabled_floor_rejects\":"<<rejected<<",\"two_step_Minkowski_drift\":"<<drift
 <<",\"clock_error\":"<<clock<<",\"equilibrium_floor_clamps\":"<<e.floor.evolution_clamps<<"}\n";
 if(drift>1e-12||clock!=0||e.floor.evolution_clamps)throw std::runtime_error("matched equilibrium guard");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}Kokkos::finalize();return code;}
