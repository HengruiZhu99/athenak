// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
using J=cubed_sphere::Jet;
void Require(bool yes,const char *message){if(!yes)throw std::runtime_error(message);}
template<class F>void Reject(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}Require(caught,"invalid supported-engine configuration accepted");}
std::array<J,2> Stored(const Real x[3],Real chi_power,Real lapse_power){
  J y[3];for(int a=0;a<3;++a)y[a]=J::Variable(x[a],a);
  const J r2=y[0]*y[0]+y[1]*y[1]+y[2]*y[2],r=cubed_sphere::Power(r2,.5);
  const J window=(J(1)-r2)*(J(1)-r2);
  return {J(chi_power)*cubed_sphere::Log(r)+J(.17)*(r-J(1))+J(.03)*y[0]*window,
          J(lapse_power)*cubed_sphere::Log(r)-J(.2)*(r-J(1))+J(.02)*y[1]*window};
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;try{
  const auto mesh=cubed_sphere::CompositeMesh(.25,{.65,1},1,6,2,2);
  auto build=[&](int scheme,bool enriched,int storage,bool puncture){
    return std::make_unique<h::CompositeTensorEvolution>(mesh,true,1e-11,4,true,puncture,0,7,true,storage,.24,enriched,true,true,false,.12,scheme);};
  Reject([&]{build(2,true,2,true);});Reject([&]{build(1,false,2,true);});
  Reject([&]{build(1,true,1,true);});Reject([&]{build(1,true,2,false);});
  auto engine=build(1,true,2,true);engine->Initialize(0,false);
  Require(engine->supported_log_plans&& !engine->log_plans,"supported option retained/selected baseline derivative cache");
  const auto &grid=engine->shift.grid;Real maximum=0;size_t checks=0;
  // Reuse the SAME fixed geometry plans across wormhole/trumpet-like powers.
  // No exponent or rho coefficient is tracked, prescribed or evolved.
  for(auto powers:{std::array<Real,2>{4,2},std::array<Real,2>{2,1.09}}){
    for(size_t p=0;p<grid.active.size();++p){const auto q=Stored(grid.geometry[p].x,powers[0],powers[1]);
      engine->state[22*p]=q[0].f;engine->state[22*p+18]=q[1].f;}
    const auto saved=engine->state;const auto padded=engine->Padded(engine->state);
    engine->GeometryJets(engine->state,padded);
    Require(saved==engine->state,"geometry reconstruction projected evolved fields");
    for(const auto &row:engine->supported_log_plans->Rows()){
      const auto &x=grid.geometry[row.active].x;const auto q=Stored(x,powers[0],powers[1]);
      J r2;for(int a=0;a<3;++a){const J y=J::Variable(x[a],a);r2=r2+y*y;}
      const J expected[2]={manufactured::Exp(q[0]),J(.5)*(J(1)+r2)*manufactured::Exp(q[1])};
      const Real dx=1/(grid.n*std::abs(grid.geometry[row.active].J[0][0]));
      for(int f=0;f<2;++f){const auto &actual=f?engine->jets[row.active].alpha:engine->jets[row.active].chi;
        maximum=std::max(maximum,std::abs(actual.value-expected[f].f));
        for(int a=0;a<3;++a){maximum=std::max(maximum,dx*std::abs(actual.d[a]-expected[f].d[a]));
          for(int b=0;b<3;++b)maximum=std::max(maximum,dx*dx*std::abs(actual.dd[a][b]-expected[f].h[a][b]));}
        ++checks;
      }
    }
  }
  Require(maximum<2e-10,"positive physical reconstruction disagrees with independent AD");
  Require(engine->supported_log_plans->maximum_gradient_l1<=32&&engine->supported_log_plans->maximum_hessian_l1<=64,"functional guard relaxed");
  std::cout<<"{\"rows\":"<<engine->EnrichedCells()<<",\"physical_checks\":"<<checks<<",\"maximum_scaled_physical_jet_error\":"<<maximum
           <<",\"active_values_unchanged\":true,\"dynamic_exponent\":false,\"generic_moving_qualification\":false,\"status\":\"passed\"}\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
