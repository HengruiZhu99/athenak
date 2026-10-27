// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <set>
#include <limits>
#include <type_traits>
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include "z4c/hyperboloidal/positive_field_chart.hpp"
namespace h=z4c::hyperboloidal;
using J=cubed_sphere::Jet;
static_assert(!std::is_constructible<h::PunctureLogPlans,cubed_sphere::CompositeDifference&&>::value,
              "temporary grid must not leave dangling wrapper");
template<typename Exception,typename Action>void Rejected(Action action,const char *label){
  bool rejected=false;try{action();}catch(const Exception&){rejected=true;}
  if(!rejected)throw std::runtime_error(std::string("missing rejection: ")+label);
}
J Exp(const J&q){J u(std::exp(q.f));for(int a=0;a<3;++a){u.d[a]=u.f*q.d[a];
  for(int b=0;b<3;++b)u.h[a][b]=u.f*(q.h[a][b]+q.d[a]*q.d[b]);}return u;}
J LogField(const Real x[3],const h::PuncturePosition &point,Real power){
  J y[3];for(int a=0;a<3;++a)y[a]=J::Variable(x[a]-point.position[a],a);
  J q=J(.2)+J(.3)*y[0]+J(.4)*y[0]*y[1]-J(.2)*y[1]*y[2]
    +J(.13)*y[0]*y[2]*y[2]+J(.09)*y[0]*y[0]*y[1]*y[1]*y[2];
  if(power!=0)q=q+J(.5*power)*cubed_sphere::Log(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
  return q;
}
h::PositiveFieldChart<Real> Chart(const Real x[3],int field){
  h::PositiveFieldChart<Real> c(h::PositiveFieldMode::log_reference);
  if(field==18){c.reference.value=(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2;
    for(int a=0;a<3;++a){c.reference.d[a]=x[a];c.reference.dd[a][a]=1;}}
  return c;
}
Real ScaledError(const h::ScalarJet<Real>&u,const J &v,Real spacing){
  Real error=std::abs(u.value-v.f);for(int a=0;a<3;++a){error=std::max(error,spacing*std::abs(u.d[a]-v.d[a]));
    for(int b=0;b<3;++b)error=std::max(error,spacing*spacing*std::abs(u.dd[a][b]-v.h[a][b]));}return error;
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int result=0;
try{const auto start=std::chrono::steady_clock::now();
  std::cout<<std::setprecision(17)<<"{\"cases\":[";
  for(int mode=0;mode<3;++mode){
    const bool amr=mode==2;const int n=amr?4:8;
    cubed_sphere::CompositeMesh mesh(.25,amr?std::vector<Real>{.65,1}:std::vector<Real>{1},1,n,2,amr?2:1);
    if(amr)mesh.RefineCartesian(0);
    cubed_sphere::CompositeDifference grid(mesh,4);
    h::PuncturePosition point;point.velocity={-.17,.03,.09};point.time=.2;point.stage=2;point.revision=7;
    if(mode==1)point.position={.037,.013,-.021};
    int radius_rejections=0;
    for(Real radius:{Real(0),Real(-.1),std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity(),std::numeric_limits<Real>::quiet_NaN()}){
      Rejected<std::invalid_argument>([&]{h::PunctureLogPlans invalid(grid,point,radius);},"invalid radius");
      ++radius_rejections;
    }
    h::PunctureLogPlans plans(grid,point);
    if(plans.rows.empty())throw std::runtime_error("no selected core rows");
    const int components=22,block_size=grid.padded*grid.padded*grid.padded;
    std::vector<Real> state(components*grid.active.size(),.731);
    std::set<size_t> targets;
    bool mixed_levels=false;Real error=0,stored_error=0,value_error=0,nearest_error=0;
    int donor_count=0,checks=0,stale_rejections=0;
    for(const auto &row:plans.rows){
      if(!targets.insert(row.active).second)throw std::runtime_error("duplicate target");
      if(grid.mesh.blocks[grid.active[row.active]/block_size].face>=0)throw std::runtime_error("shell target enriched");
      if(row.donors.size()!=125)throw std::runtime_error("wrong donor count");
      std::set<size_t> unique;Real maximum_selected=0,minimum_other=1.e100;std::set<int> levels;
      const auto &x=grid.geometry[row.active].x;
      for(size_t donor:row.donors){if(!unique.insert(donor).second)throw std::runtime_error("duplicate donor");
        const auto &block=grid.mesh.blocks[grid.active[donor]/block_size];
        if(block.face>=0)throw std::runtime_error("noncartesian donor");levels.insert(block.refinement);
        Real r2=0;for(int a=0;a<3;++a)r2+=std::pow(grid.geometry[donor].x[a]-x[a],2);
        maximum_selected=std::max(maximum_selected,r2);++donor_count;}
      mixed_levels=mixed_levels||levels.size()>1;
      for(size_t p=0;p<grid.active.size();++p)if(!unique.count(p)&&grid.mesh.blocks[grid.active[p]/block_size].face<0){
        Real r2=0;for(int a=0;a<3;++a)r2+=std::pow(grid.geometry[p].x[a]-x[a],2);
        minimum_other=std::min(minimum_other,r2);}
      nearest_error=std::max(nearest_error,std::max(Real(0),maximum_selected-minimum_other));
    }
    if(nearest_error>1.e-15)throw std::runtime_error("donors are not nearest active Cartesian points");
    int constant_checks=0;
    for(size_t p=0;p<grid.active.size();++p)for(int field:{0,18})state[p*components+field]=1.23;
    for(const auto &row:plans.rows)for(int field:{0,18}){
      const auto q=plans.Sample(row,state,components,field);
      if(q.value!=1.23)throw std::runtime_error("constant value changed");
      for(int a=0;a<3;++a){if(q.d[a]!=0)throw std::runtime_error("constant gradient changed");
        for(int b=0;b<3;++b)if(q.dd[a][b]!=0)throw std::runtime_error("constant Hessian changed");}
      ++constant_checks;
    }
    for(Real power:{0.,4.,2.,1.09,.5}){
      for(size_t p=0;p<grid.active.size();++p){const auto q=LogField(grid.geometry[p].x,point,power);
        state[p*components]=q.f;state[p*components+18]=q.f;}
      const auto saved=state;
      for(const auto &row:plans.rows){const auto &x=grid.geometry[row.active].x;
        const auto exactq=LogField(x,point,power);
        const Real spacing=1/(grid.n*std::abs(grid.geometry[row.active].J[0][0]));
        for(int field:{0,18}){
          const auto q=plans.Sample(row,state,components,field);h::ScalarJet<Real> physical{};
          if(q.value!=state[row.active*components+field])throw std::runtime_error("sample projected stored value");
          J exact=Exp(exactq);
          if(field==18){J z(0);for(int a=0;a<3;++a){J y=J::Variable(x[a],a);z=z+y*y;}exact=J(.5)*(J(1)+z)*exact;}
          if(Chart(x,field).Reconstruct(q,physical)!=h::PositiveFieldStatus::valid)
            throw std::runtime_error("positive reconstruction failed");
          error=std::max(error,ScaledError(physical,exact,spacing));
          h::ScalarJet<Real> exactstored{};exactstored.value=exactq.f;
          for(int a=0;a<3;++a){exactstored.d[a]=exactq.d[a];for(int b=0;b<3;++b)exactstored.dd[a][b]=exactq.h[a][b];}
          stored_error=std::max(stored_error,ScaledError(q,exactq,spacing));
          h::ScalarJet<Real> baseline{};Chart(x,field).Reconstruct(exactstored,baseline);
          value_error=std::max(value_error,std::abs(physical.value-baseline.value));++checks;
        }
      }
      if(state!=saved)throw std::runtime_error("sampling mutated active state");
    }
    // Non-polynomial donor data make fitted and actual query values differ.
    // The wrapper must still retain the actual current value at every target.
    for(size_t p=0;p<grid.active.size();++p)state[p*components]=std::sin(3.1*p);
    const auto saved=state;
    for(const auto &row:plans.rows)if(plans.Sample(row,state,components,0).value!=state[row.active*components])
      throw std::runtime_error("non-polynomial active value projected");
    if(state!=saved)throw std::runtime_error("all-cell non-projection control mutated values");
    for(int which=0;which<5;++which){plans.position=point;
      if(which==0)plans.position.position[0]+=.001;
      if(which==1)plans.position.velocity[1]+=.001;
      if(which==2)plans.position.time+=.001;
      if(which==3)++plans.position.stage;
      if(which==4)++plans.position.revision;
      bool rejected=false;try{plans.Sample(plans.rows.front(),state,components,0);}
      catch(const std::runtime_error&){rejected=true;}
      if(!rejected)throw std::runtime_error("stale position metadata accepted");++stale_rejections;
    }
    plans.position=point;
    int storage_rejections=0;
    auto storage_reject=[&](const std::vector<Real> &data,int count,int field){
      Rejected<std::invalid_argument>([&]{plans.Sample(plans.rows.front(),data,count,field);},"storage contract");
      ++storage_rejections;
    };
    storage_reject(state,0,0);storage_reject(state,-1,0);
    storage_reject(state,components,-1);storage_reject(state,components,components);
    storage_reject(state,components,100);storage_reject(state,components-1,0);
    auto short_state=state;short_state.pop_back();storage_reject(short_state,components,0);
    auto long_state=state;long_state.push_back(0);storage_reject(long_state,components,0);
    int mesh_rejections=0;
    auto mesh_reject=[&](auto mutate,const char *label){
      const auto saved_mesh=grid.mesh;const auto saved_n=grid.n;const auto saved_active=grid.active;
      mutate();bool rejected=false;
      try{plans.Sample(plans.rows.front(),state,components,0);}catch(const std::runtime_error&){rejected=true;}
      grid.mesh=saved_mesh;grid.n=saved_n;grid.active=saved_active;
      if(!rejected)throw std::runtime_error(std::string("missing mesh rejection: ")+label);
      ++mesh_rejections;
      // Restoring the complete original metadata restores a valid cache.
      plans.Sample(plans.rows.front(),state,components,0);
    };
    mesh_reject([&]{++grid.mesh.revision;},"revision");
    mesh_reject([&]{grid.mesh.half_width+=.001;},"half_width");
    mesh_reject([&]{grid.mesh.radii[0]+=.001;},"radii");
    mesh_reject([&]{grid.mesh.blocks[0].lo[0]+=.001;},"lower map bound");
    mesh_reject([&]{grid.mesh.blocks[0].hi[1]+=.001;},"upper map bound");
    mesh_reject([&]{++grid.mesh.blocks[0].face;},"face");
    mesh_reject([&]{++grid.mesh.blocks[0].layer;},"layer");
    mesh_reject([&]{++grid.mesh.blocks[0].refinement;},"refinement");
    mesh_reject([&]{grid.mesh.blocks.pop_back();},"block count");
    mesh_reject([&]{++grid.n;},"active cell shape");
    mesh_reject([&]{grid.active.pop_back();},"active cell count");
    mesh_reject([&]{++grid.mesh.cells;},"descriptor cell shape");
    if(amr&&!mixed_levels)throw std::runtime_error("AMR test did not cross refinement levels");
    if(error>2.e-8||stored_error>2.e-8||value_error!=0)throw std::runtime_error("composite enrichment reconstruction failed");
    std::cout<<(mode?",":"")<<"{\"mode\":"<<mode<<",\"n\":"<<n<<",\"amr\":"<<amr
      <<",\"translated\":"<<(mode==1)<<",\"active_cells\":"<<grid.active.size()<<",\"selected_rows\":"<<plans.rows.size()
      <<",\"untouched_cells\":"<<grid.active.size()-plans.rows.size()<<",\"donors_checked\":"<<donor_count
      <<",\"nearest_distance_violation\":"<<nearest_error<<",\"cross_level_donors\":"<<mixed_levels
      <<",\"constant_checks\":"<<constant_checks<<",\"reconstruction_checks\":"<<checks<<",\"maximum_condition\":"<<plans.maximum_condition
      <<",\"maximum_scaled_stored_jet_error\":"<<stored_error<<",\"maximum_scaled_physical_jet_error\":"<<error
      <<",\"physical_value_change\":"<<value_error<<",\"stale_point_rejections\":"<<stale_rejections
      <<",\"invalid_radius_rejections\":"<<radius_rejections<<",\"storage_contract_rejections\":"<<storage_rejections
      <<",\"mesh_mutation_rejections\":"<<mesh_rejections<<",\"all_active_values_unchanged\":true,\"stale_mesh_is_rejected\":true}";
  }
  std::cout<<"],\"elapsed_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
    <<",\"implementation_checks_passed\":true,\"qualification\":\"local composite wrapper algebra only; no scientific evolution gate\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
Kokkos::finalize();return result;}
