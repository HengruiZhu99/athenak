// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#define ELLIPTIC_SHIFT_3D_AMR_CORE_NO_MAIN
#include "test_amr_core_derivatives.cpp"
#include "z4c/hyperboloidal/supported_puncture_log_plans.hpp"
#include <chrono>
#include <type_traits>
static_assert(!std::is_constructible<h::SupportedPunctureLogPlans,Grid&&>::value,"temporary grid rejected");
cubed_sphere::Jet AdapterField(const Real x[3],Real power){using J=cubed_sphere::Jet;J y[3];for(int a=0;a<3;++a)y[a]=J::Variable(x[a],a);const J r2=y[0]*y[0]+y[1]*y[1]+y[2]*y[2];return J(.5*power)*cubed_sphere::Log(r2)-J(.7)*cubed_sphere::Power(r2,.5)+J(.17)+J(.3)*y[0]+J(.2)*y[0]*y[1]-J(.11)*y[1]*y[2]+J(.15)*y[0]*y[0]*y[2]+J(.3)*cubed_sphere::Power(y[1],4)+J(.07)*cubed_sphere::Power(y[0],5);}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
  const auto start=std::chrono::steady_clock::now();auto grid=MakeGrid();const Real grid_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();h::SupportedPunctureLogPlans plans(grid);const Real constructor=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();V state(2*grid.active.size());Real maximum_error=0;size_t checks=0;
  for(Real power:{4.,2.,1.09,.5,2+.4*std::sin(.2),2+.4*std::sin(.8)}){for(size_t p=0;p<grid.active.size();++p)state[2*p]=state[2*p+1]=AdapterField(grid.geometry[p].x,power).f;const auto saved=state;
    for(const auto &row:plans.Rows())for(int field=0;field<2;++field){const auto q=plans.Sample(row,state,2,field);const auto exact=AdapterField(grid.geometry[row.active].x,power);const Real dx=Spacing(grid,row.active);if(q.value!=state[2*row.active+field])throw std::runtime_error("adapter projected active value");for(int a=0;a<3;++a){maximum_error=std::max(maximum_error,dx*std::abs(q.d[a]-exact.d[a]));for(int b=0;b<3;++b)maximum_error=std::max(maximum_error,dx*dx*std::abs(q.dd[a][b]-exact.h[a][b]));}++checks;}
    if(state!=saved)throw std::runtime_error("adapter mutated input state");}
  if(maximum_error>2e-8)throw std::runtime_error("adapter exact-space derivative failed");state.assign(state.size(),1.23);for(const auto &row:plans.Rows()){const auto q=plans.Sample(row,state,2,0);if(q.value!=1.23)throw std::runtime_error("adapter constant value changed");for(int a=0;a<3;++a){if(q.d[a]!=0)throw std::runtime_error("adapter constant gradient changed");for(int b=0;b<3;++b)if(q.dd[a][b]!=0)throw std::runtime_error("adapter constant Hessian changed");}}
  // Actual support/query identity against the original wrapper, constructed
  // temporarily in the test only and released before sampling-cost measurement.
  Real baseline_seconds=0;size_t baseline_donors=0;
  {const auto t=std::chrono::steady_clock::now();h::PunctureLogPlans reference(grid);
    baseline_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
    if(reference.rows.size()!=plans.Rows().size())throw std::runtime_error("query selection changed");
    for(size_t i=0;i<reference.rows.size();++i){const auto &a=reference.rows[i];const auto &b=plans.Rows()[i];
      if(a.active!=b.active||!std::equal(a.donors.begin(),a.donors.end(),b.donors.begin()))
        throw std::runtime_error("reference125prefix changed");baseline_donors+=a.donors.size();}}
  const auto sample_start=std::chrono::steady_clock::now();Real checksum=0;
  for(const auto &row:plans.Rows())for(int field=0;field<2;++field){const auto q=plans.Sample(row,state,2,field);checksum+=q.value+q.d[0]+q.dd[1][2];}
  const Real sample_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-sample_start).count();
  if(!std::isfinite(checksum))throw std::runtime_error("sample timing checksum");
  size_t rejected=0;auto reject=[&](auto action){bool caught=false;try{action();}catch(const std::exception&){caught=true;}if(!caught)throw std::runtime_error("adapter invalid state accepted");++rejected;};const auto original=plans.position;
  for(int which=0;which<5;++which){plans.position=original;if(which==0)plans.position.position[0]+=.001;if(which==1)plans.position.velocity[1]+=.001;if(which==2)plans.position.time+=.001;if(which==3)++plans.position.stage;if(which==4)++plans.position.revision;reject([&]{plans.Sample(plans.Rows().front(),state,2,0);});}plans.position=original;
  auto sample=[&]{plans.Sample(plans.Rows().front(),state,2,0);};
  auto change_int=[&](int &member){const int old=member;++member;reject(sample);member=old;};
  for(int *member:{&grid.n,&grid.ng,&grid.padded,&grid.count,&grid.order,&grid.mesh.cells,&grid.mesh.ghosts,&grid.mesh.face_subdivisions})change_int(*member);
  ++grid.mesh.revision;reject(sample);--grid.mesh.revision;
  grid.mesh.half_width+=.001;reject(sample);grid.mesh.half_width-=.001;
  const auto saved_radii=grid.mesh.radii;grid.mesh.radii[0]+=.001;reject(sample);grid.mesh.radii=saved_radii;
  const auto saved_block=grid.mesh.blocks.front();for(int which=0;which<5;++which){grid.mesh.blocks.front()=saved_block;
    if(which==0)++grid.mesh.blocks.front().face;if(which==1)++grid.mesh.blocks.front().layer;
    if(which==2)++grid.mesh.blocks.front().refinement;if(which==3)grid.mesh.blocks.front().lo[0]+=.001;
    if(which==4)grid.mesh.blocks.front().hi[2]+=.001;reject(sample);}grid.mesh.blocks.front()=saved_block;
  const auto last_block=grid.mesh.blocks.back();grid.mesh.blocks.pop_back();reject(sample);grid.mesh.blocks.push_back(last_block);
  std::swap(grid.active.front(),grid.active.back());reject(sample);std::swap(grid.active.front(),grid.active.back());
  const int last_active=grid.active.back();grid.active.pop_back();reject(sample);grid.active.push_back(last_active);
  const auto last_geometry=grid.geometry.back();grid.geometry.pop_back();reject(sample);grid.geometry.push_back(last_geometry);
  auto short_state=state;short_state.pop_back();reject([&]{plans.Sample(plans.Rows().front(),short_state,2,0);});
  reject([&]{plans.Sample(plans.Rows().front(),state,2,-1);});
  {const Real saved=state[2*plans.Rows().front().active];state[2*plans.Rows().front().active]=std::numeric_limits<Real>::infinity();reject(sample);state[2*plans.Rows().front().active]=saved;}
  sample();reject([&]{plans.Sample(plans.Rows().front(),state,0,0);});reject([&]{plans.Sample(plans.Rows().front(),state,2,2);});auto nonfinite=state;nonfinite[2*plans.Rows().front().donors.front()]=std::numeric_limits<Real>::quiet_NaN();reject([&]{plans.Sample(plans.Rows().front(),nonfinite,2,0);});
  std::cout<<std::setprecision(17)<<"{\"rows\":"<<plans.Rows().size()<<",\"enlarged_rows\":"<<plans.enlarged_rows<<",\"maximum_condition\":"<<plans.maximum_condition<<",\"maximum_reproduction\":"<<plans.maximum_reproduction<<",\"maximum_gradient_l1\":"<<plans.maximum_gradient_l1<<",\"maximum_hessian_l1\":"<<plans.maximum_hessian_l1<<",\"exact_checks\":"<<checks<<",\"maximum_scaled_derivative_error\":"<<maximum_error<<",\"invalid_state_rejections\":"<<rejected<<",\"constructor_and_grid_seconds\":"<<constructor<<",\"grid_seconds\":"<<grid_seconds<<",\"supported_constructor_seconds\":"<<constructor-grid_seconds<<",\"baseline_constructor_seconds_test_only\":"<<baseline_seconds<<",\"two_field_sample_sweep_seconds\":"<<sample_seconds<<",\"sample_sweep_rows\":"<<2*plans.Rows().size()<<",\"weight_payload_bytes\":"<<plans.ScalarWeightPayloadBytes()<<",\"donor_payload_bytes\":"<<plans.DonorPayloadBytes()<<",\"active_index_snapshot_bytes\":"<<plans.ActiveIndexPayloadBytes()<<",\"baseline_weight_bytes_removed\":"<<baseline_donors*11*sizeof(Real)<<",\"baseline_plans_retained\":false,\"baseline_sample_calls\":0,\"total_seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"active_values_changed\":false,\"generic_translations_qualified\":false,\"production_integrated\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
