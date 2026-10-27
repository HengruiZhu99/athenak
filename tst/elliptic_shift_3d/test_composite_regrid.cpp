// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <iostream>
#include "z4c/hyperboloidal/composite_regrid.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool yes,const char*message){if(!yes)throw std::runtime_error(message);}
template<class F>void Reject(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}Require(caught,"bad accepted transfer configuration not rejected");}
Real Polynomial(int f,const Real*x){return .0001*(f+1)*(1+x[0]-.3*x[1]+.2*x[2]+x[0]*x[1]+x[2]*x[2]);}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{
  cubed_sphere::CompositeMesh mesh(.25,{.65,1},1,6,2,2);
  auto factory=[&](bool wormhole){return std::make_unique<h::CompositeTensorEvolution>(mesh,true,1e-11,4,true,wormhole,0,7,true,wormhole?2:0,.24,false,true,true,false,.12,0,1);};
  auto source=factory(false);source->Initialize(0,false);
  for(size_t p=0;p<source->shift.grid.active.size();++p){const auto&x=source->shift.grid.geometry[p].x;
    for(int f=0;f<22;++f)source->state[22*p+f]=Polynomial(f,x);
    for(int v=0;v<3;++v)source->beta_delta[3*p+v]=.01*(v+1);}
  for(size_t p=0;p<source->scri->rows.size();++p){const auto&x=source->scri->rows[p].geometry.x;
    for(int f=0;f<22;++f)source->scri->values[22*p+f]=f==18?0:Polynomial(f,x);
    for(int v=0;v<3;++v)source->scri->beta[3*p+v]=.01*(v+1);}
  const auto saved=source->state,warm=source->beta_delta,boundary=source->scri->values,surface_warm=source->scri->beta;
  const auto parent=mesh.blocks[0];auto refined=mesh;refined.RefineCartesian(0);
  auto transaction=h::RebuildCompositeAtAcceptedState(*source,refined);Real error=0;
  Require(source->state==saved&&source->beta_delta==warm&&source->scri->values==boundary,"accepted transfer mutated source");
  Require(transaction.copied_cells+transaction.interpolated_cells==transaction.engine->shift.grid.active.size(),"transfer omitted cells");
  Require(transaction.interpolated_cells==8*216,"refined children not all interpolated");
  for(size_t p=0;p<transaction.engine->shift.grid.active.size();++p){const auto&x=transaction.engine->shift.grid.geometry[p].x;
    for(int f=0;f<22;++f)error=std::max(error,std::abs(transaction.engine->state[22*p+f]-Polynomial(f,x)));
    for(int v=0;v<3;++v)Require(transaction.engine->beta_delta[3*p+v]==.01*(v+1),"translation mode changed");}
  Require(transaction.engine->scri->values==boundary&&transaction.engine->scri->beta==surface_warm,"unchanged shell boundary not bitwise preserved");
  auto coarsened=refined;coarsened.CoarsenCartesian(parent);
  auto coarse=h::RebuildCompositeAtAcceptedState(*transaction.engine,coarsened);
  for(size_t p=0;p<coarse.engine->shift.grid.active.size();++p)for(int f=0;f<22;++f)
    error=std::max(error,std::abs(coarse.engine->state[22*p+f]-Polynomial(f,coarse.engine->shift.grid.geometry[p].x)));
  Require(error<1e-12,"physical Cartesian polynomial transfer inaccurate");
  Reject([&]{h::RebuildCompositeAtAcceptedState(*source,mesh);});
  auto changed=refined;changed.radii[0]=.66;Reject([&]{h::RebuildCompositeAtAcceptedState(*source,changed);});
  changed=refined;changed.ghosts=3;Reject([&]{h::RebuildCompositeAtAcceptedState(*source,changed);});
  changed=refined;changed.blocks.pop_back();Reject([&]{h::RebuildCompositeAtAcceptedState(*source,changed);});
  source->state[0]=std::numeric_limits<Real>::quiet_NaN();Reject([&]{h::RebuildCompositeAtAcceptedState(*source,refined);});source->state=saved;
  Real log_error=0,global_log_error=0,physical_error=0,physical_square=0,physical_volume=0,maximum_l1=0;size_t log_rows=0,enlarged_rows=0;
  for(const auto powers:{std::array<Real,2>{4,2},std::array<Real,2>{2,1.09}}){
    auto positive=factory(true);positive->Initialize(0,false);
    for(size_t p=0;p<positive->shift.grid.active.size();++p){const auto&x=positive->shift.grid.geometry[p].x;const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      positive->state[22*p]=powers[0]*std::log(r)+Polynomial(0,x);
      positive->state[22*p+18]=powers[1]*std::log(r)+Polynomial(18,x);}
    const auto before=positive->state;auto mapped=h::RebuildCompositeAtAcceptedState(*positive,refined);
    Require(before==positive->state,"positive transfer changed source logs");log_rows+=mapped.positive_log_rows;maximum_l1=std::max(maximum_l1,mapped.maximum_log_value_l1);
    for(const auto&item:mapped.positive_support_counts)if(item.first>175)enlarged_rows+=item.second;
    for(size_t p=0;p<mapped.engine->shift.grid.active.size();++p){const auto&x=mapped.engine->shift.grid.geometry[p].x;const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      const Real true_chi=std::exp(powers[0]*std::log(r)+Polynomial(0,x));
      const Real weight=mapped.engine->shift.grid.volumes[p]/std::pow(true_chi,1.5);physical_volume+=2*weight;
      for(int q=0;q<2;++q){const int f=q?18:0;const Real truth=powers[q]*std::log(r)+Polynomial(f,x);
        const Real difference=std::abs(mapped.engine->state[22*p+f]-truth);global_log_error=std::max(global_log_error,difference);
        if(r<.1)log_error=std::max(log_error,difference); // basis-reproduction region, not a norm exclusion
        const Real reference=q?(1+r*r)/2:1;
        const Real physical=reference*(std::exp(mapped.engine->state[22*p+f])-std::exp(truth));
        physical_error=std::max(physical_error,std::abs(physical));physical_square+=weight*physical*physical;}}
  }
  Require(log_rows>0&&log_error<1e-10&&maximum_l1<=8,"fixed logarithmic transfer failed");
  // Actual free tensor evolution before and after interface-adjacent refinement
  // and coarsening; this is a small equilibrium lifecycle test, not collapse.
  auto evolving=factory(false);evolving->Initialize(0,false);evolving->Step(.0001);
  auto evolved=h::RebuildCompositeAtAcceptedState(*evolving,refined);evolved.engine->Step(.0001);
  auto unrefined=h::RebuildCompositeAtAcceptedState(*evolved.engine,coarsened);unrefined.engine->Step(.0001);
  Real equilibrium=0;for(Real value:unrefined.engine->state)equilibrium=std::max(equilibrium,std::abs(value));
  Require(equilibrium<1e-11,"accepted regrid damaged Minkowski evolution");
  auto pulse=factory(false);pulse->Initialize(.001,true);pulse->Step(.0001);
  auto pulse_refined=h::RebuildCompositeAtAcceptedState(*pulse,refined);pulse_refined.engine->Step(.0001);
  auto pulse_coarse=h::RebuildCompositeAtAcceptedState(*pulse_refined.engine,coarsened);pulse_coarse.engine->Step(.0001);
  Real flux=0,metric=0;for(size_t p=0;p<pulse_coarse.engine->shift.grid.active.size();++p){
    for(int v=19;v<22;++v)flux=std::max(flux,std::abs(pulse_coarse.engine->state[22*p+v]));
    for(int v=1;v<7;++v)metric=std::max(metric,std::abs(pulse_coarse.engine->state[22*p+v]));}
  Require(flux>0&&metric>0&&pulse_coarse.engine->max_shift_residual<1e-10,"nonzero tensor/flux regrid evolution failed");
  std::cout<<"{\"all_field_polynomial_max\":"<<error<<",\"fixed_log_value_max\":"<<log_error
    <<",\"all_cell_log_transfer_max\":"<<global_log_error<<",\"all_cell_positive_physical_transfer_max\":"<<physical_error
    <<",\"all_cell_positive_physical_transfer_rms\":"<<std::sqrt(physical_square/physical_volume)
    <<",\"log_rows\":"<<log_rows<<",\"log_rows_256_donors\":"<<enlarged_rows<<",\"log_value_l1_max\":"<<maximum_l1<<",\"equilibrium_after_regrid_max\":"<<equilibrium
    <<",\"pulse_flux_max\":"<<flux<<",\"pulse_metric_max\":"<<metric<<",\"pulse_original_vector_rms_max\":"<<pulse_coarse.engine->max_shift_residual
    <<",\"unchanged_shells_bitwise\":true,\"constant_translation_preserved\":true,\"native_adaptive_enabled\":false,\"collapse_accuracy_qualified\":false,\"status\":\"passed\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}Kokkos::finalize();return status;}
