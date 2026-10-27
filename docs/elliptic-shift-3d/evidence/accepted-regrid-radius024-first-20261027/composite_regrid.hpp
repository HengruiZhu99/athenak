// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_COMPOSITE_REGRID_HPP_
#define Z4C_HYPERBOLOIDAL_COMPOSITE_REGRID_HPP_
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "z4c/hyperboloidal/native_core_map.hpp"
#include "z4c/hyperboloidal/enriched_log_value.hpp"

namespace z4c {namespace hyperboloidal {
// Host accepted-state transaction. Native pack replacement must commit this
// result and rebuild its field/map/stage adapters before exposing a new pack.
// This helper alone does NOT enable the native adaptive task path. Shell maps,
// logical shape, chart and puncture position must be unchanged. RK substages
// cannot be transferred through this accepted-state interface.
struct CompositeRegridResult {
  std::unique_ptr<CompositeTensorEvolution> engine;
  size_t copied_cells=0,interpolated_cells=0,positive_log_rows=0;
  std::map<int,size_t> positive_support_counts;
  std::map<std::string,size_t> positive_rejections;
  Real maximum_log_value_l1=0;
};
namespace composite_regrid_detail {
inline bool Same(const cubed_sphere::CompositeBlock&a,const cubed_sphere::CompositeBlock&b){
  if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)return false;
  for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])return false;
  return true;
}
inline void Validate(const cubed_sphere::CompositeMesh&m){
  std::vector<NativeCartesianBlock> records;
  for(const auto&b:m.blocks)if(b.face<0)records.push_back({int(records.size()),b.refinement,
    {b.lo[0],b.lo[1],b.lo[2]},{b.hi[0],b.hi[1],b.hi[2]}});
  NativeCartesianCoreMap partition(m,records,m.cells,m.ghosts);
  for(size_t a=0;a<m.blocks.size();++a)if(m.blocks[a].face<0)
    for(size_t b=0;b<a;++b)if(m.blocks[b].face<0){bool touch=true;
      for(int d=0;d<3;++d)touch=touch&&std::min(m.blocks[a].hi[d],m.blocks[b].hi[d])>=std::max(m.blocks[a].lo[d],m.blocks[b].lo[d]);
      if(touch&&std::abs(m.blocks[a].refinement-m.blocks[b].refinement)>1)
        throw std::invalid_argument("accepted regrid violates Cartesian2:1 face/edge/corner balance");
    }
}
}
inline CompositeRegridResult RebuildCompositeAtAcceptedState(
    const CompositeTensorEvolution&old,cubed_sphere::CompositeMesh mesh,Real positive_value_radius=.24){
  using namespace composite_regrid_detail;
  const auto&source=old.shift.grid;
  if(!(positive_value_radius>0&&positive_value_radius<=source.mesh.half_width)||!std::isfinite(positive_value_radius))
    throw std::invalid_argument("accepted positive-log value support radius");
  if(mesh.cells!=source.n||mesh.ghosts!=source.ng||mesh.half_width!=source.mesh.half_width||
      mesh.face_subdivisions!=source.mesh.face_subdivisions||mesh.radii!=source.mesh.radii||
      mesh.revision<=source.mesh.revision||mesh.blocks.size()*size_t(mesh.cells)*mesh.cells*mesh.cells>50000)
    throw std::invalid_argument("accepted regrid descriptor/revision/budget mismatch");
  if(old.state.size()!=source.active.size()*old.fields||old.beta_delta.size()!=source.active.size()*3)
    throw std::invalid_argument("accepted regrid source field shape");
  for(Real v:old.state)if(!std::isfinite(v))throw std::invalid_argument("nonfinite accepted regrid source");
  for(Real v:old.beta_delta)if(!std::isfinite(v))throw std::invalid_argument("nonfinite accepted regrid shift");
  if(old.scri){for(Real v:old.scri->values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite accepted regrid scri field");
    for(Real v:old.scri->beta)if(!std::isfinite(v))throw std::invalid_argument("nonfinite accepted regrid scri shift");}
  if(old.wormhole&&old.positive_storage!=2)
    throw std::invalid_argument("puncture regrid is qualified only for fixed positive logs");
  Validate(source.mesh);Validate(mesh);
  std::vector<int> same(mesh.blocks.size(),-1);size_t shell_count=0,old_shell_count=0;
  for(const auto&b:source.mesh.blocks)old_shell_count+=b.face>=0;
  for(size_t b=0;b<mesh.blocks.size();++b){
    for(size_t a=0;a<source.mesh.blocks.size();++a)if(Same(mesh.blocks[b],source.mesh.blocks[a])){
      if(same[b]>=0)throw std::invalid_argument("ambiguous accepted regrid block identity");same[b]=a;}
    if(mesh.blocks[b].face>=0){++shell_count;if(same[b]<0)throw std::invalid_argument("shell-map changes require a separate transfer transaction");}
  }
  if(shell_count!=old_shell_count)throw std::invalid_argument("accepted regrid dropped a shell");
  CompositeRegridResult result;
  result.engine=std::make_unique<CompositeTensorEvolution>(std::move(mesh),old.telegraph,old.elliptic_atol,
    source.order,bool(old.scri),old.wormhole,old.isotropic_reference,old.scri?old.scri->radial_nodes:6,
    source.radial_interface_stencils,old.positive_storage,old.telegraph_width,bool(old.log_plans||old.supported_log_plans),
    old.shift.preconditioner.UsesRadialLines(),old.shift.preconditioner.UsesTopologyCoarsening(),bool(old.log_halo_plans),
    old.log_halo_radius,old.puncture_derivative_scheme,old.boundary_derivative_scheme);
  auto&target=*result.engine;target.pole_sat=old.pole_sat;target.regularity_sat=old.regularity_sat;
  const size_t per=size_t(source.n)*source.n*source.n;
  std::vector<int> ordinal(source.count,-1);for(size_t p=0;p<source.active.size();++p)ordinal[source.active[p]]=p;
  PuncturePosition position;if(old.wormhole&&(old.log_plans||old.supported_log_plans)){
    position=old.PositivePlanPosition();
    if(position.position!=std::array<Real,3>{0,0,0}||position.velocity!=std::array<Real,3>{0,0,0})
      throw std::invalid_argument("moving puncture chart regrid requires transport-aware stage integration");
  }
  for(size_t p=0;p<target.shift.grid.active.size();++p){const int block=p/per;
    if(same[block]>=0){const size_t donor=size_t(same[block])*per+p%per;
      std::copy_n(old.state.begin()+donor*old.fields,old.fields,target.state.begin()+p*old.fields);
      std::copy_n(old.beta_delta.begin()+3*donor,3,target.beta_delta.begin()+3*p);++result.copied_cells;continue;}
    ++result.interpolated_cells;const auto&x=target.shift.grid.geometry[p].x;
    const auto entries=source.Interpolate(x,std::min(6,source.n));
    // Anchor differences preserve constants exactly, including every signed
    // tensor/connection/flux component and a constant translational shift.
    const int anchor=ordinal.at(entries.front().point);if(anchor<0)throw std::logic_error("nonactive regrid donor");
    for(int f=0;f<old.fields+3;++f){const bool shift=f>=old.fields;const int v=shift?f-old.fields:f;
      const Real initial=shift?old.beta_delta[3*anchor+v]:old.state[old.fields*anchor+v];Real value=initial;
      for(const auto&e:entries){const int a=ordinal.at(e.point);if(a<0)throw std::logic_error("nonactive regrid donor");
        value+=e.weight*((shift?old.beta_delta[3*a+v]:old.state[old.fields*a+v])-initial);}
      if(shift)target.beta_delta[3*p+v]=value;else target.state[old.fields*p+v]=value;
    }
    if(old.wormhole&&position.RadiusSquared(x)<positive_value_radius*positive_value_radius){
      const int owner=source.mesh.Owner(x);const auto&box=source.mesh.blocks[owner];
      if(box.face>=0)throw std::logic_error("puncture transfer query outside Cartesian core");
      const Real h=(box.hi[0]-box.lo[0])/source.n;
      std::vector<std::pair<Real,size_t>> nearest;
      for(size_t a=0;a<source.active.size();++a)if(source.mesh.blocks[a/per].face<0){Real d2=0;
        for(int d=0;d<3;++d)d2+=std::pow(source.geometry[a].x[d]-x[d],2);nearest.push_back({d2,a});}
      const size_t maximum=std::min(size_t(256),nearest.size());
      std::partial_sort(nearest.begin(),nearest.begin()+maximum,nearest.end());nearest.resize(maximum);
      EnrichedLogValue value_plan;std::vector<size_t> donors;bool ready=false;
      for(int count:{125,150,175,256})if(nearest.size()>=size_t(count)){
        std::vector<EnrichedLogValue::Point> points;donors.clear();
        for(int j=0;j<count;++j){donors.push_back(nearest[j].second);const auto&y=source.geometry[nearest[j].second].x;points.push_back({y[0],y[1],y[2]});}
        if(value_plan.Build({x[0],x[1],x[2]},points,h,position,source.mesh.revision,8)){
          ready=true;++result.positive_support_counts[count];break;}
        ++result.positive_rejections[value_plan.diagnostics().status];}
      if(!ready)throw std::runtime_error("positive-log accepted transfer rejected at cell"+std::to_string(p)+
        " rho="+std::to_string(std::sqrt(position.RadiusSquared(x)))+" h="+std::to_string(h)+
        " valueL1="+std::to_string(value_plan.diagnostics().value_weight_l1)+": "+value_plan.diagnostics().status);
      for(int f:{0,18}){std::vector<Real> values;for(size_t a:donors)values.push_back(old.state[old.fields*a+f]);Real value;
        if(!value_plan.Evaluate(values,position,source.mesh.revision,value))throw std::runtime_error("positive-log accepted transfer failed");target.state[old.fields*p+f]=value;}
      ++result.positive_log_rows;result.maximum_log_value_l1=std::max(result.maximum_log_value_l1,value_plan.diagnostics().value_weight_l1);
    }
  }
  if(old.scri){const size_t face_per=size_t(source.n)*source.n;
    if(old.scri->values.size()!=old.scri->rows.size()*old.fields||old.scri->beta.size()!=old.scri->rows.size()*3)
      throw std::invalid_argument("accepted regrid scri payload shape");
    for(size_t p=0;p<target.scri->rows.size();++p){const auto&r=target.scri->rows[p];
      const size_t a=old.scri->base[same[r.block]]+r.j+source.n*r.k;
      if(a>=old.scri->rows.size()||a/face_per>=old.scri->rows.size()/face_per)throw std::logic_error("accepted regrid scri map");
      std::copy_n(old.scri->values.begin()+a*old.fields,old.fields,target.scri->values.begin()+p*old.fields);
      std::copy_n(old.scri->beta.begin()+3*a,3,target.scri->beta.begin()+3*p);}
  }
  for(Real v:target.state)if(!std::isfinite(v))throw std::runtime_error("nonfinite rebuilt state");
  return result;
}
}}
#endif
