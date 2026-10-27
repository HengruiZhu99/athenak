// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_PUNCTURE_LOG_PLANS_HPP_
#define Z4C_HYPERBOLOIDAL_PUNCTURE_LOG_PLANS_HPP_
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "z4c/hyperboloidal/enriched_log_derivative.hpp"
namespace z4c { namespace hyperboloidal {
// Active Cartesian donors only. Shell and outer derivatives retain the mapped
// tensor stencil. This is a local spatial reconstruction, not a moving unknown
// chart: no exponent or fitted value is evolved and no time source is added.
class PunctureLogPlans {
 public:
  struct Row {size_t active;std::vector<size_t> donors;EnrichedLogDerivative plan;};
  std::vector<Row> rows;
  PuncturePosition position;
  Real maximum_condition=0;
  explicit PunctureLogPlans(const cubed_sphere::CompositeDifference &grid,
      PuncturePosition point={},Real radius=.1):position(point),grid_(&grid),descriptor_(grid.mesh),active_count_(grid.active.size()){
    if(!(radius>0)||!std::isfinite(radius))throw std::invalid_argument("puncture enrichment radius must be finite and positive");
    std::vector<size_t> cartesian;
    const int block_size=grid.padded*grid.padded*grid.padded;
    for(size_t p=0;p<grid.active.size();++p)
      if(grid.mesh.blocks[grid.active[p]/block_size].face<0)cartesian.push_back(p);
    for(size_t p:cartesian){
      const auto &x=grid.geometry[p].x;
      const Real rho=std::sqrt(position.RadiusSquared(x));
      const Real spacing=1/(grid.n*std::abs(grid.geometry[p].J[0][0]));
      if(rho>=radius||rho/spacing>8)continue;
      if(cartesian.size()<125)throw std::runtime_error("puncture enrichment needs 125 active Cartesian donors");
      std::vector<std::pair<Real,size_t>> nearest;nearest.reserve(cartesian.size());
      for(size_t donor:cartesian){Real distance=0;for(int a=0;a<3;++a)
        distance+=std::pow(grid.geometry[donor].x[a]-x[a],2);nearest.emplace_back(distance,donor);}
      std::partial_sort(nearest.begin(),nearest.begin()+125,nearest.end());
      Row row;row.active=p;std::vector<EnrichedLogDerivative::Point> points;
      for(int i=0;i<125;++i){row.donors.push_back(nearest[i].second);const auto &d=grid.geometry[nearest[i].second].x;
        points.push_back({d[0],d[1],d[2]});}
      if(!row.plan.Build({x[0],x[1],x[2]},points,spacing,position))
        throw std::runtime_error("puncture enrichment rejected plan: "+row.plan.diagnostics().status);
      maximum_condition=std::max(maximum_condition,row.plan.diagnostics().condition);rows.push_back(std::move(row));
    }
  }
  ScalarJet<Real> Sample(const Row &row,const std::vector<Real> &state,int components,int field)const{
    if(components<1||field<0||field>=components||state.size()!=active_count_*components)
      throw std::invalid_argument("puncture enrichment state shape/field");
    if(grid_->mesh.revision!=descriptor_.revision||grid_->mesh.radii!=descriptor_.radii||grid_->mesh.half_width!=descriptor_.half_width
        ||grid_->n!=descriptor_.cells||grid_->mesh.cells!=descriptor_.cells
        ||grid_->active.size()!=active_count_||grid_->mesh.blocks.size()!=descriptor_.blocks.size())
      throw std::runtime_error("puncture enrichment mesh changed; rebuild plans");
    for(size_t m=0;m<descriptor_.blocks.size();++m){const auto &a=descriptor_.blocks[m],&b=grid_->mesh.blocks[m];
      if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)
        throw std::runtime_error("puncture enrichment topology changed; rebuild plans");
      for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])
        throw std::runtime_error("puncture enrichment map changed; rebuild plans");}
    std::vector<Real> q;for(size_t p:row.donors)q.push_back(state.at(p*components+field));
    EnrichedLogDerivative::Result fit;
    if(!row.plan.Evaluate(q,position,fit))throw std::runtime_error("stale/invalid puncture enrichment data");
    ScalarJet<Real> out{};out.value=state.at(row.active*components+field);
    for(int a=0;a<3;++a){out.d[a]=fit.gradient[a];for(int b=0;b<3;++b)out.dd[a][b]=fit.hessian[a][b];}
    return out;
  }
  PunctureLogPlans(cubed_sphere::CompositeDifference&&,PuncturePosition={},Real=.1)=delete;
 private:
  const cubed_sphere::CompositeDifference *grid_;
  cubed_sphere::CompositeMesh descriptor_;
  size_t active_count_;
};
}} // namespace z4c::hyperboloidal
#endif
