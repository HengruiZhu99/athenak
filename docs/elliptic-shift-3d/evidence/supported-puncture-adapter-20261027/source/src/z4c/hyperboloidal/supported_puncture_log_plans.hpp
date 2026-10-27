// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SUPPORTED_PUNCTURE_LOG_PLANS_HPP_
#define Z4C_HYPERBOLOIDAL_SUPPORTED_PUNCTURE_LOG_PLANS_HPP_
#include <map>
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include "z4c/hyperboloidal/supported_rho_log_derivative.hpp"
namespace z4c { namespace hyperboloidal {
// Standalone HOST adapter for an explicitly experimental static radial-first
// option. It does not modify/select the production/default wrapper. Unsupported
// translated footprints throw; the original scalar value is always retained.
class SupportedPunctureLogPlans {
 public:
  struct Row {size_t active,base_row;std::vector<size_t>donors;SupportedRhoLogDerivative plan;};
  PuncturePosition position;
  size_t enlarged_rows=0;Real maximum_condition=0,maximum_reproduction=0,maximum_gradient_l1=0,maximum_hessian_l1=0;
  std::map<size_t,size_t>support_counts;std::map<std::string,size_t>rejected_attempts;
  explicit SupportedPunctureLogPlans(const cubed_sphere::CompositeDifference&grid,PuncturePosition point={},Real radius=.1)
      :position(point),base_(grid,point,radius){
    std::vector<size_t>cartesian;const size_t per=size_t(grid.n)*grid.n*grid.n;for(size_t p=0;p<grid.active.size();++p)if(grid.mesh.blocks[p/per].face<0)cartesian.push_back(p);
    const size_t available=std::min(size_t(256),cartesian.size());if(available<125)throw std::runtime_error("supported positive-log adapter needs125Cartesian donors");
    rows_.reserve(base_.rows.size());for(size_t index=0;index<base_.rows.size();++index){const auto &base_row=base_.rows[index];const size_t p=base_row.active;const auto &x=grid.geometry[p].x;std::vector<std::pair<Real,size_t>>near;near.reserve(cartesian.size());
      for(size_t donor:cartesian){Real d=0;for(int a=0;a<3;++a)d+=std::pow(grid.geometry[donor].x[a]-x[a],2);near.emplace_back(d,donor);}std::partial_sort(near.begin(),near.begin()+available,near.end());
      for(int i=0;i<125;++i)if(near[i].second!=base_row.donors[i])throw std::runtime_error("supported adapter changed reference125prefix");
      std::vector<SupportedRhoLogDerivative::Point>points;for(size_t i=0;i<available;++i){const auto &y=grid.geometry[near[i].second].x;points.push_back({y[0],y[1],y[2]});}Row row;row.active=p;row.base_row=index;const Real spacing=1/(grid.n*std::abs(grid.geometry[p].J[0][0]));
      if(!row.plan.Build({x[0],x[1],x[2]},points,spacing,position))throw std::runtime_error("supported positive-log query"+std::to_string(p)+" unsupported: "+row.plan.diagnostics().status);
      for(const auto &attempt:row.plan.attempts())if(attempt.diagnostic.status!="ready")++rejected_attempts[attempt.diagnostic.status];
      const size_t count=row.plan.selected_count();for(size_t i=0;i<count;++i)row.donors.push_back(near[i].second);++support_counts[count];enlarged_rows+=count>125;const auto &d=row.plan.diagnostics();maximum_condition=std::max(maximum_condition,d.condition);maximum_reproduction=std::max(maximum_reproduction,d.reproduction);maximum_gradient_l1=std::max(maximum_gradient_l1,d.scaled_gradient_weight_l1);maximum_hessian_l1=std::max(maximum_hessian_l1,d.scaled_hessian_weight_l1);rows_.push_back(std::move(row));
    }
  }
  const std::vector<Row>&Rows()const{return rows_;}
  ScalarJet<Real>Sample(const Row&row,const std::vector<Real>&state,int components,int field)const{
    // Reuse the unchanged wrapper's complete map/index/field guards. Its
    // derivative result is discarded; it is never a numerical fallback.
    auto out=base_.Sample(base_.rows.at(row.base_row),state,components,field);
    if(base_.rows[row.base_row].active!=row.active)throw std::runtime_error("supported row identity mismatch");std::vector<Real>values;values.reserve(row.donors.size());for(size_t p:row.donors)values.push_back(state.at(p*components+field));SupportedRhoLogDerivative::Result fit;
    if(!row.plan.Evaluate(values,position,fit))throw std::runtime_error("stale/nonfinite supported positive-log data");out.value=state.at(row.active*components+field);for(int a=0;a<3;++a){out.d[a]=fit.gradient[a];for(int b=0;b<3;++b)out.dd[a][b]=fit.hessian[a][b];}return out;
  }
  SupportedPunctureLogPlans(cubed_sphere::CompositeDifference&&,PuncturePosition={},Real=.1)=delete;
 private:
  PunctureLogPlans base_;std::vector<Row>rows_;
};
}}  // namespace z4c::hyperboloidal
#endif
