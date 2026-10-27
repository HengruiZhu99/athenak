// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SUPPORTED_PUNCTURE_LOG_PLANS_HPP_
#define Z4C_HYPERBOLOIDAL_SUPPORTED_PUNCTURE_LOG_PLANS_HPP_
#include <map>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "z4c/hyperboloidal/supported_rho_log_derivative.hpp"
namespace z4c { namespace hyperboloidal {
// Separate HOST adapter for the experimental static radial-first option. It
// never selects/modifies the default wrapper or falls back to its derivatives.
// Geometry/Jacobians are immutable derived arrays under the full descriptor
// contract. A descriptor, logical-storage or active-index edit requires rebuild.
class SupportedPunctureLogPlans {
 public:
  struct Row {size_t active;std::vector<size_t>donors;SupportedRhoLogDerivative plan;};
  PuncturePosition position;
  size_t enlarged_rows=0,donor_entries=0;
  Real maximum_condition=0,maximum_reproduction=0,maximum_gradient_l1=0,maximum_hessian_l1=0;
  std::map<size_t,size_t>support_counts;std::map<std::string,size_t>rejected_attempts;
  explicit SupportedPunctureLogPlans(const cubed_sphere::CompositeDifference&grid,
      PuncturePosition point={},Real radius=.1)
      :position(point),grid_(&grid),descriptor_(grid.mesh),active_(grid.active),
       n_(grid.n),ng_(grid.ng),padded_(grid.padded),count_(grid.count),order_(grid.order){
    position.Validate();
    if(!(radius>0)||!std::isfinite(radius))
      throw std::invalid_argument("supported positive-log radius must be finite and positive");
    if(grid.geometry.size()!=active_.size())
      throw std::invalid_argument("supported positive-log geometry shape");
    std::vector<size_t>cartesian;
    const size_t block_size=size_t(grid.padded)*grid.padded*grid.padded;
    for(size_t p=0;p<active_.size();++p){
      if(active_[p]<0||size_t(active_[p])>=size_t(grid.count))
        throw std::invalid_argument("supported positive-log active index");
      if(grid.mesh.blocks.at(size_t(active_[p])/block_size).face<0)cartesian.push_back(p);
    }
    const size_t available=std::min(size_t(256),cartesian.size());
    for(size_t p:cartesian){
      const auto &x=grid.geometry[p].x;
      const Real rho=std::sqrt(position.RadiusSquared(x));
      const Real spacing=1/(grid.n*std::abs(grid.geometry[p].J[0][0]));
      // Identical selection and deterministic nearest-prefix ordering to the
      // original PunctureLogPlans; build ONLY the supported58 scalar plan.
      if(rho>=radius||rho/spacing>8)continue;
      if(available<125)throw std::runtime_error("supported positive-log needs125Cartesian donors");
      std::vector<std::pair<Real,size_t>>near;near.reserve(cartesian.size());
      for(size_t donor:cartesian){Real d=0;for(int a=0;a<3;++a)
        d+=std::pow(grid.geometry[donor].x[a]-x[a],2);near.emplace_back(d,donor);}
      std::partial_sort(near.begin(),near.begin()+available,near.end());
      std::vector<SupportedRhoLogDerivative::Point>points;
      for(size_t i=0;i<available;++i){const auto &y=grid.geometry[near[i].second].x;
        points.push_back({y[0],y[1],y[2]});}
      Row row;row.active=p;
      if(!row.plan.Build({x[0],x[1],x[2]},points,spacing,position))
        throw std::runtime_error("supported positive-log query"+std::to_string(p)+
                                 " unsupported: "+row.plan.diagnostics().status);
      for(const auto &attempt:row.plan.attempts())if(attempt.diagnostic.status!="ready")
        ++rejected_attempts[attempt.diagnostic.status];
      const size_t count=row.plan.selected_count();
      row.donors.reserve(count);for(size_t i=0;i<count;++i)row.donors.push_back(near[i].second);
      donor_entries+=count;++support_counts[count];enlarged_rows+=count>125;
      const auto &d=row.plan.diagnostics();maximum_condition=std::max(maximum_condition,d.condition);
      maximum_reproduction=std::max(maximum_reproduction,d.reproduction);
      maximum_gradient_l1=std::max(maximum_gradient_l1,d.scaled_gradient_weight_l1);
      maximum_hessian_l1=std::max(maximum_hessian_l1,d.scaled_hessian_weight_l1);
      rows_.push_back(std::move(row));
    }
  }
  const std::vector<Row>&Rows()const{return rows_;}
  // Exact scalar/donor payload sizes, excluding allocator capacity, row/map
  // bookkeeping and descriptor vectors. No baseline weights or QR retained.
  size_t ScalarWeightPayloadBytes()const{return donor_entries*RhoLogDerivative::functionals*sizeof(Real);}
  size_t DonorPayloadBytes()const{return donor_entries*sizeof(size_t);}
  size_t ActiveIndexPayloadBytes()const{return active_.size()*sizeof(int);}
  void Validate(const std::vector<Real>&state,int components,int field)const{
    if(components<1||field<0||field>=components||
       state.size()/size_t(components)!=active_.size()||state.size()%size_t(components))
      throw std::invalid_argument("supported positive-log state shape/field");
    if(grid_->n!=n_||grid_->ng!=ng_||grid_->padded!=padded_||grid_->count!=count_||grid_->order!=order_||
       grid_->mesh.cells!=descriptor_.cells||grid_->mesh.ghosts!=descriptor_.ghosts||
       grid_->mesh.face_subdivisions!=descriptor_.face_subdivisions||
       grid_->mesh.revision!=descriptor_.revision||grid_->mesh.radii!=descriptor_.radii||
       grid_->mesh.half_width!=descriptor_.half_width||grid_->active!=active_||
       grid_->geometry.size()!=active_.size()||grid_->mesh.blocks.size()!=descriptor_.blocks.size())
      throw std::runtime_error("supported positive-log mesh/storage changed; rebuild plans");
    for(size_t m=0;m<descriptor_.blocks.size();++m){const auto&a=descriptor_.blocks[m],&b=grid_->mesh.blocks[m];
      if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)
        throw std::runtime_error("supported positive-log topology changed; rebuild plans");
      for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])
        throw std::runtime_error("supported positive-log map changed; rebuild plans");}
  }
  ScalarJet<Real>Sample(const Row&row,const std::vector<Real>&state,int components,int field)const{
    Validate(state,components,field);
    if(row.active>=active_.size()||row.donors.size()!=row.plan.selected_count())
      throw std::invalid_argument("supported positive-log row shape");
    const Real value=state.at(row.active*components+field);
    if(!std::isfinite(value))throw std::runtime_error("nonfinite supported positive-log query value");
    std::vector<Real>values;values.reserve(row.donors.size());
    for(size_t p:row.donors){if(p>=active_.size())throw std::out_of_range("supported positive-log donor");
      values.push_back(state[p*components+field]);}
    SupportedRhoLogDerivative::Result fit;
    if(!row.plan.Evaluate(values,position,fit))throw std::runtime_error("stale/nonfinite supported positive-log data");
    ScalarJet<Real>out{};out.value=value;
    for(int a=0;a<3;++a){out.d[a]=fit.gradient[a];for(int b=0;b<3;++b)out.dd[a][b]=fit.hessian[a][b];}
    return out;
  }
  SupportedPunctureLogPlans(cubed_sphere::CompositeDifference&&,PuncturePosition={},Real=.1)=delete;
 private:
  const cubed_sphere::CompositeDifference*grid_;
  cubed_sphere::CompositeMesh descriptor_;std::vector<int>active_;
  int n_,ng_,padded_,count_,order_;std::vector<Row>rows_;
};
}}  // namespace z4c::hyperboloidal
#endif
