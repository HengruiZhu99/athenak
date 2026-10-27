// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_ENRICHED_LOG_VALUE_HPP_
#define Z4C_HYPERBOLOIDAL_ENRICHED_LOG_VALUE_HPP_
#include "z4c/hyperboloidal/enriched_log_derivative.hpp"

namespace z4c { namespace hyperboloidal {
// Host prototype: cache ONLY scalar VALUE weights, not derivative weights or a
// fitted/evolved exponent. Intended for positive-field logarithmic ghost values.
// The original active values and mapped first/second derivative stencils remain.
class EnrichedLogValue {
 public:
  using Point=EnrichedLogDerivative::Point;
  struct Diagnostics {
    std::string status="unbuilt";int rank=0;
    Real basis_condition=0,value_weight_l1=0,value_reproduction=0;
    bool coincident_active_identity=false;
  };
  const Diagnostics &diagnostics()const{return diagnostic_;}
  const std::vector<Real> &weights()const{return weights_;}
  bool Build(const Point &query,const std::vector<Point> &points,Real spacing,
             const PuncturePosition &puncture,unsigned long mesh_revision,
             Real maximum_value_l1=8,Real gaussian_width=2){
    diagnostic_=Diagnostics{};weights_.clear();stamp_=puncture.Capture();mesh_revision_=mesh_revision;
    auto fail=[&](const char *reason){diagnostic_.status=reason;weights_.clear();return false;};
    if(!(maximum_value_l1>=1)||!std::isfinite(maximum_value_l1)||!(spacing>0)||!std::isfinite(spacing)
       ||!(gaussian_width>0)||!std::isfinite(gaussian_width))return fail("invalid_geometry");
    if(!(puncture.RadiusSquared(query.data())>0))return fail("puncture_collocation");
    if(points.size()<57||points.size()>256)return fail("sample_count");
    size_t coincident=points.size();
    for(size_t i=0;i<points.size();++i){if(!(puncture.RadiusSquared(points[i].data())>0))return fail("puncture_collocation");
      Real d2=0;for(int a=0;a<3;++a)d2+=std::pow(points[i][a]-query[a],2);
      if(d2<1e-28*spacing*spacing)coincident=i;}
    if(coincident<points.size()){
      weights_.assign(points.size(),0);weights_[coincident]=1;diagnostic_.status="ready_identity";
      diagnostic_.coincident_active_identity=true;diagnostic_.value_weight_l1=1;return true;
    }
    // Reuse the audited rank-revealing basis construction, but release all its
    // derivative/coefficient rows after recovering functional0 by impulses.
    EnrichedLogDerivative basis;
    EnrichedLogDerivative::Options options;options.gaussian_width=gaussian_width;
    if(!basis.Build(query,points,spacing,puncture,options)){diagnostic_.rank=basis.diagnostics().rank;diagnostic_.basis_condition=basis.diagnostics().condition;
      diagnostic_.status="basis_"+basis.diagnostics().status;return false;}
    diagnostic_.rank=basis.diagnostics().rank;diagnostic_.basis_condition=basis.diagnostics().condition;
    weights_.resize(points.size());
    for(size_t i=0;i<points.size();++i){std::vector<Real> unit(points.size(),0);unit[i]=1;EnrichedLogDerivative::Result r;
      if(!basis.Evaluate(unit,puncture,r))return fail("basis_evaluation");weights_[i]=r.fitted_value;}
    Real sum=0;for(Real w:weights_)sum+=w;weights_[0]+=1-sum;
    for(Real w:weights_)diagnostic_.value_weight_l1+=std::abs(w);
    if(!std::isfinite(diagnostic_.value_weight_l1)||diagnostic_.value_weight_l1>maximum_value_l1)return fail("value_weight_amplification");
    auto reproduce=[&](const std::vector<Real> &values,Real target){Real result;
      if(!Evaluate(values,puncture,mesh_revision,result))return false;
      diagnostic_.value_reproduction=std::max(diagnostic_.value_reproduction,std::abs(result-target));return true;};
    for(int degree=0;degree<=5;++degree)for(int a=0;a<=degree;++a)for(int b=0;b<=degree-a;++b){const int c=degree-a-b;std::vector<Real> values;
      for(const auto &x:points)values.push_back(std::pow((x[0]-query[0])/spacing,a)*std::pow((x[1]-query[1])/spacing,b)*std::pow((x[2]-query[2])/spacing,c));
      if(!reproduce(values,degree==0?1:0))return fail("value_reproduction_nonfinite");}
    std::vector<Real> logarithm;const Real r2=puncture.RadiusSquared(query.data());
    for(const auto &x:points)logarithm.push_back(.5*(std::log(puncture.RadiusSquared(x.data()))-std::log(r2)));
    if(!reproduce(logarithm,0)||!std::isfinite(diagnostic_.value_reproduction)||diagnostic_.value_reproduction>1e-7)return fail("value_reproduction_failure");
    diagnostic_.status="ready";return true;
  }
  bool Evaluate(const std::vector<Real> &log_values,const PuncturePosition &puncture,
                unsigned long mesh_revision,Real &result)const{
    if(weights_.empty()||log_values.size()!=weights_.size()||!puncture.Matches(stamp_)||mesh_revision!=mesh_revision_)return false;
    for(Real x:log_values)if(!std::isfinite(x))return false;
    const Real anchor=log_values[0];Real value=anchor;
    for(size_t i=0;i<weights_.size();++i)value+=weights_[i]*(log_values[i]-anchor);
    if(!std::isfinite(value))return false;result=value;return true;
  }
 private:
  Diagnostics diagnostic_;std::vector<Real> weights_;
  PuncturePosition::Stamp stamp_{};unsigned long mesh_revision_=0;
};
}}  // namespace z4c::hyperboloidal
#endif
