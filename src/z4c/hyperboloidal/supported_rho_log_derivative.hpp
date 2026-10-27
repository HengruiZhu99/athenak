// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SUPPORTED_RHO_LOG_DERIVATIVE_HPP_
#define Z4C_HYPERBOLOIDAL_SUPPORTED_RHO_LOG_DERIVATIVE_HPP_
#include "z4c/hyperboloidal/rho_log_derivative.hpp"
namespace z4c { namespace hyperboloidal {
// Separate HOST prototype. Keep ALL58 basis columns and the SAME16/32/64
// functional caps. Only nearest-active geometric support can grow. No field-
// dependent decision, lower-degree fallback, exponent state or adoption.
class SupportedRhoLogDerivative {
 public:
  using Point=RhoLogDerivative::Point;using Result=RhoLogDerivative::Result;
  struct Attempt {int donors=0;RhoLogDerivative::Diagnostics diagnostic;};
  const std::vector<Attempt>&attempts()const{return attempts_;}
  const RhoLogDerivative::Diagnostics&diagnostics()const{return diagnostic_;}
  size_t selected_count()const{return selected_count_;}
  bool Build(const Point &query,const std::vector<Point>&ordered_nearest,Real spacing,const PuncturePosition&p){
    selected_count_=0;attempts_.clear();diagnostic_=RhoLogDerivative::Diagnostics{};
    if(ordered_nearest.size()<125||ordered_nearest.size()>256){diagnostic_.status="ordered_catalog_count";return false;}
    // Verify the promised distance ordering rather than silently reorder a
    // caller's associated donor indices. Ties are the caller's stable order.
    Real previous=-1;for(const auto &x:ordered_nearest){Real d=0;for(int a=0;a<3;++a)d+=std::pow(x[a]-query[a],2);if(!std::isfinite(d)||d<previous-1e-12*spacing*spacing){diagnostic_.status="catalog_not_nearest_ordered";return false;}previous=d;}
    for(int count:{125,150,175,256})if(size_t(count)<=ordered_nearest.size()){
      std::vector<Point> points(ordered_nearest.begin(),ordered_nearest.begin()+count);
      const bool ready=plan_.Build(query,points,spacing,p);attempts_.push_back({count,plan_.diagnostics()});diagnostic_=plan_.diagnostics();
      if(ready){selected_count_=count;return true;}
    }
    diagnostic_.status="support_exhausted:"+diagnostic_.status;return false;
  }
  bool Evaluate(const std::vector<Real>&values,const PuncturePosition&p,Result&r)const{return selected_count_&&values.size()==selected_count_&&plan_.Evaluate(values,p,r);}
 private:
  RhoLogDerivative plan_;RhoLogDerivative::Diagnostics diagnostic_;std::vector<Attempt>attempts_;size_t selected_count_=0;
};
}}  // namespace z4c::hyperboloidal
#endif
