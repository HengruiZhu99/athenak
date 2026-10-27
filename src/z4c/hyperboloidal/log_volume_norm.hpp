// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_LOG_VOLUME_NORM_HPP_
#define Z4C_HYPERBOLOIDAL_LOG_VOLUME_NORM_HPP_
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "athena.hpp"
namespace z4c { namespace hyperboloidal {
inline Real LogSumPositive(const std::vector<Real> &logs){
  if(logs.empty())throw std::invalid_argument("empty physical measure");
  const Real largest=*std::max_element(logs.begin(),logs.end());
  if(!std::isfinite(largest))throw std::invalid_argument("nonfinite physical measure");
  Real sum=0;for(Real q:logs){if(!std::isfinite(q))throw std::invalid_argument("nonfinite physical cell measure");sum+=std::exp(q-largest);}
  return largest+std::log(sum);
}
// Each cell remains in log form. Even a cell whose normalized weight would
// underflow contributes when multiplied by a sufficiently large residual.
// The signed accumulation is also used for Krylov cross products.
inline Real LogWeightedDot(const std::vector<Real> &a,const std::vector<Real> &b,
                          const std::vector<Real> &logs,Real total){
  if(a.size()!=b.size()||a.size()!=3*logs.size())throw std::invalid_argument("log norm shape");
  Real largest=-std::numeric_limits<Real>::infinity();
  for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i])||!std::isfinite(b[i]))throw std::runtime_error("nonfinite residual in physical norm");
    if(a[i]!=0&&b[i]!=0)largest=std::max(largest,logs[i/3]-total+std::log(std::abs(a[i]))+std::log(std::abs(b[i])));}
  if(!std::isfinite(largest))return 0;
  Real sum=0,correction=0;
  for(size_t i=0;i<a.size();++i)if(a[i]!=0&&b[i]!=0){
    Real term=std::exp(logs[i/3]-total+std::log(std::abs(a[i]))+std::log(std::abs(b[i]))-largest);
    if(std::signbit(a[i])!=std::signbit(b[i]))term=-term;
    const Real y=term-correction,next=sum+y;correction=(next-sum)-y;sum=next;}
  if(sum==0)return 0;
  const Real magnitude=largest+std::log(std::abs(sum));
  if(magnitude>std::log(std::numeric_limits<Real>::max()))throw std::overflow_error("physical norm result exceeds arithmetic range");
  return std::copysign(std::exp(magnitude),sum);
}
}} // namespace z4c::hyperboloidal
#endif
