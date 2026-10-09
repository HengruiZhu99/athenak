#ifndef RESEARCH_INNER_CONFORMAL_TRACE_HPP_
#define RESEARCH_INNER_CONFORMAL_TRACE_HPP_
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace z4c { namespace hyperboloidal {
// Regular interior addition on the physical-P branch. Gauge cutoff c=1-W,
// not the geometry cutoff. Caller certifies positive reference lapse and
// positive Omega wherever c>0. Return before division in the outer collar.
template<typename T>KOKKOS_INLINE_FUNCTION T ResearchInnerConformalTrace(
    const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g) {
  if(!g.physical_trace_lapse)return T(0);
  const T c=T(1)-SmoothCutoff(p.radius,T(g.r0),T(g.r1)).value;
  if(c==T(0))return T(0);
  T difference{};
  for(int i=0;i<3;++i)
    difference+=(-(u.beta.value[i]-p.beta[i])
        +p.beta[i]*(u.alpha.value-p.alpha)/p.alpha)*p.domega[i];
  return T(3)*c*(u.alpha.value+T(2)*c)*difference/p.omega;
}
}}
#endif
