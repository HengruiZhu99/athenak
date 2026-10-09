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
  if(u.alpha.value/p.alpha>T(.5)) {
    for(int i=0;i<3;++i)
      difference+=(-(u.beta.value[i]-p.beta[i])
          +p.beta[i]*(u.alpha.value-p.alpha)/p.alpha)*p.domega[i];
  } else {
    T ref_contraction{},live_contraction{};
    for(int i=0;i<3;++i) {
      ref_contraction+=p.beta[i]*p.domega[i];
      live_contraction+=u.beta.value[i]*p.domega[i];
    }
    difference=u.alpha.value*ref_contraction/p.alpha-live_contraction;
  }
  return T(3)*c*(u.alpha.value+T(2)*c)*difference/p.omega;
}
// Apply trace-only, or replace the complete regular lapse with the direct
// combined advection blend plus its existing log relaxation and trace source.
// This preserves the original pole and every shift component. In particular,
// the combined form never adds two order-one advection terms at tiny alpha.
template<typename T>KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> ResearchInnerTraceGauge(
    const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g,
    GaugeRHSParts<T> parts,bool combined) {
  if(!g.physical_trace_lapse)return parts;
  const T W=SmoothCutoff(p.radius,T(g.r0),T(g.r1)).value,c=T(1)-W;
  if(c==T(0))return parts;
  if(combined) {
    const T alpha=u.alpha.value,relative=(alpha-p.alpha)/p.alpha;
    const T loga=Kokkos::abs(relative)<T(.5)?Kokkos::log1p(relative)
        :Kokkos::log(alpha)-Kokkos::log(p.alpha);
    const T nu=T(g.lapse_inner)+T(g.lapse_outer-g.lapse_inner)*W;
    parts.regular.alpha=-alpha*nu*loga;
    for(int i=0;i<3;++i)
      parts.regular.alpha+=c*u.beta.value[i]*(u.alpha.d[i]-alpha/p.alpha*p.dalpha[i])
          +W*(u.beta.value[i]*u.alpha.d[i]-p.beta[i]*p.dalpha[i]);
  }
  parts.regular.alpha+=ResearchInnerConformalTrace(p,u,g);
  return parts;
}
}}
#endif
