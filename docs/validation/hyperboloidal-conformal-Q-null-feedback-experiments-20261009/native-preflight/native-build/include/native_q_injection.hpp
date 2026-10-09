#ifndef PRIVATE_Q_NULL_NATIVE_INJECTION_HPP_
#define PRIVATE_Q_NULL_NATIVE_INJECTION_HPP_
#include <Kokkos_Core.hpp>
#include <type_traits>
#include "q_null_feedback.hpp"
namespace z4c { namespace hyperboloidal {
template<typename T> KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> ResearchNativeQNullGauge(
    const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g) {
  if constexpr(std::is_same<T,double>::value) {
    if(!g.physical_trace_lapse && g.preferred_source)
      return qnf::Gauge(p,u,g,qnf::Parameters{.85,.95,5,true});
  }
  return InteriorLayerGauge(p,u,g);
}
template<typename T> KOKKOS_INLINE_FUNCTION bool ResearchNativeQNullAssemble(
    const GaugeRHSParts<T>&p,T omega,GaugeRHS<T>&r) {
  if(!AssembleGaugeInterior(p,omega,r)) return false;
  for(int i=0;i<3;++i) {r.beta[i]+=p.pole.beta[i]/omega;
    if(!Kokkos::isfinite(r.beta[i])) return false;}
  return true;
}
}}
#define InteriorLayerGauge ResearchNativeQNullGauge
#define AssembleGaugeInterior ResearchNativeQNullAssemble
#endif
