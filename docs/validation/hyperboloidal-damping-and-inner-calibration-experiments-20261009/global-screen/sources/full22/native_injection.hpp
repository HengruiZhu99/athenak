#ifndef SCRATCH_NATIVE_SPATIAL_NORM_INJECTION_HPP_
#define SCRATCH_NATIVE_SPATIAL_NORM_INJECTION_HPP_
#include <Kokkos_Core.hpp>
#include <type_traits>
#include "spatial_norm_control.hpp"
namespace z4c {namespace hyperboloidal {
// This private binary is restricted by its saved input to S=1,a=.5,rho=1.5.
// It adds no production runtime option. All mathematical sources are captured
// separately from the source-at-launch checkout's documentation commit.
constexpr double research_norm_a=.5,research_norm_s=1.,research_norm_rho=1.5;
constexpr double research_norm_xi=1/research_norm_a;
constexpr double research_norm_eta=research_norm_rho*research_norm_s/(research_norm_a*research_norm_a);
constexpr double research_norm_C=(research_norm_s/research_norm_a)*(1-1/research_norm_rho);
template<typename T>
KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> ResearchNativeSpatialGauge(
    const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g) {
  if constexpr(std::is_same<T,double>::value) {
    if(g.physical_trace_lapse&&!g.preferred_source)
      return spatial_norm::Gauge(p,u,g,{research_norm_xi,research_norm_eta,research_norm_C});
  }
  return InteriorLayerGauge(p,u,g);
}
template<typename T>
KOKKOS_INLINE_FUNCTION bool ResearchNativeSpatialAssemble(
    const GaugeRHSParts<T>&p,T omega,GaugeRHS<T>&r) {
  if(!AssembleGaugeInterior(p,omega,r))return false;
  for(int i=0;i<3;++i){r.beta[i]+=p.pole.beta[i]/omega;if(!Kokkos::isfinite(r.beta[i]))return false;}
  return true;
}
}}
#define InteriorLayerGauge ResearchNativeSpatialGauge
#define AssembleGaugeInterior ResearchNativeSpatialAssemble
#endif
