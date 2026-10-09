#ifndef SCRATCH_SHIFT_POLE_INJECTION_HPP_
#define SCRATCH_SHIFT_POLE_INJECTION_HPP_
#include <Kokkos_Core.hpp>
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace z4c { namespace hyperboloidal {
constexpr double research_shift_pole_rate=10.;
template <typename T>
KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> ResearchOuterShiftGauge(
    const LayerPoint<T>&p,const Z4cJet<T>&u,const LayerGaugeParameters&g) {
  auto out=InteriorLayerGauge(p,u,g);
  if(out.valid&&g.physical_trace_lapse&&!g.preferred_source) {
    const T W=LayerCoefficients(p.radius,u.alpha.value,g).weight;
    for(int i=0;i<3;++i)
      out.pole.beta[i]-=T(research_shift_pole_rate)*W*(u.beta.value[i]-p.beta[i]);
  }
  return out;
}
template <typename T>
KOKKOS_INLINE_FUNCTION bool ResearchAssembleOuterShift(
    const GaugeRHSParts<T>&p,T omega,GaugeRHS<T>&rhs) {
  if(!AssembleGaugeInterior(p,omega,rhs))return false;
  for(int i=0;i<3;++i){rhs.beta[i]+=p.pole.beta[i]/omega;if(!Kokkos::isfinite(rhs.beta[i]))return false;}
  return true;
}
}}
#define InteriorLayerGauge ResearchOuterShiftGauge
#define AssembleGaugeInterior ResearchAssembleOuterShift
#endif
