#ifndef SCRATCH_NATIVE_FEEDBACK_INJECTION_HPP_
#define SCRATCH_NATIVE_FEEDBACK_INJECTION_HPP_
#include <Kokkos_Core.hpp>
#include <type_traits>
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "null_feedback.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> ResearchLayerGauge(
    const LayerPoint<T>& p, const Z4cJet<T>& u, const LayerGaugeParameters& g) {
  if constexpr (std::is_same<T, double>::value) {
    if (g.physical_trace_lapse && !g.preferred_source)
      return research::Gauge(p, u, g);
  }
  return InteriorLayerGauge(p, u, g);
}
template <typename T>
KOKKOS_INLINE_FUNCTION bool ResearchAssembleGaugeInterior(
    const GaugeRHSParts<T>& parts, T omega, GaugeRHS<T>& rhs) {
  if (!AssembleGaugeInterior(parts, omega, rhs)) return false;
  for (int i=0; i<3; ++i) {
    rhs.beta[i] += parts.pole.beta[i]/omega;
    if (!Kokkos::isfinite(rhs.beta[i])) return false;
  }
  return true;
}
}}
// Only call sites parsed after the production headers are redirected.
#define InteriorLayerGauge ResearchLayerGauge
#define AssembleGaugeInterior ResearchAssembleGaugeInterior
#endif
