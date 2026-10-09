#ifndef RESEARCH_INNER_LAPSE_ADVECTION_HPP_
#define RESEARCH_INNER_LAPSE_ADVECTION_HPP_

#include "z4c/hyperboloidal/layer_gauge.hpp"

namespace z4c { namespace hyperboloidal {

// Lower-order regular lapse source, applied only to the physical-P gauge.
// c=1-W_gauge uses the gauge radii, not the geometry/Cauchy layer radii.
// At c=1 this restores log-relative lapse advection; c=0 retains the full
// analytic-reference advection subtraction. The physical-P pole and shift
// equations are untouched. The caller retains its existing validity checks.
template <typename T>
KOKKOS_INLINE_FUNCTION T ResearchInnerLapseAdvection(
    const LayerPoint<T>& p, const Z4cJet<T>& u,
    const LayerGaugeParameters& g) {
  if (!g.physical_trace_lapse) return T(0);
  const T c = 1-LayerCoefficients(p.radius,u.alpha.value,g).weight;
  if (c == 0) return T(0);
  const T relative = (u.alpha.value-p.alpha)/p.alpha;
  T source = 0;
  for (int i=0;i<3;++i) {
    source -= c*((u.beta.value[i]-p.beta[i])+u.beta.value[i]*relative)*p.dalpha[i];
  }
  return source;
}

}}  // namespace z4c::hyperboloidal
#endif
