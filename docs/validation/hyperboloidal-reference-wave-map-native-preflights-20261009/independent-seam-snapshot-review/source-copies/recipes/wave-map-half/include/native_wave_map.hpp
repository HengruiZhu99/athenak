// Private actual-native seam. This is not a production gauge option.
#ifndef RESEARCH_NATIVE_REFERENCE_WAVE_MAP_HPP_
#define RESEARCH_NATIVE_REFERENCE_WAVE_MAP_HPP_
#include <type_traits>
#include "reference_wave_map.hpp"
namespace z4c { namespace hyperboloidal {
// CartesianPatch supplies actual stored-cell coordinates. LayerPoint does not
// store an orientation, and no orientation is reconstructed from dOmega.
template<class T> KOKKOS_INLINE_FUNCTION bool ResearchNativeWaveMapGauge(
    const LayerPoint<T>& p, const Z4cJet<T>& u, const T xyz[3],
    GaugeRHS<T>& rhs) {
  static_assert(std::is_same<T,double>::value,
      "private native wave-map control is admitted only for double");
  const auto connection=rwm::ReferenceConnection(p,xyz);
  const auto parts=rwm::Gauge(p,u,connection);
  // rwm::Assemble adds BOTH alpha and beta poles, exactly once. The caller
  // does not call production AssembleGaugeInterior or add a beta pole again.
  return rwm::Assemble(parts,p.omega,rhs);
}
}}
#endif
