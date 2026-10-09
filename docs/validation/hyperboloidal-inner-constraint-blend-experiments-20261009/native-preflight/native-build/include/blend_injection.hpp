// SCRATCH ONLY. Bulk-C1 additions after existing C0 reference subtraction.
#ifndef SCRATCH_NATIVE_BLEND_INJECTION_HPP_
#define SCRATCH_NATIVE_BLEND_INJECTION_HPP_
#include "bulk_c1_additions.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
KOKKOS_INLINE_FUNCTION bool ResearchAddBlendedC1(
    const Z4cJet<T>&u,const OmegaJet<T>&o,T radius,
    const LayerGaugeParameters&g,Z4cRHS<T>&rhs) {
  Z4cRHS<T> delta;
  if(!AssembleC1Interior(BulkC1Additions(u,o,radius,g),o.omega,delta))return false;
  rhs.chi+=delta.chi;rhs.trace+=delta.trace;rhs.theta+=delta.theta;
  for(int i=0;i<3;++i){rhs.lambda[i]+=delta.lambda[i];
    for(int j=0;j<3;++j){rhs.metric[i][j]+=delta.metric[i][j];rhs.a[i][j]+=delta.a[i][j];}}
  return true;
}
// Diagnostic numerator for all terms divided by Omega; double poles contribute /Omega.
template <typename T>
KOKKOS_INLINE_FUNCTION void ResearchAddBlendedPole(
    const Z4cJet<T>&u,const OmegaJet<T>&o,T radius,
    const LayerGaugeParameters&g,Z4cRHS<T>&pole) {
  const auto q=BulkC1Additions(u,o,radius,g);
  pole.chi+=q.pole.chi+q.double_pole.chi/o.omega;
  pole.trace+=q.pole.trace+q.double_pole.trace/o.omega;
  pole.theta+=q.pole.theta+q.double_pole.theta/o.omega;
  for(int i=0;i<3;++i){pole.lambda[i]+=q.pole.lambda[i]+q.double_pole.lambda[i]/o.omega;
    for(int j=0;j<3;++j){pole.metric[i][j]+=q.pole.metric[i][j]+q.double_pole.metric[i][j]/o.omega;
      pole.a[i][j]+=q.pole.a[i][j]+q.double_pole.a[i][j]/o.omega;}}
}
}}
#endif
