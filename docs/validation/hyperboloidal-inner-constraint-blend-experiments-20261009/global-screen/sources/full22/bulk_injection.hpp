// SCRATCH ONLY: prescribed bulk C1 additions, no fully covariant-system claim.
#ifndef SCRATCH_GLOBAL_BULK_C1_INJECTION_HPP_
#define SCRATCH_GLOBAL_BULK_C1_INJECTION_HPP_
#include "bulk_c1_additions.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
KOKKOS_INLINE_FUNCTION bool ResearchAddBulkC1(
    const Z4cJet<T>&u,const OmegaJet<T>&o,T radius,
    const LayerGaugeParameters&g,Z4cRHS<T>&rhs) {
  Z4cRHS<T> delta;
  if(!AssembleC1Interior(BulkC1Additions(u,o,radius,g),o.omega,delta))return false;
  rhs.chi+=delta.chi;rhs.trace+=delta.trace;rhs.theta+=delta.theta;
  for(int i=0;i<3;++i){rhs.lambda[i]+=delta.lambda[i];
    for(int j=0;j<3;++j){rhs.metric[i][j]+=delta.metric[i][j];rhs.a[i][j]+=delta.a[i][j];}}
  return true;
}
}}
#endif
