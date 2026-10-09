// SCRATCH ONLY. Prescribed bulk tensor blend; no fully covariant-system claim.
// All mechanical C1 terms AND the covector repair share the same coefficient.
#ifndef SCRATCH_BULK_C1_ADDITIONS_HPP_
#define SCRATCH_BULK_C1_ADDITIONS_HPP_
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "c1_additions.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
C1Parts<T> BulkC1Additions(const Z4cJet<T>&u,const OmegaJet<T>&o,T radius,
                          const LayerGaugeParameters&g=LayerGaugeParameters{}) {
  const T coefficient=1-LayerCoefficients(radius,u.alpha.value,g).weight;
  return TensorC1Additions(u,o,coefficient,true);
}
}}
#endif
