//========================================================================================
// Principal-symbol helpers for the 50-field reference-frame FO-GH system.
// Licensed under the 3-clause BSD License, see LICENSE for details.
//========================================================================================
#ifndef REF_GH_REF_GH_CHARACTERISTICS_HPP_
#define REF_GH_REF_GH_CHARACTERISTICS_HPP_

#include "athena.hpp"

namespace ref_gh {

struct CharacteristicSpeeds {
  Real metric;
  Real transverse;
  Real plus;
  Real minus;
};

// s_cov is normalized with G^{IJ}s_I s_J = 1.  beta_ref contains coframe
// components beta_ref^I, hence beta^s = beta_ref^I s_I.
KOKKOS_INLINE_FUNCTION
CharacteristicSpeeds GetCharacteristicSpeeds(const Real alpha,
                                              const Real beta_ref[3],
                                              const Real s_cov[3]) {
  Real beta_s = 0.0;
  for (int i = 0; i < 3; ++i) beta_s += beta_ref[i]*s_cov[i];
  return {0.0, -beta_s, -beta_s + alpha, -beta_s - alpha};
}

}  // namespace ref_gh

#endif  // REF_GH_REF_GH_CHARACTERISTICS_HPP_
