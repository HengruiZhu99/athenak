//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file sommerfeld_derivatives.hpp
//! \brief Local boundary derivatives for Z4c RHS closures.

#ifndef Z4C_SOMMERFELD_DERIVATIVES_HPP_
#define Z4C_SOMMERFELD_DERIVATIVES_HPP_

#include "athena.hpp"

namespace z4c {

template <typename Sampler>
KOKKOS_INLINE_FUNCTION
Real BoundaryCenteredFirst(const Real inverse_spacing,
                           const Sampler &value) {
  return 0.5 * inverse_spacing * (value(1) - value(-1));
}

template <int NGHOST, typename Sampler>
KOKKOS_INLINE_FUNCTION
Real SommerfeldOneSidedFirst(const int side, const Real inverse_spacing,
                             const Sampler &value) {
  static_assert(NGHOST >= 2 && NGHOST <= 4,
                "Z4c Sommerfeld derivatives support NGHOST=2, 3, or 4");
  Real derivative = 0.0;
  if constexpr (NGHOST == 2) {
    derivative = (3.0 * value(0) - 4.0 * value(1) + value(2)) / 2.0;
  } else if constexpr (NGHOST == 3) {
    derivative = (25.0 * value(0) - 48.0 * value(1) + 36.0 * value(2) -
                  16.0 * value(3) + 3.0 * value(4)) / 12.0;
  } else if constexpr (NGHOST == 4) {
    derivative = (147.0 * value(0) - 360.0 * value(1) +
                  450.0 * value(2) - 400.0 * value(3) +
                  225.0 * value(4) - 72.0 * value(5) +
                  10.0 * value(6)) / 60.0;
  }
  return side * inverse_spacing * derivative;
}

}  // namespace z4c

#endif  // Z4C_SOMMERFELD_DERIVATIVES_HPP_
