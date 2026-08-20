//========================================================================================
//! \file cartoon_lattice_index.hpp
//! \brief Device-portable nearest lattice-index conversion for Cartoon test kernels.
//========================================================================================
#ifndef Z4C_CARTOON_LATTICE_INDEX_HPP_
#define Z4C_CARTOON_LATTICE_INDEX_HPP_

#include <cmath>

#include "athena.hpp"

namespace z4c {

// This has llround's nearest-integer, halfway-away-from-zero semantics without
// lowering to the unsupported llvm.llround intrinsic in Aurora SYCL device code.
KOKKOS_INLINE_FUNCTION
int NearestLatticeIndex(const Real value) {
  return value >= 0.0 ? static_cast<int>(floor(value + 0.5))
                      : -static_cast<int>(floor(-value + 0.5));
}

} // namespace z4c

#endif // Z4C_CARTOON_LATTICE_INDEX_HPP_
