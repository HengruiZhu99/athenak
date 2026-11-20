#ifndef BOLTZMANN_MOMENTS_HPP_
#define BOLTZMANN_MOMENTS_HPP_

#include "athena.hpp"

namespace boltzmann {
// The four cell moments are at the vertex's own level, including exchanged
// and prolonged ghosts. At the axis, project the tensor onto its SO(2) limit.
KOKKOS_INLINE_FUNCTION void TransferVertex(const DvceArray5D<Real>& stress,
                                           const DvceArray5D<Real>& target, int m, int j, int i,
                                           bool axis) {
  for (int l = 0; l < 10; ++l) {
    target(m, l, 0, j, i) = 0.25 * (stress(m, l, 0, j - 1, i - 1) + stress(m, l, 0, j, i - 1) +
                                    stress(m, l, 0, j - 1, i) + stress(m, l, 0, j, i));
  }
  if (axis) {
    const Real transverse = 0.5 * (target(m, 0, 0, j, i) + target(m, 5, 0, j, i));
    target(m, 0, 0, j, i) = target(m, 5, 0, j, i) = transverse;
    target(m, 1, 0, j, i) = target(m, 2, 0, j, i) = target(m, 4, 0, j, i) = 0;
    target(m, 7, 0, j, i) = target(m, 9, 0, j, i) = 0;
  }
}
}  // namespace boltzmann
#endif  // BOLTZMANN_MOMENTS_HPP_
