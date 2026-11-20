#ifndef BOLTZMANN_PROLONGATION_HPP_
#define BOLTZMANN_PROLONGATION_HPP_
#include "athena.hpp"

namespace boltzmann {
// Tensor product of the three-cell quadratic finite-volume subdivision rule.
// The four deviations sum to zero. A common positivity factor preserves this
// identity even when an unlimited child would undershoot in a beam's tail.
KOKKOS_INLINE_FUNCTION void ProlongOrdinates(int m, int n, int j, int i, int fj, int fi,
                                             const DvceArray5D<Real>& parent,
                                             const DvceArray5D<Real>& child) {
  const Real mean = parent(m, n, 0, j, i);
  const Real x = (parent(m, n, 0, j, i + 1) - parent(m, n, 0, j, i - 1)) / 8;
  const Real y = (parent(m, n, 0, j + 1, i) - parent(m, n, 0, j - 1, i)) / 8;
  const Real xy = (parent(m, n, 0, j + 1, i + 1) - parent(m, n, 0, j + 1, i - 1) -
                   parent(m, n, 0, j - 1, i + 1) + parent(m, n, 0, j - 1, i - 1)) /
                  64;
  const Real deviation[4] = {-x - y + xy, x - y - xy, -x + y - xy, x + y + xy};
  Real low = fmin(fmin(deviation[0], deviation[1]), fmin(deviation[2], deviation[3]));
  Real scale = mean + low < 0 ? mean / (-low) : 1;
  for (int t = 0; t < 4; ++t)
    child(m, n, 0, fj + t / 2, fi + t % 2) = fmax(0., mean + scale * deviation[t]);
}
}  // namespace boltzmann
#endif  // BOLTZMANN_PROLONGATION_HPP_
