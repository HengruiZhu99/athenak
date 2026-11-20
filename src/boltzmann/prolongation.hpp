#ifndef BOLTZMANN_PROLONGATION_HPP_
#define BOLTZMANN_PROLONGATION_HPP_
#include "athena.hpp"

namespace boltzmann {
KOKKOS_INLINE_FUNCTION void StoreChildren(int m, int n, int fj, int fi,
                                          Real mean, Real x, Real y, Real xy,
                                          const DvceArray5D<Real>& child) {
  const Real deviation[4] = {-x - y + xy, x - y - xy, -x + y - xy, x + y + xy};
  const Real low =
      fmin(fmin(deviation[0], deviation[1]), fmin(deviation[2], deviation[3]));
  const Real scale = mean + low < 0 ? mean / (-low) : 1;
  for (int t = 0; t < 4; ++t)
    child(m, n, 0, fj + t / 2, fi + t % 2) =
        fmax(0., mean + scale * deviation[t]);
}

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
  StoreChildren(m, n, fj, fi, mean, x, y, xy, child);
}

// Integrate the polynomial matching the coarse cell averages over the right
// half cell and subtract the parent average. The left half has opposite
// deviation. Slide the stencil inside the authoritative parent rectangle.
struct HalfCell {
  int first, count;
  Real weights[5]{};
  KOKKOS_INLINE_FUNCTION HalfCell(int center, int lo, int hi) {
    count = Kokkos::min(5, hi - lo + 1);
    if (count < 4) Kokkos::abort("FV4 subdivision needs four parent averages");
    first = Kokkos::max(lo, Kokkos::min(center - 2, hi - count + 1));
    const int k = center - first;
    constexpr int cubic[4][4] = {
        {-29, 47, -23, 5}, {-5, -9, 17, -3}, {3, -17, 9, 5}, {-5, 23, -47, 29}};
    constexpr int quartic[5][5] = {{-65, 122, -88, 38, -7},
                                   {-7, -30, 52, -18, 3},
                                   {3, -22, 0, 22, -3},
                                   {-3, 18, -52, 30, 7},
                                   {7, -38, 88, -122, 65}};
    for (int q = 0; q < count; ++q)
      weights[q] = count == 5 ? quartic[k][q] / 128. : cubic[k][q] / 64.;
  }
};

KOKKOS_INLINE_FUNCTION void ProlongOrdinatesFourth(
    int m, int n, int j, int i, int fj, int fi, int il, int iu, int jl, int ju,
    const DvceArray5D<Real>& parent, const DvceArray5D<Real>& child) {
  const Real mean = parent(m, n, 0, j, i);
  if (mean == 0) {
    StoreChildren(m, n, fj, fi, 0, 0, 0, 0, child);
    return;
  }
  const HalfCell wx(i, il, iu), wy(j, jl, ju);
  Real x = 0, y = 0, xy = 0;
  for (int b = 0; b < wy.count; ++b) {
    const int row = wy.first + b;
    const Real center = parent(m, n, 0, row, i);
    Real row_detail = 0;
    for (int a = 0; a < wx.count; ++a)
      row_detail +=
          wx.weights[a] * (parent(m, n, 0, row, wx.first + a) - center);
    if (row == j) x = row_detail;
    y += wy.weights[b] * (center - mean);
    xy += wy.weights[b] * row_detail;
  }
  StoreChildren(m, n, fj, fi, mean, x, y, xy, child);
}
}  // namespace boltzmann
#endif  // BOLTZMANN_PROLONGATION_HPP_
