#ifndef BOLTZMANN_SOURCE_PROJECTION_HPP_
#define BOLTZMANN_SOURCE_PROJECTION_HPP_

#include <limits>

#include "boltzmann/geometry.hpp"
#include "boltzmann/moments.hpp"

namespace boltzmann {
// Point evaluation of the cubic matching four cell means. Each block uses its
// active cells, including at the axis and at refinement or physical boundaries.
struct SourceStencil {
  int first;
  Real w[4];
};

KOKKOS_INLINE_FUNCTION int SourceFirst(int vertex, int lo, int hi) {
  return Kokkos::max(lo, Kokkos::min(vertex - 2, hi - 3));
}

KOKKOS_INLINE_FUNCTION SourceStencil VerticalSourceStencil(int vertex, int lo,
                                                           int hi) {
  const int first = SourceFirst(vertex, lo, hi);
  constexpr int coeff[5][4] = {{25, -23, 13, -3},
                               {3, 13, -5, 1},
                               {-1, 7, 7, -1},
                               {1, -5, 13, 3},
                               {-3, 13, -23, 25}};
  SourceStencil result{first, {}};
  for (int j = 0; j < 4; ++j) result.w[j] = coeff[vertex - first][j] / 12.;
  return result;
}

KOKKOS_INLINE_FUNCTION SourceStencil
RadialSourceStencil(int vertex, int lo, int hi, Real left_radius_over_dr) {
  SourceStencil result{SourceFirst(vertex, lo, hi), {}};
  // The raw moment divided by the cell-center radius is the rho-weighted
  // cell mean of sqrt(gamma_cart)*T_hat. Match the moments of 1,x,x^2,x^3
  // about the vertex (x in cell widths). There is no 1/rho_vertex at the axis.
  Real a[4][5]{};
  a[0][4] = 1;
  for (int q = 0; q < 4; ++q) {
    const Real x = result.first + q - vertex + .5;
    const Real r = left_radius_over_dr + (result.first - lo + q) + .5;
    a[0][q] = 1;
    a[1][q] = x + 1 / (12 * r);
    a[2][q] = x * x + 1. / 12 + x / (6 * r);
    a[3][q] = x * (x * x + .25) + (x * x / 4 + 1. / 80) / r;
  }
  // Gaussian elimination of the transposed moment system. Its leading
  // principal minors are nonzero for positively weighted, ordered intervals.
  for (int k = 0; k < 4; ++k) {
    const Real pivot = a[k][k];
    for (int j = k + 1; j <= 4; ++j) a[k][j] /= pivot;
    for (int i = 0; i < 4; ++i) {
      if (i == k) continue;
      const Real factor = a[i][k];
      for (int j = k + 1; j <= 4; ++j) a[i][j] -= factor * a[k][j];
    }
  }
  for (int q = 0; q < 4; ++q) result.w[q] = a[q][4];
  return result;
}

// In the local orthonormal frame, P/E - (S/E)(S/E)^T is the direction
// covariance of a nonnegative angular distribution. Check all principal minors.
KOKKOS_INLINE_FUNCTION bool RealizableSource(const Real v[10]) {
  if (!(v[6] > 0)) {
    for (int l = 0; l < 10; ++l)
      if (v[l] != 0) return false;
    return true;
  }
  const Real x = v[7] / v[6], y = v[8] / v[6], z = v[9] / v[6];
  const Real a = v[0] / v[6] - x * x, b = v[3] / v[6] - y * y,
             c = v[5] / v[6] - z * z;
  const Real d = v[1] / v[6] - x * y, e = v[2] / v[6] - x * z,
             f = v[4] / v[6] - y * z;
  constexpr Real tol = 128 * std::numeric_limits<Real>::epsilon();
  return a >= -tol && b >= -tol && c >= -tol && a * b - d * d >= -tol &&
         a * c - e * e >= -tol && b * c - f * f >= -tol &&
         a * b * c + 2 * d * e * f - a * f * f - b * e * e - c * d * d >= -tol;
}

KOKKOS_INLINE_FUNCTION bool ProjectSourceMoments(const DvceArray5D<Real>& means,
                                                 int m, int j, int i,
                                                 const SourceStencil& radial,
                                                 const SourceStencil& vertical,
                                                 Real source[10]) {
  const int il = Kokkos::clamp(i - 1, radial.first, radial.first + 3);
  const int ir = Kokkos::clamp(i, radial.first, radial.first + 3);
  const int jb = Kokkos::clamp(j - 1, vertical.first, vertical.first + 3);
  const int jt = Kokkos::clamp(j, vertical.first, vertical.first + 3);
  if (means(m, 6, 0, jb, il) == 0 && means(m, 6, 0, jb, ir) == 0 &&
      means(m, 6, 0, jt, il) == 0 && means(m, 6, 0, jt, ir) == 0) {
    for (int l = 0; l < 10; ++l) source[l] = 0;
    return true;
  }
  for (int l = 0; l < 10; ++l) {
    Real v = 0;
    for (int z = 0; z < 4; ++z) {
      Real row = 0;
      for (int r = 0; r < 4; ++r)
        row +=
            radial.w[r] * means(m, l, 0, vertical.first + z, radial.first + r);
      v += vertical.w[z] * row;
    }
    source[l] = v;
  }
  if (RealizableSource(source)) return true;
  // A positive convex combination of native orthonormal cell moments remains
  // realizable even at a sharp beam edge. Clamp to authoritative active cells.
  for (int l = 0; l < 10; ++l)
    source[l] = .25 * (means(m, l, 0, jb, il) + means(m, l, 0, jb, ir) +
                       means(m, l, 0, jt, il) + means(m, l, 0, jt, ir));
  return false;
}

KOKKOS_INLINE_FUNCTION void StoreVertexSource(const Real source[10],
                                              const Frame& frame,
                                              const DvceArray5D<Real>& target,
                                              int m, int j, int i, bool axis) {
  constexpr int sym[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  target(m, 6, 0, j, i) = source[6] / frame.volume;
  for (int a = 0; a < 3; ++a) {
    Real s = 0;
    for (int p = a; p < 3; ++p) s += frame.co[a][p] * source[7 + p];
    target(m, 7 + a, 0, j, i) = s / frame.volume;
    for (int b = a; b < 3; ++b) {
      Real t = 0;
      for (int p = a; p < 3; ++p)
        for (int q = b; q < 3; ++q)
          t += frame.co[a][p] * frame.co[b][q] * source[sym[p][q]];
      target(m, sym[a][b], 0, j, i) = t / frame.volume;
    }
  }
  if (axis) ProjectAxisStress(target, m, j, i);
}
}  // namespace boltzmann
#endif
