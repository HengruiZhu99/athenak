//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef UTILS_CIRCUMBINARY_BUDGET_HPP_
#define UTILS_CIRCUMBINARY_BUDGET_HPP_
#include "athena.hpp"

class MeshBlockPack;
namespace circumbinary {
constexpr int kRegions = 4;
// A disjoint partition inside outer: near1, near2, cavity, annulus. The actual
// puncture excision ellipsoids are excluded. No origin or axis motion is implied.
struct BudgetGeometry {
  Real inner = 0, outer = 0, radius[2] = {}, center[2][3] = {}, excision_H[2][3][3] = {};
  int holes = 0, subcells = 2;
  KOKKOS_INLINE_FUNCTION int Region(Real x, Real y, Real z) const {
    Real pos[3] = {x, y, z};
    Real r2 = x * x + y * y + z * z;
    if (r2 >= outer * outer) {
      return -1;
    }
    for (int h = 0; h < holes; ++h) {
      Real d[3], d2 = 0, f = 0;
      for (int a = 0; a < 3; ++a) {
        d[a] = pos[a] - center[h][a];
        d2 += d[a] * d[a];
      }
      for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
          f += d[a] * excision_H[h][a][b] * d[b];
        }
      }
      if (f < 1.0) {
        return -1;
      }
      if (d2 < radius[h] * radius[h]) {
        return h;
      }
    }
    return r2 < inner * inner ? 2 : 3;
  }
};
// A barycentric spherical shell. The first shell subtracts one or two moving
// component horizons, each represented by its equal-time Cartesian quadric.
struct HorizonShell {
  Real inner = 0, outer = 0, center[2][3] = {}, H[2][3][3] = {};
  Real inscribed[2] = {}, circumscribed[2] = {}, frame_omega = 0;
  int holes = 0, subcells = 8;
  // Classify a cell using an enclosing ball. Only ambiguous cut cells need chords.
  KOKKOS_INLINE_FUNCTION int CellCoverage(const Real x[3], const Real dx[3]) const {
    Real d = 0.5 * sqrt(SQR(dx[0]) + SQR(dx[1]) + SQR(dx[2]));
    Real r = sqrt(SQR(x[0]) + SQR(x[1]) + SQR(x[2]));
    if (r - d >= outer || r + d <= inner) return 0;
    bool full = r + d <= outer && (inner == 0 || r - d >= inner);
    for (int h = 0; h < holes; ++h) {
      r = sqrt(SQR(x[0] - center[h][0]) + SQR(x[1] - center[h][1]) +
               SQR(x[2] - center[h][2]));
      if (r + d <= inscribed[h]) return 0;
      full &= r - d >= circumscribed[h];
    }
    return full ? 1 : -1;
  }
  // Integrate the vertical chord analytically. Excluded quadrics are disjoint
  // and contained in the outer sphere, as checked when geometry is constructed.
  KOKKOS_INLINE_FUNCTION Real ColumnLength(Real x, Real y, Real lo, Real hi) const {
    const Real xy = x * x + y * y;
    if (xy >= outer * outer) return 0;
    Real z = sqrt(outer * outer - xy);
    lo = fmax(lo, -z);
    hi = fmin(hi, z);
    if (hi <= lo) return 0;
    Real length = hi - lo;
    if (xy < inner * inner) {
      z = sqrt(inner * inner - xy);
      length -= fmax(0.0, fmin(hi, z) - fmax(lo, -z));
    }
    for (int h = 0; h < holes; ++h) {
      Real X = x - center[h][0], Y = y - center[h][1];
      const auto& q = H[h];
      Real A = q[2][2], B = q[0][2] * X + q[1][2] * Y;
      Real C = q[0][0] * X * X + 2 * q[0][1] * X * Y + q[1][1] * Y * Y - 1;
      Real disc = B * B - A * C;
      if (disc <= 0) continue;
      Real mid = center[h][2] - B / A, half = sqrt(disc) / A;
      length -= fmax(0.0, fmin(hi, mid + half) - fmax(lo, mid - half));
    }
    return fmax(0.0, length);
  }
  // Coordinate volume and horizontal first moments in the shell's coordinate system.
  KOKKOS_INLINE_FUNCTION Real CellMoments(const Real center[3], const Real dx[3],
                                          Real first[3]) const {
    int n = subcells;
    first[0] = first[1] = 0;
    Real area = dx[0] * dx[1] / (n * n), volume = 0;
    Real lo = center[2] - 0.5 * dx[2], hi = center[2] + 0.5 * dx[2];
    int coverage = CellCoverage(center, dx);
    if (coverage == 0) return 0;
    if (coverage == 1) {
      volume = dx[0] * dx[1] * dx[2];
      first[0] = center[0] * volume;
      first[1] = center[1] * volume;
    } else {
      for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b) {
          Real x = center[0] + dx[0] * ((a + 0.5) / n - 0.5);
          Real y = center[1] + dx[1] * ((b + 0.5) / n - 0.5);
          Real dv = area * ColumnLength(x, y, lo, hi);
          volume += dv;
          first[0] += x * dv;
          first[1] += y * dv;
        }
    }
    return volume;
  }
};
struct ShellMoments {
  Real stored[4] = {}, source[4] = {}, sampled = 0, rejected = 0;
  KOKKOS_INLINE_FUNCTION ShellMoments& operator+=(const ShellMoments& other) {
    for (int d = 0; d < 4; ++d) {
      stored[d] += other.stored[d];
      source[d] += other.source[d];
    }
    sampled += other.sampled;
    rejected += other.rejected;
    return *this;
  }
};
struct RegionMoments {
  Real J[kRegions] = {}, torque[kRegions] = {}, volume[kRegions] = {}, bad[kRegions] = {};
  KOKKOS_INLINE_FUNCTION RegionMoments& operator+=(const RegionMoments& s) {
    for (int r = 0; r < kRegions; ++r) {
      J[r] += s.J[r];
      torque[r] += s.torque[r];
      volume[r] += s.volume[r];
      bad[r] += s.bad[r];
    }
    return *this;
  }
};
}  // namespace circumbinary
namespace Kokkos {
template <>
struct reduction_identity<circumbinary::ShellMoments> {
  KOKKOS_INLINE_FUNCTION static circumbinary::ShellMoments sum() { return {}; }
};
template <>
struct reduction_identity<circumbinary::RegionMoments> {
  KOKKOS_INLINE_FUNCTION static circumbinary::RegionMoments sum() { return {}; }
};
}  // namespace Kokkos
// Local sums only. MPI reduction is deliberately outside retryable task functions.
circumbinary::RegionMoments ReduceBinaryRegions(MeshBlockPack*,
                                                circumbinary::BudgetGeometry,
                                                bool with_torque = false);
circumbinary::ShellMoments ReduceHorizonShell(MeshBlockPack*, circumbinary::HorizonShell,
                                              bool with_sources = true);
// Global inventory between a component-centered coordinate sphere and its horizon.
// Values are accepted volume, rejected volume, baryon mass and barycentric Jz.
void MeasureTransferInventory(MeshBlockPack*, const Real origin[3],
                              circumbinary::HorizonShell, Real values[4]);
#endif  // UTILS_CIRCUMBINARY_BUDGET_HPP_
