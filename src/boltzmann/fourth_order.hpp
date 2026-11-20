// Finite-volume face averages in (rho,z,mu,psi). Each ordinate stores a
// phase-cell average. Directions 0..3 label faces, 4 labels the volume source.
#ifndef BOLTZMANN_FOURTH_ORDER_HPP_
#define BOLTZMANN_FOURTH_ORDER_HPP_
#include "athena.hpp"

namespace boltzmann {
struct PhaseCell {
  int i, j, band, sector;
  KOKKOS_INLINE_FUNCTION PhaseCell Shift(int d, int step, int sectors) const {
    PhaseCell p = *this;
    if (d == 0) p.i += step;
    if (d == 1) p.j += step;
    if (d == 2) p.band += step;
    if (d == 3) p.sector = (p.sector + sectors + step) % sectors;
    return p;
  }
  KOKKOS_INLINE_FUNCTION Real Read(const DvceArray5D<Real>& a, int m,
                                   int sectors) const {
    return a(m, band * sectors + sector, 0, j, i);
  }
};

struct PhaseFace {
  int d, m, hemisphere, sectors;
  DvceArray5D<Real> state, speed;

  KOKKOS_INLINE_FUNCTION Real Mean(PhaseCell p) const {
    const Real right = p.Read(state, m, sectors);
    return d == 4 ? right
                  : 0.5 * (right +
                           p.Shift(d, -1, sectors).Read(state, m, sectors));
  }

  KOKKOS_INLINE_FUNCTION Real Reconstruct(PhaseCell p, Real velocity) const {
    if (d == 4) return p.Read(state, m, sectors);
    int start = velocity >= 0 ? -3 : -2;
    if (d == 2) {
      int k = p.band % hemisphere;
      start = Kokkos::max(-k, Kokkos::min(start, hemisphere - k - 5));
    }
    // Integral-constrained quartic evaluated at the common face. Latitude
    // stencils slide into the same hemisphere at its two zero-flux boundaries.
    constexpr int w[4][5] = {{-3, 17, -43, 77, 12},
                             {2, -13, 47, 27, -3},
                             {-3, 27, 47, -13, 2},
                             {12, 77, -43, 17, -3}};
    Real result = 0;
    for (int k = 0; k < 5; ++k)
      result += w[start + 4][k] *
                p.Shift(d, start + k, sectors).Read(state, m, sectors);
    return result / 60;
  }

  template <bool latitude_integrated = false>
  KOKKOS_INLINE_FUNCTION Real Product(PhaseCell p) const {
    Real a = p.Read(speed, m, sectors);
    Real laplace = 0, covariance = 0;
    for (int t = 0; t < 4; ++t) {
      if (t == d || (latitude_integrated && t == 2)) continue;
      int k = p.band % hemisphere;
      if (t == 2 && (k == 0 || k == hemisphere - 1)) {
        const int sign = k == 0 ? 1 : -1;
        auto p1 = p.Shift(t, sign, sectors), p2 = p.Shift(t, 2 * sign, sectors);
        Real a1 = p1.Read(speed, m, sectors), a2 = p2.Read(speed, m, sectors);
        laplace += 2 * a - 5 * a1 + 4 * a2 -
                   p.Shift(t, 3 * sign, sectors).Read(speed, m, sectors);
        covariance +=
            (-3 * a + 4 * a1 - a2) * (-3 * Mean(p) + 4 * Mean(p1) - Mean(p2));
      } else {
        auto left = p.Shift(t, -1, sectors), right = p.Shift(t, 1, sectors);
        Real lo = left.Read(speed, m, sectors),
             hi = right.Read(speed, m, sectors);
        laplace += hi - 2 * a + lo;
        covariance += (hi - lo) * (Mean(right) - Mean(left));
      }
    }
    Real average = a + laplace / 24;
    return average * Reconstruct(p, average) + covariance / 48;
  }
  KOKKOS_INLINE_FUNCTION Real
  PolarFlux(PhaseCell p, int offset,
            const Kokkos::Array<DvceArray5D<Real>, 3>& coeff) const {
    Real result = Product<true>(p);
    const Real v = p.Read(speed, m, sectors), center = Reconstruct(p, v);
    int c = 0;
    for (int k = 0; k < 4; ++k)
      if (offset + k != 0) {
        result += p.Read(coeff[c++], m, sectors) *
                  (Reconstruct(p.Shift(2, offset + k, sectors), v) - center);
      }
    return result;
  }
};
}  // namespace boltzmann
#endif
