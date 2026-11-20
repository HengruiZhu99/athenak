#ifndef BOLTZMANN_RECONSTRUCTION_HPP_
#define BOLTZMANN_RECONSTRUCTION_HPP_
#include "athena.hpp"
namespace boltzmann {
// Monotonized central reconstruction. A nonnegative cell average has
// nonnegative face states, including at a smooth/vacuum interface.
KOKKOS_INLINE_FUNCTION Real FaceState(Real left, Real center, Real right, int side) {
  Real dl = center - left, dr = right - center;
  if (dl * dr <= 0) return center;
  Real half = fmin(0.25 * fabs(dl + dr), fmin(fabs(dl), fabs(dr)));
  return center + side * copysign(half, dl);
}

KOKKOS_INLINE_FUNCTION Real AngularFlux(const DvceArray5D<Real>& u, int m, int j, int i,
                                        int band, int sector, int hemisphere_bands, int sectors,
                                        bool latitude, Real speed) {
  const int side = speed >= 0 ? 1 : -1;
  const int upstream = (speed >= 0 ? -1 : 0);
  if (latitude) {
    int b = band + upstream;
    const int first = b / hemisphere_bands * hemisphere_bands;
    int left = (b > first ? b - 1 : b) * sectors + sector;
    int right = (b + 1 < first + hemisphere_bands ? b + 1 : b) * sectors + sector;
    return speed * FaceState(u(m, left, 0, j, i), u(m, b * sectors + sector, 0, j, i),
                             u(m, right, 0, j, i), side);
  }
  int s = (sector + upstream + sectors) % sectors, base = band * sectors;
  return speed * FaceState(u(m, base + (s + sectors - 1) % sectors, 0, j, i),
                           u(m, base + s, 0, j, i), u(m, base + (s + 1) % sectors, 0, j, i),
                           side);
}
}  // namespace boltzmann
#endif  // BOLTZMANN_RECONSTRUCTION_HPP_
