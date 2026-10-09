// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CMC_REFERENCE_HPP_
#define Z4C_HYPERBOLOIDAL_CMC_REFERENCE_HPP_

#include <cmath>
#include <stdexcept>
#include <Kokkos_Core.hpp>

namespace z4c {
namespace hyperboloidal {

// Penrose factor omega is NOT Z4c chi. All fields below belong to the regular
// metric bar(g)=omega^2 g_phys. Cartesian bar(gamma)_ij=delta_ij, chi=1,
// A_ij=0, Gamma^i=0. The physical CMC trace is -3/a, not k_bar.
template <typename T>
struct CMCPoint {
  T omega, alpha, k_bar, k_physical;
  T beta[3], domega[3], dalpha[3];
  T hessian_omega, hessian_alpha;  // diagonal entries; off-diagonals vanish
  // Einstein tensor projections, with the factor 8*pi absorbed. These are
  // analytic REFERENCE sources, not sources for an arbitrary evolved metric.
  T einstein_nn, einstein_momentum[3], einstein_stress_diagonal;
};

template <typename T>
struct CMCReference {
  T scri_radius, curvature_radius;

  // Call on the host before capture into a device lambda.
  void Validate() const {
    if (!std::isfinite(scri_radius) || scri_radius <= 0 ||
        !std::isfinite(curvature_radius) || curvature_radius <= 0) {
      throw std::invalid_argument("CMC reference requires finite positive S and a");
    }
  }

  KOKKOS_INLINE_FUNCTION
  CMCPoint<T> At(T x, T y, T z) const {
    const T s = scri_radius, a = curvature_radius;
    const T xyz[3] = {x, y, z};
    const T r2 = x*x + y*y + z*z;
    CMCPoint<T> p;
    p.omega = (s*s-r2)/(2*a*s);
    p.alpha = (s*s+r2)/(2*a*s);
    p.k_bar = -3/(a*p.alpha);
    p.k_physical = -3/a;
    p.hessian_omega = -1/(a*s);
    p.hessian_alpha = 1/(a*s);
    p.einstein_nn = 3/(a*a*p.alpha*p.alpha);
    p.einstein_stress_diagonal = -p.einstein_nn + 2/(a*s*p.alpha)
        + 2*r2/(a*a*a*s*p.alpha*p.alpha*p.alpha);
    for (int i = 0; i < 3; ++i) {
      p.beta[i] = -xyz[i]/a;
      p.domega[i] = -xyz[i]/(a*s);
      p.dalpha[i] = xyz[i]/(a*s);
      p.einstein_momentum[i] = -2*p.dalpha[i]/(a*p.alpha*p.alpha);
    }
    return p;
  }

  // True radial light speeds. Factorization avoids cancellation at scri.
  KOKKOS_INLINE_FUNCTION
  T OutgoingSpeed(T r) const {
    const T u = scri_radius+r;
    return u*u/(2*curvature_radius*scri_radius);
  }
  KOKKOS_INLINE_FUNCTION
  T IngoingSpeed(T r) const {
    const T u = scri_radius-r;
    return -u*u/(2*curvature_radius*scri_radius);
  }

  // Coordinates relative to the sphere's center; classification of the whole
  // closed box, not just its center. This does not construct a boundary stencil.
  // -1: strictly inside; 0: intersects/touches; +1: strictly outside.
  KOKKOS_INLINE_FUNCTION
  int ClassifyBox(const T lo[3], const T hi[3]) const {
    T near2 = 0, far2 = 0;
    for (int d = 0; d < 3; ++d) {
      const T nearest = lo[d] > 0 ? lo[d] : (hi[d] < 0 ? hi[d] : 0);
      near2 += nearest*nearest;
      far2 += lo[d]*lo[d] > hi[d]*hi[d] ? lo[d]*lo[d] : hi[d]*hi[d];
    }
    return near2 > scri_radius*scri_radius ? 1 :
           (far2 < scri_radius*scri_radius ? -1 : 0);
  }
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CMC_REFERENCE_HPP_
