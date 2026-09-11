#ifndef Z4C_PHYSICAL_EXTRAPOLATION_HPP_
#define Z4C_PHYSICAL_EXTRAPOLATION_HPP_
#include "athena.hpp"

namespace z4c {
// A simple function for doing one-sided extrapolation.
// The off[xyz] variables control the direction of the extrapolation,
// and delta specifies how far to extrapolate to.
template<int order>
KOKKOS_INLINE_FUNCTION
Real Extrapolate(DvceArray5D<Real> u, const int m, const int n,
                 const int k, const int j, const int i,
                 const int offz, const int offy, const int offx,
                 const int delta);

// Linear extrapolation
template<>
KOKKOS_INLINE_FUNCTION
Real Extrapolate<2>(DvceArray5D<Real> u, const int m, const int n,
                    const int k, const int j, const int i,
                    const int offz, const int offy, const int offx,
                    const int delta) {
  Real f0 = u(m,n,k,j,i);
  Real f1 = u(m,n,k+offz,j+offy,i+offx);
  return f0 + (delta)*(f0 - f1);
}

// Quadratic extrapolation
template<>
KOKKOS_INLINE_FUNCTION
Real Extrapolate<3>(DvceArray5D<Real> u, const int m, const int n,
                    const int k, const int j, const int i,
                    const int offz, const int offy, const int offx,
                    const int delta) {
  Real f0 = u(m,n,k,j,i);
  Real f1 = u(m,n,k+offz,j+offy,i+offx);
  Real f2 = u(m,n,k+2*offz,j+2*offy,i+2*offx);
  return 0.5*(f0 * (1 + delta) * (2 + delta) +
              delta*(f2 + delta*f2 - 2*f1*(2 + delta)));
}

// Cubic extrapolation
template<>
KOKKOS_INLINE_FUNCTION
Real Extrapolate<4>(DvceArray5D<Real> u, const int m, const int n,
                    const int k, const int j, const int i,
                    const int offz, const int offy, const int offx,
                    const int delta) {
  Real f0 = u(m,n,k,j,i);
  Real f1 = u(m,n,k+offz,j+offy,i+offx);
  Real f2 = u(m,n,k+2*offz,j+2*offy,i+2*offx);
  Real f3 = u(m,n,k+3*offz,j+3*offy,i+3*offx);
  return (-3.0*f1*delta*(2 + delta)*(3 + delta) +
          f0*(1 + delta)*(2 + delta)*(3 + delta) +
          delta*(1 + delta)*(-f3*(2 + delta) + 3*f2*(3 + delta)))/6.0;
}

}  // namespace z4c
#endif  // Z4C_PHYSICAL_EXTRAPOLATION_HPP_
