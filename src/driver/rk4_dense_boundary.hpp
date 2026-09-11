#ifndef DRIVER_RK4_DENSE_BOUNDARY_HPP_
#define DRIVER_RK4_DENSE_BOUNDARY_HPP_
#include <Kokkos_Core.hpp>

namespace subcycling {
// Classical-RK4 coarse data for one component. k1..k4 are RHS values, NOT
// increments. StageBoundary returns fine RK stage vectors, not merely the
// physical solution evaluated at a stage's nominal time.
// Ji et al., arXiv:2503.09629v2, equations (11)-(19), following Mongwane2015.
// Cubic dense output has O(H^4) local interpolation error; its use here is
// not a standalone claim of fourth-order global AMR convergence.
struct RK4DenseBoundary {
  double y0, H, k1, k2, k3, k4;
  KOKKOS_INLINE_FUNCTION double Value(double q) const {
    const double q2=q*q, q3=q2*q;
    return y0+H*((q-1.5*q2+2*q3/3)*k1+
                 (q2-2*q3/3)*(k2+k3)+(-0.5*q2+2*q3/3)*k4);
  }
  KOKKOS_INLINE_FUNCTION double First(double q) const {
    return (1-3*q+2*q*q)*k1+(2*q-2*q*q)*(k2+k3)+(-q+2*q*q)*k4;
  }
  KOKKOS_INLINE_FUNCTION double Second(double q) const {
    return ((-3+4*q)*k1+(2-4*q)*(k2+k3)+(-1+4*q)*k4)/H;
  }
  KOKKOS_INLINE_FUNCTION double Third() const {
    return 4*(k1-k2-k3+k4)/(H*H);
  }
  KOKKOS_INLINE_FUNCTION double StageBoundary(double start_fraction,
                                             double fine_dt, int stage) const {
    const double y=Value(start_fraction);
    if (stage == 1) return y;
    const double first=First(start_fraction);
    if (stage == 2) return y+0.5*fine_dt*first;
    const double second=Second(start_fraction), third=Third();
    const double jacobian_second=4*(k3-k2)/(H*H);
    const double h2=fine_dt*fine_dt, h3=h2*fine_dt;
    if (stage == 3) {
      return y+0.5*fine_dt*first+0.25*h2*second+
             h3*(third-jacobian_second)/16;
    }
    return y+fine_dt*first+0.5*h2*second+h3*(third+jacobian_second)/8;
  }
};
}  // namespace subcycling
#endif  // DRIVER_RK4_DENSE_BOUNDARY_HPP_
