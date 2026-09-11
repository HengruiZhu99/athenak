#ifndef DRIVER_CLASSICAL_RK4_HPP_
#define DRIVER_CLASSICAL_RK4_HPP_
#include <Kokkos_Core.hpp>

namespace classical_rk4 {
// One-based stages, matching Driver and Z4c tasks. Stage vectors must not
// themselves be used as dense-output physical solutions at these times.
KOKKOS_INLINE_FUNCTION constexpr double StageTime(int stage) {
  return stage == 1 ? 0.0 : (stage == 4 ? 1.0 : 0.5);
}
KOKKOS_INLINE_FUNCTION constexpr double Weight(int stage) {
  return stage == 1 || stage == 4 ? 1.0/6.0 : 1.0/3.0;
}
KOKKOS_INLINE_FUNCTION constexpr double NextFraction(int stage) {
  return stage < 3 ? 0.5 : 1.0;
}
}  // namespace classical_rk4
#endif  // DRIVER_CLASSICAL_RK4_HPP_
