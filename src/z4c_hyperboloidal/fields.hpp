// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_FIELDS_HPP_
#define Z4C_HYPERBOLOIDAL_FIELDS_HPP_

#include <array>
#include <string_view>
#include "athena.hpp"

namespace z4c_hyperboloidal {
// Point values in the shared Cartesian tensor frame on every mesh chart.
// These are not the 25-variable standard Z4c layout. Slot 13 is P+3,
// where P=K-2*Theta; slot 18 and slots 22:24 have analytic offsets.
struct Fields {
  static constexpr int count=31, independent_count=26;
  static constexpr int chi=0, metric=1, curvature=7, trace=13, theta=14;
  static constexpr int lambda=15, lapse=18, B=19, shift=22, D=25, source=28;
  static constexpr std::array<std::string_view,count> names={
    "chi","gxx-1","gxy","gxz","gyy-1","gyz","gzz-1",
    "Axx","Axy","Axz","Ayy","Ayz","Azz","P+3","Theta",
    "Lambda_x","Lambda_y","Lambda_z","alpha-(1+r2)/2",
    "B_x","B_y","B_z","beta_x+x","beta_y+y","beta_z+z",
    "D_x","D_y","D_z","J_x","J_y","J_z"};
  KOKKOS_INLINE_FUNCTION static bool Independent(int f) {
    return f>=0 && f<28 && f!=6 && f!=12;
  }
};

enum class PointStatus : int {
  success=0, invalid_input=1, invalid_geometry=2, invalid_assembly=3,
  invalid_gauge=4, invalid_auxiliary=5, cfl_exceeded=6, nonfinite_rhs=7
};
}  // namespace z4c_hyperboloidal
#endif
