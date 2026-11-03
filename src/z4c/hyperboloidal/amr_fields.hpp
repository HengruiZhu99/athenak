// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_AMR_FIELDS_HPP_
#define Z4C_HYPERBOLOIDAL_AMR_FIELDS_HPP_
#include <array>
#include <string_view>
namespace z4c::hyperboloidal {
// Native point samples, all tensor/vector components in the shared Cartesian
// frame. Register this complete array; never cast it to Z4c::Z4c_vars (25 fields).
// Additive analytic reference terms commute with degree-four interpolation.
// RK compensation uses the same layout but is an independent registered array.
struct AMRFields {
  static constexpr int count=31;
  static constexpr int chi=0,metric=1,curvature=7,trace=13,theta=14;
  static constexpr int lambda=15,lapse=18,B=19,shift=22,D=25,source=28;
  static constexpr std::array<std::string_view,count> names={
    "chi","gxx-1","gxy","gxz","gyy-1","gyz","gzz-1",
    "Axx","Axy","Axz","Ayy","Ayz","Azz","P+3","Theta",
    "Lambda_x","Lambda_y","Lambda_z","alpha-(1+r2)/2",
    "B_x","B_y","B_z","beta_x+x","beta_y+y","beta_z+z",
    "D_x","D_y","D_z","J_x","J_y","J_z"};
};
}
#endif
