// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SCRI_HISTORY_HPP_
#define Z4C_HYPERBOLOIDAL_SCRI_HISTORY_HPP_
#include <array>
#include <string>
namespace z4c_hyperboloidal {
// Signed extrema are encoded as (-minimum, maximum) until the MPI_MAX reduction.
// This preserves signs and handles ranks owning no scri points correctly.
struct ScriHistory {
  enum Signed { s_error,z_before,z_after,clock_original,clock_modified,
    Bx,By,Bz,invariant,I_geom,I_difference,theta_n,scalar_normal,N,L,
    p_t,theta_t,ell,qx,qy,qz,source,pen_s_t,pen_chi,pen_p,pen_theta,
    clock_before,reference_source,signed_count };
  enum Absolute { reg_lambda,reg_a,pole_chi_h,pole_p_h,pole_theta_h,
    pole_lambda_h,pole_a_h,pole_alpha_h,pole_beta_h,pen_lambda,pen_a,
    lift_alpha_hessian,lift_beta_hessian,absolute_count };
  static constexpr int count=2*signed_count+absolute_count;
  static constexpr int groups=(count+19)/20;
  static constexpr bool Minimum(int c) {return c<2*signed_count && c%2==0;}
  static std::string Label(int c) {
    constexpr const char *signed_names[]={"s-1","z-pre","z-post","c-orig","c-new",
      "Bx","By","Bz","I","Igeom","I-Igeom","Th-n","scalar-n","N","L",
      "P-t","Th-t","ell","qx","qy","qz","source","pen-z","pen-chi",
      "pen-P","pen-Th","c-pre","src-ref"};
    constexpr const char *absolute_names[]={"reg-Lam","reg-A","Nchi/h","NP/h","NTh/h",
      "NLam/h","NA/h","Nalp/h","Nbet/h","pen-Lam","pen-A","Hess-alp","Hess-bet"};
    if(c<2*signed_count)return std::string(signed_names[c/2])+(Minimum(c)?"-min":"-max");
    return absolute_names[c-2*signed_count];
  }
};
}  // namespace z4c_hyperboloidal
#endif
