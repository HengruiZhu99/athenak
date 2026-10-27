// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_SURFACE_MASS_HPP_
#define Z4C_HYPERBOLOIDAL_SURFACE_MASS_HPP_
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c {namespace hyperboloidal {
// A coordinate sphere in global Cartesian components. relative=x−center; the
// metric/Omega jets must be evaluated at x, not at relative. Physical null
// normals are n±s with inner product−2, K_ij=−(1/2)L_n gamma_ij.
template<class T>struct SphereExpansion {
  bool valid=false;
  T area_per_solid_angle{},mean_curvature{},curvature_trace{},theta_plus{},theta_minus{},product{};
};
template<class T>KOKKOS_INLINE_FUNCTION
SphereExpansion<T> CoordinateSphereExpansion(const Z4cJet<T>&u,const OmegaJet<T>&o,const T relative[3]){
  SphereExpansion<T>out{};T r2{};for(int i=0;i<3;++i)r2+=relative[i]*relative[i];
  if(!(r2>0&&o.omega>0&&u.chi.value>0)||!Kokkos::isfinite(r2+o.omega+u.chi.value))return out;
  const auto geometry=Geometry(u.metric);if(!geometry.valid)return out;
  const T radius=Kokkos::sqrt(r2),chi=u.chi.value;T normal[3]{},up[3]{},q{};
  for(int i=0;i<3;++i)normal[i]=relative[i]/radius;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)up[i]+=geometry.inverse[i][j]*normal[j];
  for(int i=0;i<3;++i)q+=normal[i]*up[i];if(!(q>0)||!Kokkos::isfinite(q))return out;
  T geometric{},gradient_chi{},gradient_omega{},a_normal{},a_trace{};
  for(int i=0;i<3;++i){gradient_chi+=up[i]*u.chi.d[i];gradient_omega+=up[i]*o.gradient[i];
    for(int j=0;j<3;++j){const T tangent=geometry.inverse[i][j]-up[i]*up[j]/q;
      T hessian=((i==j?T(1):T(0))-normal[i]*normal[j])/radius;
      for(int k=0;k<3;++k)hessian-=geometry.connection[k][i][j]*normal[k];
      geometric+=tangent*hessian;a_normal+=u.a.k[i][j]*up[i]*up[j]/q;
      a_trace+=geometry.inverse[i][j]*u.a.k[i][j];}}
  // gamma_phys=g_tilde/(chi Omega²). Expand the conformal connection before
  // evaluation, so no 1/Omega intermediate is formed in the mean curvature.
  out.mean_curvature=Kokkos::sqrt(chi/q)*(o.omega*(geometric-gradient_chi/chi)-2*gradient_omega);
  // Same physical curvature normalization as ToPhysicalADM; retain any trace
  // of A rather than projecting algebraic or differential constraints.
  out.curvature_trace=o.omega*(a_normal-a_trace)-T(2./3.)*(u.trace.value+2*u.theta.value);
  out.area_per_solid_angle=r2*Kokkos::sqrt(geometry.determinant*q)/(chi*o.omega*o.omega);
  out.theta_plus=out.curvature_trace+out.mean_curvature;
  out.theta_minus=out.curvature_trace-out.mean_curvature;
  out.product=out.theta_plus*out.theta_minus;
  out.valid=out.area_per_solid_angle>0&&Kokkos::isfinite(out.area_per_solid_angle)
    &&Kokkos::isfinite(out.mean_curvature)&&Kokkos::isfinite(out.curvature_trace)&&Kokkos::isfinite(out.product);
  return out;
}
template<class T>struct SurfaceMass {bool valid=false;T area{},areal_radius{},mass{};};
template<class T>KOKKOS_INLINE_FUNCTION
SurfaceMass<T> HawkingSurfaceMass(T area,T integrated_expansion_product){
  SurfaceMass<T>out{};out.area=area;
  if(!(area>0)||!Kokkos::isfinite(area)||!Kokkos::isfinite(integrated_expansion_product))return out;
  const T pi=T(3.141592653589793238462643383279502884L);
  out.areal_radius=Kokkos::sqrt(area/(4*pi));
  out.mass=out.areal_radius/T(2)*(1+integrated_expansion_product/(16*pi));
  out.valid=Kokkos::isfinite(out.areal_radius)&&Kokkos::isfinite(out.mass);return out;
}
}}
#endif
