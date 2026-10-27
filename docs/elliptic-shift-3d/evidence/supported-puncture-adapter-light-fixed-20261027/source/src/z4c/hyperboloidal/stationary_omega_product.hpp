// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_STATIONARY_OMEGA_PRODUCT_HPP_
#define Z4C_HYPERBOLOIDAL_STATIONARY_OMEGA_PRODUCT_HPP_

#include <Kokkos_Core.hpp>

namespace z4c { namespace hyperboloidal {

// The regular COORDINATE product N=alpha*n_bar(Omega), not a unit normal.
// For n_bar=(partial_t-beta^i partial_i)/alpha at alpha>0,
//
//   N=Omega_t-beta^i Omega_i,
//   partial_d N=partial_d Omega_t-(partial_d beta^i)Omega_i
//                                      -beta^i Omega_di.
//
// Evaluate this product before any division by a collapsed lapse. T may also
// be a directional-AD scalar: only scalar construction and arithmetic enter.
template<typename T> struct OmegaNormalProductJet {
  T value{},d[3]{};
  KOKKOS_INLINE_FUNCTION T Directional(const T direction[3])const{
    T out{};for(int a=0;a<3;++a)out+=direction[a]*d[a];return out;
  }
};

template<typename T> KOKKOS_INLINE_FUNCTION
T StationaryOmegaProductValue(const T beta[3],const T omega_gradient[3]){
  T product{};for(int i=0;i<3;++i)product-=beta[i]*omega_gradient[i];
  return product;
}

// EXPLICITLY stationary coordinate Omega. dbeta[d][i] has derivative first,
// matching VectorJet. This does not inspect or override OmegaJet::normal.
// Calling it for arbitrary independently prescribed normal data would change
// the equations. A nonstationary coordinate Omega needs the separate helper
// below and the caller's Omega_t and spatial gradient of Omega_t.
template<typename T> KOKKOS_INLINE_FUNCTION
OmegaNormalProductJet<T> StationaryOmegaProduct(const T beta[3],
    const T dbeta[3][3],const T omega_gradient[3],const T omega_hessian[3][3]){
  OmegaNormalProductJet<T> out;
  out.value=StationaryOmegaProductValue(beta,omega_gradient);
  for(int d=0;d<3;++d)for(int i=0;i<3;++i)
    out.d[d]-=dbeta[d][i]*omega_gradient[i]+beta[i]*omega_hessian[d][i];
  return out;
}

template<typename T> KOKKOS_INLINE_FUNCTION
OmegaNormalProductJet<T> CoordinateOmegaProduct(const T beta[3],
    const T dbeta[3][3],const T omega_gradient[3],const T omega_hessian[3][3],
    T omega_coordinate_time,const T domega_coordinate_time[3]){
  auto out=StationaryOmegaProduct(beta,dbeta,omega_gradient,omega_hessian);
  out.value+=omega_coordinate_time;
  for(int d=0;d<3;++d)out.d[d]+=domega_coordinate_time[d];
  return out;
}

// Linear normal products in the existing pole numerators can be evaluated as:
//
// chi:   (2/3)*chi*(alpha*K-3*N),
// Theta: alpha*(K*K/3-3*|grad Omega|_bar^2-kappa*Theta)-3*N*Theta,
// A_ij:  [unchanged trace-free Omega-Hessian terms]+A_ij*(alpha*K-N).
//
// Their first directional derivatives follow by the product rule with D N
// from this jet. For example D[A_ij*(alpha*K-N)] is
// D A_ij*(alpha*K-N)+A_ij*(D alpha*K+alpha*D K-D N).
// Appendix-C's +3*alpha*w*Theta is +3*N*Theta; where the chosen formulation
// cancels that term, combine the selected source algebraically before losing
// a much smaller remainder. This file does NOT choose that source variant.
// There is no background-RHS subtraction, lapse modification or floor here.
//
// At alpha=0 the coordinate contraction N (and D N) can remain defined from
// beta, coordinate-Omega and their jets. The 3+1 unit normal and w=N/alpha
// are NOT thereby defined. Uncancelled w, D w or w^2 expressions still need
// independent limiting data/regularity. Even N=0 and alpha=0 do not specify a
// path-independent ratio; no 0/0, limiting w or finite normal is manufactured.
// A time derivative of N additionally needs beta_t and coordinate-Omega time
// jets; this helper supplies spatial derivatives only.

}}  // namespace z4c::hyperboloidal
#endif
