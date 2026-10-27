// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_REGULAR_PENROSE_HPP_
#define Z4C_HYPERBOLOIDAL_REGULAR_PENROSE_HPP_

#include "z4c/hyperboloidal/conformal_constraints.hpp"

namespace z4c { namespace hyperboloidal {

enum class RegularPenroseStatus { valid, invalid_metric, nonpositive_chi,
                                  nonfinite_input, nonfinite_result };
KOKKOS_INLINE_FUNCTION
const char *RegularPenroseStatusName(RegularPenroseStatus status){
  switch(status){
    case RegularPenroseStatus::valid:return "valid";
    case RegularPenroseStatus::invalid_metric:return "invalid_metric";
    case RegularPenroseStatus::nonpositive_chi:return "nonpositive_chi";
    case RegularPenroseStatus::nonfinite_input:return "nonfinite_input";
    case RegularPenroseStatus::nonfinite_result:return "nonfinite_result";
  }
  return "nonfinite_result";
}

template<typename T> struct RegularPenroseCurvature {
  bool valid=false;
  RegularPenroseStatus status=RegularPenroseStatus::nonfinite_input;
  T chi_ricci[3][3]{}; // chi Rbar_ij, NOT Rbar_ij
  T scalar{};         // Rbar, NOT chi Rbar
  T gradient_ratio[3]{}; // chi_i/chi; no inverse chi squared/cubed
  T hessian_chi[3][3]{},laplacian_chi{},gradient_chi_squared_over_chi{};
};
template<typename T> struct RegularPenroseCovector {
  bool valid=false;
  RegularPenroseStatus status=RegularPenroseStatus::nonfinite_input;
  T chi_derivative[3][3]{}; // chi Dbar_i z_j, with arbitrary covector z
  T divergence{};          // bar g^ij Dbar_i z_j
};
template<typename T> struct RegularPenroseScalar {
  bool valid=false;
  RegularPenroseStatus status=RegularPenroseStatus::nonfinite_input;
  T chi_hessian[3][3]{}; // chi Dbar_i Dbar_j f
  T laplacian{};        // bar Delta f
};

template<typename Jet> KOKKOS_INLINE_FUNCTION
bool RegularPenroseFiniteJet(const Jet &q){
  bool valid=Kokkos::isfinite(q.value);
  for(int i=0;i<3;++i){valid=valid&&Kokkos::isfinite(q.d[i]);
    for(int j=0;j<3;++j)valid=valid&&Kokkos::isfinite(q.dd[i][j]);}
  return valid;
}

// In three spatial dimensions, bar g=chi^-1 g gives
// chi Rbar_ij=chi Rg_ij+.5(D_iD_j chi+g_ij Delta chi)
//             -.25 chi_i(chi_j/chi)-.75 g_ij gradchi.(gradchi/chi),
// Rbar=chi Rg+2 Delta chi-2.5 gradchi.(gradchi/chi).
// No determinant-one, conformal-flatness, constraint, radial-power or background
// subtraction assumption enters. Chi/F templates avoid the ScalarJet include
// cycle: only the value/d/dd contract is needed. The original positive physical
// chi is required; zero-endpoint limits cannot be inferred by replacing it.
template<typename T,typename ChiJet> KOKKOS_INLINE_FUNCTION
RegularPenroseCurvature<T> RegularPenroseGeometry(const MetricJet<T> &g,
    const SpatialGeometry<T> &geometry,const ChiJet &chi){
  RegularPenroseCurvature<T> out{};
  if(!geometry.valid){out.status=RegularPenroseStatus::invalid_metric;return out;}
  if(!RegularPenroseFiniteJet(chi)){out.status=RegularPenroseStatus::nonfinite_input;return out;}
  if(!(chi.value>T(0))){out.status=RegularPenroseStatus::nonpositive_chi;return out;}
  for(int i=0;i<3;++i){out.gradient_ratio[i]=chi.d[i]/chi.value;
    if(!Kokkos::isfinite(out.gradient_ratio[i])){out.status=RegularPenroseStatus::nonfinite_result;return out;}}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    T h=chi.dd[i][j];for(int k=0;k<3;++k)h-=geometry.connection[k][i][j]*chi.d[k];
    out.hessian_chi[i][j]=h;
    out.laplacian_chi+=geometry.inverse[i][j]*h;
    out.gradient_chi_squared_over_chi+=geometry.inverse[i][j]*chi.d[i]*out.gradient_ratio[j];
  }
  out.scalar=chi.value*geometry.scalar+T(2)*(out.laplacian_chi-out.gradient_chi_squared_over_chi)
      -T(.5)*out.gradient_chi_squared_over_chi;
  bool valid=Kokkos::isfinite(out.scalar)&&Kokkos::isfinite(out.laplacian_chi)
      &&Kokkos::isfinite(out.gradient_chi_squared_over_chi);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    out.chi_ricci[i][j]=chi.value*geometry.ricci[i][j]+T(.5)*out.hessian_chi[i][j]
        +T(.5)*g.g[i][j]*out.laplacian_chi-T(.25)*chi.d[i]*out.gradient_ratio[j]
        -T(.75)*g.g[i][j]*out.gradient_chi_squared_over_chi;
    valid=valid&&Kokkos::isfinite(out.hessian_chi[i][j])&&Kokkos::isfinite(out.chi_ricci[i][j]);
  }
  out.valid=valid;out.status=valid?RegularPenroseStatus::valid:RegularPenroseStatus::nonfinite_result;
  return out;
}

// The same scaled connection identity applies to an independent covector,
// including Z_i. derivative[i][j]=partial_i z_j need not be symmetric.
// This avoids chi*(dz-Gamma_bar*z) with a separately constructed bar metric.
template<typename T,typename ChiJet> KOKKOS_INLINE_FUNCTION
RegularPenroseCovector<T> RegularPenroseCovectorDerivative(const MetricJet<T> &g,
    const SpatialGeometry<T> &geometry,const ChiJet &chi,const T z[3],
    const T derivative[3][3]){
  RegularPenroseCovector<T> out{};
  if(!geometry.valid){out.status=RegularPenroseStatus::invalid_metric;return out;}
  if(!RegularPenroseFiniteJet(chi)){out.status=RegularPenroseStatus::nonfinite_input;return out;}
  if(!(chi.value>T(0))){out.status=RegularPenroseStatus::nonpositive_chi;return out;}
  T cross{},scaled_divergence_g{};
  for(int i=0;i<3;++i){if(!Kokkos::isfinite(z[i]))return out;
    for(int j=0;j<3;++j){if(!Kokkos::isfinite(derivative[i][j]))return out;
      cross+=geometry.inverse[i][j]*chi.d[i]*z[j];}}
  bool valid=Kokkos::isfinite(cross);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    T scaled_covariant=chi.value*derivative[i][j];
    for(int k=0;k<3;++k)scaled_covariant-=geometry.connection[k][i][j]*(chi.value*z[k]);
    out.chi_derivative[i][j]=scaled_covariant+T(.5)*(chi.d[i]*z[j]+chi.d[j]*z[i]-g.g[i][j]*cross);
    scaled_divergence_g+=geometry.inverse[i][j]*scaled_covariant;
    valid=valid&&Kokkos::isfinite(out.chi_derivative[i][j]);
  }
  out.divergence=scaled_divergence_g-T(.5)*cross;
  valid=valid&&Kokkos::isfinite(out.divergence);
  out.valid=valid;out.status=valid?RegularPenroseStatus::valid:RegularPenroseStatus::nonfinite_result;
  return out;
}

// chi Hess_bar(f)=chi Hess_g(f)+.5(chi_i f_j+chi_j f_i-g_ij gradchi.gradf),
// Delta_bar(f)=chi Delta_g(f)-.5 gradchi.gradf. The scalar value need not be
// positive: f may be Omega, including actual Omega=0, or another scalar.
template<typename T,typename ChiJet,typename FJet> KOKKOS_INLINE_FUNCTION
RegularPenroseScalar<T> RegularPenroseHessian(const MetricJet<T> &g,
    const SpatialGeometry<T> &geometry,const ChiJet &chi,const FJet &f){
  RegularPenroseScalar<T> out{};
  if(!RegularPenroseFiniteJet(f))return out;
  const auto derivative=RegularPenroseCovectorDerivative(g,geometry,chi,f.d,f.dd);
  out.valid=derivative.valid;out.status=derivative.status;out.laplacian=derivative.divergence;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.chi_hessian[i][j]=derivative.chi_derivative[i][j];
  return out;
}

}}  // namespace z4c::hyperboloidal
#endif
