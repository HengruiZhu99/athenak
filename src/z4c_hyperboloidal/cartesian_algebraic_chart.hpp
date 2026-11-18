// Ported from scri-ko01 src/z4c/hyperboloidal/cartesian_algebraic_chart.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_ALGEBRAIC_CHART_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_ALGEBRAIC_CHART_HPP_
#include <array>
#include <type_traits>
#include "z4c_hyperboloidal/mapped_geometry.hpp"
namespace z4c_hyperboloidal {
// Five independent Cartesian metric components and five curvature components.
// Slot 6 (gzz-1) and 12 (Azz) are dependent coordinates, never RK variables.
// This is a local chart on SPD determinant-one metrics. It neither reconstructs
// Lambda from the metric nor imposes any differential Einstein constraint.
using CartesianChartJets=std::array<mapped::Jet,31>;
// Stage values and spatial derivative jets use the same algebraic chart.
// Preserve the jet quotient's multiply-by-reciprocal rounding on scalar inputs.
template <class T>
KOKKOS_INLINE_FUNCTION decltype(auto) ChartValue(T& v) {
  if constexpr (std::is_same_v<std::remove_cv_t<T>,mapped::Jet>) return (v.f);
  else return (v);
}
template <class T>
KOKKOS_INLINE_FUNCTION T ChartQuotient(const T& numerator,const T& denominator) {
  if constexpr (std::is_arithmetic_v<T>) return numerator*pow(denominator,Real(-1));
  else return numerator/denominator;
}
template <class J>
KOKKOS_INLINE_FUNCTION bool TryExpandCartesianChart(std::array<J,31>&v) {
  const J a=v[1]+J(1),b=v[2],c=v[3],d=v[4]+J(1),e=v[5];
  const J minor=a*d-b*b;
  if(!(ChartValue(a)>0 && ChartValue(minor)>0))return false;
  const J f=ChartQuotient(J(1)-J(2)*b*c*e+a*e*e+d*c*c,minor);
  v[6]=f-J(1);
  v[12]=ChartQuotient(J(0)-(d*f-e*e)*v[7]-J(2)*(c*e-b*f)*v[8]
      -J(2)*(b*e-c*d)*v[9]-(a*f-c*c)*v[10]-J(2)*(b*c-a*e)*v[11],minor);
  return true;
}
inline void ExpandCartesianChart(CartesianChartJets&v){
  if(!TryExpandCartesianChart(v))throw std::runtime_error("nonpositive Cartesian chart metric");
}
inline void ExpandCartesianChart(double*v) {
  CartesianChartJets j{};for(int f=0;f<31;++f)j[f].f=v[f];
  ExpandCartesianChart(j);v[6]=j[6].f;v[12]=j[12].f;
}
KOKKOS_INLINE_FUNCTION bool CartesianIndependent(int f){return f!=6 && f!=12 && f<28;}
} // namespace z4c_hyperboloidal
#endif
