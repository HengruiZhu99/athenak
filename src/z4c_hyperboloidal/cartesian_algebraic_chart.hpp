// Ported from scri-ko01 src/z4c/hyperboloidal/cartesian_algebraic_chart.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_ALGEBRAIC_CHART_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_ALGEBRAIC_CHART_HPP_
#include <array>
#include "z4c_hyperboloidal/mapped_geometry.hpp"
namespace z4c_hyperboloidal {
// Five independent Cartesian metric components and five curvature components.
// Slot 6 (gzz-1) and 12 (Azz) are dependent coordinates, never RK variables.
// This is a local chart on SPD determinant-one metrics. It neither reconstructs
// Lambda from the metric nor imposes any differential Einstein constraint.
using CartesianChartJets=std::array<mapped::Jet,31>;
KOKKOS_INLINE_FUNCTION bool TryExpandCartesianChart(CartesianChartJets&v) {
  using J=mapped::Jet;
  const J a=v[1]+J(1),b=v[2],c=v[3],d=v[4]+J(1),e=v[5];
  const J minor=a*d-b*b;
  if(!(a.f>0 && minor.f>0))return false;
  const J f=(J(1)-J(2)*b*c*e+a*e*e+d*c*c)/minor;
  v[6]=f-J(1);
  v[12]=(J(0)-(d*f-e*e)*v[7]-J(2)*(c*e-b*f)*v[8]
      -J(2)*(b*e-c*d)*v[9]-(a*f-c*c)*v[10]-J(2)*(b*c-a*e)*v[11])/minor;
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
