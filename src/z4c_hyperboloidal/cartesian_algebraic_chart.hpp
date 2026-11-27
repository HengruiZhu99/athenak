// Ported from scri-ko01 src/z4c/hyperboloidal/cartesian_algebraic_chart.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_ALGEBRAIC_CHART_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_ALGEBRAIC_CHART_HPP_
#include <array>
#include <type_traits>
#include "z4c_hyperboloidal/conformal_constraints.hpp"
#include "z4c_hyperboloidal/fields.hpp"
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
struct CartesianChartMetric {
  J a, b, c, d, e, minor, f;

  KOKKOS_INLINE_FUNCTION bool Complete() {
    minor = a * d - b * b;
    if (!(ChartValue(a) > 0 && ChartValue(minor) > 0)) return false;
    f = ChartQuotient(J(1) - J(2) * b * c * e + a * e * e + d * c * c, minor);
    return true;
  }

  KOKKOS_INLINE_FUNCTION J TraceFreeZZ(const J &xx, const J &xy, const J &xz, const J &yy,
                                       const J &yz) const {
    return ChartQuotient(J(0) - (d * f - e * e) * xx - J(2) * (c * e - b * f) * xy -
                             J(2) * (b * e - c * d) * xz - (a * f - c * c) * yy -
                             J(2) * (b * c - a * e) * yz,
                         minor);
  }
};

template <class J>
KOKKOS_INLINE_FUNCTION bool TryExpandCartesianChart(std::array<J, 31> &v) {
  CartesianChartMetric<J> metric{
      v[1] + J(1), v[2], v[3], v[4] + J(1), v[5], {}, {},
  };
  if (!metric.Complete()) return false;
  v[6] = metric.f - J(1);
  v[12] = metric.TraceFreeZZ(v[7], v[8], v[9], v[10], v[11]);
  return true;
}

// Expand point values with constant input jets and return the inverse physical chart
// metric. The implicit metric component carries the derivative overflow check.
KOKKOS_INLINE_FUNCTION bool ProjectConstantChart(
    std::array<Real, Fields::count>& fields, Real inverse[3][3]) {
  using J = mapped::Jet;
  CartesianChartMetric<J> jets{
      J(fields[1] + 1), J(fields[2]), J(fields[3]), J(fields[4] + 1), J(fields[5]),
      {}, {},
  };
  if (!jets.Complete()) return false;
  // Every input is constant. Each component of the implicit metric's gradient
  // and Hessian therefore follows the same arithmetic, even on overflow.
  if (jets.f.d[0] != 0 || jets.f.h[0][0] != 0) return false;
  const CartesianChartMetric<Real> chart{jets.a.f, jets.b.f, jets.c.f, jets.d.f, jets.e.f,
                                         jets.minor.f, jets.f.f};
  fields[6] = chart.f - 1;
  fields[12] = chart.TraceFreeZZ(fields[7], fields[8], fields[9], fields[10], fields[11]);
  const Real g[3][3] = {
      {chart.a, chart.b, chart.c},
      {chart.b, chart.d, chart.e},
      {chart.c, chart.e, fields[6] + 1},
  };
  Real determinant;
  if (!InvertMetric(g, inverse, determinant)) return false;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (!Kokkos::isfinite(inverse[i][j])) return false;
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

// Covariant enforcement of det(gamma)=1 and gamma^ij A_ij=0 on six stored components each:
// gamma_ij -> det(gamma)^(-1/3) gamma_ij, then A_ij -> A_ij - gamma_ij gamma^kl A_kl / 3.
// Values only. violation returns det(gamma)-1 and gamma^kl A_kl before the projection.
KOKKOS_INLINE_FUNCTION bool ProjectAlgebraicConstraints(
    std::array<Real, Fields::count>& fields, Real inverse[3][3],
    Real violation[2] = nullptr) {
  constexpr int ti[6] = {0, 0, 0, 1, 1, 2}, tj[6] = {0, 1, 2, 1, 2, 2};
  Real g[3][3], determinant;
  for (int t = 0; t < 6; ++t)
    g[ti[t]][tj[t]] = g[tj[t]][ti[t]] = fields[1 + t] + (ti[t] == tj[t]);
  if (!(g[0][0] > 0 && g[0][0] * g[1][1] - g[0][1] * g[0][1] > 0)) return false;
  if (!InvertMetric(g, inverse, determinant) || !(determinant > 0)) return false;
  const Real scale = Kokkos::cbrt(1 / determinant);
  Real trace = 0;
  for (int t = 0; t < 6; ++t)
    trace += (ti[t] == tj[t] ? 1 : 2) * inverse[ti[t]][tj[t]] * fields[7 + t];
  if (violation) {
    violation[0] = determinant - 1;
    violation[1] = trace;
  }
  // gamma^kl A_kl of the unscaled metric times gamma_ij is invariant under the rescaling.
  for (int t = 0; t < 6; ++t) {
    fields[7 + t] -= g[ti[t]][tj[t]] * trace / 3;
    fields[1 + t] = scale * g[ti[t]][tj[t]] - (ti[t] == tj[t]);
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      inverse[i][j] /= scale;
      if (!Kokkos::isfinite(inverse[i][j])) return false;
    }
  return true;
}

// The algebraic closure of a stored point: expand the two dependent components, or
// project all twelve when every component is evolved.
KOKKOS_INLINE_FUNCTION bool CloseAlgebraicConstraints(
    std::array<Real, Fields::count>& fields, Real inverse[3][3], bool evolve_all,
    Real violation[2] = nullptr) {
  if (evolve_all) return ProjectAlgebraicConstraints(fields, inverse, violation);
  if (violation) violation[0] = violation[1] = 0;
  return ProjectConstantChart(fields, inverse);
}
} // namespace z4c_hyperboloidal
#endif
