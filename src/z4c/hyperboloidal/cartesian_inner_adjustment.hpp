// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_INNER_ADJUSTMENT_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_INNER_ADJUSTMENT_HPP_
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c::hyperboloidal {
// The tensor conformal system evolves covariant derivatives of Z_i. AthenaK's
// Cartesian Z4c instead inserts the evolved connection only in the derivative
// term of conformal Ricci. They agree on the connection constraint, but a
// constant Omega does not make their off-constraint equations identical.
//
// This explicit experimental RHS blend replaces those terms on the inner
// plateau and vanishes before the outer gauge/scri region. It is not a change
// of the physical spatial map. Weight multiplies complete local source terms;
// no spatial derivatives of the blend enter. Split pole numerators avoid an
// artificial Omega floor. The caller must leave the legacy formulation alone.
template <class T>
KOKKOS_INLINE_FUNCTION bool CartesianInnerAdjustment(const Z4cJet<T>& u, const OmegaJet<T>& o,
                                                     T weight, Z4cRHSParts<T>& parts) {
  if (weight == 0)
    return true;
  const auto gt = Geometry(u.metric);
  const auto gb = Geometry(PenroseMetric(u.metric, u.chi));
  if (!gt.valid || !gb.valid || !parts.valid)
    return false;
  T z[3]{}, dz[3][3]{}, correction[3][3]{}, pole[3][3]{};
  T trace = 0, pole_trace = 0, ricci_trace = 0, divz = 0, divbeta = 0;
  for (int i = 0; i < 3; ++i) {
    divbeta += u.beta.d[i][i];
    for (int j = 0; j < 3; ++j) {
      const T delta = u.lambda.value[j] - gt.contracted[j];
      z[i] += T(.5) * u.metric.g[i][j] * delta;
      for (int d = 0; d < 3; ++d)
        dz[d][i] += T(.5) * (u.metric.dg[d][i][j] * delta +
                             u.metric.g[i][j] * (u.lambda.d[d][j] - gt.dcontracted[d][j]));
    }
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      T evolved_ricci = 0, covariant = dz[i][j] + dz[j][i], tilde_dz = dz[i][j];
      for (int k = 0; k < 3; ++k) {
        evolved_ricci += T(.5) * (u.metric.g[k][i] * (u.lambda.d[j][k] - gt.dcontracted[j][k]) +
                                  u.metric.g[k][j] * (u.lambda.d[i][k] - gt.dcontracted[i][k]));
        covariant -= 2 * gb.connection[k][i][j] * z[k];
        tilde_dz -= gt.connection[k][i][j] * z[k];
      }
      correction[i][j] = u.alpha.value * u.chi.value * (evolved_ricci - covariant);
      pole[i][j] = -2 * u.alpha.value * u.chi.value * (z[i] * o.gradient[j] + z[j] * o.gradient[i]);
      trace += gt.inverse[i][j] * correction[i][j];
      pole_trace += gt.inverse[i][j] * pole[i][j];
      ricci_trace += gt.inverse[i][j] * evolved_ricci;
      divz += gt.inverse[i][j] * tilde_dz;
    }
  parts.regular.theta +=
      weight * o.omega * u.alpha.value * u.chi.value * (ricci_trace - 2 * divz) / 2;
  for (int i = 0; i < 3; ++i) {
    const T delta = u.lambda.value[i] - gt.contracted[i];
    parts.regular.lambda[i] -= weight * T(2. / 3.) * delta * divbeta;
    parts.pole.lambda[i] +=
        weight * T(2. / 3.) * u.alpha.value * (u.trace.value + 2 * u.theta.value) * delta;
    for (int j = 0; j < 3; ++j) {
      parts.regular.a[i][j] += weight * (correction[i][j] - u.metric.g[i][j] * trace / 3);
      parts.pole.a[i][j] += weight * (pole[i][j] - u.metric.g[i][j] * pole_trace / 3);
    }
  }
  return true;
}
}  // namespace z4c::hyperboloidal
#endif
