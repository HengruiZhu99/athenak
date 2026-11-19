// Ported from scri-ko01 src/z4c/hyperboloidal/cartesian_inner_adjustment.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_INNER_ADJUSTMENT_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_CARTESIAN_INNER_ADJUSTMENT_HPP_
#include "z4c_hyperboloidal/conformal_rhs.hpp"

namespace z4c_hyperboloidal {
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
KOKKOS_INLINE_FUNCTION void CartesianInnerAdjustment(
    const Z4cJet<T>& u, const OmegaJet<T>& o, T weight, const SpatialGeometry<T>& gt,
    const SpatialGeometry<T>& gb, Z4cRHSParts<T>& parts) {
  T delta[3], z[3]{}, correction[3][3]{}, pole[3][3]{};
  T trace = 0, pole_trace = 0, constraint_trace = 0, divbeta = 0;
  for (int k = 0; k < 3; ++k) {
    delta[k] = u.lambda.value[k] - gt.contracted[k];
    divbeta += u.beta.d[k][k];
    for (int i = 0; i < 3; ++i) z[i] += T(.5) * u.metric.g[i][k] * delta[k];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      // With 2 Z_i = g_ik delta^k, derivatives of delta cancel between
      // evolved Ricci and the symmetrized covariant derivative of Z.
      T gradient = 0, conformal_connection = 0, penrose_connection = 0;
      for (int k = 0; k < 3; ++k) {
        gradient -= T(.5) * (u.metric.dg[i][j][k] + u.metric.dg[j][i][k]) * delta[k];
        conformal_connection += gt.connection[k][i][j] * z[k];
        penrose_connection += gb.connection[k][i][j] * z[k];
      }
      correction[i][j] =
          u.alpha.value * u.chi.value * (gradient + 2 * penrose_connection);
      constraint_trace += gt.inverse[i][j] * (gradient + 2 * conformal_connection);
      pole[i][j] = -2 * u.alpha.value * u.chi.value * (z[i] * o.gradient[j] + z[j] * o.gradient[i]);
      trace += gt.inverse[i][j] * correction[i][j];
      pole_trace += gt.inverse[i][j] * pole[i][j];
    }
  parts.regular.theta +=
      weight * o.omega * u.alpha.value * u.chi.value * constraint_trace / 2;
  for (int i = 0; i < 3; ++i) {
    parts.regular.lambda[i] -= weight * T(2. / 3.) * delta[i] * divbeta;
    parts.pole.lambda[i] += weight * T(2. / 3.) * u.alpha.value *
                            (u.trace.value + 2 * u.theta.value) * delta[i];
    for (int j = 0; j < 3; ++j) {
      parts.regular.a[i][j] += weight * (correction[i][j] - u.metric.g[i][j] * trace / 3);
      parts.pole.a[i][j] += weight * (pole[i][j] - u.metric.g[i][j] * pole_trace / 3);
    }
  }
}
}  // namespace z4c_hyperboloidal
#endif
