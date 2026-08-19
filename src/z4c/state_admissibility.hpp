//========================================================================================
//! \file state_admissibility.hpp
//! \brief Pure, device-callable Z4c state-admissibility predicates.
//
// The Z4c algebraic projection is a projection onto an already admissible
// conformal state.  It must never turn a nonpositive/NaN determinant into a
// synthetic valid state.  Keep this header independent of MeshBlock storage so
// unit tests and every state writer use precisely the same predicate.
//========================================================================================

#ifndef Z4C_STATE_ADMISSIBILITY_HPP_
#define Z4C_STATE_ADMISSIBILITY_HPP_

#include <Kokkos_Core.hpp>

#include "athena.hpp"

namespace z4c {

enum class Z4cStateFailureReason : int {
  valid = 0,
  nonfinite_component,
  nonpositive_chi,
  nonpositive_lapse,
  nonfinite_determinant,
  nonpositive_metric_pivot_0,
  nonpositive_metric_pivot_1,
  nonpositive_metric_pivot_2,
};

KOKKOS_INLINE_FUNCTION
const char *Z4cStateFailureReasonName(const Z4cStateFailureReason reason) {
  switch (reason) {
    case Z4cStateFailureReason::valid: return "valid";
    case Z4cStateFailureReason::nonfinite_component: return "nonfinite_component";
    case Z4cStateFailureReason::nonpositive_chi: return "nonpositive_chi";
    case Z4cStateFailureReason::nonpositive_lapse: return "nonpositive_lapse";
    case Z4cStateFailureReason::nonfinite_determinant: return "nonfinite_determinant";
    case Z4cStateFailureReason::nonpositive_metric_pivot_0:
      return "nonpositive_metric_pivot_0";
    case Z4cStateFailureReason::nonpositive_metric_pivot_1:
      return "nonpositive_metric_pivot_1";
    case Z4cStateFailureReason::nonpositive_metric_pivot_2:
      return "nonpositive_metric_pivot_2";
  }
  return "unknown";
}

struct Z4cMetricAdmissibility {
  Real determinant = 0.0;
  Real pivot0 = 0.0;
  Real pivot1 = 0.0;
  Real pivot2 = 0.0;
  Z4cStateFailureReason reason = Z4cStateFailureReason::valid;
};

// Sylvester/Cholesky test for a symmetric 3x3 metric.  The pivots are exposed
// so an invalid-state record can say exactly which condition failed.
KOKKOS_INLINE_FUNCTION
Z4cMetricAdmissibility EvaluateConformalMetric(const Real gxx, const Real gxy,
                                                const Real gxz, const Real gyy,
                                                const Real gyz, const Real gzz) {
  Z4cMetricAdmissibility result;
  result.pivot0 = gxx;
  const Real minor2 = gxx * gyy - gxy * gxy;
  result.pivot1 = minor2;
  result.determinant =
      gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gxz * gyz) +
      gxz * (gxy * gyz - gxz * gyy);
  result.pivot2 = result.determinant;

  if (!Kokkos::isfinite(result.determinant)) {
    result.reason = Z4cStateFailureReason::nonfinite_determinant;
  } else if (!(result.pivot0 > 0.0)) {
    result.reason = Z4cStateFailureReason::nonpositive_metric_pivot_0;
  } else if (!(result.pivot1 > 0.0)) {
    result.reason = Z4cStateFailureReason::nonpositive_metric_pivot_1;
  } else if (!(result.pivot2 > 0.0)) {
    result.reason = Z4cStateFailureReason::nonpositive_metric_pivot_2;
  }
  return result;
}

struct Z4cStateAdmissibility {
  Z4cMetricAdmissibility metric;
  int first_nonfinite_component = -1;
  Z4cStateFailureReason reason = Z4cStateFailureReason::valid;
};

KOKKOS_INLINE_FUNCTION
unsigned long long SelectFirstZ4cFailureKey(const unsigned long long left,
                                            const unsigned long long right) {
  return left < right ? left : right;
}

// Project only an already SPD conformal metric.  The arrays use the canonical
// symmetric ordering (xx, xy, xz, yy, yz, zz).  Returning false leaves both
// inputs byte-for-byte unchanged, which is important when a caller turns the
// result into a failure record.
KOKKOS_INLINE_FUNCTION
bool ProjectAdmissibleConformalState(Real *metric, Real *atracefree) {
  const Z4cMetricAdmissibility before = EvaluateConformalMetric(
      metric[0], metric[1], metric[2], metric[3], metric[4], metric[5]);
  if (before.reason != Z4cStateFailureReason::valid) return false;

  const Real scale = Kokkos::cbrt(1.0 / before.determinant);
  if (!Kokkos::isfinite(scale) || !(scale > 0.0)) return false;
  for (int component = 0; component < 6; ++component) metric[component] *= scale;

  // det(metric) is one after the conformal projection.  The adjugate is
  // therefore its inverse (up to floating-point projection error).
  const Real inv_xx = metric[3] * metric[5] - metric[4] * metric[4];
  const Real inv_xy = metric[2] * metric[4] - metric[1] * metric[5];
  const Real inv_xz = metric[1] * metric[4] - metric[2] * metric[3];
  const Real inv_yy = metric[0] * metric[5] - metric[2] * metric[2];
  const Real inv_yz = metric[1] * metric[2] - metric[0] * metric[4];
  const Real inv_zz = metric[0] * metric[3] - metric[1] * metric[1];
  const Real trace = inv_xx * atracefree[0] + 2.0 * inv_xy * atracefree[1] +
                     2.0 * inv_xz * atracefree[2] + inv_yy * atracefree[3] +
                     2.0 * inv_yz * atracefree[4] + inv_zz * atracefree[5];
  for (int component = 0; component < 6; ++component) {
    atracefree[component] -= (1.0 / 3.0) * trace * metric[component];
  }
  return true;
}

// values is the canonical 25-component Z4c state in Z4c::u0 order.  Keep the
// indices explicit here to avoid a header dependency on Z4c itself.
KOKKOS_INLINE_FUNCTION
Z4cStateAdmissibility EvaluateZ4cState(const Real *values, const int nvalues,
                                       const bool require_positive_lapse = true) {
  Z4cStateAdmissibility result;
  for (int variable = 0; variable < nvalues; ++variable) {
    if (!Kokkos::isfinite(values[variable])) {
      result.first_nonfinite_component = variable;
      result.reason = Z4cStateFailureReason::nonfinite_component;
      return result;
    }
  }
  if (!(values[0] > 0.0)) {
    result.reason = Z4cStateFailureReason::nonpositive_chi;
    return result;
  }
  if (require_positive_lapse && !(values[18] > 0.0)) {
    result.reason = Z4cStateFailureReason::nonpositive_lapse;
    return result;
  }
  result.metric = EvaluateConformalMetric(values[1], values[2], values[3],
                                           values[4], values[5], values[6]);
  result.reason = result.metric.reason;
  return result;
}

}  // namespace z4c

#endif  // Z4C_STATE_ADMISSIBILITY_HPP_
