//========================================================================================
//! \file reference_residual.hpp
//! \brief Cancellation-free arithmetic for reference-relative Ref-GH quantities.
//========================================================================================
#ifndef REF_GH_REFERENCE_RESIDUAL_HPP_
#define REF_GH_REFERENCE_RESIDUAL_HPP_

#include "athena.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

// Carry the reference value, physical value, and their algebraically evaluated
// difference together.  `delta` is authoritative: it is not recomputed as
// physical-reference.  The redundant physical member keeps later nonlinear
// residual identities from reconstructing a perturbed full value by adding a
// small residual to a singular reference value.
struct ReferenceResidualValue {
  Real reference;
  Real physical;
  Real delta;
};

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue MakeReferenceResidual(const Real reference,
                                             const Real physical,
                                             const Real delta) {
  return {reference, physical, delta};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue ReferenceResidualConstant(const Real value) {
  return {value, value, 0.0};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator+(const ReferenceResidualValue &x,
                                 const ReferenceResidualValue &y) {
  return {x.reference + y.reference, x.physical + y.physical,
          x.delta + y.delta};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator-(const ReferenceResidualValue &x,
                                 const ReferenceResidualValue &y) {
  return {x.reference - y.reference, x.physical - y.physical,
          x.delta - y.delta};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator-(const ReferenceResidualValue &x) {
  return {-x.reference, -x.physical, -x.delta};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator*(const ReferenceResidualValue &x,
                                 const ReferenceResidualValue &y) {
  // (xy)-(xbar ybar) = (x-xbar)y + xbar(y-ybar).
  return {x.reference*y.reference, x.physical*y.physical,
          x.delta*y.physical + x.reference*y.delta};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator*(const Real coefficient,
                                 const ReferenceResidualValue &x) {
  return {coefficient*x.reference, coefficient*x.physical,
          coefficient*x.delta};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator*(const ReferenceResidualValue &x,
                                 const Real coefficient) {
  return coefficient*x;
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator/(const ReferenceResidualValue &x,
                                 const ReferenceResidualValue &y) {
  // x/y-xbar/ybar=(dx*ybar-xbar*dy)/(ybar*y).  This is exact and returns
  // bitwise zero for dx=dy=0.
  return {x.reference/y.reference, x.physical/y.physical,
          (x.delta*y.reference - x.reference*y.delta)
              /(y.reference*y.physical)};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator/(const ReferenceResidualValue &x,
                                 const Real denominator) {
  return {x.reference/denominator, x.physical/denominator,
          x.delta/denominator};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue operator/(const Real numerator,
                                 const ReferenceResidualValue &x) {
  return ReferenceResidualConstant(numerator)/x;
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue ReferenceResidualSqrt(
    const ReferenceResidualValue &x) {
  const Real reference_root = Kokkos::sqrt(x.reference);
  const Real physical_root = Kokkos::sqrt(x.physical);
  return {reference_root, physical_root,
          x.delta/(reference_root + physical_root)};
}

KOKKOS_INLINE_FUNCTION
ReferenceResidualValue ReferenceResidualCubeRoot(
    const ReferenceResidualValue &x) {
  const Real one_third = 1.0/3.0;
  const Real reference_root = Kokkos::pow(x.reference, one_third);
  const Real physical_root = Kokkos::pow(x.physical, one_third);
  return {reference_root, physical_root,
          x.delta/(physical_root*physical_root
                   + physical_root*reference_root
                   + reference_root*reference_root)};
}

struct ReferenceRelativeCoordinateData {
  Real reference_metric[4][4];          // NOLINT(runtime/arrays)
  Real reference_inverse[4][4];         // NOLINT(runtime/arrays)
  Real reference_d_metric[4][4][4];     // NOLINT(runtime/arrays)
  Real delta_metric[4][4];              // NOLINT(runtime/arrays)
  Real delta_d_metric[4][4][4];         // NOLINT(runtime/arrays)
  bool valid;
};

template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real ResidualReferenceCoframeDerivative(const Reference &reference,
                                        const int p, const int A,
                                        const int a) {
  Real derivative = 0.0;
  for (int B = 0; B < 4; ++B) {
    for (int b = 0; b < 4; ++b) {
      derivative -= ReferenceCoframe(reference, B, a)
                    *ReferenceDFrame(reference, p, B, b)
                    *ReferenceCoframe(reference, A, b);
    }
  }
  return derivative;
}

// Build g-gbar and dg-dgbar directly from the regular frame fields.  The
// reference has Psi_AB=eta_AB and dPsi_AB=Pi_AB=Phi_IAB=0.  Consequently the
// matched state produces exact binary64 zero without subtracting coordinate
// metrics whose components diverge toward the puncture.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool BuildReferenceRelativeCoordinateData(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const Real metric[4][4],
    const Real d_metric[4][4][4], const CoordinateGhGeometry &geometry,
    const Reference &reference, ReferenceRelativeCoordinateData &result) {
  result.valid = false;
  Real d_coframe[4][4][4];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 4; ++p) {
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        d_coframe[p][A][a] =
            ResidualReferenceCoframeDerivative(reference, p, A, a);
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      result.reference_metric[a][b] = 0.0;
      result.reference_inverse[a][b] = 0.0;
      result.delta_metric[a][b] = 0.0;
      for (int A = 0; A < 4; ++A) {
        const Real sign = (A == 0) ? -1.0 : 1.0;
        result.reference_metric[a][b] += sign
            *ReferenceCoframe(reference, A, a)
            *ReferenceCoframe(reference, A, b);
        result.reference_inverse[a][b] += sign
            *ReferenceFrame(reference, A, a)
            *ReferenceFrame(reference, A, b);
        for (int B = 0; B < 4; ++B) {
          const Real eta = (A == B) ? sign : 0.0;
          result.delta_metric[a][b] +=
              ReferenceCoframe(reference, A, a)
              *ReferenceCoframe(reference, B, b)*(psi[A][B] - eta);
        }
      }
      for (int p = 0; p < 4; ++p) {
        result.reference_d_metric[p][a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          const Real sign = (A == 0) ? -1.0 : 1.0;
          result.reference_d_metric[p][a][b] += sign*(
              d_coframe[p][A][a]*ReferenceCoframe(reference, A, b)
              + ReferenceCoframe(reference, A, a)*d_coframe[p][A][b]);
        }
      }
    }
  }

  Real d_delta_psi[4][4][4];  // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int p = 0; p < 3; ++p) {
        d_delta_psi[p + 1][A][B] = 0.0;
        for (int I = 0; I < 3; ++I) {
          d_delta_psi[p + 1][A][B] +=
              ReferenceSpatialCoframe(reference, I, p)*phi[I][A][B];
        }
      }
      d_delta_psi[0][A][B] = -geometry.lapse*pi[A][B];
      for (int p = 0; p < 3; ++p) {
        d_delta_psi[0][A][B] +=
            geometry.shift[p]*d_delta_psi[p + 1][A][B];
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    Real frame_corrected_delta[4][4];  // NOLINT(runtime/arrays)
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        frame_corrected_delta[A][B] = d_delta_psi[p][A][B];
        for (int a = 0; a < 4; ++a) {
          for (int b = 0; b < 4; ++b) {
            frame_corrected_delta[A][B] -=
                (ReferenceDFrame(reference, p, A, a)
                   *ReferenceFrame(reference, B, b)
                 + ReferenceFrame(reference, A, a)
                   *ReferenceDFrame(reference, p, B, b))
                    *result.delta_metric[a][b];
          }
        }
      }
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        result.delta_d_metric[p][a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            result.delta_d_metric[p][a][b] +=
                ReferenceCoframe(reference, A, a)
                *ReferenceCoframe(reference, B, b)
                *frame_corrected_delta[A][B];
          }
        }
      }
    }
  }

  result.valid = true;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      result.valid = result.valid
          && Kokkos::isfinite(result.reference_metric[a][b])
          && Kokkos::isfinite(result.reference_inverse[a][b])
          && Kokkos::isfinite(result.delta_metric[a][b]);
      for (int p = 0; p < 4; ++p) {
        result.valid = result.valid
            && Kokkos::isfinite(result.reference_d_metric[p][a][b])
            && Kokkos::isfinite(result.delta_d_metric[p][a][b]);
      }
    }
  }
  // `metric` and `d_metric` are intentionally accepted as the independently
  // reconstructed physical members used by downstream residual triples.
  (void)metric;
  (void)d_metric;
  return result.valid;
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_RESIDUAL_HPP_
