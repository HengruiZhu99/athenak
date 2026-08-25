//========================================================================================
//! \file reference_gauge_baseline.hpp
//! \brief Analytic reference-only baseline for regular gauge-driver variables.
//========================================================================================
#ifndef REF_GH_REFERENCE_GAUGE_BASELINE_HPP_
#define REF_GH_REFERENCE_GAUGE_BASELINE_HPP_

#include "athena.hpp"
#include "ref_gh/gauge_driver.hpp"
#include "ref_gh/reference_cache.hpp"

namespace ref_gh {

struct ReferenceGaugeBaseline {
  Real hhat[4];        // NOLINT(runtime/arrays)
  Real theta[4];       // NOLINT(runtime/arrays)
  Real d_hhat[4][4];   // NOLINT(runtime/arrays)
  bool valid;
};

template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real ReferenceCoframeDerivative(const Reference &reference, const int p,
                                const int A, const int a) {
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

// d_t Omega_{A lambda}^B for
// Omega_{A lambda}^B=(d_lambda e_A^a) theta^B_a.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
Real ReferenceDtFrameMotion(const Reference &reference, const int A,
                            const int lambda, const int B) {
  Real derivative = 0.0;
  for (int a = 0; a < 4; ++a) {
    derivative += ReferenceDDFrame(reference, 0, lambda, A, a)
                  *ReferenceCoframe(reference, B, a)
                  + ReferenceDFrame(reference, lambda, A, a)
                    *ReferenceCoframeDerivative(reference, 0, B, a);
  }
  return derivative;
}

// Construct the ordinary GH source of the reference metric,
//
//   Href_a = -gbar_ab gbar^{cd} barGamma^b_cd,
//
// its analytic coordinate derivatives, and the stationary improved-driver
// theta state.  This object depends only on the reference two-jet.  It is a
// change of evolved variables, not a change of physical gauge: production
// equations reconstruct Hhat=delta_Hhat+Href and theta=delta_theta+theta_ref.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
ReferenceGaugeBaseline ComputeReferenceGaugeBaseline(
    const Reference &reference) {
  ReferenceGaugeBaseline result{};
  result.valid = false;
  Real metric[4][4] = {};       // NOLINT(runtime/arrays)
  Real inverse[4][4] = {};      // NOLINT(runtime/arrays)
  Real d_metric[4][4][4] = {};  // NOLINT(runtime/arrays)
  Real d_inverse[4][4][4] = {}; // NOLINT(runtime/arrays)
  Real d_coframe[4][4][4] = {}; // NOLINT(runtime/arrays)
  for (int p = 0; p < 4; ++p) {
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        for (int B = 0; B < 4; ++B) {
          for (int b = 0; b < 4; ++b) {
            d_coframe[p][A][a] -= ReferenceCoframe(reference, B, a)
                *ReferenceDFrame(reference, p, B, b)
                *ReferenceCoframe(reference, A, b);
          }
        }
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int A = 0; A < 4; ++A) {
        const Real sign = (A == 0) ? -1.0 : 1.0;
        metric[a][b] += sign*ReferenceCoframe(reference, A, a)
                         *ReferenceCoframe(reference, A, b);
        inverse[a][b] += sign*ReferenceFrame(reference, A, a)
                          *ReferenceFrame(reference, A, b);
        for (int p = 0; p < 4; ++p) {
          d_metric[p][a][b] += sign*(
              d_coframe[p][A][a]*ReferenceCoframe(reference, A, b)
              + ReferenceCoframe(reference, A, a)*d_coframe[p][A][b]);
        }
      }
    }
  }
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] -=
                inverse[a][c]*inverse[b][d]*d_metric[p][c][d];
          }
        }
      }
    }
  }

  Real h_upper[4] = {};       // NOLINT(runtime/arrays)
  Real h_lower[4] = {};       // NOLINT(runtime/arrays)
  Real d_h_upper[4][4] = {};  // NOLINT(runtime/arrays)
  Real d_h_lower[4][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        h_upper[a] -= inverse[b][c]
                      *ReferenceChristoffel(reference, a, b, c);
        for (int p = 0; p < 4; ++p) {
          d_h_upper[p][a] -=
              d_inverse[p][b][c]*ReferenceChristoffel(reference, a, b, c)
              + inverse[b][c]
                    *ReferenceDChristoffel(reference, p, a, b, c);
        }
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      h_lower[a] += metric[a][b]*h_upper[b];
      for (int p = 0; p < 4; ++p) {
        d_h_lower[p][a] += d_metric[p][a][b]*h_upper[b]
                           + metric[a][b]*d_h_upper[p][b];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      result.hhat[A] += ReferenceFrame(reference, A, a)*h_lower[a];
      for (int p = 0; p < 4; ++p) {
        result.d_hhat[p][A] +=
            ReferenceDFrame(reference, p, A, a)*h_lower[a]
            + ReferenceFrame(reference, A, a)*d_h_lower[p][a];
      }
    }
  }

  if (!(inverse[0][0] < 0.0)) return result;
  const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real shift[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) {
    shift[p] = lapse*lapse*inverse[0][p + 1];
  }
  for (int A = 0; A < 4; ++A) {
    for (int p = 0; p < 3; ++p) {
      result.theta[A] -= shift[p]*result.d_hhat[p + 1][A];
    }
    for (int B = 0; B < 4; ++B) {
      Real frame_motion = ReferenceFrameMotion(reference, A, 0, B);
      for (int p = 0; p < 3; ++p) {
        frame_motion -= shift[p]
            *ReferenceFrameMotion(reference, A, p + 1, B);
      }
      result.theta[A] -= frame_motion*result.hhat[B];
    }
  }
  result.valid = Kokkos::isfinite(lapse);
  for (int A = 0; A < 4; ++A) {
    result.valid = result.valid && Kokkos::isfinite(result.hhat[A])
                   && Kokkos::isfinite(result.theta[A]);
    for (int p = 0; p < 4; ++p) {
      result.valid = result.valid && Kokkos::isfinite(result.d_hhat[p][A]);
    }
  }
  return result;
}

}  // namespace ref_gh

#endif  // REF_GH_REFERENCE_GAUGE_BASELINE_HPP_
