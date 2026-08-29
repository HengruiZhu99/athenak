//========================================================================================
//! \file residual_gauge_source.hpp
//! \brief Ordinary-GH Einstein increment assembled from regular residuals.
//========================================================================================
#ifndef REF_GH_RESIDUAL_GAUGE_SOURCE_HPP_
#define REF_GH_RESIDUAL_GAUGE_SOURCE_HPP_

#include "athena.hpp"
#include "ref_gh/reference_residual.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct OrdinaryGaugeResidualDiagnostics {
  Real j[4];        // NOLINT(runtime/arrays)
  Real d_j[4][4];   // NOLINT(runtime/arrays)
  Real delta_base[4];       // NOLINT(runtime/arrays)
  Real d_delta_base[4][4];  // NOLINT(runtime/arrays)
};

// Oracle/general implementation of
//
//   J_a = E^A_a delta_H_A - [B_a(g;gbar)-B_a(gbar;gbar)].
//
// Every bracket is evaluated with ReferenceResidualValue identities.  This
// function never reconstructs raw Hhat or subtracts raw Href.  The analytic
// radial-q production dispatch must use a compact generated specialization
// rather than invoking recursive ReferenceDChristoffel accessors through this
// template; until that specialization exists this function is oracle-only for
// AnalyticRadialQPoint.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool AddOrdinaryGaugeResidualPartialWaveSource(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const Real metric[4][4],
    const Real d_metric[4][4][4], const Reference &reference,
    const CoordinateGhGeometry &geometry, const Real delta_hhat[4],
    const Real d_delta_hhat[4][4], const Real gamma0, Real source[4][4],
    OrdinaryGaugeResidualDiagnostics *diagnostics = nullptr) {
  ReferenceRelativeCoordinateData relative;
  if (!BuildReferenceRelativeCoordinateData(
          psi, pi, phi, metric, d_metric, geometry, reference, relative)) {
    return false;
  }

  ReferenceResidualValue g[4][4];         // NOLINT(runtime/arrays)
  ReferenceResidualValue dg[4][4][4];     // NOLINT(runtime/arrays)
  ReferenceResidualValue inverse[4][4];   // NOLINT(runtime/arrays)
  ReferenceResidualValue d_inverse[4][4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      g[a][b] = MakeReferenceResidual(
          relative.reference_metric[a][b], metric[a][b],
          relative.delta_metric[a][b]);
      for (int p = 0; p < 4; ++p) {
        dg[p][a][b] = MakeReferenceResidual(
            relative.reference_d_metric[p][a][b], d_metric[p][a][b],
            relative.delta_d_metric[p][a][b]);
      }
      Real delta_inverse = 0.0;
      for (int c = 0; c < 4; ++c) {
        for (int d = 0; d < 4; ++d) {
          delta_inverse -= geometry.inverse_metric[a][c]
              *relative.delta_metric[c][d]*relative.reference_inverse[d][b];
        }
      }
      inverse[a][b] = MakeReferenceResidual(
          relative.reference_inverse[a][b], geometry.inverse_metric[a][b],
          delta_inverse);
    }
  }
  for (int p = 0; p < 4; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_inverse[p][a][b] = ReferenceResidualConstant(0.0);
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] = d_inverse[p][a][b]
                - inverse[a][c]*inverse[b][d]*dg[p][c][d];
          }
        }
      }
    }
  }

  ReferenceResidualValue base_upper[4];       // NOLINT(runtime/arrays)
  ReferenceResidualValue d_base_upper[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    base_upper[a] = ReferenceResidualConstant(0.0);
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        base_upper[a] = base_upper[a]
            - inverse[b][c]*ReferenceChristoffel(reference, a, b, c);
      }
    }
    for (int p = 0; p < 4; ++p) {
      d_base_upper[p][a] = ReferenceResidualConstant(0.0);
      for (int b = 0; b < 4; ++b) {
        for (int c = 0; c < 4; ++c) {
          d_base_upper[p][a] = d_base_upper[p][a]
              - d_inverse[p][b][c]*ReferenceChristoffel(reference, a, b, c)
              - inverse[b][c]*ReferenceDChristoffel(reference, p, a, b, c);
        }
      }
    }
  }

  ReferenceResidualValue base_lower[4];       // NOLINT(runtime/arrays)
  ReferenceResidualValue d_base_lower[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    base_lower[a] = ReferenceResidualConstant(0.0);
    for (int p = 0; p < 4; ++p) {
      d_base_lower[p][a] = ReferenceResidualConstant(0.0);
    }
    for (int b = 0; b < 4; ++b) {
      base_lower[a] = base_lower[a] + g[a][b]*base_upper[b];
      for (int p = 0; p < 4; ++p) {
        d_base_lower[p][a] = d_base_lower[p][a]
            + dg[p][a][b]*base_upper[b] + g[a][b]*d_base_upper[p][b];
      }
    }
  }

  Real j[4] = {};       // NOLINT(runtime/arrays)
  Real d_j[4][4] = {};  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    j[a] = -base_lower[a].delta;
    for (int A = 0; A < 4; ++A) {
      j[a] += ReferenceCoframe(reference, A, a)*delta_hhat[A];
    }
    for (int p = 0; p < 4; ++p) {
      d_j[p][a] = -d_base_lower[p][a].delta;
      for (int A = 0; A < 4; ++A) {
        d_j[p][a] +=
            ResidualReferenceCoframeDerivative(reference, p, A, a)
                *delta_hhat[A]
            + ReferenceCoframe(reference, A, a)*d_delta_hhat[p][A];
      }
    }
    if (diagnostics != nullptr) {
      diagnostics->j[a] = j[a];
      diagnostics->delta_base[a] = base_lower[a].delta;
      for (int p = 0; p < 4; ++p) {
        diagnostics->d_j[p][a] = d_j[p][a];
        diagnostics->d_delta_base[p][a] = d_base_lower[p][a].delta;
      }
    }
  }

  Real coordinate_extra[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      Real nabla_ab = d_j[a][b];
      Real nabla_ba = d_j[b][a];
      for (int c = 0; c < 4; ++c) {
        nabla_ab -= geometry.christoffel[c][a][b]*j[c];
        nabla_ba -= geometry.christoffel[c][b][a]*j[c];
      }
      coordinate_extra[a][b] = -nabla_ab - nabla_ba;
      for (int c = 0; c < 4; ++c) {
        const Real projector = ((c == a) ? geometry.normal_lower[b] : 0.0)
                               + ((c == b) ? geometry.normal_lower[a] : 0.0)
                               - metric[a][b]*geometry.normal_upper[c];
        coordinate_extra[a][b] += gamma0*projector*j[c];
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          source[A][B] += ReferenceFrame(reference, A, a)
                          *ReferenceFrame(reference, B, b)
                          *coordinate_extra[a][b];
        }
      }
    }
  }
  return true;
}

}  // namespace ref_gh

#endif  // REF_GH_RESIDUAL_GAUGE_SOURCE_HPP_
