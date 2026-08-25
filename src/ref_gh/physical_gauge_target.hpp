//========================================================================================
//! \file physical_gauge_target.hpp
//! \brief Ordinary-GH target for advective 1+log plus conformal Gamma-driver.
//========================================================================================
#ifndef REF_GH_PHYSICAL_GAUGE_TARGET_HPP_
#define REF_GH_PHYSICAL_GAUGE_TARGET_HPP_

#include "athena.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/standard_gh_source.hpp"

namespace ref_gh {

struct PhysicalGaugeTarget {
  Real coordinate[4];       // NOLINT(runtime/arrays)
  Real frame[4];            // NOLINT(runtime/arrays)
  Real conformal_gamma[3];  // NOLINT(runtime/arrays)
  Real d_alpha[3];          // NOLINT(runtime/arrays)
  Real d_shift[3][3];       // NOLINT(runtime/arrays)
  Real trace_k;
};

// This constructs the ordinary/reference-independent covector F_mu.  The
// normal component follows directly from true advective 1+log,
// n^mu F_mu=(2/alpha-1)K.  The spatial components follow from substituting
// dt beta^i=nu(tildeGamma^i-eta_beta Upsilon^i) into the exact 3+1 identity
// for -Gamma_i.  No reference quantity enters the physical target.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool ComputePhysicalGaugeTarget(
    const Real metric[4][4], const Real d_metric[4][4][4],
    const CoordinateGhGeometry &geometry, const Reference &reference,
    const Real upsilon[3], const Real nu, const Real eta_beta,
    PhysicalGaugeTarget &target) {
  Real inverse_spatial[3][3];  // NOLINT(runtime/arrays)
  Real spatial_determinant = 0.0;
  if (!InvertSpatial3(metric, inverse_spatial, spatial_determinant)) return false;

  Real d_inverse[3][4][4];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_inverse[p][a][b] = 0.0;
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] -= geometry.inverse_metric[a][c]
                *geometry.inverse_metric[b][d]*d_metric[p + 1][c][d];
          }
        }
      }
    }
    target.d_alpha[p] = 0.5*geometry.lapse*geometry.lapse*geometry.lapse
                        *d_inverse[p][0][0];
    for (int q = 0; q < 3; ++q) {
      target.d_shift[p][q] =
          2.0*geometry.lapse*target.d_alpha[p]
              *geometry.inverse_metric[0][q + 1]
          + geometry.lapse*geometry.lapse*d_inverse[p][0][q + 1];
    }
  }

  target.trace_k = 0.0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      target.trace_k -= geometry.lapse*inverse_spatial[i][j]
                        *geometry.christoffel[0][i + 1][j + 1];
    }
  }

  constexpr Real lambda = -1.0/3.0;
  const Real determinant_factor = Kokkos::pow(spatial_determinant, -lambda);
  for (int i = 0; i < 3; ++i) {
    target.conformal_gamma[i] = 0.0;
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          const Real projector = inverse_spatial[i][k]*inverse_spatial[j][l]
              - 0.5*(1.0 + lambda)*inverse_spatial[i][j]
                  *inverse_spatial[k][l];
          target.conformal_gamma[i] += determinant_factor*projector
              *d_metric[j + 1][k + 1][l + 1];
        }
      }
    }
  }

  Real desired_d0_shift[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    Real advective_shift = 0.0;
    for (int p = 0; p < 3; ++p) {
      advective_shift += geometry.shift[p]*target.d_shift[p][i];
    }
    desired_d0_shift[i] =
        nu*(target.conformal_gamma[i] - eta_beta*upsilon[i]) - advective_shift;
  }
  for (int i = 0; i < 3; ++i) {
    Real contracted_spatial_connection = 0.0;
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const Real connection_first = 0.5*(
            d_metric[j + 1][i + 1][k + 1]
            + d_metric[k + 1][i + 1][j + 1]
            - d_metric[i + 1][j + 1][k + 1]);
        contracted_spatial_connection += inverse_spatial[j][k]
                                         *connection_first;
      }
    }
    target.coordinate[i + 1] = target.d_alpha[i]/geometry.lapse
                               - contracted_spatial_connection;
    for (int j = 0; j < 3; ++j) {
      target.coordinate[i + 1] += metric[i + 1][j + 1]
                                   *desired_d0_shift[j]
                                   /(geometry.lapse*geometry.lapse);
    }
  }
  const Real normal_target = (2.0/geometry.lapse - 1.0)*target.trace_k;
  target.coordinate[0] = geometry.lapse*normal_target;
  for (int i = 0; i < 3; ++i) {
    target.coordinate[0] += geometry.shift[i]*target.coordinate[i + 1];
  }
  for (int A = 0; A < 4; ++A) {
    target.frame[A] = 0.0;
    for (int a = 0; a < 4; ++a) {
      target.frame[A] += ReferenceFrame(reference, A, a)*target.coordinate[a];
    }
  }
  return Kokkos::isfinite(target.trace_k);
}

}  // namespace ref_gh

#endif  // REF_GH_PHYSICAL_GAUGE_TARGET_HPP_
