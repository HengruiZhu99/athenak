//========================================================================================
//! \file physical_gauge_target.hpp
//! \brief Ordinary-GH target for advective 1+log plus conformal Gamma-driver.
//========================================================================================
#ifndef REF_GH_PHYSICAL_GAUGE_TARGET_HPP_
#define REF_GH_PHYSICAL_GAUGE_TARGET_HPP_

#include "athena.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_residual.hpp"
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

struct PhysicalGaugeTargetResidual {
  Real reference_coordinate[4];       // NOLINT(runtime/arrays)
  Real physical_coordinate[4];        // NOLINT(runtime/arrays)
  Real delta_coordinate[4];           // NOLINT(runtime/arrays)
  Real reference_frame[4];            // NOLINT(runtime/arrays)
  Real physical_frame[4];             // NOLINT(runtime/arrays)
  Real delta_frame[4];                // NOLINT(runtime/arrays)
  Real reference_conformal_gamma[3];  // NOLINT(runtime/arrays)
  Real physical_conformal_gamma[3];   // NOLINT(runtime/arrays)
  Real delta_conformal_gamma[3];      // NOLINT(runtime/arrays)
  Real reference_shift[3];            // NOLINT(runtime/arrays)
  Real physical_shift[3];             // NOLINT(runtime/arrays)
  Real delta_shift[3];                // NOLINT(runtime/arrays)
  Real reference_lapse;
  Real physical_lapse;
  Real delta_lapse;
  Real reference_trace_k;
  Real physical_trace_k;
  Real delta_trace_k;
  bool valid;
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

// Evaluate F(g)-F(gbar) through exact residual identities.  No output delta is
// formed by subtracting independently evaluated physical and reference gauge
// targets.  The redundant full members exist only for oracle comparison and
// for nonsingular coefficients consumed elsewhere in the existing equations.
template <typename Reference>
KOKKOS_INLINE_FUNCTION
bool ComputePhysicalGaugeTargetResidual(
    const Real psi[4][4], const Real pi[4][4],
    const Real phi[3][4][4], const Real metric[4][4],
    const Real d_metric[4][4][4], const CoordinateGhGeometry &geometry,
    const Reference &reference, const Real upsilon[3], const Real nu,
    const Real eta_beta, PhysicalGaugeTargetResidual &target) {
  target.valid = false;
  ReferenceRelativeCoordinateData relative;
  if (!BuildReferenceRelativeCoordinateData(
          psi, pi, phi, metric, d_metric, geometry, reference, relative)) {
    return false;
  }

  ReferenceResidualValue g[4][4];      // NOLINT(runtime/arrays)
  ReferenceResidualValue dg[4][4][4];  // NOLINT(runtime/arrays)
  ReferenceResidualValue inverse[4][4];  // NOLINT(runtime/arrays)
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

  ReferenceResidualValue d_inverse[3][4][4];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        d_inverse[p][a][b] = ReferenceResidualConstant(0.0);
        for (int c = 0; c < 4; ++c) {
          for (int d = 0; d < 4; ++d) {
            d_inverse[p][a][b] = d_inverse[p][a][b]
                - inverse[a][c]*inverse[b][d]*dg[p + 1][c][d];
          }
        }
      }
    }
  }

  const ReferenceResidualValue lapse = ReferenceResidualConstant(1.0)
      /ReferenceResidualSqrt(-inverse[0][0]);
  target.reference_lapse = lapse.reference;
  target.physical_lapse = lapse.physical;
  target.delta_lapse = lapse.delta;
  ReferenceResidualValue shift[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    shift[i] = lapse*lapse*inverse[0][i + 1];
    target.reference_shift[i] = shift[i].reference;
    target.physical_shift[i] = shift[i].physical;
    target.delta_shift[i] = shift[i].delta;
  }

  Real physical_spatial_inverse[3][3];  // NOLINT(runtime/arrays)
  Real physical_spatial_determinant = 0.0;
  if (!InvertSpatial3(metric, physical_spatial_inverse,
                      physical_spatial_determinant)) return false;
  Real reference_spatial_inverse[3][3] = {};  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int I = 0; I < 3; ++I) {
        reference_spatial_inverse[i][j] +=
            ReferenceSpatialFrame(reference, I, i)
            *ReferenceSpatialFrame(reference, I, j);
      }
    }
  }
  ReferenceResidualValue spatial_inverse[3][3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Real delta = 0.0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          delta -= physical_spatial_inverse[i][k]
              *relative.delta_metric[k + 1][l + 1]
              *reference_spatial_inverse[l][j];
        }
      }
      spatial_inverse[i][j] = MakeReferenceResidual(
          reference_spatial_inverse[i][j], physical_spatial_inverse[i][j],
          delta);
    }
  }

  ReferenceResidualValue christoffel_first[4][4][4];  // NOLINT(runtime/arrays)
  ReferenceResidualValue christoffel[4][4][4];        // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        christoffel_first[a][b][c] =
            0.5*(dg[b][a][c] + dg[c][a][b] - dg[a][b][c]);
      }
    }
  }
  // The first-kind array must be complete before raising its first index.
  // Interleaving these loops reads not-yet-initialized d-components.
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        christoffel[a][b][c] = ReferenceResidualConstant(0.0);
        for (int d = 0; d < 4; ++d) {
          christoffel[a][b][c] = christoffel[a][b][c]
              + inverse[a][d]*christoffel_first[d][b][c];
        }
      }
    }
  }
  ReferenceResidualValue trace_k = ReferenceResidualConstant(0.0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      trace_k = trace_k
          - lapse*spatial_inverse[i][j]*christoffel[0][i + 1][j + 1];
    }
  }
  target.reference_trace_k = trace_k.reference;
  target.physical_trace_k = trace_k.physical;
  target.delta_trace_k = trace_k.delta;

  ReferenceResidualValue d_alpha[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) {
    d_alpha[p] = 0.5*lapse*lapse*lapse*d_inverse[p][0][0];
  }

  const ReferenceResidualValue a = g[1][1];
  const ReferenceResidualValue b = g[1][2];
  const ReferenceResidualValue c = g[1][3];
  const ReferenceResidualValue d = g[2][2];
  const ReferenceResidualValue e = g[2][3];
  const ReferenceResidualValue f = g[3][3];
  const ReferenceResidualValue spatial_determinant =
      a*(d*f - e*e) - b*(b*f - c*e) + c*(b*e - c*d);
  const ReferenceResidualValue determinant_factor =
      ReferenceResidualCubeRoot(spatial_determinant);

  ReferenceResidualValue conformal_gamma[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    conformal_gamma[i] = ReferenceResidualConstant(0.0);
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          const ReferenceResidualValue projector =
              spatial_inverse[i][k]*spatial_inverse[j][l]
              - (1.0/3.0)*spatial_inverse[i][j]*spatial_inverse[k][l];
          conformal_gamma[i] = conformal_gamma[i]
              + determinant_factor*projector*dg[j + 1][k + 1][l + 1];
        }
      }
    }
    target.reference_conformal_gamma[i] = conformal_gamma[i].reference;
    target.physical_conformal_gamma[i] = conformal_gamma[i].physical;
    target.delta_conformal_gamma[i] = conformal_gamma[i].delta;
  }

  ReferenceResidualValue desired_d0_shift[3];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    ReferenceResidualValue advective_shift = ReferenceResidualConstant(0.0);
    for (int p = 0; p < 3; ++p) {
      const ReferenceResidualValue d_shift =
          2.0*lapse*d_alpha[p]*inverse[0][i + 1]
          + lapse*lapse*d_inverse[p][0][i + 1];
      advective_shift = advective_shift + shift[p]*d_shift;
    }
    const ReferenceResidualValue upsilon_value =
        MakeReferenceResidual(0.0, upsilon[i], upsilon[i]);
    desired_d0_shift[i] =
        nu*(conformal_gamma[i] - eta_beta*upsilon_value) - advective_shift;
  }

  ReferenceResidualValue coordinate[4];  // NOLINT(runtime/arrays)
  for (int i = 0; i < 3; ++i) {
    ReferenceResidualValue contracted_spatial_connection =
        ReferenceResidualConstant(0.0);
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        const ReferenceResidualValue connection_first = 0.5*(
            dg[j + 1][i + 1][k + 1]
            + dg[k + 1][i + 1][j + 1]
            - dg[i + 1][j + 1][k + 1]);
        contracted_spatial_connection = contracted_spatial_connection
            + spatial_inverse[j][k]*connection_first;
      }
    }
    coordinate[i + 1] = d_alpha[i]/lapse
                        - contracted_spatial_connection;
    for (int j = 0; j < 3; ++j) {
      coordinate[i + 1] = coordinate[i + 1]
          + g[i + 1][j + 1]*desired_d0_shift[j]/(lapse*lapse);
    }
  }
  const ReferenceResidualValue normal_target =
      (2.0/lapse - ReferenceResidualConstant(1.0))*trace_k;
  coordinate[0] = lapse*normal_target;
  for (int i = 0; i < 3; ++i) {
    coordinate[0] = coordinate[0] + shift[i]*coordinate[i + 1];
  }

  for (int A = 0; A < 4; ++A) {
    ReferenceResidualValue frame = ReferenceResidualConstant(0.0);
    for (int a_index = 0; a_index < 4; ++a_index) {
      frame = frame
          + ReferenceFrame(reference, A, a_index)*coordinate[a_index];
    }
    target.reference_frame[A] = frame.reference;
    target.physical_frame[A] = frame.physical;
    target.delta_frame[A] = frame.delta;
    target.reference_coordinate[A] = coordinate[A].reference;
    target.physical_coordinate[A] = coordinate[A].physical;
    target.delta_coordinate[A] = coordinate[A].delta;
  }
  // Do not reconstruct the legacy full target here.  The `.physical` members
  // above are carried by the same residual arithmetic and exist only for
  // independent oracle comparisons and nonsingular shift advection.  Calling
  // ComputePhysicalGaugeTarget here would both duplicate production work and
  // make the oracle comparison tautological by overwriting its candidate.
  target.valid = Kokkos::isfinite(physical_spatial_determinant)
                 && physical_spatial_determinant > 0.0;
  for (int A = 0; A < 4; ++A) {
    target.valid = target.valid
        && Kokkos::isfinite(target.delta_frame[A])
        && Kokkos::isfinite(target.physical_frame[A]);
  }
  return target.valid;
}

}  // namespace ref_gh

#endif  // REF_GH_PHYSICAL_GAUGE_TARGET_HPP_
