//========================================================================================
//! \file ref_gh_calcrhs.cpp
//! \brief Flat-reference nonlinear GH RHS and compatible Phi update.
//========================================================================================
#include <cmath>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "ref_gh/covariant_gh_source.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"
#include "utils/finite_diff.hpp"

namespace ref_gh {

template <int FDNG>
TaskStatus RefGh::CalcRHS(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int radius = FDNG - 1;
  const int nmb = pmy_pack->nmb_thispack;
  const auto state = u0;
  const auto state_rhs = u_rhs;
  const auto table = reference_table;
  const int reference_kind = opt.reference_kind;
  const int source_kind = opt.source_kind;
  const Real reference_mass = opt.reference_mass;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const Real stage_time = pmy_pack->pmesh->time;
  const Real gamma0 = opt.gamma0;
  Kokkos::deep_copy(state_rhs, 0.0);
  DebugFence("ref_gh CalcRHS zero");

  // Psi_t is required on a stencil halo by the compatible Phi update.  Pi_t is only
  // consumed on physical cells and is therefore not evaluated outside that region.
  par_for("ref_gh flat primary rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks - radius, indcs.ke + radius,
  indcs.js - radius, indcs.je + radius,
  indcs.is - radius, indcs.ie + radius,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const ReferenceGeometry reference = GetReferenceGeometry(
        reference_kind, table, reference_mass, center_x, center_y, center_z,
        stage_time, x, y, z);
    Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4]; // NOLINT
    Real d_psi[4][4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    const bool valid = LoadPointGeometry(state, reference, m, k, j, i, psi, pi,
                                         phi, d_psi, metric, d_metric, geometry,
                                         determinant);
    if (!valid) {
      for (int n = 0; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
      return;
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        Real psi_rhs = -geometry.lapse*pi[a][b];
        for (int p = 0; p < 3; ++p) {
          psi_rhs += geometry.shift[p]*d_psi[p + 1][a][b];
        }
        state_rhs(m, PsiIndex(a, b), k, j, i) = psi_rhs;
      }
    }
    const bool physical = k >= indcs.ks && k <= indcs.ke
                          && j >= indcs.js && j <= indcs.je
                          && i >= indcs.is && i <= indcs.ie;
    if (!physical) return;

    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    Real spatial_inverse[3][3];  // NOLINT(runtime/arrays)
    Real spatial_determinant = 0.0;
    if (!InvertSpatial3(metric, spatial_inverse, spatial_determinant)) {
      for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
      return;
    }
    Real scalar_source[4][4];  // NOLINT(runtime/arrays)
    CovariantSourceSectors source_sectors;
    if (source_kind == 0) {
      if (!CovariantGhScalarWaveSource(psi, pi, phi, reference, geometry, gamma0,
                                       scalar_source, source_sectors)) {
        for (int n = 10; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
        return;
      }
    } else {
      Real partial_source[4][4];  // NOLINT(runtime/arrays)
      StandardGhPartialWaveSource(metric, d_metric, reference, geometry, gamma0,
                                  partial_source);
      TransformPartialWaveSource(metric, d_metric, partial_source, d_psi,
                                 reference, geometry, scalar_source);
    }

    Real spatial_connection[3][3][3];  // NOLINT(runtime/arrays)
    for (int q = 0; q < 3; ++q) {
      for (int p = 0; p < 3; ++p) {
        for (int r = 0; r < 3; ++r) {
          spatial_connection[q][p][r] = 0.0;
          for (int ell = 0; ell < 3; ++ell) {
            spatial_connection[q][p][r] += 0.5*spatial_inverse[q][ell]
              *(d_metric[p + 1][ell + 1][r + 1]
                + d_metric[r + 1][ell + 1][p + 1]
                - d_metric[ell + 1][p + 1][r + 1]);
          }
        }
      }
    }
    Real trace_k = 0.0;
    for (int p = 0; p < 3; ++p) {
      for (int q = 0; q < 3; ++q) {
        trace_k -= geometry.lapse*spatial_inverse[p][q]
                   *geometry.christoffel[0][p + 1][q + 1];
      }
    }
    Real d_alpha[3];  // NOLINT(runtime/arrays)
    for (int p = 0; p < 3; ++p) {
      Real d_inverse_00 = 0.0;
      for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
          d_inverse_00 -= geometry.inverse_metric[0][a]
                          *geometry.inverse_metric[0][b]*d_metric[p + 1][a][b];
        }
      }
      d_alpha[p] = 0.5*geometry.lapse*geometry.lapse*geometry.lapse*d_inverse_00;
    }

    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        Real divergence = 0.0;
        Real lapse_gradient_term = 0.0;
        for (int p = 0; p < 3; ++p) {
          for (int q = 0; q < 3; ++q) {
            Real partial_tilde_phi = 0.0;
            Real tilde_phi_q = 0.0;
            for (int I = 0; I < 3; ++I) {
              partial_tilde_phi +=
                  CoframeDerivative(reference, p + 1, I + 1, q + 1)
                    *phi[I][a][b]
                  + reference.spatial_coframe[I][q]
                    *Dx<FDNG>(p, idx, state, m, PhiIndex(I, a, b), k, j, i);
              tilde_phi_q += reference.spatial_coframe[I][q]*phi[I][a][b];
            }
            Real covariant_derivative = partial_tilde_phi;
            for (int r = 0; r < 3; ++r) {
              Real tilde_phi_r = 0.0;
              for (int I = 0; I < 3; ++I) {
                tilde_phi_r += reference.spatial_coframe[I][r]*phi[I][a][b];
              }
              covariant_derivative -= spatial_connection[r][p][q]*tilde_phi_r;
            }
            divergence += spatial_inverse[p][q]*covariant_derivative;
            lapse_gradient_term += spatial_inverse[p][q]*d_alpha[p]*tilde_phi_q;
          }
        }
        Real pi_rhs = geometry.lapse*(trace_k*pi[a][b] - divergence
                                      + scalar_source[a][b])
                      - lapse_gradient_term;
        for (int p = 0; p < 3; ++p) {
          pi_rhs += geometry.shift[p]
                    *Dx<FDNG>(p, idx, state, m, PiIndex(a, b), k, j, i);
        }
        state_rhs(m, PiIndex(a, b), k, j, i) = pi_rhs;
      }
    }
  });
  DebugFence("ref_gh CalcRHS primary");

  par_for("ref_gh compatible phi rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const ReferenceGeometry reference = GetReferenceGeometry(
        reference_kind, table, reference_mass, center_x, center_y, center_z,
        stage_time, x, y, z);
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    for (int I = 0; I < 3; ++I) {
      for (int component = 0; component < kSymmetric4Size; ++component) {
        Real phi_rhs = 0.0;
        for (int p = 0; p < 3; ++p) {
          phi_rhs += reference.spatial_frame[I][p]
                       *Dx<FDNG>(p, idx, state_rhs, m,
                                  kPsiOffset + component, k, j, i);
          Real coordinate_d_psi = 0.0;
          for (int J = 0; J < 3; ++J) {
            coordinate_d_psi += reference.spatial_coframe[J][p]
                *state(m, kPhiOffset + J*kSymmetric4Size + component, k, j, i);
          }
          phi_rhs += reference.dt_spatial_frame[I][p]*coordinate_d_psi;
        }
        state_rhs(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i) = phi_rhs;
      }
    }
  });
  DebugFence("ref_gh CalcRHS compatible_phi");

  if (opt.diss > 0.0) {
    const Real sign = (FDNG % 2 == 0) ? -1.0 : 1.0;
    const Real coefficient = opt.diss*std::pow(2.0, -2.0*FDNG)*sign;
    par_for("ref_gh dissipation", DevExeSpace(), 0, nmb - 1, 0, nref_gh - 1,
    indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                           1.0/size.d_view(m).dx3};
      for (int p = 0; p < 3; ++p) {
        state_rhs(m, n, k, j, i) += coefficient
            *Diss<FDNG>(p, idx, state, m, n, k, j, i);
      }
    });
    DebugFence("ref_gh CalcRHS dissipation");
  }
  DebugFence("ref_gh CalcRHS");
  return TaskStatus::complete;
}

template TaskStatus RefGh::CalcRHS<2>(Driver *, int);
template TaskStatus RefGh::CalcRHS<3>(Driver *, int);
template TaskStatus RefGh::CalcRHS<4>(Driver *, int);

template <int FDNG>
void RefGh::CalcConstraints() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const auto constraints = u_con;
  const auto table = reference_table;
  const int reference_kind = opt.reference_kind;
  const int source_kind = opt.source_kind;
  const Real reference_mass = opt.reference_mass;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const Real time = pmy_pack->pmesh->time;
  Kokkos::deep_copy(constraints, 0.0);
  par_for("ref_gh flat constraints", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const ReferenceGeometry reference = GetReferenceGeometry(
        reference_kind, table, reference_mass, center_x, center_y, center_z,
        time, x, y, z);
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4]; // NOLINT
    Real d_psi[4][4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    if (!LoadPointGeometry(state, reference, m, k, j, i, psi, pi, phi, d_psi,
                           metric, d_metric, geometry, determinant)) {
      for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
      return;
    }
    Real scalar_source[4][4];  // NOLINT(runtime/arrays)
    CovariantSourceSectors source_sectors;
    if (!CovariantGhScalarWaveSource(psi, pi, phi, reference, geometry, 0.0,
                                     scalar_source, source_sectors)) {
      for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
      return;
    }
    if (source_kind == 0) {
      for (int A = 0; A < 4; ++A) {
        constraints(m, A, k, j, i) = source_sectors.delta[A];
      }
    } else {
      for (int a = 0; a < 4; ++a) {
        constraints(m, a, k, j, i) = geometry.gauge_constraint[a];
      }
    }

    Real q2 = 0.0;
    Real delta2 = 0.0;
    Real frame_ricci2 = 0.0;
    Real coordinate_ricci2 = 0.0;
    Real curvature2 = 0.0;
    Real qq2 = 0.0;
    Real delta_product2 = 0.0;
    Real damping2 = 0.0;
    Real frame_correction2 = 0.0;
    for (int A = 0; A < 4; ++A) {
      for (int B = 0; B < 4; ++B) {
        frame_ricci2 += reference.ricci_frame[A][B]*reference.ricci_frame[A][B];
        curvature2 += source_sectors.curvature[A][B]*source_sectors.curvature[A][B];
        qq2 += source_sectors.qq[A][B]*source_sectors.qq[A][B];
        delta_product2 += source_sectors.delta_product[A][B]
                          *source_sectors.delta_product[A][B];
        damping2 += source_sectors.damping[A][B]*source_sectors.damping[A][B];
        frame_correction2 += source_sectors.frame_correction[A][B]
                             *source_sectors.frame_correction[A][B];
        for (int C = 0; C < 4; ++C) {
          q2 += source_sectors.q[C][A][B]*source_sectors.q[C][A][B];
          delta2 += source_sectors.delta_lower[A][B][C]
                    *source_sectors.delta_lower[A][B][C];
        }
        Real coordinate_ricci = 0.0;
        for (int C = 0; C < 4; ++C) {
          coordinate_ricci += reference.d_christoffel[C][C][A][B]
                              - reference.d_christoffel[B][C][A][C];
          for (int D = 0; D < 4; ++D) {
            coordinate_ricci += reference.christoffel[C][C][D]
                                *reference.christoffel[D][A][B]
                              - reference.christoffel[C][B][D]
                                *reference.christoffel[D][A][C];
          }
        }
        coordinate_ricci2 += coordinate_ricci*coordinate_ricci;
      }
    }
    constraints(m, kDiagnosticOffset + 0, k, j, i) = Kokkos::sqrt(q2);
    constraints(m, kDiagnosticOffset + 1, k, j, i) = Kokkos::sqrt(delta2);
    constraints(m, kDiagnosticOffset + 2, k, j, i) = Kokkos::sqrt(frame_ricci2);
    constraints(m, kDiagnosticOffset + 3, k, j, i) =
        Kokkos::sqrt(coordinate_ricci2);
    constraints(m, kDiagnosticOffset + 4, k, j, i) = Kokkos::sqrt(curvature2);
    constraints(m, kDiagnosticOffset + 5, k, j, i) = Kokkos::sqrt(qq2);
    constraints(m, kDiagnosticOffset + 6, k, j, i) = Kokkos::sqrt(delta_product2);
    constraints(m, kDiagnosticOffset + 7, k, j, i) = Kokkos::sqrt(damping2);
    constraints(m, kDiagnosticOffset + 8, k, j, i) =
        Kokkos::sqrt(frame_correction2);
    Real reduction2 = 0.0;
    Real curl2 = 0.0;
    for (int I = 0; I < 3; ++I) {
      for (int component = 0; component < kSymmetric4Size; ++component) {
        Real reduction =
            -state(m, kPhiOffset + I*kSymmetric4Size + component, k, j, i);
        for (int p = 0; p < 3; ++p) {
          reduction += reference.spatial_frame[I][p]
              *Dx<FDNG>(p, idx, state, m, kPsiOffset + component, k, j, i);
        }
        reduction2 += reduction*reduction;
        for (int J = I + 1; J < 3; ++J) {
          Real curl = 0.0;
          for (int p = 0; p < 3; ++p) {
            curl += reference.spatial_frame[I][p]
                      *Dx<FDNG>(p, idx, state, m,
                                kPhiOffset + J*kSymmetric4Size + component,
                                k, j, i)
                    - reference.spatial_frame[J][p]
                      *Dx<FDNG>(p, idx, state, m,
                                kPhiOffset + I*kSymmetric4Size + component,
                                k, j, i);
          }
          for (int K = 0; K < 3; ++K) {
            curl -= reference.structure[I][J][K]
                    *state(m, kPhiOffset + K*kSymmetric4Size + component,
                           k, j, i);
          }
          curl2 += curl*curl;
        }
      }
    }
    constraints(m, 4, k, j, i) = Kokkos::sqrt(reduction2);
    constraints(m, 5, k, j, i) = Kokkos::sqrt(curl2);
  });
}

template void RefGh::CalcConstraints<2>();
template void RefGh::CalcConstraints<3>();
template void RefGh::CalcConstraints<4>();

}  // namespace ref_gh
