//========================================================================================
//! \file ref_gh_calcrhs.cpp
//! \brief Flat-reference nonlinear GH RHS and compatible Phi update.
//========================================================================================
#include <cmath>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "mesh/mesh.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/reference_geometry.hpp"
#include "ref_gh/standard_gh_source.hpp"
#include "utils/finite_diff.hpp"

namespace ref_gh {
namespace {

KOKKOS_INLINE_FUNCTION
void LoadSymmetric(const DvceArray5D<Real> &state, const int offset, const int m,
                   const int k, const int j, const int i, Real tensor[4][4]) {
  for (int a = 0; a < 4; ++a) {
    for (int b = a; b < 4; ++b) {
      tensor[a][b] = tensor[b][a] = state(m, offset + Symmetric4Index(a, b), k, j, i);
    }
  }
}

KOKKOS_INLINE_FUNCTION
bool LoadFlatPointGeometry(const DvceArray5D<Real> &state, const int m, const int k,
                           const int j, const int i, Real metric[4][4],
                           Real pi[4][4], Real phi[3][4][4],
                           Real d_metric[4][4][4],
                           CoordinateGhGeometry &geometry, Real &determinant) {
  LoadSymmetric(state, kPsiOffset, m, k, j, i, metric);
  LoadSymmetric(state, kPiOffset, m, k, j, i, pi);
  for (int p = 0; p < 3; ++p) {
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        phi[p][a][b] = phi[p][b][a] =
            state(m, PhiIndex(p, a, b), k, j, i);
        d_metric[p + 1][a][b] = d_metric[p + 1][b][a] = phi[p][a][b];
      }
    }
  }
  Real inverse[4][4];  // NOLINT(runtime/arrays)
  if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) return false;
  const Real alpha = 1.0/Kokkos::sqrt(-inverse[0][0]);
  Real beta[3];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 3; ++p) beta[p] = alpha*alpha*inverse[0][p + 1];
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      d_metric[0][a][b] = -alpha*pi[a][b];
      for (int p = 0; p < 3; ++p) d_metric[0][a][b] += beta[p]*phi[p][a][b];
    }
  }
  const MinkowskiReference provider;
  const ReferenceGeometry reference = provider(0.0, 0.0, 0.0, 0.0);
  return ComputeCoordinateGhGeometry(metric, d_metric, reference, geometry, determinant);
}

}  // namespace

template <int FDNG>
TaskStatus RefGh::CalcRHS(Driver *driver, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int radius = FDNG - 1;
  const int nmb = pmy_pack->nmb_thispack;
  const auto state = u0;
  const auto state_rhs = u_rhs;
  const Real gamma0 = opt.gamma0;
  Kokkos::deep_copy(state_rhs, 0.0);

  // Psi_t is required on a stencil halo by the compatible Phi update.  Pi_t is only
  // consumed on physical cells and is therefore not evaluated outside that region.
  par_for("ref_gh flat primary rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks - radius, indcs.ke + radius,
  indcs.js - radius, indcs.je + radius,
  indcs.is - radius, indcs.ie + radius,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real metric[4][4], pi[4][4], phi[3][4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    const bool valid = LoadFlatPointGeometry(state, m, k, j, i, metric, pi, phi,
                                             d_metric, geometry, determinant);
    if (!valid) {
      for (int n = 0; n < 20; ++n) state_rhs(m, n, k, j, i) = NAN;
      return;
    }
    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        Real psi_rhs = -geometry.lapse*pi[a][b];
        for (int p = 0; p < 3; ++p) psi_rhs += geometry.shift[p]*phi[p][a][b];
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
    const MinkowskiReference provider;
    const ReferenceGeometry reference = provider(0.0, 0.0, 0.0, 0.0);
    Real partial_source[4][4], covariant_source[4][4]; // NOLINT(runtime/arrays)
    StandardGhPartialWaveSource(metric, d_metric, reference, geometry, gamma0,
                                partial_source);
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        covariant_source[a][b] = partial_source[a][b];
        for (int c = 0; c < 4; ++c) {
          covariant_source[a][b] -= geometry.contracted_upper[c]*d_metric[c][a][b];
        }
      }
    }

    Real spatial_connection[3][3][3];  // NOLINT(runtime/arrays)
    for (int q = 0; q < 3; ++q) {
      for (int p = 0; p < 3; ++p) {
        for (int r = 0; r < 3; ++r) {
          spatial_connection[q][p][r] = 0.0;
          for (int ell = 0; ell < 3; ++ell) {
            spatial_connection[q][p][r] += 0.5*spatial_inverse[q][ell]
              *(phi[p][ell + 1][r + 1] + phi[r][ell + 1][p + 1]
                - phi[ell][p + 1][r + 1]);
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
                          *geometry.inverse_metric[0][b]*phi[p][a][b];
        }
      }
      d_alpha[p] = 0.5*geometry.lapse*geometry.lapse*geometry.lapse*d_inverse_00;
    }

    for (int a = 0; a < 4; ++a) {
      for (int b = a; b < 4; ++b) {
        Real pi_rhs = geometry.lapse*(trace_k*pi[a][b] + covariant_source[a][b]);
        for (int p = 0; p < 3; ++p) {
          pi_rhs += geometry.shift[p]
                    *Dx<FDNG>(p, idx, state, m, PiIndex(a, b), k, j, i);
          for (int q = 0; q < 3; ++q) {
            pi_rhs -= geometry.lapse*spatial_inverse[p][q]
                      *Dx<FDNG>(p, idx, state, m, PhiIndex(q, a, b), k, j, i);
            pi_rhs -= spatial_inverse[p][q]*d_alpha[p]*phi[q][a][b];
            for (int r = 0; r < 3; ++r) {
              pi_rhs += geometry.lapse*spatial_inverse[p][q]
                        *spatial_connection[r][p][q]*phi[r][a][b];
            }
          }
        }
        state_rhs(m, PiIndex(a, b), k, j, i) = pi_rhs;
      }
    }
  });

  par_for("ref_gh compatible phi rhs", DevExeSpace(), 0, nmb - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    for (int p = 0; p < 3; ++p) {
      for (int component = 0; component < kSymmetric4Size; ++component) {
        state_rhs(m, kPhiOffset + p*kSymmetric4Size + component, k, j, i) =
            Dx<FDNG>(p, idx, state_rhs, m, kPsiOffset + component, k, j, i);
      }
    }
  });

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
  }
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
  Kokkos::deep_copy(constraints, 0.0);
  par_for("ref_gh flat constraints", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real idx[3] = {1.0/size.d_view(m).dx1, 1.0/size.d_view(m).dx2,
                         1.0/size.d_view(m).dx3};
    Real metric[4][4], pi[4][4], phi[3][4][4], d_metric[4][4][4]; // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    if (!LoadFlatPointGeometry(state, m, k, j, i, metric, pi, phi, d_metric,
                               geometry, determinant)) {
      for (int n = 0; n < ncon; ++n) constraints(m, n, k, j, i) = NAN;
      return;
    }
    for (int a = 0; a < 4; ++a) {
      constraints(m, a, k, j, i) = geometry.gauge_constraint[a];
    }
    Real reduction2 = 0.0;
    Real curl2 = 0.0;
    for (int p = 0; p < 3; ++p) {
      for (int component = 0; component < kSymmetric4Size; ++component) {
        const Real reduction =
            Dx<FDNG>(p, idx, state, m, kPsiOffset + component, k, j, i)
            - state(m, kPhiOffset + p*kSymmetric4Size + component, k, j, i);
        reduction2 += reduction*reduction;
        for (int q = p + 1; q < 3; ++q) {
          const Real curl =
              Dx<FDNG>(p, idx, state, m,
                       kPhiOffset + q*kSymmetric4Size + component, k, j, i)
              - Dx<FDNG>(q, idx, state, m,
                         kPhiOffset + p*kSymmetric4Size + component, k, j, i);
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
