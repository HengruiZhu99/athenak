//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef DYN_GRMHD_VALENCIA_DIAGNOSTICS_HPP_
#define DYN_GRMHD_VALENCIA_DIAGNOSTICS_HPP_
#include <limits>
#include "dyn_grmhd/valencia_stress_energy.hpp"
#include "dyn_grmhd/valencia_sources.hpp"

// Diagnostic gamma-law stress reconstruction shared by spatial output and region sums.
// Geometry derivatives/contractions are the evolution helpers; retain the product rule.
template <class ADM, class Prim, class Bcc>
KOKKOS_INLINE_FUNCTION void ValenciaTorqueCell(const ADM& adm, const Prim& prim,
                                               const Bcc& bcc, int torque_ng, int ndim,
                                               Real gamma_gas, const Real idx[3], int m,
                                               int k, int j, int i, Real x1v, Real x2v,
                                               Real x3v, Real torque[3],
                                               Real* force = nullptr,
                                               const Real* shift_offset = nullptr) {
  if (force)
    for (int d = 0; d < 3; ++d) force[d] = std::numeric_limits<Real>::quiet_NaN();
  // --- 1. Metric, det, inverse ---
  Real g_dd_1d[6];
  g_dd_1d[0] = adm.g_dd(m, 0, 0, k, j, i);
  g_dd_1d[1] = adm.g_dd(m, 0, 1, k, j, i);
  g_dd_1d[2] = adm.g_dd(m, 0, 2, k, j, i);
  g_dd_1d[3] = adm.g_dd(m, 1, 1, k, j, i);
  g_dd_1d[4] = adm.g_dd(m, 1, 2, k, j, i);
  g_dd_1d[5] = adm.g_dd(m, 2, 2, k, j, i);

  Real detg = adm::SpatialDet(g_dd_1d[0], g_dd_1d[1], g_dd_1d[2], g_dd_1d[3], g_dd_1d[4],
                              g_dd_1d[5]);

  // Guard against bad metric
  if (!(Kokkos::isfinite(detg) && detg > 0.0 && g_dd_1d[0] > 0.0 &&
        g_dd_1d[0] * g_dd_1d[3] - g_dd_1d[1] * g_dd_1d[1] > 0.0)) {
    torque[0] = std::numeric_limits<Real>::quiet_NaN();
    torque[1] = std::numeric_limits<Real>::quiet_NaN();
    torque[2] = std::numeric_limits<Real>::quiet_NaN();
    return;
  }
  Real sqrt_gamma = sqrt(detg);

  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
  adm::SpatialInv(1.0 / detg, g_dd_1d[0], g_dd_1d[1], g_dd_1d[2], g_dd_1d[3], g_dd_1d[4],
                  g_dd_1d[5], &g_uu(0, 0), &g_uu(0, 1), &g_uu(0, 2), &g_uu(1, 1),
                  &g_uu(1, 2), &g_uu(2, 2));

  Real alpha = adm.alpha(m, k, j, i);
  if (!(Kokkos::isfinite(alpha) && alpha > 0.0)) {
    for (int d = 0; d < 3; ++d) {
      torque[d] = std::numeric_limits<Real>::quiet_NaN();
    }
    return;
  }

  // --- 2. Metric derivatives, lapse/shift derivatives (index DOWN: ∂_i) ---
  Real dg_ddd[3][3][3], dalpha_d[3], dbeta_du[3][3];
  ValenciaGeometryDerivatives(torque_ng, adm, ndim, idx, m, k, j, i, dalpha_d, dbeta_du,
                              dg_ddd);

  // --- 3. Primitives + B field as Valencia objects ---
  Real rho = prim(m, IDN, k, j, i);
  Real pres = prim(m, IPR, k, j, i);

  Real prim_Wv_u[3] = {prim(m, IVX, k, j, i), prim(m, IVY, k, j, i),
                       prim(m, IVZ, k, j, i)};

  Real B_tilde_u[3] = {bcc(m, 0, k, j, i), bcc(m, 1, k, j, i), bcc(m, 2, k, j, i)};

  bool admissible =
      Kokkos::isfinite(rho) && rho > 0.0 && Kokkos::isfinite(pres) && pres >= 0.0;
  for (int d = 0; d < 3; ++d) {
    admissible &= Kokkos::isfinite(prim_Wv_u[d]) && Kokkos::isfinite(B_tilde_u[d]);
  }
  if (!admissible) {
    for (int d = 0; d < 3; ++d) {
      torque[d] = std::numeric_limits<Real>::quiet_NaN();
    }
    return;
  }
  Real W, v_u[3], v_d[3];
  ValenciaPrimsToEulerianVelocity(prim_Wv_u, g_dd_1d, W, v_u, v_d);

  Real sqrt_gamma_chk, B_u[3], B_d[3];
  ValenciaEulerianMagneticFromDensitized(B_tilde_u, g_dd_1d, sqrt_gamma_chk, B_u, B_d);

  Real Bv, b0, b_u[3], b_d[3], b2;
  ValenciaComovingB(v_u, v_d, B_u, B_d, W, Bv, b0, b_u, b_d, b2);

  // Build the total Eulerian stress in the index placements the source uses.
  // b_u is the spatial projection of b, and b0 = alpha b^t.
  const Real pstar = pres + 0.5 * b2;
  const Real A = (rho * GammaLawSpecificEnthalpy(rho, pres, gamma_gas) + b2) * W * W;
  const Real E_tot = A - pstar - b0 * b0;
  Real S_d[3], W_uu[3][3];
  for (int a = 0; a < 3; ++a) {
    S_d[a] = A * v_d[a] - b0 * b_d[a];
    for (int b = a; b < 3; ++b) {
      W_uu[a][b] = A * v_u[a] * v_u[b] + pstar * g_uu(a, b) - b_u[a] * b_u[b];
      W_uu[b][a] = W_uu[a][b];
    }
  }
  Real G_d[3] = {0.0, 0.0, 0.0};
  ValenciaMomentumSources(E_tot, S_d, W_uu, alpha, dalpha_d, dbeta_du, dg_ddd, 1.0,
                          [&](int a, Real term) { G_d[a] += term; });

  if (force)
    for (int d = 0; d < 3; ++d) force[d] = sqrt_gamma * G_d[d];

  // The isotropic mixed-stress term cancels in F^a_b - F^b_a.
  const Real x[3] = {x1v, x2v, x3v};
  Real beta[3];
  for (int a = 0; a < 3; ++a)
    beta[a] = adm.beta_u(m, a, k, j, i) + (shift_offset ? shift_offset[a] : 0);
  for (int d = 0; d < 3; ++d) {
    int a = (d + 1) % 3, b = (d + 2) % 3;
    Real antisymmetric_flux = alpha * (A * (v_u[a] * v_d[b] - v_u[b] * v_d[a])
                                      - b_u[a] * b_d[b] + b_u[b] * b_d[a])
                            - beta[a] * S_d[b]
                            + beta[b] * S_d[a];
    torque[d] = sqrt_gamma * (x[a] * G_d[b] - x[b] * G_d[a] + antisymmetric_flux);
  }
}
#endif  // DYN_GRMHD_VALENCIA_DIAGNOSTICS_HPP_
