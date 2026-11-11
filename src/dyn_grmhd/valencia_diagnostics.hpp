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
                                               Real x3v, Real torque[3]) {
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

  // Total stress-energy (fluid + EM), non-densitized
  Real E_tot;
  Real S_d[3];
  Real S_dd[3][3];
  ValenciaStressEnergyTotal(rho, pres, gamma_gas, v_u, v_d, W, B_u, B_d, b0, b_d, b2,
                            g_dd_1d, E_tot, S_d, S_dd);

  // --- 4. Build stresses with needed index placements ---
  // Mixed spatial stress: S^k{}_j = g^{km} S_{mj}
  AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> S_ud;
  for (int kk = 0; kk < 3; ++kk) {
    for (int jj = 0; jj < 3; ++jj) {
      Real sum = 0.0;
      for (int mm = 0; mm < 3; ++mm) {
        sum += g_uu(kk, mm) * S_dd[mm][jj];
      }
      S_ud(kk, jj) = sum;
    }
  }

  // Fully contravariant spatial stress: S^{jk} = g^{jm} g^{kn} S_{mn}
  AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> S_uu;
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      Real sum = 0.0;
      for (int m2 = 0; m2 < 3; ++m2) {
        for (int n2 = 0; n2 < 3; ++n2) {
          sum += g_uu(a, m2) * g_uu(b, n2) * S_dd[m2][n2];
        }
      }
      S_uu(a, b) = sum;
    }
  }

  Real W_uu[3][3];
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      W_uu[a][b] = S_uu(a, b);
    }
  }
  Real G_d[3] = {0.0, 0.0, 0.0};
  ValenciaMomentumSources(E_tot, S_d, W_uu, alpha, dalpha_d, dbeta_du, dg_ddd, 1.0,
                          [&](int a, Real term) { G_d[a] += term; });

  // --- 6. Position and densitized torque density (r × G)_a ---
  // Start with sqrt(gamma) * (r × G)_a   (using Cartesian r × ...)
  torque[0] = (x2v * G_d[2] - x3v * G_d[1]) * sqrt_gamma;  // tau_x
  torque[1] = (x3v * G_d[0] - x1v * G_d[2]) * sqrt_gamma;  // tau_y
  torque[2] = (x1v * G_d[1] - x2v * G_d[0]) * sqrt_gamma;  // tau_z

  // --- 7. Add the product-rule term: F^j{}_i ∂_j φ^i = F^j{}_i ε_{a j}{}^{i} ---
  // F^j{}_i = α S^j{}_i - β^j S_i
  Real F_ud[3][3];
  for (int jdir = 0; jdir < 3; ++jdir) {
    const Real betaj = adm.beta_u(m, jdir, k, j, i);  // β^j
    for (int idir = 0; idir < 3; ++idir) {
      F_ud[jdir][idir] = alpha * S_ud(jdir, idir) - betaj * S_d[idir];
    }
  }

  // (F^j{}_i ε_{a j}{}^{i}) components
  Real miss[3] = {0.0, 0.0, 0.0};
  miss[0] = F_ud[1][2] - F_ud[2][1];  // a = x
  miss[1] = F_ud[2][0] - F_ud[0][2];  // a = y
  miss[2] = F_ud[0][1] - F_ud[1][0];  // a = z

  torque[0] += sqrt_gamma * miss[0];
  torque[1] += sqrt_gamma * miss[1];
  torque[2] += sqrt_gamma * miss[2];
}
#endif  // DYN_GRMHD_VALENCIA_DIAGNOSTICS_HPP_
