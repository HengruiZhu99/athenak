//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef DYN_GRMHD_VALENCIA_SURFACE_FLUX_HPP_
#define DYN_GRMHD_VALENCIA_SURFACE_FLUX_HPP_
#include "dyn_grmhd/valencia_stress_energy.hpp"

// Expanded accumulator for separated fluid/EM fluxes and all axes
struct FluxAccumulator {
  Real moving[19] = {};
  Real owned = 0, admissible = 0, covered_area = 0;
  Real bad[6] = {};
  Real mdot = 0, edot_fluid = 0, edot_em = 0, phiB = 0, area = 0;
  // Linear momentum fluxes (fluid, EM)
  Real pdot_x_fluid = 0, pdot_y_fluid = 0, pdot_z_fluid = 0;
  Real pdot_x_em = 0, pdot_y_em = 0, pdot_z_em = 0;
  // Angular momentum fluxes (fluid, EM for x, y, z axes)
  Real ldot_x_fluid = 0, ldot_y_fluid = 0, ldot_z_fluid = 0;
  Real ldot_x_em = 0, ldot_y_em = 0, ldot_z_em = 0;

  KOKKOS_INLINE_FUNCTION
  FluxAccumulator& operator+=(const FluxAccumulator& src) {
    for (int i = 0; i < 19; ++i) {
      moving[i] += src.moving[i];
    }
    owned += src.owned;
    admissible += src.admissible;
    covered_area += src.covered_area;
    for (int i = 0; i < 6; ++i) {
      bad[i] += src.bad[i];
    }
    mdot += src.mdot;
    edot_fluid += src.edot_fluid;
    edot_em += src.edot_em;
    phiB += src.phiB;
    area += src.area;
    pdot_x_fluid += src.pdot_x_fluid;
    pdot_y_fluid += src.pdot_y_fluid;
    pdot_z_fluid += src.pdot_z_fluid;
    pdot_x_em += src.pdot_x_em;
    pdot_y_em += src.pdot_y_em;
    pdot_z_em += src.pdot_z_em;
    ldot_x_fluid += src.ldot_x_fluid;
    ldot_y_fluid += src.ldot_y_fluid;
    ldot_z_fluid += src.ldot_z_fluid;
    ldot_x_em += src.ldot_x_em;
    ldot_y_em += src.ldot_y_em;
    ldot_z_em += src.ldot_z_em;
    return *this;
  }
};

namespace Kokkos {
template <>
struct reduction_identity<FluxAccumulator> {
  KOKKOS_INLINE_FUNCTION
  static FluxAccumulator sum() { return FluxAccumulator{}; }
};
}  // namespace Kokkos

// dS_d is the proper covector, already containing sqrt(gamma). No field repair.
KOKKOS_INLINE_FUNCTION void ValenciaSurfacePoint(
    Real rho, Real pgas, Real gamma_gas, const Real prim_Wv_u[3], const Real B_tilde_u[3],
    const Real g_dd_1d[6], Real detg, Real alp, const Real beta_u[3],
    const Real position[3], const Real surface_velocity[3], const Real dS_d[3],
    FluxAccumulator& update) {
  // Inverse metric g^{ij}
  Real g_uu_1d[6];
  adm::SpatialInv(1.0 / detg, g_dd_1d[0], g_dd_1d[1], g_dd_1d[2], g_dd_1d[3], g_dd_1d[4],
                  g_dd_1d[5], &g_uu_1d[0], &g_uu_1d[1], &g_uu_1d[2], &g_uu_1d[3],
                  &g_uu_1d[4], &g_uu_1d[5]);

  Real g_u[3][3];
  g_u[0][0] = g_uu_1d[0];
  g_u[0][1] = g_uu_1d[1];
  g_u[0][2] = g_uu_1d[2];
  g_u[1][0] = g_uu_1d[1];
  g_u[1][1] = g_uu_1d[3];
  g_u[1][2] = g_uu_1d[4];
  g_u[2][0] = g_uu_1d[2];
  g_u[2][1] = g_uu_1d[4];
  g_u[2][2] = g_uu_1d[5];

  // --- B. Eulerian velocity and Eulerian B from Valencia helpers ---
  Real W, v_u[3], v_d[3];
  ValenciaPrimsToEulerianVelocity(prim_Wv_u, g_dd_1d, W, v_u, v_d);

  Real sqrt_gamma_surf, B_u[3], B_d[3];
  ValenciaEulerianMagneticFromDensitized(B_tilde_u, g_dd_1d, sqrt_gamma_surf, B_u, B_d);

  // Comoving magnetic field
  Real Bv, b0, b_u[3], b_d[3], b2;
  ValenciaComovingB(v_u, v_d, B_u, B_d, W, Bv, b0, b_u, b_d, b2);

  // --- C. Decomposed stress-energy (fluid + EM) from one source of truth ---
  Real E_fluid, E_em;
  Real S_fluid_d[3], S_em_d[3];
  Real S_fluid_dd[3][3], S_em_dd[3][3];

  ValenciaStressEnergyDecomposed(rho, pgas, gamma_gas, v_u, v_d, W, B_u, B_d, b0, b_d, b2,
                                 g_dd_1d, E_fluid, E_em, S_fluid_d, S_em_d, S_fluid_dd,
                                 S_em_dd);

  const Real E_tot = E_fluid + E_em;

  // Raise S_j to S^i = g^{ij} S_j
  Real S_fluid_u[3] = {0.0, 0.0, 0.0};
  Real S_em_u[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      S_fluid_u[i] += g_u[i][j] * S_fluid_d[j];
      S_em_u[i] += g_u[i][j] * S_em_d[j];
    }
  }

  // Mixed stresses S^i_j = g^{ik} S_kj
  Real S_fluid_ud[3][3], S_em_ud[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Real val_f = 0.0;
      Real val_em = 0.0;
      for (int k = 0; k < 3; ++k) {
        val_f += g_u[i][k] * S_fluid_dd[k][j];
        val_em += g_u[i][k] * S_em_dd[k][j];
      }
      S_fluid_ud[i][j] = val_f;
      S_em_ud[i][j] = val_em;
    }
  }

  // --- D. Mass flux ---
  Real u_u[3] = {W * (v_u[0] - beta_u[0] / alp), W * (v_u[1] - beta_u[1] / alp),
                 W * (v_u[2] - beta_u[2] / alp)};
  const Real mass_flux_vec[3] = {alp * rho * u_u[0], alp * rho * u_u[1],
                                 alp * rho * u_u[2]};

  // Negative sign for inward accretion convention
  update.mdot -= (mass_flux_vec[0] * dS_d[0] + mass_flux_vec[1] * dS_d[1] +
                  mass_flux_vec[2] * dS_d[2]);

  // --- E. Energy fluxes (fluid + EM) ---
  Real FE_fluid[3], FE_em[3];
  for (int i = 0; i < 3; ++i) {
    FE_fluid[i] = alp * S_fluid_u[i] - beta_u[i] * E_fluid;
    FE_em[i] = alp * S_em_u[i] - beta_u[i] * E_em;
  }

  update.edot_fluid -=
      (FE_fluid[0] * dS_d[0] + FE_fluid[1] * dS_d[1] + FE_fluid[2] * dS_d[2]);
  update.edot_em -= (FE_em[0] * dS_d[0] + FE_em[1] * dS_d[1] + FE_em[2] * dS_d[2]);

  // --- F. Momentum fluxes ---
  Real FS_fluid[3][3], FS_em[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      FS_fluid[i][j] = alp * S_fluid_ud[i][j] - beta_u[i] * S_fluid_d[j];
      FS_em[i][j] = alp * S_em_ud[i][j] - beta_u[i] * S_em_d[j];
    }
  }

  Real d_pdot_fluid[3] = {0.0, 0.0, 0.0};
  Real d_pdot_em[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      d_pdot_fluid[j] += FS_fluid[i][j] * dS_d[i];
      d_pdot_em[j] += FS_em[i][j] * dS_d[i];
    }
  }

  // Store linear momentum fluxes
  update.pdot_x_fluid += d_pdot_fluid[0];
  update.pdot_y_fluid += d_pdot_fluid[1];
  update.pdot_z_fluid += d_pdot_fluid[2];

  update.pdot_x_em += d_pdot_em[0];
  update.pdot_y_em += d_pdot_em[1];
  update.pdot_z_em += d_pdot_em[2];

  // --- G. Angular-momentum fluxes ---
  // Angular momentum density uses J_a = ε_{aℓm} x^ℓ S_m (covariant S_m).
  // Therefore flux through surface is
  //   Ldot_a = ∮ ε_{aℓm} x^ℓ (F^j{}_m dΣ_j)
  // and we already computed A_m := (F^j{}_m dΣ_j) as d_pdot_*[m] above.
  {
    const Real x = position[0];
    const Real y = position[1];
    const Real z = position[2];

    // A_m = F^j{}_m dΣ_j  (covariant in m)
    const Real A_fluid_d[3] = {d_pdot_fluid[0], d_pdot_fluid[1], d_pdot_fluid[2]};
    const Real A_em_d[3] = {d_pdot_em[0], d_pdot_em[1], d_pdot_em[2]};

    // Ldot = r × A
    update.ldot_x_fluid += (y * A_fluid_d[2] - z * A_fluid_d[1]);
    update.ldot_y_fluid += (z * A_fluid_d[0] - x * A_fluid_d[2]);
    update.ldot_z_fluid += (x * A_fluid_d[1] - y * A_fluid_d[0]);

    update.ldot_x_em += (y * A_em_d[2] - z * A_em_d[1]);
    update.ldot_y_em += (z * A_em_d[0] - x * A_em_d[2]);
    update.ldot_z_em += (x * A_em_d[1] - y * A_em_d[0]);
  }

  // Every current below is UNDENSITIZED; dS_d already includes sqrt(gamma).
  Real wn = 0.0;
  for (int i = 0; i < 3; ++i) {
    wn += surface_velocity[i] * dS_d[i];
  }
  Real dmass = rho * W * wn;
  for (int i = 0; i < 3; ++i) {
    dmass -= mass_flux_vec[i] * dS_d[i];
  }
  update.moving[0] += dmass;
  update.moving[1] += fmax(dmass, 0.0);
  update.moving[2] += fmax(-dmass, 0.0);
  for (int part = 0; part < 2; ++part) {
    Real E = part == 0 ? E_fluid : E_em;
    Real S[3], P[3] = {}, FE[3], FS[3][3];
    for (int j = 0; j < 3; ++j) {
      S[j] = part == 0 ? S_fluid_d[j] : S_em_d[j];
      FE[j] = part == 0 ? FE_fluid[j] : FE_em[j];
      for (int i = 0; i < 3; ++i) {
        FS[i][j] = part == 0 ? FS_fluid[i][j] : FS_em[i][j];
      }
    }
    Real ein = E * wn, et = alp * E, etin;
    for (int j = 0; j < 3; ++j) {
      ein -= FE[j] * dS_d[j];
      et -= beta_u[j] * S[j];
      P[j] = S[j] * wn;
      for (int i = 0; i < 3; ++i) {
        P[j] -= FS[i][j] * dS_d[i];
      }
      update.moving[5 + 3 * part + j] += P[j];
    }
    update.moving[3 + part] += ein;
    Real x = position[0], y = position[1], z = position[2];
    update.moving[11 + 3 * part] += y * P[2] - z * P[1];
    update.moving[12 + 3 * part] += z * P[0] - x * P[2];
    update.moving[13 + 3 * part] += x * P[1] - y * P[0];
    etin = et * wn;
    for (int i = 0; i < 3; ++i) {
      Real ft = alp * FE[i];
      for (int j = 0; j < 3; ++j) {
        ft -= beta_u[j] * FS[i][j];
      }
      etin -= ft * dS_d[i];
    }
    update.moving[17 + part] += etin;
  }

  // --- H. Phi_B (magnetic flux) ---
  {
    // The EHT definition is Flux = 0.5 * Integral |B . dA|
    // B_u is the Eulerian magnetic field (already available).
    // dS_d is the covariant surface area element (normal vector * area).

    const Real flux_dot_dS = B_u[0] * dS_d[0] + B_u[1] * dS_d[1] + B_u[2] * dS_d[2];

    update.phiB += 0.5 * Kokkos::fabs(flux_dot_dS);
  }

  // --- I. Surface area ---
  Real dS_u[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      dS_u[i] += g_u[i][j] * dS_d[j];
    }
  }
  const Real dS_sq = dS_d[0] * dS_u[0] + dS_d[1] * dS_u[1] + dS_d[2] * dS_u[2];
  update.area += (dS_sq > 0.0 ? sqrt(dS_sq) : 0.0);
  update.covered_area += (dS_sq > 0.0 ? sqrt(dS_sq) : 0.0);
}
#endif  // DYN_GRMHD_VALENCIA_SURFACE_FLUX_HPP_
