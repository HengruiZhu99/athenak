//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file scalar_field_calcrhs.cpp
//! \brief Computes the canonical scalar-field evolution RHS.

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "mesh/mesh.hpp"
#include "scalar_field/scalar_field.hpp"
#include "utils/finite_diff.hpp"

namespace scalar_field {

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ScalarField::CalcRHS
//! \brief Compute the scalar-field RHS from the physical ADM variables.

template <int NGHOST>
TaskStatus ScalarField::CalcRHS(Driver *driver, int stage) {
  (void)driver;
  (void)stage;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  auto &adm_vars = pmy_pack->padm->adm;
  auto &excision_mask = pmy_pack->pcoord->excision_floor;
  auto &state = u0;
  auto &rhs = u_rhs;

  int const is = indcs.is;
  int const ie = indcs.ie;
  int const js = indcs.js;
  int const je = indcs.je;
  int const ks = indcs.ks;
  int const ke = indcs.ke;
  int const nmb = pmy_pack->nmb_thispack;
  int const ndim = 1 + static_cast<int>(pmy_pack->pmesh->multi_d) +
                   static_cast<int>(pmy_pack->pmesh->three_d);
  int const ncomp = ncomponents;
  PotentialData const pot = potential;
  Real const ko_diss = diss;
  bool const use_excision = excision;
  Real const target_phi = excision_phi;
  Real const target_pi = excision_pi;
  Real const inverse_damping_time = 1.0/excision_tdamp;

  par_for(
      "scalar field RHS", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(int const m, int const k, int const j, int const i) {
        // Smooth excision is a continuous-in-time relaxation, not a stagewise reset.
        // Bypassing the physical RHS also avoids inverting a singular interior metric.
        if (use_excision && excision_mask(m, k, j, i)) {
          for (int component = 0; component < ncomp; ++component) {
            const int iphi = 2*component;
            const int ipi = iphi + 1;
            rhs(m, iphi, k, j, i) =
                -(state(m, iphi, k, j, i) - target_phi) *
                inverse_damping_time;
            rhs(m, ipi, k, j, i) =
                -(state(m, ipi, k, j, i) - target_pi) *
                inverse_damping_time;
          }
          return;
        }

        Real idx[3] = {Real(1.0)/size.d_view(m).dx1,
                       Real(1.0)/size.d_view(m).dx2,
                       Real(1.0)/size.d_view(m).dx3};

        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
        dg_ddd.ZeroClear();
        Gamma_udd.ZeroClear();

        Real const detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i),
            adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i),
            adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i),
            adm_vars.g_dd(m, 2, 2, k, j, i));
        adm::SpatialInv(
            1.0/detg, adm_vars.g_dd(m, 0, 0, k, j, i),
            adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i),
            adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i),
            adm_vars.g_dd(m, 2, 2, k, j, i), &g_uu(0, 0), &g_uu(0, 1),
            &g_uu(0, 2), &g_uu(1, 1), &g_uu(1, 2), &g_uu(2, 2));

        Real dalpha_d[3] = {0.0, 0.0, 0.0};
        for (int d = 0; d < ndim; ++d) {
          dalpha_d[d] = Dx<NGHOST>(d, idx, adm_vars.alpha, m, k, j, i);
          for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
              dg_ddd(d, a, b) =
                  Dx<NGHOST>(d, idx, adm_vars.g_dd, m, a, b, k, j, i);
            }
          }
        }

        // Gamma^c_ab = g^cd (d_a g_bd + d_b g_ad - d_d g_ab)/2.
        for (int c = 0; c < 3; ++c) {
          for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
              for (int d = 0; d < 3; ++d) {
                Gamma_udd(c, a, b) +=
                    0.5*g_uu(c, d) *
                    (dg_ddd(a, b, d) + dg_ddd(b, a, d) -
                     dg_ddd(d, a, b));
              }
            }
          }
        }

        Real contracted_gamma_u[3] = {0.0, 0.0, 0.0};
        Real trace_k = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = a; b < 3; ++b) {
            Real const symm_factor = (a == b) ? 1.0 : 2.0;
            trace_k += symm_factor*g_uu(a, b) *
                       adm_vars.vK_dd(m, a, b, k, j, i);
            for (int c = 0; c < 3; ++c) {
              contracted_gamma_u[c] +=
                  symm_factor*g_uu(a, b)*Gamma_udd(c, a, b);
            }
          }
        }

        Real phi[2] = {0.0, 0.0};
        for (int component = 0; component < ncomp; ++component) {
          phi[component] = state(m, 2*component, k, j, i);
        }
        Real const q = FieldInvariant(ncomp, phi);
        Real const alpha = adm_vars.alpha(m, k, j, i);

        for (int component = 0; component < ncomp; ++component) {
          int const iphi = 2*component;
          int const ipi = iphi + 1;
          Real dphi_d[3] = {0.0, 0.0, 0.0};
          AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddphi_dd;
          ddphi_dd.ZeroClear();

          Real adv_phi = 0.0;
          Real adv_pi = 0.0;
          Real diss_phi = 0.0;
          Real diss_pi = 0.0;
          for (int a = 0; a < ndim; ++a) {
            dphi_d[a] = Dx<NGHOST>(a, idx, state, m, iphi, k, j, i);
            ddphi_dd(a, a) =
                Dxx<NGHOST>(a, idx, state, m, iphi, k, j, i);
            adv_phi += Lx<NGHOST>(a, idx, adm_vars.beta_u, state, m, a,
                                     iphi, k, j, i);
            adv_pi += Lx<NGHOST>(a, idx, adm_vars.beta_u, state, m, a,
                                    ipi, k, j, i);
            diss_phi += Diss<NGHOST>(a, idx, state, m, iphi, k, j, i);
            diss_pi += Diss<NGHOST>(a, idx, state, m, ipi, k, j, i);
          }
          for (int a = 0; a < ndim; ++a) {
            for (int b = a + 1; b < ndim; ++b) {
              ddphi_dd(a, b) =
                  Dxy<NGHOST>(a, b, idx, state, m, iphi, k, j, i);
            }
          }

          Real laplacian = 0.0;
          Real lapse_gradient = 0.0;
          for (int a = 0; a < 3; ++a) {
            for (int b = a; b < 3; ++b) {
              Real const symm_factor = (a == b) ? 1.0 : 2.0;
              laplacian += symm_factor*g_uu(a, b)*ddphi_dd(a, b);
              if (a == b) {
                lapse_gradient += g_uu(a, b)*dalpha_d[a]*dphi_d[b];
              } else {
                lapse_gradient +=
                    g_uu(a, b) *
                    (dalpha_d[a]*dphi_d[b] + dalpha_d[b]*dphi_d[a]);
              }
            }
          }
          for (int c = 0; c < 3; ++c) {
            laplacian -= contracted_gamma_u[c]*dphi_d[c];
          }

          Real const pi = state(m, ipi, k, j, i);
          rhs(m, iphi, k, j, i) = adv_phi - alpha*pi + ko_diss*diss_phi;
          rhs(m, ipi, k, j, i) =
              adv_pi - alpha*laplacian + alpha*trace_k*pi -
              lapse_gradient + alpha*pot.Derivative(phi[component], q) +
              ko_diss*diss_pi;
        }
      });

  return TaskStatus::complete;
}

template TaskStatus ScalarField::CalcRHS<2>(Driver *driver, int stage);
template TaskStatus ScalarField::CalcRHS<3>(Driver *driver, int stage);
template TaskStatus ScalarField::CalcRHS<4>(Driver *driver, int stage);

}  // namespace scalar_field
