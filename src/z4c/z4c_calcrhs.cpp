//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \fn TaskStatus Z4c::CalcRHS
//! \brief Computes the wave equation RHS

#include <math.h>

//#include <algorithm>
//#include <cinttypes>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <fstream>
//#include <limits>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/adm.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"
#include "z4c/tmunu.hpp"
#include "coordinates/cell_locations.hpp"

namespace z4c {

template <typename Centering, typename Symmetry, int NGHOST>
//! \fn void Z4c::CalcRHS(Driver *pdriver, int stage)
//! \brief compute rhs of the z4c equations
TaskStatus Z4c::CalcRHSImpl(Driver *pdriver, int stage) {
  CheckStateAdmissibility(pdriver, stage, Z4cStateCheckpoint::pre_rhs);
  const auto &layout = pmy_pack->pz4c->layout;
  auto &size = pmy_pack->pmb->mb_size;
  const int is = layout.is; const int ie = layout.ie;
  const int js = layout.js; const int je = layout.je;
  const int ks = layout.ks; const int ke = layout.ke;
  int nmb = pmy_pack->nmb_thispack;
  const int nx1 = layout.nx1;
  const int nx2 = layout.nx2;
  const int nx3 = layout.nx3;
  const int active_nx1 = ie - is + 1;
  const int active_nx2 = je - js + 1;
  const int active_nx3 = ke - ks + 1;

  if constexpr (std::is_same_v<Centering, VertexCenteredZ4c>) {
    const char *diagnostic = std::getenv("ATHENA_Z4C_VC_PRE_RHS_STATE_DIAGNOSTIC");
    if (diagnostic != nullptr && diagnostic[0] != '\0') {
      const auto host_state =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), u0);
      const auto host_coarse = pmy_pack->pmesh->multilevel
          ? Kokkos::create_mirror_view_and_copy(HostMemSpace(), coarse_u0)
          : decltype(Kokkos::create_mirror_view_and_copy(HostMemSpace(), coarse_u0))();
      pmy_pack->pmb->mb_gid.sync_host();
      std::ifstream prior(diagnostic);
      const bool exists = prior.good();
      prior.close();
      std::ofstream out(diagnostic, std::ios::app);
      if (!exists) {
        out << "cycle,time,stage,array,gid,variable,k,j,i,active,value\n";
      }
      for (int m = 0; m < nmb; ++m) {
        const int gid = pmy_pack->pmb->mb_gid.h_view(m);
        for (int v = 0; v < nz4c; ++v) {
          for (int k = 0; k < layout.n3; ++k) {
            for (int j = 0; j < layout.n2; ++j) {
              for (int i = 0; i < layout.n1; ++i) {
                const bool active = i >= is && i <= ie && j >= js && j <= je &&
                                    k >= ks && k <= ke;
                out << pmy_pack->pmesh->ncycle << ',' << std::setprecision(17)
                    << pmy_pack->pmesh->time << ',' << stage << ",fine," << gid
                    << ',' << v << ',' << k << ',' << j << ',' << i << ','
                    << active << ',' << host_state(m, v, k, j, i) << '\n';
              }
            }
          }
          if (pmy_pack->pmesh->multilevel) {
            for (int k = 0; k < layout.cn3; ++k) {
              for (int j = 0; j < layout.cn2; ++j) {
                for (int i = 0; i < layout.cn1; ++i) {
                  const bool active = i >= layout.cis && i <= layout.cie &&
                                      j >= layout.cjs && j <= layout.cje &&
                                      k >= layout.cks && k <= layout.cke;
                  out << pmy_pack->pmesh->ncycle << ',' << std::setprecision(17)
                      << pmy_pack->pmesh->time << ',' << stage << ",coarse," << gid
                      << ',' << v << ',' << k << ',' << j << ',' << i << ','
                      << active << ',' << host_coarse(m, v, k, j, i) << '\n';
                }
              }
            }
          }
        }
      }
      out.flush();
      if (!out) {
        std::cerr << "### FATAL ERROR: failed to write VC pre-RHS state diagnostic"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  auto &z4c = pmy_pack->pz4c->z4c;
  auto &rhs = pmy_pack->pz4c->rhs;
  auto &telegraph_mu = pmy_pack->pz4c->u_telegraph_mu;
  auto &opt = pmy_pack->pz4c->opt;
  Real time = pmy_pack->pmesh->time;
  bool is_vacuum = (pmy_pack->ptmunu == nullptr) ? true : false;
  Tmunu::Tmunu_vars tmunu;
  if (!is_vacuum) tmunu = pmy_pack->ptmunu->tmunu;

  // Gaussian roll for the kappa1 input coefficient (host-side; capture by value).
  // In max-|K| mode both endpoints are dimensionless multipliers.
  Real kappa1_effective = opt.damp_kappa1;
  if (opt.roll_kappa && time >= opt.kappa_roll_start_time) {
    // Gaussian stitch: S(t0)=1, S→0 as t→\infty
    Real s = (time - opt.kappa_roll_start_time) / opt.roll_window;
    Real S = exp(-2.30258509299 * s * s);  // smooth, C^\infty falloff
    // prefactor chosen to have S=0.1 at the end of the roll_window
    kappa1_effective = opt.target_kappa1
                      + (opt.damp_kappa1 - opt.target_kappa1) * S;
  }
  // A fixed inverse-length coefficient injects a preferred physical scale.  Compute one
  // global curvature scale per RHS call and share it between every opt-in scale-invariant
  // gauge/damping term.
  const bool use_max_K_scale =
      (opt.telegraph_lapse &&
       opt.telegraph_damping_prescription != TelegraphDampingPrescription::fixed) ||
      opt.shift_eta_max_K || opt.damp_kappa1_max_K;
  Real max_abs_K = 1.0;
  if (use_max_K_scale) {
    const int nmkji = nmb * active_nx3 * active_nx2 * active_nx1;
    const int nkji = active_nx3 * active_nx2 * active_nx1;
    const int nji = active_nx2 * active_nx1;
    max_abs_K = 0.0;

    Kokkos::parallel_reduce(
        "z4c global max abs K",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, Real &mb_max_abs_K) {
          int m = idx / nkji;
          int k = (idx - m * nkji) / nji;
          int j = (idx - m * nkji - k * nji) / active_nx1;
          int i = (idx - m * nkji - k * nji - j * active_nx1) + is;
          k += ks;
          j += js;

          Real K = z4c.vKhat(m, k, j, i) + 2.0 * z4c.vTheta(m, k, j, i);
          mb_max_abs_K = fmax(mb_max_abs_K, fabs(K));
        },
        Kokkos::Max<Real>(max_abs_K));

#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &max_abs_K, 1, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
  }
  const Real kappa1_eff =
      kappa1_effective * (opt.damp_kappa1_max_K ? max_abs_K : 1.0);
  const Real shift_eta_eff =
      opt.shift_eta * (opt.shift_eta_max_K ? max_abs_K : 1.0);

  // B_i is dimensionless in alpha_t = chi div(B).  The scale-invariant
  // parameterization is
  //
  //   Q(x)=mu(x)/max|K|, tau_eff=tau/max|K|,
  //   kappa_eff=kappa/max|K|,
  //
  // so Q/tau_eff=mu/tau and kappa_eff/tau_eff=kappa/tau.  The helper below
  // evaluates these cancelled coefficients directly, including when max|K|=0.

  const bool collect_rhs_stage_diagnostics =
      opt.rhs_stage_diagnostics && time >= opt.rhs_stage_diagnostics_start_time;
  const bool collect_chi_provenance =
      opt.chi_parent_provenance.enabled && time >= opt.chi_parent_provenance.start_time;
  const bool prescribed_zero_shift =
      opt.shift_mode == Z4cShiftMode::prescribed_zero;
  const bool use_o2_shift_advection =
      opt.shift_advection_order == Z4cShiftAdvectionOrder::o2;
  auto &chi_provenance_terms = pmy_pack->pz4c->chi_provenance_terms;
  DvceArray5D<Real> rhs_stage_terms;
  DvceArray5D<Real> rhs_raw_pre_axis;
  DvceArray5D<Real> rhs_post_axis_pre_ko;
  if (collect_rhs_stage_diagnostics) {
    rhs_stage_terms = DvceArray5D<Real>("z4c rhs stage terms", nmb, 75,
                                       u_rhs.extent_int(2), u_rhs.extent_int(3),
                                       u_rhs.extent_int(4));
    rhs_raw_pre_axis = DvceArray5D<Real>(
        "z4c raw pre-axis RHS", u_rhs.extent_int(0), u_rhs.extent_int(1),
        u_rhs.extent_int(2), u_rhs.extent_int(3), u_rhs.extent_int(4));
    rhs_post_axis_pre_ko = DvceArray5D<Real>(
        "z4c post-axis pre-KO RHS", u_rhs.extent_int(0), u_rhs.extent_int(1),
        u_rhs.extent_int(2), u_rhs.extent_int(3), u_rhs.extent_int(4));
  }

  // ===================================================================================
  // Main RHS calculation
  //
  par_for(
      "z4c rhs loop", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        // Define scratch arrays to be used in the following calculations

        // Gamma computed from the metric
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
        // inverse of conf. metric
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
        // g^cd A_ac A_db
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> AA_dd;
        // Ricci tensor
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> R_dd;
        // Ricci tensor, conformal contribution
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Rphi_dd;
        // 2nd differential of the lapse
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddalpha_dd;
        // 2nd differential of phi
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Ddphi_dd;

        // Christoffel symbols of 1st kind
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
        // Christoffel symbols of 2nd kind
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;

        // auxiliary derivatives

        // lapse 1st drvts
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
        // 2nd "divergence" of beta
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> ddbeta_d;
        // chi 1st drvts
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
        // phi 1st drvts
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dphi_d;
        // lapse 2nd drvts
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddalpha_dd;
        // shift 1st drvts
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;
        // chi 2nd drvts
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> ddchi_dd;
        // Gamma 1st drvts
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;

        // metric 1st drvts
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
        // shift 2nd drvts
        AthenaPointTensor<Real, TensorSymm::ISYM2, 3, 3> ddbeta_ddu;

        // metric 2nd drvts
        AthenaPointTensor<Real, TensorSymm::SYM22, 3, 4> ddg_dddd;

        // Lie derivative of conf. 3-metric
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> Lg_dd;
        // Lie derivative of A
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> LA_dd;

        Real idx[] = {1 / size.d_view(m).dx1, 1 / size.d_view(m).dx2,
                      1 / size.d_view(m).dx3};
        auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
            idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);

        // -----------------------------------------------------------------------------------
        // Initialize everything to zero
        //
        // Scalars

        // auxiliary Lie derivatives along the shift vector
        // Lie derivative of chi
        Real Lchi = 0.0;
        // Lie derivative of Khat
        Real LKhat = 0.0;
        // Lie derivative of Theta
        Real LTheta = 0.0;

        // determinant of three metric
        Real detg = 0.0;
        // bounded version of chi
        Real chi_guarded = 0.0;
        // 1/psi4
        Real oopsi4 = 0.0;
        // trace of A
        Real AA = 0.0;
        // Ricci scalar
        Real R = 0.0;
        // tilde H
        Real Ht = 0.0;
        // trace of extrinsic curvature
        Real K = 0.0;
        // Trace of S_ik
        Real S = 0.0;
        // Trace of Ddalpha_dd
        Real Ddalpha = 0.0;

        // d_a beta^a
        Real dbeta = 0.0;

        //
        // Vectors
        Gamma_u.ZeroClear();
        ddbeta_d.ZeroClear();

        //
        // Symmetric tensors
        Lg_dd.ZeroClear();
        LA_dd.ZeroClear();
        AA_dd.ZeroClear();
        R_dd.ZeroClear();
        Gamma_udd.ZeroClear();

        // -----------------------------------------------------------------------------------
        // 1st derivatives
        //
        // Scalars
        for (int a = 0; a < 3; ++a) {
          dalpha_d(a) = derivatives.ScalarFirst(a, z4c.alpha);
          dchi_d(a) = derivatives.ScalarFirst(a, z4c.chi);
        }

        // Vectors
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            dbeta_du(b, a) = derivatives.VectorFirst(b, a, z4c.beta_u);
            dGam_du(b, a) = derivatives.VectorFirst(b, a, z4c.vGam_u);
          }

        // Tensors
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b)
            for (int c = 0; c < 3; ++c) {
              dg_ddd(c, a, b) =
                  derivatives.template TensorFirst<TensorVariance::all_lower>(
                      c, a, b, z4c.g_dd);
            }

        // -----------------------------------------------------------------------------------
        // 2nd derivatives
        //
        // Scalars
        for (int a = 0; a < 3; ++a) {
          ddalpha_dd(a, a) = derivatives.ScalarSecond(a, a, z4c.alpha);
          ddchi_dd(a, a) = derivatives.ScalarSecond(a, a, z4c.chi);

          for (int b = a + 1; b < 3; ++b) {
            ddalpha_dd(a, b) = derivatives.ScalarSecond(a, b, z4c.alpha);
            ddchi_dd(a, b) = derivatives.ScalarSecond(a, b, z4c.chi);
          }
        }

        // Vectors
        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < 3; ++a) {
            ddbeta_ddu(a, a, c) =
                derivatives.VectorSecond(a, a, c, z4c.beta_u);
            for (int b = a + 1; b < 3; ++b) {
              ddbeta_ddu(a, b, c) =
                  derivatives.VectorSecond(a, b, c, z4c.beta_u);
            }
          }

        // Tensors
        for (int c = 0; c < 3; ++c)
          for (int d = c; d < 3; ++d)
            for (int a = 0; a < 3; ++a) {
              ddg_dddd(a, a, c, d) =
                  derivatives.template TensorSecond<TensorVariance::all_lower>(
                      a, a, c, d, z4c.g_dd);
              for (int b = a + 1; b < 3; ++b) {
                ddg_dddd(a, b, c, d) =
                    derivatives.template TensorSecond<TensorVariance::all_lower>(
                        a, b, c, d, z4c.g_dd);
              }
            }

        // -----------------------------------------------------------------------------------
        // Advective derivatives
        //

        //
        // Scalars
        // Keep the production expression byte-for-byte and collect directional
        // components only as shadows.  The diagnostic must never rebuild Lchi.
        Lchi = use_o2_shift_advection
                   ? derivatives.ScalarAdvectiveO2(z4c.beta_u, z4c.chi)
                   : derivatives.ScalarAdvective(z4c.beta_u, z4c.chi);
        if (collect_chi_provenance) {
          for (int direction = 0; direction < 3; ++direction) {
            const Real contribution = derivatives.DirectionalScalarAdvective(
                direction, z4c.beta_u, z4c.chi);
            chi_provenance_terms(m, chi_adv_rho + direction, k, j, i) = contribution;
          }
        }
        LKhat = use_o2_shift_advection
                    ? derivatives.ScalarAdvectiveO2(z4c.beta_u, z4c.vKhat)
                    : derivatives.ScalarAdvective(z4c.beta_u, z4c.vKhat);
        LTheta = use_o2_shift_advection
                     ? derivatives.ScalarAdvectiveO2(z4c.beta_u, z4c.vTheta)
                     : derivatives.ScalarAdvective(z4c.beta_u, z4c.vTheta);

        // Tensors
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b)
            {
              Lg_dd(a, b) = use_o2_shift_advection
                  ? derivatives.template TensorAdvectiveO2<TensorVariance::all_lower>(
                        a, b, z4c.beta_u, z4c.g_dd)
                  : derivatives.template TensorAdvective<TensorVariance::all_lower>(
                        a, b, z4c.beta_u, z4c.g_dd);
              LA_dd(a, b) = use_o2_shift_advection
                  ? derivatives.template TensorAdvectiveO2<TensorVariance::all_lower>(
                        a, b, z4c.beta_u, z4c.vA_dd)
                  : derivatives.template TensorAdvective<TensorVariance::all_lower>(
                        a, b, z4c.beta_u, z4c.vA_dd);
            }

        // -----------------------------------------------------------------------------------
        // Get K from Khat
        //
        K = z4c.vKhat(m, k, j, i) + 2. * z4c.vTheta(m, k, j, i);

        // -----------------------------------------------------------------------------------
        // Inverse metric

        detg = adm::SpatialDet(z4c.g_dd(m, 0, 0, k, j, i), z4c.g_dd(m, 0, 1, k, j, i),
                               z4c.g_dd(m, 0, 2, k, j, i), z4c.g_dd(m, 1, 1, k, j, i),
                               z4c.g_dd(m, 1, 2, k, j, i), z4c.g_dd(m, 2, 2, k, j, i));
        adm::SpatialInv(1.0 / detg, z4c.g_dd(m, 0, 0, k, j, i),
                        z4c.g_dd(m, 0, 1, k, j, i), z4c.g_dd(m, 0, 2, k, j, i),
                        z4c.g_dd(m, 1, 1, k, j, i), z4c.g_dd(m, 1, 2, k, j, i),
                        z4c.g_dd(m, 2, 2, k, j, i), &g_uu(0, 0), &g_uu(0, 1), &g_uu(0, 2),
                        &g_uu(1, 1), &g_uu(1, 2), &g_uu(2, 2));

        // -----------------------------------------------------------------------------------
        // Christoffel symbols

        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b) {
              Gamma_ddd(c, a, b) =
                  0.5 * (dg_ddd(a, b, c) + dg_ddd(b, a, c) - dg_ddd(c, a, b));
            }
        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b)
              for (int d = 0; d < 3; ++d) {
                Gamma_udd(c, a, b) += g_uu(c, d) * Gamma_ddd(d, a, b);
              }
        // Gamma's computed from the conformal metric (not evolved)
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c) {
              Gamma_u(a) += g_uu(b, c) * Gamma_udd(a, b, c);
            }

        // -----------------------------------------------------------------------------------
        // Curvature of conformal metric
        //
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            for (int c = 0; c < 3; ++c) {
              R_dd(a, b) +=
                  0.5 * (z4c.g_dd(m, c, a, k, j, i) * dGam_du(b, c) +
                         z4c.g_dd(m, c, b, k, j, i) * dGam_du(a, c) +
                         Gamma_u(c) * (Gamma_ddd(a, b, c) + Gamma_ddd(b, a, c)));
            }
            for (int c = 0; c < 3; ++c)
              for (int d = 0; d < 3; ++d) {
                R_dd(a, b) -= 0.5 * g_uu(c, d) * ddg_dddd(c, d, a, b);
              }
            for (int c = 0; c < 3; ++c)
              for (int d = 0; d < 3; ++d)
                for (int e = 0; e < 3; ++e) {
                  R_dd(a, b) += g_uu(c, d) * (Gamma_udd(e, c, a) * Gamma_ddd(b, e, d) +
                                              Gamma_udd(e, c, b) * Gamma_ddd(a, e, d) +
                                              Gamma_udd(e, a, d) * Gamma_ddd(e, c, b));
                }
          }

        // -----------------------------------------------------------------------------------
        // Derivatives of conformal factor phi
        //
        chi_guarded = (z4c.chi(m, k, j, i) > opt.chi_div_floor) ? z4c.chi(m, k, j, i)
                                                                : opt.chi_div_floor;
        oopsi4 = pow(chi_guarded, -4. / opt.chi_psi_power);
        for (int a = 0; a < 3; ++a) {
          dphi_d(a) = dchi_d(a) / (chi_guarded * opt.chi_psi_power);
        }
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            Ddphi_dd(a, b) = ddchi_dd(a, b) / (chi_guarded * opt.chi_psi_power) -
                             opt.chi_psi_power * dphi_d(a) * dphi_d(b);
            for (int c = 0; c < 3; ++c) {
              Ddphi_dd(a, b) -= Gamma_udd(c, a, b) * dphi_d(c);
            }
          }

        // -----------------------------------------------------------------------------------
        // Curvature contribution from conformal factor
        //
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            Rphi_dd(a, b) = 4. * dphi_d(a) * dphi_d(b) - 2. * Ddphi_dd(a, b);
            for (int c = 0; c < 3; ++c)
              for (int d = 0; d < 3; ++d) {
                Rphi_dd(a, b) -= 2. * z4c.g_dd(m, a, b, k, j, i) * g_uu(c, d) *
                                 (Ddphi_dd(c, d) + 2. * dphi_d(c) * dphi_d(d));
              }
          }

        // -----------------------------------------------------------------------------------
        // Trace of the matter stress tensor
        //
        if (!is_vacuum) {
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
              S += oopsi4 * g_uu(a, b) * tmunu.S_dd(m, a, b, k, j, i);
            }
        }

        // -----------------------------------------------------------------------------------
        // 2nd covariant derivative of the lapse
        // TODO(JMF): This could potentially be sped up by calculating d_i phi d^i alpha
        // beforehand.
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            Ddalpha_dd(a, b) = ddalpha_dd(a, b) -
                               2. * (dphi_d(a) * dalpha_d(b) + dphi_d(b) * dalpha_d(a));
            for (int c = 0; c < 3; ++c) {
              Ddalpha_dd(a, b) -= Gamma_udd(c, a, b) * dalpha_d(c);
              for (int d = 0; d < 3; ++d) {
                Ddalpha_dd(a, b) += 2. * z4c.g_dd(m, a, b, k, j, i) * g_uu(c, d) *
                                    dphi_d(c) * dalpha_d(d);
              }
            }
          }

        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            Ddalpha += oopsi4 * g_uu(a, b) * Ddalpha_dd(a, b);
          }

        // -----------------------------------------------------------------------------------
        // Contractions of A_ab, inverse, and derivatives
        //
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
              for (int d = 0; d < 3; ++d) {
                AA_dd(a, b) += g_uu(c, d) * z4c.vA_dd(m, a, c, k, j, i) *
                               z4c.vA_dd(m, d, b, k, j, i);
              }
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            AA += g_uu(a, b) * AA_dd(a, b);
          }
        // -----------------------------------------------------------------------------------
        // Ricci scalar
        //
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            R += oopsi4 * g_uu(a, b) * (R_dd(a, b) + Rphi_dd(a, b));
          }

        // -----------------------------------------------------------------------------------
        // Hamiltonian constraint
        //
        // Note that the matter term is *not* included here; this is included explicitly
        // when calculating d_t \Theta.
        Ht = R + (2. / 3.) * SQR(K) - AA;  // - 16.*M_PI*tmunu.E(m,k,j,i);

        // -----------------------------------------------------------------------------------
        // Finalize advective (Lie) derivatives
        //
        // Shift vector contractions
        for (int a = 0; a < 3; ++a) {
          dbeta += dbeta_du(a, a);
        }
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            ddbeta_d(a) += (1. / 3.) * ddbeta_ddu(a, b, b);
          }

        // Finalize Lchi
        Lchi += (1. / 6.) * opt.chi_psi_power * chi_guarded * dbeta;
        if (collect_chi_provenance) {
          const Real chi_lie_divergence_term =
              (1. / 6.) * opt.chi_psi_power * chi_guarded * dbeta;
          chi_provenance_terms(m, chi_lie_divergence, k, j, i) =
              chi_lie_divergence_term;
          chi_provenance_terms(m, chi_adv_total_production, k, j, i) = Lchi;
        }

        // Finalize Lg_dd and LA_dd
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            Lg_dd(a, b) -= (2. / 3.) * z4c.g_dd(m, a, b, k, j, i) * dbeta;
            for (int c = 0; c < 3; ++c) {
              Lg_dd(a, b) += dbeta_du(a, c) * z4c.g_dd(m, b, c, k, j, i);
              Lg_dd(a, b) += dbeta_du(b, c) * z4c.g_dd(m, a, c, k, j, i);
            }
          }
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            LA_dd(a, b) -= (2. / 3.) * z4c.vA_dd(m, a, b, k, j, i) * dbeta;
            for (int c = 0; c < 3; ++c) {
              LA_dd(a, b) += dbeta_du(b, c) * z4c.vA_dd(m, a, c, k, j, i);
              LA_dd(a, b) += dbeta_du(a, c) * z4c.vA_dd(m, b, c, k, j, i);
            }
          }

        // -----------------------------------------------------------------------------------
        // Assemble RHS
        //
        // Khat, chi, and Theta
        rhs.vKhat(m, k, j, i) =
            -Ddalpha + z4c.alpha(m, k, j, i) * (AA + (1. / 3.) * SQR(K)) + LKhat +
            kappa1_eff * (1 - opt.damp_kappa2) * z4c.alpha(m, k, j, i) *
                z4c.vTheta(m, k, j, i);
        // Matter term
        if (!is_vacuum) {
          rhs.vKhat(m, k, j, i) +=
              4. * M_PI * z4c.alpha(m, k, j, i) * (S + tmunu.E(m, k, j, i));
        }
        rhs.chi(m, k, j, i) =
            Lchi - (1. / 6.) * opt.chi_psi_power * chi_guarded *
                       z4c.alpha(m, k, j, i) * K;
        if (collect_chi_provenance) {
          const Real chi_curvature_term = rhs.chi(m, k, j, i) - Lchi;
          chi_provenance_terms(m, chi_curvature_source, k, j, i) =
              chi_curvature_term;
          chi_provenance_terms(m, chi_rhs_before_ko, k, j, i) =
              rhs.chi(m, k, j, i);
        }
        rhs.vTheta(m, k, j, i) =
            LTheta +
            z4c.alpha(m, k, j, i) *
                (0.5 * Ht - (2. + opt.damp_kappa2) * kappa1_eff * z4c.vTheta(m, k, j, i));
        // Matter term
        if (!is_vacuum) {
          rhs.vTheta(m, k, j, i) -=
              8. * M_PI * z4c.alpha(m, k, j, i) * tmunu.E(m, k, j, i);
        }
        // If BSSN is enabled, theta is disabled.
        rhs.vTheta(m, k, j, i) *= opt.use_z4c;
        // g and A
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) {
            rhs.g_dd(m, a, b, k, j, i) =
                -2. * z4c.alpha(m, k, j, i) * z4c.vA_dd(m, a, b, k, j, i) + Lg_dd(a, b);
            const Real a_geometric =
                oopsi4 * (-Ddalpha_dd(a, b) +
                          z4c.alpha(m, k, j, i) * (R_dd(a, b) + Rphi_dd(a, b)));
            const Real a_trace_subtraction =
                -(1. / 3.) * z4c.g_dd(m, a, b, k, j, i) *
                (-Ddalpha + z4c.alpha(m, k, j, i) * R);
            const Real a_nonlinear =
                z4c.alpha(m, k, j, i) *
                (K * z4c.vA_dd(m, a, b, k, j, i) - 2. * AA_dd(a, b));
            const Real a_lie = LA_dd(a, b);
            rhs.vA_dd(m, a, b, k, j, i) =
                a_geometric + a_trace_subtraction + a_nonlinear + a_lie;
            if (collect_rhs_stage_diagnostics) {
              const int pair = (a == 0) ? b : ((a == 1) ? 3 + (b - 1) : 5);
              rhs_stage_terms(m, pair, k, j, i) = a_geometric;
              rhs_stage_terms(m, 6 + pair, k, j, i) = a_trace_subtraction;
              rhs_stage_terms(m, 12 + pair, k, j, i) = a_nonlinear;
              rhs_stage_terms(m, 18 + pair, k, j, i) = a_lie;
              rhs_stage_terms(m, 36 + pair, k, j, i) =
                  -oopsi4 * Ddalpha_dd(a, b);
              rhs_stage_terms(m, 42 + pair, k, j, i) =
                  oopsi4 * z4c.alpha(m, k, j, i) *
                  (R_dd(a, b) + Rphi_dd(a, b));
              rhs_stage_terms(m, 48 + pair, k, j, i) =
                  (1. / 3.) * z4c.g_dd(m, a, b, k, j, i) * Ddalpha;
              rhs_stage_terms(m, 54 + pair, k, j, i) =
                  -(1. / 3.) * z4c.g_dd(m, a, b, k, j, i) *
                  z4c.alpha(m, k, j, i) * R;
            }
            // Matter term
            if (!is_vacuum) {
              rhs.vA_dd(m, a, b, k, j, i) -= 8. * M_PI * z4c.alpha(m, k, j, i) *
                                             (oopsi4 * tmunu.S_dd(m, a, b, k, j, i) -
                                              (1. / 3.) * S * z4c.g_dd(m, a, b, k, j, i));
            }
          }
      });

  par_for(
      "z4c Gamma rhs loop", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Gamma_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> DA_u;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> A_uu;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_ddd;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> Gamma_udd;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dKhat_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dTheta_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> ddbeta_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_advective_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_expansion_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_ddiv_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_contraction_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LGam_second_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dbeta_du;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 2> dGam_du;
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 3> dg_ddd;
        AthenaPointTensor<Real, TensorSymm::ISYM2, 3, 3> ddbeta_ddu;

        Gamma_u.ZeroClear();
        DA_u.ZeroClear();
        A_uu.ZeroClear();
        Gamma_udd.ZeroClear();
        ddbeta_d.ZeroClear();
        LGam_u.ZeroClear();
        LGam_advective_u.ZeroClear();
        LGam_expansion_u.ZeroClear();
        LGam_ddiv_u.ZeroClear();
        LGam_contraction_u.ZeroClear();
        LGam_second_u.ZeroClear();

        Real idx[] = {1 / size.d_view(m).dx1, 1 / size.d_view(m).dx2,
                      1 / size.d_view(m).dx3};
        auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
            idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);
        Real dbeta = 0.0;
        Real chi_guarded = (z4c.chi(m, k, j, i) > opt.chi_div_floor) ? z4c.chi(m, k, j, i)
                                                                     : opt.chi_div_floor;

        for (int a = 0; a < 3; ++a) {
          dalpha_d(a) = derivatives.ScalarFirst(a, z4c.alpha);
          dchi_d(a) = derivatives.ScalarFirst(a, z4c.chi);
          dKhat_d(a) = derivatives.ScalarFirst(a, z4c.vKhat);
          dTheta_d(a) = derivatives.ScalarFirst(a, z4c.vTheta);
        }

        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            dbeta_du(b, a) = derivatives.VectorFirst(b, a, z4c.beta_u);
            dGam_du(b, a) = derivatives.VectorFirst(b, a, z4c.vGam_u);
          }

        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b)
            for (int c = 0; c < 3; ++c) {
              dg_ddd(c, a, b) =
                  derivatives.template TensorFirst<TensorVariance::all_lower>(
                      c, a, b, z4c.g_dd);
            }

        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < 3; ++a) {
            ddbeta_ddu(a, a, c) =
                derivatives.VectorSecond(a, a, c, z4c.beta_u);
            for (int b = a + 1; b < 3; ++b) {
              ddbeta_ddu(a, b, c) =
                  derivatives.VectorSecond(a, b, c, z4c.beta_u);
            }
          }

        for (int b = 0; b < 3; ++b) {
          LGam_advective_u(b) = use_o2_shift_advection
              ? derivatives.VectorAdvectiveO2(b, z4c.beta_u, z4c.vGam_u)
              : derivatives.VectorAdvective(b, z4c.beta_u, z4c.vGam_u);
          LGam_u(b) = LGam_advective_u(b);
        }

        Real detg =
            adm::SpatialDet(z4c.g_dd(m, 0, 0, k, j, i), z4c.g_dd(m, 0, 1, k, j, i),
                            z4c.g_dd(m, 0, 2, k, j, i), z4c.g_dd(m, 1, 1, k, j, i),
                            z4c.g_dd(m, 1, 2, k, j, i), z4c.g_dd(m, 2, 2, k, j, i));
        adm::SpatialInv(1.0 / detg, z4c.g_dd(m, 0, 0, k, j, i),
                        z4c.g_dd(m, 0, 1, k, j, i), z4c.g_dd(m, 0, 2, k, j, i),
                        z4c.g_dd(m, 1, 1, k, j, i), z4c.g_dd(m, 1, 2, k, j, i),
                        z4c.g_dd(m, 2, 2, k, j, i), &g_uu(0, 0), &g_uu(0, 1), &g_uu(0, 2),
                        &g_uu(1, 1), &g_uu(1, 2), &g_uu(2, 2));

        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b) {
              Gamma_ddd(c, a, b) =
                  0.5 * (dg_ddd(a, b, c) + dg_ddd(b, a, c) - dg_ddd(c, a, b));
            }
        for (int c = 0; c < 3; ++c)
          for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b)
              for (int d = 0; d < 3; ++d) {
                Gamma_udd(c, a, b) += g_uu(c, d) * Gamma_ddd(d, a, b);
              }
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c) {
              Gamma_u(a) += g_uu(b, c) * Gamma_udd(a, b, c);
            }

        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b)
            for (int c = 0; c < 3; ++c)
              for (int d = 0; d < 3; ++d) {
                A_uu(a, b) += g_uu(a, c) * g_uu(b, d) * z4c.vA_dd(m, c, d, k, j, i);
              }
        // TODO(JMF): dchi_d/chi_guarded is opt.chi_psi_power * dphi_d.
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            DA_u(a) -= (3. / 2.) * A_uu(a, b) * dchi_d(b) / chi_guarded;
            DA_u(a) -= (1. / 3.) * g_uu(a, b) * (2. * dKhat_d(b) + dTheta_d(b));
          }
          for (int b = 0; b < 3; ++b)
            for (int c = 0; c < 3; ++c) {
              DA_u(a) += Gamma_udd(a, b, c) * A_uu(b, c);
            }
        }

        for (int a = 0; a < 3; ++a) {
          dbeta += dbeta_du(a, a);
        }
        for (int a = 0; a < 3; ++a)
          for (int b = 0; b < 3; ++b) {
            ddbeta_d(a) += (1. / 3.) * ddbeta_ddu(a, b, b);
          }

        // Finalize LGam_u (note that this is not a real Lie derivative)
        for (int a = 0; a < 3; ++a) {
          LGam_expansion_u(a) = (2. / 3.) * Gamma_u(a) * dbeta;
          LGam_u(a) += LGam_expansion_u(a);
          for (int b = 0; b < 3; ++b) {
            LGam_ddiv_u(a) += g_uu(a, b) * ddbeta_d(b);
            LGam_contraction_u(a) -= Gamma_u(b) * dbeta_du(b, a);
            for (int c = 0; c < 3; ++c) {
              LGam_second_u(a) += g_uu(b, c) * ddbeta_ddu(b, c, a);
            }
          }
          LGam_u(a) += LGam_ddiv_u(a) + LGam_contraction_u(a) + LGam_second_u(a);
        }

        // Gamma's
        for (int a = 0; a < 3; ++a) {
          const Real gamma_divergence = 2. * z4c.alpha(m, k, j, i) * DA_u(a);
          const Real gamma_lie = LGam_u(a);
          const Real gamma_damping =
              -2. * z4c.alpha(m, k, j, i) * kappa1_eff *
              (z4c.vGam_u(m, a, k, j, i) - Gamma_u(a));
          Real gamma_lapse_gradient = 0.0;
          for (int b = 0; b < 3; ++b) {
            gamma_lapse_gradient -= 2. * A_uu(a, b) * dalpha_d(b);
            // Matter term
            if (!is_vacuum) {
              gamma_lapse_gradient -= 16. * M_PI * z4c.alpha(m, k, j, i) *
                                      g_uu(a, b) * tmunu.S_d(m, b, k, j, i);
            }
          }
          rhs.vGam_u(m, a, k, j, i) =
              gamma_divergence + gamma_lie + gamma_damping + gamma_lapse_gradient;
          if (collect_rhs_stage_diagnostics) {
            rhs_stage_terms(m, 24 + a, k, j, i) = gamma_divergence;
            rhs_stage_terms(m, 27 + a, k, j, i) = gamma_lie;
            rhs_stage_terms(m, 30 + a, k, j, i) = gamma_damping;
            rhs_stage_terms(m, 33 + a, k, j, i) = gamma_lapse_gradient;
            rhs_stage_terms(m, 60 + a, k, j, i) = LGam_advective_u(a);
            rhs_stage_terms(m, 63 + a, k, j, i) = LGam_expansion_u(a);
            rhs_stage_terms(m, 66 + a, k, j, i) = LGam_ddiv_u(a);
            rhs_stage_terms(m, 69 + a, k, j, i) = LGam_contraction_u(a);
            rhs_stage_terms(m, 72 + a, k, j, i) = LGam_second_u(a);
          }
        }
      });

  par_for(
      "z4c gauge rhs loop", DevExeSpace(), 0, nmb - 1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dalpha_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> dchi_d;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> Lbeta_u;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> LB_d;
        Lbeta_u.ZeroClear();
        LB_d.ZeroClear();

        Real idx[] = {1 / size.d_view(m).dx1, 1 / size.d_view(m).dx2,
                      1 / size.d_view(m).dx3};
        auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
            idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);
        Real Lalpha = 0.0;
        Real dB = 0.0;
        Real const alpha = z4c.alpha(m, k, j, i);
        Real const chi = z4c.chi(m, k, j, i);
        Real chi_guarded = (chi > opt.chi_div_floor) ? chi : opt.chi_div_floor;

        Real detg =
            adm::SpatialDet(z4c.g_dd(m, 0, 0, k, j, i), z4c.g_dd(m, 0, 1, k, j, i),
                            z4c.g_dd(m, 0, 2, k, j, i), z4c.g_dd(m, 1, 1, k, j, i),
                            z4c.g_dd(m, 1, 2, k, j, i), z4c.g_dd(m, 2, 2, k, j, i));
        adm::SpatialInv(1.0 / detg, z4c.g_dd(m, 0, 0, k, j, i),
                        z4c.g_dd(m, 0, 1, k, j, i), z4c.g_dd(m, 0, 2, k, j, i),
                        z4c.g_dd(m, 1, 1, k, j, i), z4c.g_dd(m, 1, 2, k, j, i),
                        z4c.g_dd(m, 2, 2, k, j, i), &g_uu(0, 0), &g_uu(0, 1), &g_uu(0, 2),
                        &g_uu(1, 1), &g_uu(1, 2), &g_uu(2, 2));

        for (int a = 0; a < 3; ++a) {
          dalpha_d(a) = derivatives.ScalarFirst(a, z4c.alpha);
          dchi_d(a) = derivatives.ScalarFirst(a, z4c.chi);
        }
        Lalpha = use_o2_shift_advection
                     ? derivatives.ScalarAdvectiveO2(z4c.beta_u, z4c.alpha)
                     : derivatives.ScalarAdvective(z4c.beta_u, z4c.alpha);

        for (int b = 0; b < 3; ++b) {
          Lbeta_u(b) = use_o2_shift_advection
                           ? derivatives.VectorAdvectiveO2(
                                 b, z4c.beta_u, z4c.beta_u)
                           : derivatives.VectorAdvective(
                                 b, z4c.beta_u, z4c.beta_u);
          if (opt.telegraph_lapse) {
            LB_d(b) = use_o2_shift_advection
                          ? derivatives.VectorAdvectiveO2(
                                b, z4c.beta_u, z4c.vB_d)
                          : derivatives.VectorAdvective(
                                b, z4c.beta_u, z4c.vB_d);
          }
        }
        // Preserve the legacy Cartesian a-major accumulation order exactly.
        if (opt.telegraph_lapse) {
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              dB += g_uu(a, b) * derivatives.VectorFirst(a, b, z4c.vB_d);
            }
          }
        }

        // lapse function
        Real const f = opt.lapse_oplog * opt.lapse_harmonicf + opt.lapse_harmonic * alpha;
        if (opt.lapse_shock_avoiding) {
          rhs.alpha(m, k, j, i) =
              opt.lapse_advect * Lalpha -
              (alpha * alpha + opt.lapse_shock_avoiding_kappa) *
                  (z4c.vKhat(m, k, j, i) + 2.0 * z4c.vTheta(m, k, j, i));
        } else {
          // Preserve the existing vanilla/telegraph lapse arithmetic exactly.
          rhs.alpha(m, k, j, i) =
              opt.lapse_advect * Lalpha - f * alpha * z4c.vKhat(m, k, j, i);
        }
        if (opt.slow_start_lapse) {
          Real W2 = (chi > opt.chi_min_floor) ? chi : opt.chi_min_floor;
          Real W = pow(W2, 0.5);
          rhs.alpha(m, k, j, i) += opt.ssl_damping_amp * (W - alpha) *
                                   pow(W, opt.ssl_damping_index) *
                                   exp(-0.5 * pow(time / (opt.ssl_damping_time), 2));
        }
        if (opt.telegraph_lapse) {
          Real W = (chi > 0) ? chi : 0;
          rhs.alpha(m, k, j, i) += W * dB;
          const Real K =
              z4c.vKhat(m, k, j, i) + 2.0 * z4c.vTheta(m, k, j, i);
          Real local_mu = 1.0;
          if (opt.telegraph_damping_prescription ==
              TelegraphDampingPrescription::max_domain_abs_K) {
            local_mu = max_abs_K;
          } else if (opt.telegraph_damping_prescription ==
                     TelegraphDampingPrescription::local_abs_K) {
            local_mu = LocalAbsKTelegraphMu(K);
          } else if (opt.telegraph_damping_prescription ==
                     TelegraphDampingPrescription::local_extrinsic_curvature_norm) {
            local_mu = LocalExtrinsicCurvatureNormTelegraphMu(
                K,
                g_uu(0, 0), g_uu(0, 1), g_uu(0, 2),
                g_uu(1, 1), g_uu(1, 2), g_uu(2, 2),
                z4c.vA_dd(m, 0, 0, k, j, i),
                z4c.vA_dd(m, 0, 1, k, j, i),
                z4c.vA_dd(m, 0, 2, k, j, i),
                z4c.vA_dd(m, 1, 1, k, j, i),
                z4c.vA_dd(m, 1, 2, k, j, i),
                z4c.vA_dd(m, 2, 2, k, j, i));
          } else if (opt.telegraph_damping_prescription ==
                     TelegraphDampingPrescription::local_chi_gradient_norm) {
            local_mu = LocalChiGradientNormTelegraphMu(
                chi, opt.chi_psi_power,
                g_uu(0, 0), g_uu(0, 1), g_uu(0, 2),
                g_uu(1, 1), g_uu(1, 2), g_uu(2, 2),
                dchi_d(0), dchi_d(1), dchi_d(2));
          }
          telegraph_mu(m, 0, k, j, i) = local_mu;
          const auto coefficients = ScaleInvariantTelegraphCoefficients(
              local_mu, max_abs_K, opt.telegraph_tau, opt.telegraph_kappa);
          for (int a = 0; a < 3; ++a) {
            rhs.vB_d(m, a, k, j, i) =
                opt.lapse_advect * LB_d(a) +
                -coefficients.damping * z4c.vB_d(m, a, k, j, i) +
                coefficients.gradient * dalpha_d(a);
          }
        }
        Real const shift_gamma =
            (1 -
             opt.sss_damping_amp * exp(-0.5 * pow(time / (opt.sss_damping_time), 2))) *
            opt.shift_ggamma;
        Real const alpha_sq = SQR(alpha);
        Real const shift_alpha2ggamma = opt.shift_alpha2ggamma * alpha_sq;
        Real const shift_hh_alpha_chi = opt.shift_hh * alpha * chi_guarded;
        // shift vector
        for (int a = 0; a < 3; ++a) {
          if (prescribed_zero_shift) {
            rhs.beta_u(m, a, k, j, i) = 0.0;
            if (!opt.telegraph_lapse) rhs.vB_d(m, a, k, j, i) = 0.0;
            continue;
          }
          rhs.beta_u(m, a, k, j, i) =
              shift_gamma * z4c.vGam_u(m, a, k, j, i) + opt.shift_advect * Lbeta_u(a);
          rhs.beta_u(m, a, k, j, i) -=
              shift_eta_eff * z4c.beta_u(m, a, k, j, i);
          // FORCE beta = 0
          // rhs.beta_u(m,a,k,j,i) = 0;
        }
        // harmonic gauge terms
        for (int a = 0; a < 3; ++a) {
          rhs.beta_u(m, a, k, j, i) += shift_alpha2ggamma * z4c.vGam_u(m, a, k, j, i);
          for (int b = 0; b < 3; ++b) {
            rhs.beta_u(m, a, k, j, i) +=
                shift_hh_alpha_chi * (0.5 * alpha * dchi_d(b) - dalpha_d(b)) * g_uu(a, b);
          }
        }
      });

  // ===================================================================================
  // Add dissipation for stability
  //
  Real &diss = pmy_pack->pz4c->diss;
  auto &u0 = pmy_pack->pz4c->u0;
  auto &u_rhs = pmy_pack->pz4c->u_rhs;
  if (collect_rhs_stage_diagnostics) {
    Kokkos::deep_copy(rhs_raw_pre_axis, u_rhs);
  }
  if constexpr (std::is_same_v<Centering, VertexCenteredZ4c> &&
                std::is_same_v<Symmetry, CartoonSO2>) {
    // The non-KO continuum RHS must already preserve the evolved-axis
    // subspace. This strict pre-gate prevents the KO projection below from
    // hiding a geometric, gauge, or boundary inconsistency.
    ApplyVertexAxisRegularity(u_rhs, stage, "pre_ko_rhs");
    if (collect_rhs_stage_diagnostics) {
      Kokkos::deep_copy(rhs_post_axis_pre_ko, u_rhs);
    }
    auto &mb_bcs = pmy_pack->pmb->mb_bcs;
    par_for("SO2-invariant vertex K-O dissipation", DevExeSpace(), 0, nmb - 1,
            ks, ke, js, je, is, ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          Real values[Z4c::nz4c];
          Real idx[] = {1 / size.d_view(m).dx1, 1 / size.d_view(m).dx2,
                        1 / size.d_view(m).dx3};
          auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
              idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);
          for (int n = 0; n < Z4c::nz4c; ++n) {
            values[n] = u_rhs(m, n, k, j, i);
            for (int direction = 0; direction < 3; ++direction) {
              const Real before = values[n];
              values[n] += derivatives.DirectionalComponentDissipation(
                               direction, n, u0) * diss;
              if (collect_chi_provenance && n == Z4c::I_Z4C_CHI) {
                const int term = direction == 0 ? chi_ko_rho
                                 : (direction == 1 ? chi_ko_z : chi_ko_y);
                const int cumulative = direction == 0 ? chi_rhs_after_ko_rho
                                       : (direction == 1 ? chi_rhs_after_ko_z
                                                         : chi_rhs_after_ko_y);
                chi_provenance_terms(m, term, k, j, i) = values[n] - before;
                chi_provenance_terms(m, cumulative, k, j, i) = values[n];
              }
            }
          }
          if (i == is &&
              mb_bcs.d_view(m, BoundaryFace::inner_x1) == BoundaryFlag::axis) {
            ProjectVertexAxisZ4cValues(values);
          }
          for (int n = 0; n < Z4c::nz4c; ++n) {
            u_rhs(m, n, k, j, i) = values[n];
          }
          if (collect_chi_provenance) {
            chi_provenance_terms(m, chi_rhs_after_ko, k, j, i) =
                values[Z4c::I_Z4C_CHI];
          }
        });
  } else {
    if (collect_rhs_stage_diagnostics) {
      Kokkos::deep_copy(rhs_post_axis_pre_ko, u_rhs);
    }
    par_for("K-O Dissipation",
    DevExeSpace(),0,nmb-1,0,nz4c-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) {
      Real idx[] = {1/size.d_view(m).dx1, 1/size.d_view(m).dx2, 1/size.d_view(m).dx3};
      auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
          idx, size.d_view, nx1, is, m, k, j, i, nx3 == 1);
      // Keep the established multiply-then-accumulate order for Cartesian roundoff.
      for (int direction = 0; direction < 3; ++direction) {
        if (collect_chi_provenance && n == Z4c::I_Z4C_CHI) {
          const Real rhs_before = u_rhs(m,n,k,j,i);
          u_rhs(m,n,k,j,i) +=
              derivatives.DirectionalComponentDissipation(direction, n, u0) * diss;
          const Real contribution = u_rhs(m,n,k,j,i) - rhs_before;
          const int term = direction == 0 ? chi_ko_rho
                           : (direction == 1 ? chi_ko_z : chi_ko_y);
          const int cumulative = direction == 0 ? chi_rhs_after_ko_rho
                                 : (direction == 1 ? chi_rhs_after_ko_z
                                                   : chi_rhs_after_ko_y);
          chi_provenance_terms(m, term, k, j, i) = contribution;
          chi_provenance_terms(m, cumulative, k, j, i) = u_rhs(m,n,k,j,i);
        } else {
          u_rhs(m,n,k,j,i) +=
              derivatives.DirectionalComponentDissipation(direction, n, u0) * diss;
        }
      }
      if (collect_chi_provenance && n == Z4c::I_Z4C_CHI) {
        chi_provenance_terms(m, chi_rhs_after_ko, k, j, i) = u_rhs(m,n,k,j,i);
      }
    });
  }

  // This intentionally expensive host-side census is default-off and exists only for
  // bounded causal audits.  It reports the pre-projection complete RHS so the
  // exact axis correction remains attributable. Restricting the census to a
  // physical meridional tube keeps the evidence focused on the failure region.
  if (collect_rhs_stage_diagnostics) {
    Kokkos::fence();
    auto host_u0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u0);
    auto host_rhs = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_rhs);
    auto host_terms =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), rhs_stage_terms);
    auto host_raw_pre_axis =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), rhs_raw_pre_axis);
    auto host_post_axis_pre_ko = Kokkos::create_mirror_view_and_copy(
        HostMemSpace(), rhs_post_axis_pre_ko);
    pmy_pack->pmb->mb_size.sync_host();
    pmy_pack->pmb->mb_gid.sync_host();
    auto host_size = pmy_pack->pmb->mb_size.h_view;
    auto host_gid = pmy_pack->pmb->mb_gid.h_view;
    const std::string diagnostic_path =
        "z4c_rhs_stage_rank" + std::to_string(global_variable::my_rank) + ".log";
    std::ofstream diagnostic_output(diagnostic_path, std::ios::app);
    if (!diagnostic_output) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unable to open " << diagnostic_path << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int n = 0; n < nz4c; ++n) {
      Real state_max = 0.0;
      Real rhs_max = 0.0;
      Real state_value = 0.0;
      Real rhs_value = 0.0;
      Real state_rho = 0.0;
      Real state_z = 0.0;
      Real rhs_rho = 0.0;
      Real rhs_z = 0.0;
      int state_gid = -1;
      int rhs_gid = -1;
      int state_i = -1;
      int state_j = -1;
      int rhs_i = -1;
      int rhs_j = -1;
      std::uint64_t nonfinite_state = 0;
      std::uint64_t nonfinite_rhs = 0;
      std::uint64_t selected_cells = 0;
      for (int m = 0; m < nmb; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            const Real z = Z4cPointX<Centering>(j - js, nx2,
                                                host_size(m).x2min,
                                                host_size(m).x2max);
            if (fabs(z) > opt.rhs_stage_diagnostics_abs_z_max) continue;
            for (int i = is; i <= ie; ++i) {
              const Real rho = Z4cPointX<Centering>(i - is, nx1,
                                                    host_size(m).x1min,
                                                    host_size(m).x1max);
              if (rho < 0.0 || rho > opt.rhs_stage_diagnostics_rho_max) continue;
              ++selected_cells;
              const Real state = host_u0(m, n, k, j, i);
              const Real rhs_value_here = host_rhs(m, n, k, j, i);
              if (!isfinite(state)) {
                ++nonfinite_state;
              } else if (fabs(state) >= state_max) {
                state_max = fabs(state);
                state_value = state;
                state_rho = rho;
                state_z = z;
                state_gid = host_gid(m);
                state_i = i;
                state_j = j;
              }
              if (!isfinite(rhs_value_here)) {
                ++nonfinite_rhs;
              } else if (fabs(rhs_value_here) >= rhs_max) {
                rhs_max = fabs(rhs_value_here);
                rhs_value = rhs_value_here;
                rhs_rho = rho;
                rhs_z = z;
                rhs_gid = host_gid(m);
                rhs_i = i;
                rhs_j = j;
              }
            }
          }
        }
      }
      diagnostic_output << std::setprecision(17)
                << "Z4C_RHS_STAGE_DIAGNOSTIC rank=" << global_variable::my_rank
                << " cycle=" << pmy_pack->pmesh->ncycle
                << " time=" << time << " stage=" << stage
                << " variable=" << Z4c_names[n]
                << " selected_cells=" << selected_cells
                << " nonfinite_state=" << nonfinite_state
                << " nonfinite_rhs=" << nonfinite_rhs
                << " state_abs_max=" << state_max
                << " state_value=" << state_value
                << " state_gid=" << state_gid
                << " state_i=" << state_i << " state_j=" << state_j
                << " state_rho=" << state_rho << " state_z=" << state_z
                << " rhs_abs_max=" << rhs_max
                << " rhs_value=" << rhs_value
                << " rhs_gid=" << rhs_gid
                << " rhs_i=" << rhs_i << " rhs_j=" << rhs_j
                << " rhs_rho=" << rhs_rho << " rhs_z=" << rhs_z << '\n';
    }
    static const char * const term_names[75] = {
        "A_geometric_xx", "A_geometric_xy", "A_geometric_xz",
        "A_geometric_yy", "A_geometric_yz", "A_geometric_zz",
        "A_trace_xx", "A_trace_xy", "A_trace_xz",
        "A_trace_yy", "A_trace_yz", "A_trace_zz",
        "A_nonlinear_xx", "A_nonlinear_xy", "A_nonlinear_xz",
        "A_nonlinear_yy", "A_nonlinear_yz", "A_nonlinear_zz",
        "A_lie_xx", "A_lie_xy", "A_lie_xz",
        "A_lie_yy", "A_lie_yz", "A_lie_zz",
        "Gamma_divergence_x", "Gamma_divergence_y", "Gamma_divergence_z",
        "Gamma_lie_x", "Gamma_lie_y", "Gamma_lie_z",
        "Gamma_damping_x", "Gamma_damping_y", "Gamma_damping_z",
        "Gamma_lapse_gradient_x", "Gamma_lapse_gradient_y",
        "Gamma_lapse_gradient_z",
        "A_hessian_xx", "A_hessian_xy", "A_hessian_xz",
        "A_hessian_yy", "A_hessian_yz", "A_hessian_zz",
        "A_ricci_tensor_xx", "A_ricci_tensor_xy", "A_ricci_tensor_xz",
        "A_ricci_tensor_yy", "A_ricci_tensor_yz", "A_ricci_tensor_zz",
        "A_trace_lapse_xx", "A_trace_lapse_xy", "A_trace_lapse_xz",
        "A_trace_lapse_yy", "A_trace_lapse_yz", "A_trace_lapse_zz",
        "A_trace_ricci_xx", "A_trace_ricci_xy", "A_trace_ricci_xz",
        "A_trace_ricci_yy", "A_trace_ricci_yz", "A_trace_ricci_zz",
        "Gamma_advective_x", "Gamma_advective_y", "Gamma_advective_z",
        "Gamma_expansion_x", "Gamma_expansion_y", "Gamma_expansion_z",
        "Gamma_ddiv_x", "Gamma_ddiv_y", "Gamma_ddiv_z",
        "Gamma_contraction_x", "Gamma_contraction_y", "Gamma_contraction_z",
        "Gamma_second_x", "Gamma_second_y", "Gamma_second_z"};

    // Preserve all phase values and every named term at the physical central-axis
    // vertex.  Maxima alone are insufficient because their locations can move with
    // resolution and the pre-axis projection deliberately makes the final A_rr-A_yy
    // difference vanish.
    if constexpr (std::is_same_v<Centering, VertexCenteredZ4c> &&
                  std::is_same_v<Symmetry, CartoonSO2>) {
      int central_m = -1;
      int central_j = -1;
      Real central_abs_z = std::numeric_limits<Real>::max();
      for (int m = 0; m < nmb; ++m) {
        if (host_size(m).x1min != 0.0) continue;
        for (int j = js; j <= je; ++j) {
          const Real z = Z4cPointX<Centering>(j - js, nx2,
                                              host_size(m).x2min,
                                              host_size(m).x2max);
          if (fabs(z) < central_abs_z ||
              (fabs(z) == central_abs_z && host_gid(m) < host_gid(central_m))) {
            central_abs_z = fabs(z);
            central_m = m;
            central_j = j;
          }
        }
      }
      if (central_m >= 0) {
        const Real z = Z4cPointX<Centering>(central_j - js, nx2,
                                            host_size(central_m).x2min,
                                            host_size(central_m).x2max);
        for (int n = 0; n < nz4c; ++n) {
          const Real raw = host_raw_pre_axis(central_m, n, ks, central_j, is);
          const Real projected =
              host_post_axis_pre_ko(central_m, n, ks, central_j, is);
          const Real post_ko = host_rhs(central_m, n, ks, central_j, is);
          diagnostic_output << std::setprecision(17)
                            << "Z4C_AXIS_RHS_PHASE_DIAGNOSTIC rank="
                            << global_variable::my_rank
                            << " cycle=" << pmy_pack->pmesh->ncycle
                            << " time=" << time << " stage=" << stage
                            << " variable=" << Z4c_names[n]
                            << " gid=" << host_gid(central_m) << " rho=0"
                            << " z=" << z << " raw_pre_axis=" << raw
                            << " post_axis_pre_ko=" << projected
                            << " axis_correction=" << projected - raw
                            << " ko_contribution=" << post_ko - projected
                            << " post_ko=" << post_ko << '\n';
        }
        for (int term = 0; term < 75; ++term) {
          diagnostic_output << std::setprecision(17)
                            << "Z4C_AXIS_TERM_POINT_DIAGNOSTIC rank="
                            << global_variable::my_rank
                            << " cycle=" << pmy_pack->pmesh->ncycle
                            << " time=" << time << " stage=" << stage
                            << " term=" << term_names[term]
                            << " gid=" << host_gid(central_m) << " rho=0"
                            << " z=" << z << " value="
                            << host_terms(central_m, term, ks, central_j, is) << '\n';
        }
      }
    }
    for (int term = 0; term < 75; ++term) {
      Real term_max = 0.0;
      Real term_value = 0.0;
      Real term_rho = 0.0;
      Real term_z = 0.0;
      int term_gid = -1;
      int term_i = -1;
      int term_j = -1;
      std::uint64_t nonfinite_term = 0;
      std::uint64_t selected_cells = 0;
      for (int m = 0; m < nmb; ++m) {
        for (int k = ks; k <= ke; ++k) {
          for (int j = js; j <= je; ++j) {
            const Real z = Z4cPointX<Centering>(j - js, nx2,
                                                host_size(m).x2min,
                                                host_size(m).x2max);
            if (fabs(z) > opt.rhs_stage_diagnostics_abs_z_max) continue;
            for (int i = is; i <= ie; ++i) {
              const Real rho = Z4cPointX<Centering>(i - is, nx1,
                                                    host_size(m).x1min,
                                                    host_size(m).x1max);
              if (rho < 0.0 || rho > opt.rhs_stage_diagnostics_rho_max) continue;
              ++selected_cells;
              const Real value = host_terms(m, term, k, j, i);
              if (!isfinite(value)) {
                ++nonfinite_term;
              } else if (fabs(value) >= term_max) {
                term_max = fabs(value);
                term_value = value;
                term_rho = rho;
                term_z = z;
                term_gid = host_gid(m);
                term_i = i;
                term_j = j;
              }
            }
          }
        }
      }
      diagnostic_output << std::setprecision(17)
                        << "Z4C_RHS_TERM_DIAGNOSTIC rank="
                        << global_variable::my_rank
                        << " cycle=" << pmy_pack->pmesh->ncycle
                        << " time=" << time << " stage=" << stage
                        << " term=" << term_names[term]
                        << " selected_cells=" << selected_cells
                        << " nonfinite=" << nonfinite_term
                        << " abs_max=" << term_max
                        << " value=" << term_value
                        << " gid=" << term_gid
                        << " i=" << term_i << " j=" << term_j
                        << " rho=" << term_rho << " z=" << term_z << '\n';
    }

    // The VC Cartoon axis projection compares A_rhorho with the suppressed
    // A_yy component.  Preserve a paired, same-point decomposition so a
    // rejected correction can be attributed to geometry, trace subtraction,
    // nonlinear/Lie terms, or KO without relying on independent maxima.
    if constexpr (std::is_same_v<Centering, VertexCenteredZ4c> &&
                  std::is_same_v<Symmetry, CartoonSO2>) {
      Real selected_abs = 0.0;
      int selected_m = -1;
      int selected_j = -1;
      for (int m = 0; m < nmb; ++m) {
        if (host_size(m).x1min != 0.0) continue;
        for (int j = js; j <= je; ++j) {
          const Real z = Z4cPointX<Centering>(j - js, nx2,
                                              host_size(m).x2min,
                                              host_size(m).x2max);
          if (fabs(z) > opt.rhs_stage_diagnostics_abs_z_max) continue;
          const Real difference =
              host_rhs(m, Z4c::I_Z4C_AXX, ks, j, is) -
              host_rhs(m, Z4c::I_Z4C_AZZ, ks, j, is);
          if (isfinite(difference) && fabs(difference) >= selected_abs) {
            selected_abs = fabs(difference);
            selected_m = m;
            selected_j = j;
          }
        }
      }
      if (selected_m >= 0) {
        constexpr int pair_rhorho = 0;
        constexpr int pair_suppressed = 5;
        constexpr int bases[] = {0, 6, 12, 18, 36, 42, 48, 54};
        const int m = selected_m;
        const int j = selected_j;
        const Real z = Z4cPointX<Centering>(j - js, nx2,
                                            host_size(m).x2min,
                                            host_size(m).x2max);
        Real main_difference = 0.0;
        for (int family = 0; family < 4; ++family) {
          main_difference +=
              host_terms(m, bases[family] + pair_rhorho, ks, j, is) -
              host_terms(m, bases[family] + pair_suppressed, ks, j, is);
        }
        const Real full_difference =
            host_rhs(m, Z4c::I_Z4C_AXX, ks, j, is) -
            host_rhs(m, Z4c::I_Z4C_AZZ, ks, j, is);
        diagnostic_output << std::setprecision(17)
                          << "Z4C_AXIS_RHS_PAIR_DIAGNOSTIC rank="
                          << global_variable::my_rank
                          << " cycle=" << pmy_pack->pmesh->ncycle
                          << " time=" << time << " stage=" << stage
                          << " gid=" << host_gid(m) << " z=" << z
                          << " rhs_difference=" << full_difference
                          << " geometric_difference="
                          << host_terms(m, bases[0] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[0] + pair_suppressed, ks, j, is)
                          << " trace_difference="
                          << host_terms(m, bases[1] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[1] + pair_suppressed, ks, j, is)
                          << " nonlinear_difference="
                          << host_terms(m, bases[2] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[2] + pair_suppressed, ks, j, is)
                          << " lie_difference="
                          << host_terms(m, bases[3] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[3] + pair_suppressed, ks, j, is)
                          << " hessian_difference="
                          << host_terms(m, bases[4] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[4] + pair_suppressed, ks, j, is)
                          << " ricci_tensor_difference="
                          << host_terms(m, bases[5] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[5] + pair_suppressed, ks, j, is)
                          << " trace_lapse_difference="
                          << host_terms(m, bases[6] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[6] + pair_suppressed, ks, j, is)
                          << " trace_ricci_difference="
                          << host_terms(m, bases[7] + pair_rhorho, ks, j, is) -
                                 host_terms(m, bases[7] + pair_suppressed, ks, j, is)
                          << " ko_difference=" << full_difference - main_difference
                          << '\n';
      }
    }
  }

  // The rho=0 vertex is evolved.  Project only its exact SO(2) identities on the
  // complete RHS before any RK stage consumes it.  In diagnostic mode the host
  // census above observes, but does not modify, the pre-projection state first.
  ApplyVertexAxisRegularity(u_rhs, stage, "post_rhs");

  return TaskStatus::complete;
}

template <int NGHOST>
TaskStatus Z4c::CalcRHS(Driver *pdriver, int stage) {
  const bool is_vertex = layout.centering == Z4cGridCentering::vertex;
  if (pmy_pack->z4c_symmetry.mode == Z4cSymmetryMode::cartoon_so2) {
    if (is_vertex) {
      return CalcRHSImpl<VertexCenteredZ4c, CartoonSO2, NGHOST>(pdriver, stage);
    }
    return CalcRHSImpl<CellCenteredZ4c, CartoonSO2, NGHOST>(pdriver, stage);
  }
  if (is_vertex) {
    return CalcRHSImpl<VertexCenteredZ4c, Cartesian3D, NGHOST>(pdriver, stage);
  }
  return CalcRHSImpl<CellCenteredZ4c, Cartesian3D, NGHOST>(pdriver, stage);
}

template TaskStatus Z4c::CalcRHSImpl<CellCenteredZ4c, Cartesian3D, 2>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<CellCenteredZ4c, Cartesian3D, 3>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<CellCenteredZ4c, Cartesian3D, 4>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<CellCenteredZ4c, CartoonSO2, 2>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<CellCenteredZ4c, CartoonSO2, 3>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<CellCenteredZ4c, CartoonSO2, 4>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<VertexCenteredZ4c, Cartesian3D, 2>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<VertexCenteredZ4c, Cartesian3D, 3>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<VertexCenteredZ4c, Cartesian3D, 4>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<VertexCenteredZ4c, CartoonSO2, 2>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<VertexCenteredZ4c, CartoonSO2, 3>(Driver *, int);
template TaskStatus Z4c::CalcRHSImpl<VertexCenteredZ4c, CartoonSO2, 4>(Driver *, int);
template TaskStatus Z4c::CalcRHS<2>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<3>(Driver *pdriver, int stage);
template TaskStatus Z4c::CalcRHS<4>(Driver *pdriver, int stage);
} // namespace z4c
