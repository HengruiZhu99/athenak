#ifndef Z4C_BULK_RHS_HPP_
#define Z4C_BULK_RHS_HPP_
#include <cmath>
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/z4c.hpp"
#include "z4c/tmunu.hpp"

namespace z4c {
// Continuum bulk operator shared by leaf and auxiliary-parent consumers.
// Boundary RHS replacement, KO, axis checks/projection and diagnostics are
// deliberately separate. Gauge coefficients must describe this stage's time.
template<typename Centering,typename Symmetry,int NGHOST>
void EvaluateZ4cBulkRHS(const Z4cGridLayout &layout,
    const DualArray1D<RegionSize> &size, const subcycling::BlockBatches &rhs_batches,
    const Z4c::Z4c_vars &z4c, const Z4c::Z4c_vars &rhs,
    const DvceArray5D<Real> &telegraph_mu, const Z4c::Options &opt,
    bool is_vacuum, const Tmunu::Tmunu_vars &tmunu, Real time,
    Real kappa1_eff,Real shift_eta_eff,Real max_abs_K,
    bool collect_chi_provenance,const DvceArray5D<Real> &chi_provenance_terms,
    bool collect_rhs_stage_diagnostics,const DvceArray5D<Real> &rhs_stage_terms) {
  const int is=layout.is,ie=layout.ie,js=layout.js,je=layout.je;
  const int ks=layout.ks,ke=layout.ke,nx1=layout.nx1,nx3=layout.nx3;
  const bool prescribed_zero_shift=opt.shift_mode==Z4cShiftMode::prescribed_zero;
  const bool use_o2_shift_advection=opt.shift_advection_order==Z4cShiftAdvectionOrder::o2;
  // ===================================================================================
  // Main RHS calculation
  //
  rhs_batches.For4(
      "z4c rhs loop", ks, ke, js, je, is, ie,
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

  rhs_batches.For4(
      "z4c Gamma rhs loop", ks, ke, js, je, is, ie,
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

  rhs_batches.For4(
      "z4c gauge rhs loop", ks, ke, js, je, is, ie,
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

}
}  // namespace z4c
#endif  // Z4C_BULK_RHS_HPP_
