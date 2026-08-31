//========================================================================================
//! \file ref_gh_diagnostics.cpp
//! \brief ADM reconstruction and constraint refresh for reference-frame GH.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "ref_gh/analytic_radial_q_production.hpp"
#include "ref_gh/puncture_exponent.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace ref_gh {

void RefGh::SetADMVariables(MeshBlockPack *pack) { pack->prefgh->RefGhToADM(); }

void RefGh::RefGhToADM() {
  if (opt.reference_backend == 1) {
    RefGhToADMImpl<true>();
  } else {
    RefGhToADMImpl<false>();
  }
}

template <bool Analytic>
void RefGh::RefGhToADMImpl() {
  if (pmy_pack->padm == nullptr) return;
  FillReferenceCache(pmy_pack->pmesh->time, false);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const auto state = u0;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto analytic_static = reference_static;
  const auto analytic_stage = reference_stage;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  auto &size = pmy_pack->pmb->mb_size;
  const auto adm_vars = pmy_pack->padm->adm;
  par_for("ref_gh to ADM", DevExeSpace(), 0, pmy_pack->nmb_thispack - 1,
  0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const auto reference = MakeTypedProductionReferencePoint<Analytic>(
        reference_cache, reference_extra, analytic_static, analytic_stage,
        m, k, j, i, x, y, z, center_x, center_y, center_z);
    Real metric[4][4];  // NOLINT(runtime/arrays)
    Real determinant = 0.0;
    for (int a = 0; a < 4; ++a) {
      for (int b = 0; b < 4; ++b) {
        metric[a][b] = 0.0;
        for (int A = 0; A < 4; ++A) {
          for (int B = 0; B < 4; ++B) {
            metric[a][b] += ReferenceCoframe(reference, A, a)
                            *ReferenceCoframe(reference, B, b)
                            *state(m, PsiIndex(A, B), k, j, i);
          }
        }
      }
    }
    Real inverse[4][4];  // NOLINT(runtime/arrays)
    if (!Invert4(metric, inverse, determinant) || !(inverse[0][0] < 0.0)) {
      adm_vars.alpha(m, k, j, i) = NAN;
      adm_vars.psi4(m, k, j, i) = NAN;
      for (int a = 0; a < 3; ++a) {
        adm_vars.beta_u(m, a, k, j, i) = NAN;
        for (int b = a; b < 3; ++b) {
          adm_vars.g_dd(m, a, b, k, j, i) = NAN;
          adm_vars.vK_dd(m, a, b, k, j, i) = NAN;
        }
      }
      return;
    }
    const Real lapse = 1.0/Kokkos::sqrt(-inverse[0][0]);
    Real shift[3];  // NOLINT(runtime/arrays)
    for (int p = 0; p < 3; ++p) {
      shift[p] = lapse*lapse*inverse[0][p + 1];
    }
    adm_vars.alpha(m, k, j, i) = lapse;
    for (int a = 0; a < 3; ++a) {
      adm_vars.beta_u(m, a, k, j, i) = shift[a];
      for (int b = a; b < 3; ++b) {
        adm_vars.g_dd(m, a, b, k, j, i) = metric[a + 1][b + 1];
        Real christoffel = 0.0;
        for (int ell = 0; ell < 4; ++ell) {
          christoffel += 0.5*inverse[0][ell]*(
              ProductionCoordinateMetricDerivative(
                  state, reference, m, k, j, i, metric, lapse, shift,
                  a + 1, ell, b + 1)
              + ProductionCoordinateMetricDerivative(
                  state, reference, m, k, j, i, metric, lapse, shift,
                  b + 1, ell, a + 1)
              - ProductionCoordinateMetricDerivative(
                  state, reference, m, k, j, i, metric, lapse, shift,
                  ell, a + 1, b + 1));
        }
        adm_vars.vK_dd(m, a, b, k, j, i) = -lapse*christoffel;
      }
    }
    const Real det_spatial = adm::SpatialDet(
        metric[1][1], metric[1][2], metric[1][3], metric[2][2],
        metric[2][3], metric[3][3]);
    adm_vars.psi4(m, k, j, i) = Kokkos::pow(det_spatial, 1.0/3.0);
  });
}

void RefGh::CacheMetricCondition() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  const auto constraints = u_con;
  const auto adm_vars = pmy_pack->padm->adm;
  Kokkos::parallel_for(
      "ref_gh cache metric condition", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pmy_pack->nmb_thispack*ncells), KOKKOS_LAMBDA(const int idx) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real frame_scale = constraints(
            m, kMetricConditionDiagnostic, k, j, i);
        const Real scale2 = frame_scale*frame_scale;
        constraints(m, kMetricConditionDiagnostic, k, j, i) =
            SymmetricConditionNumber3(
                scale2*adm_vars.g_dd(m, 0, 0, k, j, i),
                scale2*adm_vars.g_dd(m, 0, 1, k, j, i),
                scale2*adm_vars.g_dd(m, 0, 2, k, j, i),
                scale2*adm_vars.g_dd(m, 1, 1, k, j, i),
                scale2*adm_vars.g_dd(m, 1, 2, k, j, i),
                scale2*adm_vars.g_dd(m, 2, 2, k, j, i));
      });
}

void RefGh::UpdateDiagnostics() {
  FillReferenceCache(pmy_pack->pmesh->time, true);
  DebugFence("ref_gh diagnostics reference");
  RefGhToADM();
  DebugFence("ref_gh diagnostics ADM reconstruction");
  switch (opt.fd_order) {
    case 2: CalcConstraints<2>(); break;
    case 4: CalcConstraints<3>(); break;
    case 6: CalcConstraints<4>(); break;
  }
  DebugFence("ref_gh diagnostics constraints");
  CacheMetricCondition();
  DebugFence("ref_gh diagnostics metric condition");
}

void RefGh::AppendMaxLocationDiagnostics() {
  if (opt.reference_backend == 1) {
    AppendMaxLocationDiagnosticsImpl<true>();
  } else {
    AppendMaxLocationDiagnosticsImpl<false>();
  }
}

template <bool Analytic>
void RefGh::AppendMaxLocationDiagnosticsImpl() {
  if (!opt.max_location_diagnostics) return;
  if (max_location_diagnostic_time == pmy_pack->pmesh->time
      && max_location_diagnostic_cycle == pmy_pack->pmesh->ncycle) return;
  max_location_diagnostic_time = pmy_pack->pmesh->time;
  max_location_diagnostic_cycle = pmy_pack->pmesh->ncycle;

  enum Diagnostic : int {
    kReferenceRicci, kReferenceRiemann, kSpin, kSpinDerivative,
    kReferenceDtFrame, kReferenceDtConnection,
    kReferenceSpatialFrameGradient, kReferenceWindowGradient,
    kPsi, kQ, kDelta, kPi, kPhi, kGhConstraint, kReductionConstraint,
    kCurlConstraint, kSourceCurvature, kSourceQq, kSourceDeltaDelta,
    kSourceDamping, kSourceFrameCorrection,
    kPsiRhs, kPiRhs, kPhiRhs, kHhatRhs, kThetaRhs, kUpsilonRhs,
    kChiBeta, kChiBetaNear, kChiBetaInner, kChiBetaAnnulus0,
    kChiBetaAnnulus1, kChiBetaAnnulus2, kChiBetaOuter,
    kChiBetaFirstSuperluminal,
    kDiagnosticCount
  };
  constexpr const char *names[kDiagnosticCount] = {
    "reference_Ricci", "reference_Riemann", "spin_connection",
    "spin_derivative", "reference_dt_frame", "reference_dt_connection",
    "reference_spatial_frame_gradient", "reference_window_gradient",
    "Psi", "Q", "Delta", "Pi", "Phi",
    "GH_constraint", "reduction_constraint", "curl_constraint",
    "source_curvature", "source_QQ", "source_DeltaDelta",
    "source_damping", "source_frame_correction",
    "Psi_RHS_Linf", "Pi_RHS_Linf", "Phi_RHS_Linf", "Hhat_RHS_Linf",
    "theta_RHS_Linf", "Upsilon_RHS_Linf",
    "chi_beta", "chi_beta_near_r_lt_1", "chi_beta_inner_r_lt_0p5",
    "chi_beta_annulus_0p5_1", "chi_beta_annulus_1_1p5",
    "chi_beta_annulus_1p5_2", "chi_beta_outer_r_ge_2",
    "chi_beta_first_ge_1"
  };
  constexpr int kRecordFields = 12;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const auto state = u0;
  const auto state_rhs = u_rhs;
  const auto constraints = u_con;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto analytic_static = reference_static;
  const auto analytic_stage = reference_stage;
  const auto adm_vars = pmy_pack->padm->adm;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  const int reference_kind = opt.reference_kind;
  const int transition_path = opt.transition_path;
  const Real reference_mass = opt.reference_mass;
  const Real r_core0 = opt.r_core0;
  const Real tau_core = opt.tau_core;
  const Real kappa_core = opt.kappa_core;
  const Real transition_width = opt.transition_width;
  const Real regularization_outer_start = opt.regularization_outer_start;
  const Real regularization_outer_end = opt.regularization_outer_end;
  const Real diagnostic_time = pmy_pack->pmesh->time;
  const bool exclude_puncture_stencils =
      opt.exclude_puncture_stencil_diagnostics;
  const int puncture_stencil_radius =
      PunctureEvolutionStencilRadius(opt.fd_order, opt.diss);
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real local_records[kDiagnosticCount*kRecordFields] = {};  // NOLINT

  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  for (int diagnostic_index = 0; diagnostic_index < kDiagnosticCount;
       ++diagnostic_index) {
    MaxLoc::value_type maximum;
    Kokkos::parallel_reduce(
        "ref_gh diagnostic maximum location",
        Kokkos::RangePolicy<>(DevExeSpace(),
            0, pmy_pack->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min,
                                     size.d_view(m).x1max);
          const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                     size.d_view(m).x2min,
                                     size.d_view(m).x2max);
          const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                     size.d_view(m).x3min,
                                     size.d_view(m).x3max);
          const Real dx = x - center_x;
          const Real dy = y - center_y;
          const Real dz = z - center_z;
          const Real cell_radius = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
          const bool stencil_based =
              diagnostic_index == kReductionConstraint
              || diagnostic_index == kCurlConstraint
              || (diagnostic_index >= kPsiRhs
                  && diagnostic_index <= kUpsilonRhs);
          if (exclude_puncture_stencils && stencil_based) {
            const Real displacement[3] = {dx, dy, dz};
            const Real spacing[3] = {
              size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
            if (!PunctureStencilIsClear(
                    displacement, spacing, puncture_stencil_radius)) return;
          }
          const auto reference = MakeTypedProductionReferencePoint<Analytic>(
              reference_cache, reference_extra, analytic_static, analytic_stage,
              m, k, j, i, x, y, z, center_x, center_y, center_z);
          Real value2 = 0.0;
          if (diagnostic_index == kReferenceRicci) {
            const Real value = constraints(
                m, kDiagnosticOffset + 2, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kReferenceRiemann) {
            // The compact analytic backend deliberately does not materialize
            // recursive oracle spin/Riemann tensors in ordinary diagnostics.
            if constexpr (!Analytic) {
              for (int A = 0; A < 4; ++A) {
                for (int B = 0; B < 4; ++B) {
                  for (int C = 0; C < 4; ++C) {
                    for (int D = 0; D < 4; ++D) {
                      const Real value =
                          ReferenceRiemann(reference, A, B, C, D);
                      value2 += value*value;
                    }
                  }
                }
              }
            }
          } else if (diagnostic_index == kSpin) {
            if constexpr (!Analytic) {
              for (int A = 0; A < 4; ++A) {
                for (int B = 0; B < 4; ++B) {
                  for (int C = 0; C < 4; ++C) {
                    const Real value = ReferenceSpin(reference, A, B, C);
                    value2 += value*value;
                  }
                }
              }
            }
          } else if (diagnostic_index == kSpinDerivative) {
            if constexpr (!Analytic) {
              for (int D = 0; D < 4; ++D) {
                for (int A = 0; A < 4; ++A) {
                  for (int B = 0; B < 4; ++B) {
                    for (int C = 0; C < 4; ++C) {
                      const Real value =
                          ReferenceSpinDerivative(reference, D, A, B, C);
                      value2 += value*value;
                    }
                  }
                }
              }
            }
          } else if (diagnostic_index == kReferenceDtFrame) {
            if constexpr (!Analytic) {
              for (int A = 0; A < 4; ++A) {
                for (int a = 0; a < 4; ++a) {
                  const Real value = ReferenceDFrame(reference, 0, A, a);
                  value2 += value*value;
                }
              }
            }
          } else if (diagnostic_index == kReferenceDtConnection) {
            if constexpr (!Analytic) {
              for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                  for (int c = b; c < 4; ++c) {
                    const Real value =
                        ReferenceDChristoffel(reference, 0, a, b, c);
                    value2 += value*value;
                  }
                }
              }
            }
          } else if (diagnostic_index == kReferenceSpatialFrameGradient) {
            if constexpr (!Analytic) {
              for (int p = 1; p < 4; ++p) {
                for (int A = 0; A < 4; ++A) {
                  for (int a = 0; a < 4; ++a) {
                    const Real value = ReferenceDFrame(reference, p, A, a);
                    value2 += value*value;
                  }
                }
              }
            }
          } else if (diagnostic_index == kReferenceWindowGradient) {
            if (reference_kind == 5) {
              const ReferenceJet radius = ControlledRadiusJet(
                  x, y, z, center_x, center_y, center_z);
              const ReferenceJet time_jet = CoordinateJet(diagnostic_time, 0);
              const ReferenceJet r_core = transition_path == kFixedCorePath
                  ? ConstantJet(r_core0*reference_mass)
                  : ConstantJet(r_core0*reference_mass)*Exp(
                      ConstantJet(-1.0/(tau_core*reference_mass))*time_jet);
              const ReferenceJet transition_coordinate =
                  transition_path == kFixedWidthPath
                  ? (radius + (-r_core))*ConstantJet(
                      1.0/(transition_width*reference_mass))
                  : (radius*Reciprocal(r_core) + ConstantJet(-1.0))
                      *ConstantJet(1.0/kappa_core);
              const ReferenceJet core_blend =
                  QuinticSmoothstep(transition_coordinate);
              const ReferenceJet outer_coordinate =
                  (radius + ConstantJet(
                      -regularization_outer_start*reference_mass))
                  *ConstantJet(1.0/((regularization_outer_end
                                     - regularization_outer_start)
                                    *reference_mass));
              const ReferenceJet window = core_blend
                  *(ConstantJet(1.0)
                    + (-QuinticSmoothstep(outer_coordinate)));
              for (int p = 1; p < 4; ++p) {
                value2 += window.d[p]*window.d[p];
              }
            }
          } else if (diagnostic_index == kPsi) {
            for (int n = kPsiOffset; n < kPiOffset; ++n) {
              const Real value = state(m, n, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kQ) {
            const Real value = constraints(
                m, kDiagnosticOffset + 0, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kDelta) {
            const Real value = constraints(
                m, kDiagnosticOffset + 1, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kPi) {
            for (int n = kPiOffset; n < kPhiOffset; ++n) {
              const Real value = state(m, n, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kPhi) {
            for (int n = kPhiOffset; n < kHhatOffset; ++n) {
              const Real value = state(m, n, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kGhConstraint) {
            for (int A = 0; A < 4; ++A) {
              const Real value = constraints(m, A, k, j, i);
              value2 += value*value;
            }
          } else if (diagnostic_index == kReductionConstraint) {
            const Real value = constraints(m, 4, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index == kCurlConstraint) {
            const Real value = constraints(m, 5, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index >= kSourceCurvature
                     && diagnostic_index <= kSourceFrameCorrection) {
            const int source_slot = kDiagnosticOffset + 4
                                    + diagnostic_index - kSourceCurvature;
            const Real value = constraints(m, source_slot, k, j, i);
            value2 = value*value;
          } else if (diagnostic_index >= kPsiRhs
                     && diagnostic_index <= kUpsilonRhs) {
            int begin = kPsiOffset;
            int end = kPiOffset;
            if (diagnostic_index == kPiRhs) {
              begin = kPiOffset;
              end = kPhiOffset;
            } else if (diagnostic_index == kPhiRhs) {
              begin = kPhiOffset;
              end = kHhatOffset;
            } else if (diagnostic_index == kHhatRhs) {
              begin = kHhatOffset;
              end = kThetaOffset;
            } else if (diagnostic_index == kThetaRhs) {
              begin = kThetaOffset;
              end = kUpsilonOffset;
            } else if (diagnostic_index == kUpsilonRhs) {
              begin = kUpsilonOffset;
              end = nvar;
            }
            for (int n = begin; n < end; ++n) {
              const Real value = state_rhs(m, n, k, j, i);
              value2 = fmax(value2, value*value);
            }
          } else {
            const Real alpha = adm_vars.alpha(m, k, j, i);
            Real shift2 = 0.0;
            for (int a = 0; a < 3; ++a) {
              for (int b = 0; b < 3; ++b) {
                shift2 += adm_vars.g_dd(m, a, b, k, j, i)
                          *adm_vars.beta_u(m, a, k, j, i)
                          *adm_vars.beta_u(m, b, k, j, i);
              }
            }
            const Real chi_beta = Kokkos::sqrt(fmax(shift2, 0.0))/alpha;
            bool include = true;
            if (diagnostic_index == kChiBetaNear) {
              include = cell_radius < 1.0;
            } else if (diagnostic_index == kChiBetaInner) {
              include = cell_radius < 0.5;
            } else if (diagnostic_index == kChiBetaAnnulus0) {
              include = cell_radius >= 0.5 && cell_radius < 1.0;
            } else if (diagnostic_index == kChiBetaAnnulus1) {
              include = cell_radius >= 1.0 && cell_radius < 1.5;
            } else if (diagnostic_index == kChiBetaAnnulus2) {
              include = cell_radius >= 1.5 && cell_radius < 2.0;
            } else if (diagnostic_index == kChiBetaOuter) {
              include = cell_radius >= 2.0;
            }
            if (diagnostic_index == kChiBetaFirstSuperluminal) {
              // Maximize this monotone proxy to select the minimum radius.
              const Real proxy = (chi_beta >= 1.0)
                  ? 1.0/(1.0 + cell_radius) : 0.0;
              value2 = proxy*proxy;
            } else if (include) {
              value2 = chi_beta*chi_beta;
            }
          }
          const Real value = Kokkos::sqrt(value2);
          const Real comparable = Kokkos::isfinite(value)
              ? value : std::numeric_limits<Real>::max();
          if (comparable >= local_maximum.val) {
            local_maximum.val = comparable;
            local_maximum.loc = idx;
          }
        }, MaxLoc(maximum));

    const bool first_superluminal =
        diagnostic_index == kChiBetaFirstSuperluminal;
    const bool local_valid = !first_superluminal || maximum.val > 0.0;
    int work = local_valid ? maximum.loc : 0;
    const int ii = work % indcs.nx1; work /= indcs.nx1;
    const int jj = work % indcs.nx2; work /= indcs.nx2;
    const int kk = work % indcs.nx3;
    const int m = work/indcs.nx3;
    const Real x = CellCenterX(ii, indcs.nx1,
                               size.h_view(m).x1min, size.h_view(m).x1max);
    const Real y = CellCenterX(jj, indcs.nx2,
                               size.h_view(m).x2min, size.h_view(m).x2max);
    const Real z = CellCenterX(kk, indcs.nx3,
                               size.h_view(m).x3min, size.h_view(m).x3max);
    const Real dx = x - opt.reference_center[0];
    const Real dy = y - opt.reference_center[1];
    const Real dz = z - opt.reference_center[2];
    const Real radius = std::sqrt(dx*dx + dy*dy + dz*dz);
    const int offset = diagnostic_index*kRecordFields;
    local_records[offset + 0] = first_superluminal
        ? (local_valid ? 1.0 : 0.0) : maximum.val;
    local_records[offset + 1] = local_valid
        ? radius : std::numeric_limits<Real>::infinity();
    local_records[offset + 2] = 0.0;
    local_records[offset + 3] = pmy_pack->pmb->mb_lev.h_view(m);
    local_records[offset + 4] = global_variable::my_rank;
    local_records[offset + 5] = pmy_pack->pmb->mb_gid.h_view(m);
    local_records[offset + 6] = x;
    local_records[offset + 7] = y;
    local_records[offset + 8] = z;
    local_records[offset + 9] = ii;
    local_records[offset + 10] = jj;
    local_records[offset + 11] = kk;
  }

  const Real time = pmy_pack->pmesh->time;
  const Real r_core = opt.transition_path == kFixedCorePath
      ? opt.r_core0*opt.reference_mass
      : opt.r_core0*opt.reference_mass
          *std::exp(-time/(opt.tau_core*opt.reference_mass));
  for (int n = 0; n < kDiagnosticCount; ++n) {
    const int offset = n*kRecordFields;
    local_records[offset + 2] = local_records[offset + 1]/r_core;
  }

  std::vector<Real> gathered;
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    gathered.resize(global_variable::nranks*kDiagnosticCount*kRecordFields);
  }
  MPI_Gather(local_records, kDiagnosticCount*kRecordFields, MPI_ATHENA_REAL,
             gathered.data(), kDiagnosticCount*kRecordFields, MPI_ATHENA_REAL,
             0, MPI_COMM_WORLD);
#else
  gathered.assign(local_records,
                  local_records + kDiagnosticCount*kRecordFields);
#endif
  if (global_variable::my_rank != 0) return;

  const std::string filename =
      pinput->GetString("job", "basename") + ".ref_gh_maxloc.tsv";
  FILE *file = std::fopen(filename.c_str(), "a+");
  if (file == nullptr) {
    std::cout << "### FATAL ERROR: unable to open Ref-GH max-location file "
              << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::fseek(file, 0, SEEK_END);
  if (std::ftell(file) == 0) {
    std::fprintf(file, "time\tcycle\tdiagnostic\tmaximum\tradius\t"
                       "r_over_r_core\tlevel\trank\tgid\tx\ty\tz\ti\tj\tk\n");
  }
  for (int n = 0; n < kDiagnosticCount; ++n) {
    const Real *best = nullptr;
    for (int rank = 0; rank < global_variable::nranks; ++rank) {
      const Real *candidate = gathered.data()
          + (rank*kDiagnosticCount + n)*kRecordFields;
      if (n == kChiBetaFirstSuperluminal) {
        if (candidate[0] >= 1.0
            && (best == nullptr || candidate[1] < best[1])) best = candidate;
      } else if (best == nullptr || candidate[0] > best[0]) {
        best = candidate;
      }
    }
    Real absent_record[kRecordFields] = {};
    absent_record[1] = std::numeric_limits<Real>::infinity();
    if (best == nullptr) best = absent_record;
    std::fprintf(file,
        "%.17e\t%d\t%s\t%.17e\t%.17e\t%.17e\t%d\t%d\t%d\t"
        "%.17e\t%.17e\t%.17e\t%d\t%d\t%d\n",
        time, pmy_pack->pmesh->ncycle, names[n], best[0], best[1], best[2],
        static_cast<int>(best[3]), static_cast<int>(best[4]),
        static_cast<int>(best[5]), best[6], best[7], best[8],
        static_cast<int>(best[9]), static_cast<int>(best[10]),
        static_cast<int>(best[11]));
  }
  std::fclose(file);
}

void RefGh::WriteInitialRhsSectorDiagnostics() {
  if (!pinput->GetOrAddBoolean(
          "problem", "initial_rhs_sector_diagnostics", false)) return;
  if (opt.reference_backend == 1) {
    WriteInitialRhsSectorDiagnosticsImpl<true>();
  } else {
    WriteInitialRhsSectorDiagnosticsImpl<false>();
  }
}

template <bool Analytic>
void RefGh::WriteInitialRhsSectorDiagnosticsImpl() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  const int nmb = pmy_pack->nmb_thispack;
  const int n1 = u_rhs.extent_int(4);
  const int n2 = u_rhs.extent_int(3);
  const int n3 = u_rhs.extent_int(2);
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;

  DvceArray5D<Real> actual("initial RHS actual", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> with_gamma2(
      "initial RHS no KO", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> base(
      "initial RHS no gamma2", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> no_gauge(
      "initial RHS no gauge", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> no_gamma0(
      "initial RHS no gamma0", nmb, nvar, n3, n2, n1);
  DvceArray5D<Real> vacuum_pi(
      "initial RHS vacuum Pi", nmb, kSymmetric4Size, n3, n2, n1);
  Kokkos::deep_copy(actual, u_rhs);

  const Real saved_gamma0 = opt.gamma0;
  const Real saved_gamma2 = opt.gamma2;
  const Real saved_diss = opt.diss;
  const bool saved_gauge_driver = opt.gauge_driver_enabled;
  const bool saved_gauge_subtraction = opt.gauge_reference_subtraction;
  auto calculate_rhs = [this]() {
    switch (opt.fd_order) {
      case 2: (void)CalcRHS<2>(nullptr, 1); break;
      case 4: (void)CalcRHS<3>(nullptr, 1); break;
      case 6: (void)CalcRHS<4>(nullptr, 1); break;
    }
    Kokkos::fence("ref_gh initial RHS sector evaluation");
  };

  opt.diss = 0.0;
  calculate_rhs();
  Kokkos::deep_copy(with_gamma2, u_rhs);
  opt.gamma2 = 0.0;
  calculate_rhs();
  Kokkos::deep_copy(base, u_rhs);
  opt.gauge_driver_enabled = false;
  opt.gauge_reference_subtraction = false;
  calculate_rhs();
  Kokkos::deep_copy(no_gauge, u_rhs);
  opt.gamma0 = 0.0;
  calculate_rhs();
  Kokkos::deep_copy(no_gamma0, u_rhs);

  opt.gamma0 = saved_gamma0;
  opt.gamma2 = saved_gamma2;
  opt.diss = saved_diss;
  opt.gauge_driver_enabled = saved_gauge_driver;
  opt.gauge_reference_subtraction = saved_gauge_subtraction;
  calculate_rhs();

  const auto state = u0;
  const auto restored_rhs = u_rhs;
  const auto reference_cache = reference_evolution;
  const auto reference_extra = reference_diagnostic;
  const auto analytic_static = reference_static;
  const auto analytic_stage = reference_stage;
  const Real center_x = opt.reference_center[0];
  const Real center_y = opt.reference_center[1];
  const Real center_z = opt.reference_center[2];
  Kokkos::deep_copy(vacuum_pi, 0.0);
  par_for("ref_gh initial vacuum Pi source", DevExeSpace(), 0, nmb - 1,
  indcs.ks, indcs.ke, indcs.js, indcs.je, indcs.is, indcs.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const auto reference = MakeTypedProductionReferencePoint<Analytic>(
        reference_cache, reference_extra, analytic_static, analytic_stage,
        m, k, j, i, x, y, z, center_x, center_y, center_z);
    Real psi[4][4], metric[4][4], pi[4][4], phi[3][4][4];  // NOLINT
    Real d_psi[4][4][4], d_metric[4][4][4];  // NOLINT
    CoordinateGhGeometry geometry;
    Real determinant = 0.0;
    Real source[4][4];  // NOLINT(runtime/arrays)
    if (!LoadProductionPointGeometry(
            state, reference, m, k, j, i, psi, pi, phi, d_psi, metric,
            d_metric, geometry, determinant)
        || !ProductionCovariantScalarWaveSource(
            psi, pi, phi, reference, geometry, 0.0, source)) {
      for (int component = 0; component < kSymmetric4Size; ++component) {
        vacuum_pi(m, component, k, j, i) = NAN;
      }
      return;
    }
    for (int A = 0; A < 4; ++A) {
      for (int B = A; B < 4; ++B) {
        vacuum_pi(m, Symmetric4Index(A, B), k, j, i) =
            geometry.lapse*source[A][B];
      }
    }
  });
  Kokkos::fence("ref_gh initial vacuum Pi source");

  const int stencil_radius =
      PunctureEvolutionStencilRadius(opt.fd_order, opt.diss);
  Real decomposition_error = 0.0;
  Real rerun_error = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh initial RHS decomposition reproduction",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nvar*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_decomposition,
                    Real &local_rerun) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
        const int n = work % nvar;
        const int m = work/nvar;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min,
                                   size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min,
                                   size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min,
                                   size.d_view(m).x3max);
        const Real displacement[3] = {
          x - center_x, y - center_y, z - center_z};
        const Real spacing[3] = {
          size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
        if (!PunctureStencilIsClear(
                displacement, spacing, stencil_radius)) return;
        Real sum = 0.0;
        if (n < kPiOffset) {
          sum = no_gamma0(m, n, k, j, i)
                + actual(m, n, k, j, i) - with_gamma2(m, n, k, j, i);
        } else if (n < kPhiOffset) {
          const int component = n - kPiOffset;
          const Real vacuum = vacuum_pi(m, component, k, j, i);
          const Real principal = no_gamma0(m, n, k, j, i) - vacuum;
          const Real gamma0 = no_gauge(m, n, k, j, i)
                              - no_gamma0(m, n, k, j, i);
          const Real gauge = base(m, n, k, j, i)
                             - no_gauge(m, n, k, j, i);
          const Real gamma2 = with_gamma2(m, n, k, j, i)
                              - base(m, n, k, j, i);
          const Real ko = actual(m, n, k, j, i)
                          - with_gamma2(m, n, k, j, i);
          sum = principal + vacuum + gamma0 + gauge + gamma2 + ko;
        } else if (n < kHhatOffset) {
          sum = base(m, n, k, j, i)
                + with_gamma2(m, n, k, j, i) - base(m, n, k, j, i)
                + actual(m, n, k, j, i) - with_gamma2(m, n, k, j, i);
        } else {
          sum = base(m, n, k, j, i)
                + actual(m, n, k, j, i) - with_gamma2(m, n, k, j, i);
        }
        const Real expected = actual(m, n, k, j, i);
        const Real scale = fmax(1.0, Kokkos::abs(expected));
        local_decomposition = fmax(
            local_decomposition, Kokkos::abs(sum - expected)/scale);
        local_rerun = fmax(
            local_rerun,
            Kokkos::abs(restored_rhs(m, n, k, j, i) - expected)/scale);
      }, Kokkos::Max<Real>(decomposition_error),
         Kokkos::Max<Real>(rerun_error));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &decomposition_error, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &rerun_error, 1, MPI_ATHENA_REAL,
                MPI_MAX, MPI_COMM_WORLD);
#endif
  constexpr Real kReproductionTolerance = 5.0e-13;
  if (!std::isfinite(decomposition_error) || !std::isfinite(rerun_error)
      || decomposition_error > kReproductionTolerance
      || rerun_error > kReproductionTolerance) {
    std::cout << "### FATAL ERROR: Ref-GH initial RHS sector decomposition "
              << "failed: reproduction=" << decomposition_error
              << " rerun=" << rerun_error
              << " tolerance=" << kReproductionTolerance << std::endl;
    std::exit(EXIT_FAILURE);
  }

  enum SectorMode : int {
    kActual, kPrincipalDirect, kPrincipalPi, kVacuumPi, kGaugePi,
    kGamma0Pi, kGamma2, kGaugeDriver, kKo
  };
  struct SectorRecordSpec {
    const char *sector;
    const char *family;
    int begin;
    int end;
    int mode;
  };
  constexpr SectorRecordSpec specs[] = {
    {"actual", "Psi", kPsiOffset, kPiOffset, kActual},
    {"actual", "Pi", kPiOffset, kPhiOffset, kActual},
    {"actual", "Phi", kPhiOffset, kHhatOffset, kActual},
    {"actual", "Hhat", kHhatOffset, kThetaOffset, kActual},
    {"actual", "theta", kThetaOffset, kUpsilonOffset, kActual},
    {"actual", "Upsilon", kUpsilonOffset, nvar, kActual},
    {"principal", "Psi", kPsiOffset, kPiOffset, kPrincipalDirect},
    {"principal", "Pi", kPiOffset, kPhiOffset, kPrincipalPi},
    {"principal", "Phi", kPhiOffset, kHhatOffset, kPrincipalDirect},
    {"covariant_vacuum_source", "Pi", kPiOffset, kPhiOffset, kVacuumPi},
    {"ordinary_gauge_increment", "Pi", kPiOffset, kPhiOffset, kGaugePi},
    {"gamma0_damping", "Pi", kPiOffset, kPhiOffset, kGamma0Pi},
    {"gamma2_damping", "Pi", kPiOffset, kPhiOffset, kGamma2},
    {"gamma2_damping", "Phi", kPhiOffset, kHhatOffset, kGamma2},
    {"driver", "Hhat", kHhatOffset, kThetaOffset, kGaugeDriver},
    {"driver", "theta", kThetaOffset, kUpsilonOffset, kGaugeDriver},
    {"driver", "Upsilon", kUpsilonOffset, nvar, kGaugeDriver},
    {"KO", "Psi", kPsiOffset, kPiOffset, kKo},
    {"KO", "Pi", kPiOffset, kPhiOffset, kKo},
    {"KO", "Phi", kPhiOffset, kHhatOffset, kKo},
    {"KO", "Hhat", kHhatOffset, kThetaOffset, kKo},
    {"KO", "theta", kThetaOffset, kUpsilonOffset, kKo},
    {"KO", "Upsilon", kUpsilonOffset, nvar, kKo}
  };
  constexpr int kRecordCount = sizeof(specs)/sizeof(specs[0]);
  constexpr int kRecordFields = 11;
  Real local_records[kRecordCount*kRecordFields] = {};  // NOLINT
  using MaxLoc = Kokkos::MaxLoc<Real, int>;
  for (int record = 0; record < kRecordCount; ++record) {
    const int begin = specs[record].begin;
    const int end = specs[record].end;
    const int mode = specs[record].mode;
    MaxLoc::value_type maximum;
    Kokkos::parallel_reduce(
        "ref_gh initial RHS sector maximum",
        Kokkos::RangePolicy<>(DevExeSpace(),
            0, nmb*(end - begin)*ncells),
        KOKKOS_LAMBDA(const int idx, MaxLoc::value_type &local_maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks; work /= indcs.nx3;
          const int n = work % (end - begin) + begin;
          const int m = work/(end - begin);
          const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min,
                                     size.d_view(m).x1max);
          const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                     size.d_view(m).x2min,
                                     size.d_view(m).x2max);
          const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                     size.d_view(m).x3min,
                                     size.d_view(m).x3max);
          const Real displacement[3] = {
            x - center_x, y - center_y, z - center_z};
          const Real spacing[3] = {
            size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
          if (!PunctureStencilIsClear(
                  displacement, spacing, stencil_radius)) return;
          Real value = actual(m, n, k, j, i);
          if (mode == kPrincipalDirect) {
            value = (n < kPiOffset) ? no_gamma0(m, n, k, j, i)
                                    : base(m, n, k, j, i);
          } else if (mode == kPrincipalPi) {
            value = no_gamma0(m, n, k, j, i)
                    - vacuum_pi(m, n - kPiOffset, k, j, i);
          } else if (mode == kVacuumPi) {
            value = vacuum_pi(m, n - kPiOffset, k, j, i);
          } else if (mode == kGaugePi) {
            value = base(m, n, k, j, i) - no_gauge(m, n, k, j, i);
          } else if (mode == kGamma0Pi) {
            value = no_gauge(m, n, k, j, i)
                    - no_gamma0(m, n, k, j, i);
          } else if (mode == kGamma2) {
            value = with_gamma2(m, n, k, j, i) - base(m, n, k, j, i);
          } else if (mode == kGaugeDriver) {
            value = base(m, n, k, j, i);
          } else if (mode == kKo) {
            value = actual(m, n, k, j, i)
                    - with_gamma2(m, n, k, j, i);
          }
          const Real comparable = Kokkos::isfinite(value)
              ? Kokkos::abs(value) : std::numeric_limits<Real>::max();
          if (comparable > local_maximum.val) {
            local_maximum.val = comparable;
            local_maximum.loc = idx;
          }
        }, MaxLoc(maximum));
    int work = maximum.loc;
    const int ii = work % indcs.nx1; work /= indcs.nx1;
    const int jj = work % indcs.nx2; work /= indcs.nx2;
    const int kk = work % indcs.nx3; work /= indcs.nx3;
    const int component = work % (end - begin) + begin;
    const int m = work/(end - begin);
    const Real x = CellCenterX(ii, indcs.nx1,
                               size.h_view(m).x1min, size.h_view(m).x1max);
    const Real y = CellCenterX(jj, indcs.nx2,
                               size.h_view(m).x2min, size.h_view(m).x2max);
    const Real z = CellCenterX(kk, indcs.nx3,
                               size.h_view(m).x3min, size.h_view(m).x3max);
    const Real dx = x - center_x;
    const Real dy = y - center_y;
    const Real dz = z - center_z;
    const int offset = record*kRecordFields;
    local_records[offset + 0] = maximum.val;
    local_records[offset + 1] = std::sqrt(dx*dx + dy*dy + dz*dz);
    local_records[offset + 2] = component;
    local_records[offset + 3] = global_variable::my_rank;
    local_records[offset + 4] = pmy_pack->pmb->mb_gid.h_view(m);
    local_records[offset + 5] = x;
    local_records[offset + 6] = y;
    local_records[offset + 7] = z;
    local_records[offset + 8] = ii;
    local_records[offset + 9] = jj;
    local_records[offset + 10] = kk;
  }

  std::vector<Real> gathered;
#if MPI_PARALLEL_ENABLED
  if (global_variable::my_rank == 0) {
    gathered.resize(global_variable::nranks*kRecordCount*kRecordFields);
  }
  MPI_Gather(local_records, kRecordCount*kRecordFields, MPI_ATHENA_REAL,
             gathered.data(), kRecordCount*kRecordFields, MPI_ATHENA_REAL,
             0, MPI_COMM_WORLD);
#else
  gathered.assign(local_records,
                  local_records + kRecordCount*kRecordFields);
#endif
  int global_maximum_location[6] = {};
  Real global_maximum_metadata[2] = {};
  if (global_variable::my_rank == 0) {
    const Real *overall = nullptr;
    for (int record = 0; record < 6; ++record) {
      for (int rank = 0; rank < global_variable::nranks; ++rank) {
        const Real *candidate = gathered.data()
            + (rank*kRecordCount + record)*kRecordFields;
        if (overall == nullptr || candidate[0] > overall[0]
            || (candidate[0] == overall[0] && candidate[1] < overall[1])) {
          overall = candidate;
        }
      }
    }
    global_maximum_location[0] = static_cast<int>(overall[3]);
    global_maximum_location[1] = static_cast<int>(overall[4]);
    global_maximum_location[2] = static_cast<int>(overall[2]);
    global_maximum_location[3] = static_cast<int>(overall[8]);
    global_maximum_location[4] = static_cast<int>(overall[9]);
    global_maximum_location[5] = static_cast<int>(overall[10]);
    global_maximum_metadata[0] = overall[0];
    global_maximum_metadata[1] = overall[1];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(global_maximum_location, 6, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(global_maximum_metadata, 2, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif
  constexpr int kPointSectorCount = 9;
  DvceArray1D<Real> point_sectors(
      "initial RHS sectors at global maximum", kPointSectorCount);
  Kokkos::deep_copy(point_sectors, 0.0);
  if (global_variable::my_rank == global_maximum_location[0]) {
    int target_m = -1;
    for (int m = 0; m < nmb; ++m) {
      if (pmy_pack->pmb->mb_gid.h_view(m) == global_maximum_location[1]) {
        target_m = m;
        break;
      }
    }
    if (target_m < 0) {
      std::cout << "### FATAL ERROR: global initial RHS maximum gid was not "
                   "found on its reported rank." << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const int target_n = global_maximum_location[2];
    const int target_i = global_maximum_location[3] + indcs.is;
    const int target_j = global_maximum_location[4] + indcs.js;
    const int target_k = global_maximum_location[5] + indcs.ks;
    Kokkos::parallel_for(
        "ref_gh decompose global initial RHS maximum",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
        KOKKOS_LAMBDA(const int) {
          const Real actual_value =
              actual(target_m, target_n, target_k, target_j, target_i);
          Real principal = 0.0;
          Real vacuum = 0.0;
          Real gauge = 0.0;
          Real gamma0 = 0.0;
          Real gamma2 = 0.0;
          Real driver = 0.0;
          const Real ko = actual_value
              - with_gamma2(target_m, target_n, target_k, target_j, target_i);
          if (target_n < kPiOffset) {
            principal = no_gamma0(
                target_m, target_n, target_k, target_j, target_i);
          } else if (target_n < kPhiOffset) {
            vacuum = vacuum_pi(
                target_m, target_n - kPiOffset,
                target_k, target_j, target_i);
            principal = no_gamma0(
                target_m, target_n, target_k, target_j, target_i) - vacuum;
            gamma0 = no_gauge(
                target_m, target_n, target_k, target_j, target_i)
                - no_gamma0(target_m, target_n, target_k, target_j, target_i);
            gauge = base(target_m, target_n, target_k, target_j, target_i)
                - no_gauge(target_m, target_n, target_k, target_j, target_i);
            gamma2 = with_gamma2(
                target_m, target_n, target_k, target_j, target_i)
                - base(target_m, target_n, target_k, target_j, target_i);
          } else if (target_n < kHhatOffset) {
            principal = base(target_m, target_n, target_k, target_j, target_i);
            gamma2 = with_gamma2(
                target_m, target_n, target_k, target_j, target_i)
                - principal;
          } else {
            driver = base(target_m, target_n, target_k, target_j, target_i);
          }
          point_sectors(0) = actual_value;
          point_sectors(1) = principal;
          point_sectors(2) = vacuum;
          point_sectors(3) = gauge;
          point_sectors(4) = gamma0;
          point_sectors(5) = gamma2;
          point_sectors(6) = driver;
          point_sectors(7) = ko;
          point_sectors(8) = principal + vacuum + gauge + gamma0
                             + gamma2 + driver + ko;
        });
    Kokkos::fence("ref_gh decompose global initial RHS maximum");
  }
  auto point_sectors_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), point_sectors);
  Real global_point_sectors[kPointSectorCount] = {};
  for (int sector = 0; sector < kPointSectorCount; ++sector) {
    global_point_sectors[sector] = point_sectors_host(sector);
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(global_point_sectors, kPointSectorCount, MPI_ATHENA_REAL,
            global_maximum_location[0], MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank != 0) return;
  const std::string filename =
      pinput->GetString("job", "basename") + ".ref_gh_rhs_sectors.tsv";
  FILE *file = std::fopen(filename.c_str(), "w");
  if (file == nullptr) {
    std::cout << "### FATAL ERROR: unable to write Ref-GH RHS sectors "
              << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::fprintf(file,
      "# reproduction_conditioned_linf=%.17e\n"
      "# production_rerun_conditioned_linf=%.17e\n"
      "# tolerance=%.17e\n"
      "# global_actual_maximum=%.17e component=%d radius=%.17e "
      "rank=%d gid=%d\n"
      "# at_global_max_actual=%.17e principal=%.17e vacuum=%.17e "
      "ordinary_gauge=%.17e gamma0=%.17e gamma2=%.17e driver=%.17e "
      "KO=%.17e sum=%.17e\n",
      decomposition_error, rerun_error, kReproductionTolerance,
      global_maximum_metadata[0], global_maximum_location[2],
      global_maximum_metadata[1], global_maximum_location[0],
      global_maximum_location[1], global_point_sectors[0],
      global_point_sectors[1], global_point_sectors[2],
      global_point_sectors[3], global_point_sectors[4],
      global_point_sectors[5], global_point_sectors[6],
      global_point_sectors[7], global_point_sectors[8]);
  std::fprintf(file,
      "sector\tfamily\tmaximum\tcomponent\tradius\trank\tgid\t"
      "x\ty\tz\ti\tj\tk\n");
  for (int record = 0; record < kRecordCount; ++record) {
    const Real *best = nullptr;
    for (int rank = 0; rank < global_variable::nranks; ++rank) {
      const Real *candidate = gathered.data()
          + (rank*kRecordCount + record)*kRecordFields;
      if (best == nullptr || candidate[0] > best[0]
          || (candidate[0] == best[0] && candidate[1] < best[1])) {
        best = candidate;
      }
    }
    std::fprintf(file,
        "%s\t%s\t%.17e\t%d\t%.17e\t%d\t%d\t%.17e\t%.17e\t"
        "%.17e\t%d\t%d\t%d\n",
        specs[record].sector, specs[record].family, best[0],
        static_cast<int>(best[2]), best[1], static_cast<int>(best[3]),
        static_cast<int>(best[4]), best[5], best[6], best[7],
        static_cast<int>(best[8]), static_cast<int>(best[9]),
        static_cast<int>(best[10]));
  }
  std::fclose(file);
  std::cout << "reference-GH initial RHS sector decomposition passed: "
            << "conditioned error=" << decomposition_error
            << ", production rerun=" << rerun_error
            << ", file=" << filename << std::endl;
}

}  // namespace ref_gh
