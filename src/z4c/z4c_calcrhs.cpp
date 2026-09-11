#include "driver/execution_profile.hpp"
#include "driver/classical_rk4.hpp"
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
#include "mesh/mesh_refinement.hpp"
#include "coordinates/adm.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/cartoon_vertex_axis.hpp"
#include "z4c/z4c.hpp"
#include "z4c/bulk_rhs.hpp"
#include "z4c/dissipation.hpp"
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
  rhs_batches.Update(pdriver->level_batch_rhs, pmy_pack->pmb->mb_lev, nmb);
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
  if (pdriver->integrator == "rk4_classical") {
    time += pmy_pack->pmesh->dt * classical_rk4::StageTime(stage);
  }
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
    execution_profile::Scope profile("gauge/global_max_K");
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

  EvaluateZ4cBulkRHS<Centering,Symmetry,NGHOST>(layout,size,rhs_batches,
      z4c,rhs,telegraph_mu,opt,is_vacuum,tmunu,time,kappa1_eff,shift_eta_eff,
      max_abs_K,collect_chi_provenance,chi_provenance_terms,
      collect_rhs_stage_diagnostics,rhs_stage_terms);

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
    // Do not let KO projection conceal a continuum-axis inconsistency.
    ApplyVertexAxisRegularity(u_rhs, stage, "pre_ko_rhs");
  }
  if (collect_rhs_stage_diagnostics) Kokkos::deep_copy(rhs_post_axis_pre_ko,u_rhs);
  AddZ4cDissipation<Centering,Symmetry,NGHOST>(layout,size,pmy_pack->pmb->mb_bcs,
      rhs_batches,u0,u_rhs,diss,collect_chi_provenance,chi_provenance_terms);

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
  if constexpr (std::is_same_v<Centering, VertexCenteredZ4c>) {
    if (pmy_pack->pmesh->pmr != nullptr) {
      pmy_pack->pmesh->pmr->VCAMRLifecycleMarkFirstPostEventRHS(u_rhs, stage);
    }
  }

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
