//========================================================================================
//! \file controlled_transition.cpp
//! \brief Wormhole-matched data for the controlled Ref-GH Schwarzschild transition.
//========================================================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "pgen/pgen.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/ref_gh_geometry.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

struct InitialMatchEvidence {
  Real regular_state_linf = 0.0;
  Real relative_spatial_linf = 0.0;
  Real relative_lapse_linf = 0.0;
  Real relative_shift_linf = 0.0;
  Real minimum_cell_radius = std::numeric_limits<Real>::max();
} initial_match;

void ControlledTransitionHistory(HistoryData *pdata, Mesh *mesh) {
  enum Index {
    kDeltaQ, kDeltaQDot, kDeltaP, kDeltaPDot, kEG, kEAlpha, kFitCount,
    kLambdaMin, kLambdaMax, kDetThirdMin, kDetThirdMax, kConditionMax,
    kRelativeLapseMin, kRelativeLapseMax, kV2Max, kPsiMax, kInversePsiMax,
    kMinusPhysicalLapseMin, kPhysicalLapseMax, kCharacteristicMax,
    kRCore, kTransitionAmplitude, kFeedbackActive, kShellValid,
    kControllerGeneration, kHistoryCount
  };
  static_assert(kHistoryCount <= NHISTORY_VARIABLES,
                "controlled transition history exceeds fixed history storage");
  const char *labels[kHistoryCount] = {
    "delta-q", "delta-qdot", "delta-p", "delta-pdot", "e-G", "e-alpha",
    "fit-cells", "G-lmin", "G-lmax", "detG13-min", "detG13-max",
    "G-cond-max", "arel-min", "arel-max", "v2-max", "Psi-max",
    "invPsi-max", "minus-a-min", "a-max", "char-max", "r-core",
    "transition", "feedback", "shell-valid", "ctrl-gen"
  };
  pdata->nhist = kHistoryCount;
  for (int n = 0; n < kHistoryCount; ++n) {
    pdata->label[n] = labels[n];
    pdata->hdata[n] = 0.0;
    // All controller quantities are replicated.  Physical extrema are also
    // naturally global maxima (the minimum is stored with a minus sign).
    pdata->use_max[n] = true;
  }

  auto *module = mesh->pmb_pack->prefgh;
  if (module->opt.reference_controlled) {
    module->MeasureControllerAtTime(mesh->time);
  }
  module->UpdateDiagnostics();
  auto &diagnostics = module->controller_diagnostics;
  pdata->hdata[kDeltaQ] = module->controller.delta_q;
  pdata->hdata[kDeltaQDot] = module->controller.delta_q_dot;
  pdata->hdata[kDeltaP] = module->controller.delta_p;
  pdata->hdata[kDeltaPDot] = module->controller.delta_p_dot;
  pdata->hdata[kEG] = diagnostics.e_G;
  pdata->hdata[kEAlpha] = diagnostics.e_alpha;
  pdata->hdata[kFitCount] = diagnostics.fitting_cell_count;
  pdata->hdata[kLambdaMin] = diagnostics.lambda_min;
  pdata->hdata[kLambdaMax] = diagnostics.lambda_max;
  pdata->hdata[kDetThirdMin] = diagnostics.det_g_third_min;
  pdata->hdata[kDetThirdMax] = diagnostics.det_g_third_max;
  pdata->hdata[kConditionMax] = diagnostics.condition_max;
  pdata->hdata[kRelativeLapseMin] = diagnostics.relative_lapse_min;
  pdata->hdata[kRelativeLapseMax] = diagnostics.relative_lapse_max;
  pdata->hdata[kV2Max] = diagnostics.v2_max;
  pdata->hdata[kPsiMax] = diagnostics.psi_max;
  pdata->hdata[kInversePsiMax] = diagnostics.inverse_psi_max;
  pdata->hdata[kRCore] = diagnostics.r_core;
  pdata->hdata[kTransitionAmplitude] = diagnostics.transition_amplitude;
  pdata->hdata[kFeedbackActive] = diagnostics.feedback_active ? 1.0 : 0.0;
  pdata->hdata[kShellValid] = diagnostics.fitting_shell_valid ? 1.0 : 0.0;
  pdata->hdata[kControllerGeneration] =
      static_cast<Real>(module->controller_generation);

  auto &indcs = mesh->mb_indcs;
  const auto adm_vars = mesh->pmb_pack->padm->adm;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real minus_alpha_min = -std::numeric_limits<Real>::max();
  Real alpha_max = 0.0;
  Real characteristic_max = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh controlled physical extrema",
      Kokkos::RangePolicy<>(DevExeSpace(),
          0, mesh->pmb_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_minus_alpha_min,
                    Real &local_alpha_max, Real &local_characteristic_max) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real alpha = adm_vars.alpha(m, k, j, i);
        if (-alpha > local_minus_alpha_min) local_minus_alpha_min = -alpha;
        if (alpha > local_alpha_max) local_alpha_max = alpha;
        Real metric[4][4] = {};  // NOLINT(runtime/arrays)
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            metric[I + 1][J + 1] = adm_vars.g_dd(m, I, J, k, j, i);
          }
        }
        Real inverse[3][3];  // NOLINT(runtime/arrays)
        Real determinant = 0.0;
        if (ref_gh::InvertSpatial3(metric, inverse, determinant)) {
          for (int I = 0; I < 3; ++I) {
            const Real speed = Kokkos::abs(adm_vars.beta_u(m, I, k, j, i))
                + alpha*Kokkos::sqrt(inverse[I][I]);
            if (speed > local_characteristic_max) {
              local_characteristic_max = speed;
            }
          }
        }
      }, Kokkos::Max<Real>(minus_alpha_min), Kokkos::Max<Real>(alpha_max),
      Kokkos::Max<Real>(characteristic_max));
  pdata->hdata[kMinusPhysicalLapseMin] = minus_alpha_min;
  pdata->hdata[kPhysicalLapseMax] = alpha_max;
  pdata->hdata[kCharacteristicMax] = characteristic_max;
}

void FinishControlledTransition(ParameterInput *pin, Mesh *mesh) {
  auto *module = mesh->pmb_pack->prefgh;
  module->UpdateDiagnostics();
  const auto state = module->u0;
  auto &indcs = mesh->mb_indcs;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real state_max = 0.0;
  Real bad_state = 0.0;
  Kokkos::parallel_reduce(
      "ref_gh controlled final state",
      Kokkos::RangePolicy<>(DevExeSpace(),
          0, mesh->pmb_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_state_max, Real &local_bad_state) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        if (0.0 > local_bad_state) local_bad_state = 0.0;
        for (int n = 0; n < ref_gh::nvar; ++n) {
          const Real value = state(m, n, k, j, i);
          if (!Kokkos::isfinite(value)) local_bad_state = 1.0;
          const Real magnitude = Kokkos::abs(value);
          if (magnitude > local_state_max) local_state_max = magnitude;
        }
      }, Kokkos::Max<Real>(state_max), Kokkos::Max<Real>(bad_state));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &state_max, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &bad_state, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename")
                                 + "-controlled-transition.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# time cycles initial_state_Linf initial_G_Linf "
                       "initial_arel_Linf initial_shift_Linf min_cell_r "
                       "final_state_Linf bad_state delta_q delta_p\n");
    std::fprintf(file, "%.17e %d %.17e %.17e %.17e %.17e %.17e %.17e "
                       "%.17e %.17e %.17e\n",
                 mesh->time, mesh->ncycle, initial_match.regular_state_linf,
                 initial_match.relative_spatial_linf,
                 initial_match.relative_lapse_linf,
                 initial_match.relative_shift_linf,
                 initial_match.minimum_cell_radius, state_max, bad_state,
                 module->controller.delta_q, module->controller.delta_p);
    std::fclose(file);
    std::cout << "reference-GH controlled Schwarzschild final: time="
              << mesh->time << " state Linf=" << state_max
              << " bad-state=" << bad_state << std::endl;
  }
  if (bad_state > 0.0) {
    std::cout << "### FATAL ERROR: controlled Schwarzschild state became nonfinite."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

void FinishEstimatorCalibration(ParameterInput *pin, Mesh *mesh) {
  auto *module = mesh->pmb_pack->prefgh;
  module->MeasureControllerAtTime(mesh->time);
  const auto &diagnostics = module->controller_diagnostics;
  if (!diagnostics.fitting_shell_valid
      || !std::isfinite(diagnostics.e_G)
      || !std::isfinite(diagnostics.e_alpha)) {
    std::cout << "### FATAL ERROR: Ref-GH planted estimator shell is invalid."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    const std::string filename = pin->GetString("job", "basename")
                                 + "-estimator.dat";
    FILE *file = std::fopen(filename.c_str(), "w");
    if (file == nullptr) std::exit(EXIT_FAILURE);
    std::fprintf(file, "# nx delta_q expected_e_G measured_e_G delta_p "
                       "expected_e_alpha measured_e_alpha fit_cells shell_valid\n");
    std::fprintf(file, "%d %.17e %.17e %.17e %.17e %.17e %.17e %.17e %d\n",
                 mesh->mesh_indcs.nx1, module->controller.delta_q,
                 2.0*module->controller.delta_q, diagnostics.e_G,
                 module->controller.delta_p, -module->controller.delta_p,
                 diagnostics.e_alpha, diagnostics.fitting_cell_count,
                 diagnostics.fitting_shell_valid ? 1 : 0);
    std::fclose(file);
    std::cout << "reference-GH estimator calibration: nx="
              << mesh->mesh_indcs.nx1
              << " delta_q=" << module->controller.delta_q
              << " e_G=" << diagnostics.e_G
              << " delta_p=" << module->controller.delta_p
              << " e_alpha=" << diagnostics.e_alpha
              << " cells=" << diagnostics.fitting_cell_count << std::endl;
  }
}

}  // namespace

void ProblemGenerator::RefGhControlledTransition(ParameterInput *pin,
                                                  const bool restart) {
  const bool estimator_calibration =
      pin->GetString("problem", "pgen_name") == "ref_gh_estimator_calibration";
  pgen_final_func = estimator_calibration
      ? &FinishEstimatorCalibration : &FinishControlledTransition;
  user_hist_func = &ControlledTransitionHistory;
  if (restart) return;
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->prefgh == nullptr
      || (pack->prefgh->opt.reference_kind != 4
          && pack->prefgh->opt.reference_kind != 5)) {
    std::cout << "controlled Schwarzschild data require ref_gh/reference=wormhole "
                 "or controlled_transition." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // The puncture must be a Cartesian vertex on every logical level that is
  // actually present.  This integer lattice check is stronger than r_min>0.
  const Real center[3] = {pack->prefgh->opt.reference_center[0],
                          pack->prefgh->opt.reference_center[1],
                          pack->prefgh->opt.reference_center[2]};
  const Real domain_min[3] = {pmy_mesh_->mesh_size.x1min,
                              pmy_mesh_->mesh_size.x2min,
                              pmy_mesh_->mesh_size.x3min};
  const Real domain_length[3] = {
      pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min,
      pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min,
      pmy_mesh_->mesh_size.x3max - pmy_mesh_->mesh_size.x3min};
  const int root_cells[3] = {pmy_mesh_->mesh_indcs.nx1,
                             pmy_mesh_->mesh_indcs.nx2,
                             pmy_mesh_->mesh_indcs.nx3};
  bool active_level[64] = {};
  if (pmy_mesh_->max_level >= 64) {
    std::cout << "### FATAL ERROR: vertex audit supports fewer than 64 levels."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int m = 0; m < pmy_mesh_->nmb_total; ++m) {
    active_level[pmy_mesh_->lloc_eachmb[m].level] = true;
  }
  for (int level = pmy_mesh_->root_level; level <= pmy_mesh_->max_level; ++level) {
    if (!active_level[level]) continue;
    const Real scale = std::ldexp(1.0, level - pmy_mesh_->root_level);
    for (int direction = 0; direction < 3; ++direction) {
      const Real spacing = domain_length[direction]/(root_cells[direction]*scale);
      const Real lattice_coordinate = (center[direction] - domain_min[direction])
                                      /spacing;
      if (std::abs(lattice_coordinate - std::round(lattice_coordinate))
          > 64.0*std::numeric_limits<Real>::epsilon()
                *std::max(1.0, std::abs(lattice_coordinate))) {
        std::cout << "### FATAL ERROR: Ref-GH puncture is not a cell vertex at level "
                  << level << " direction=" << direction
                  << " lattice coordinate=" << lattice_coordinate << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }

  auto &indcs = pmy_mesh_->mb_indcs;
  auto &size = pack->pmb->mb_size;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  const auto state = pack->prefgh->u0;
  const Real center_x = center[0];
  const Real center_y = center[1];
  const Real center_z = center[2];
  const Real mass = pack->prefgh->opt.reference_mass;
  const Real start_time = pmy_mesh_->time;
  const Real r_core0 = pack->prefgh->opt.r_core0;
  const Real tau_core = pack->prefgh->opt.tau_core;
  const Real kappa_core = pack->prefgh->opt.kappa_core;
  const Real tau_transition = pack->prefgh->opt.tau_transition;
  const Real outer_start = pack->prefgh->opt.regularization_outer_start;
  const Real outer_end = pack->prefgh->opt.regularization_outer_end;
  const Real planted_delta_q = pack->prefgh->controller.delta_q;
  const Real planted_delta_p = pack->prefgh->controller.delta_p;
  const auto table = pack->prefgh->reference_table;
  par_for(estimator_calibration ? "ref_gh estimator planted data"
                                : "ref_gh wormhole matched data",
  DevExeSpace(), 0,
  pack->nmb_thispack - 1, 0, n3 - 1, 0, n2 - 1, 0, n1 - 1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int n = 0; n < ref_gh::nvar; ++n) state(m, n, k, j, i) = 0.0;
    if (!estimator_calibration) {
      state(m, ref_gh::PsiIndex(0, 0), k, j, i) = -1.0;
      state(m, ref_gh::PsiIndex(1, 1), k, j, i) = 1.0;
      state(m, ref_gh::PsiIndex(2, 2), k, j, i) = 1.0;
      state(m, ref_gh::PsiIndex(3, 3), k, j, i) = 1.0;
      return;
    }
    const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                               size.d_view(m).x1min, size.d_view(m).x1max);
    const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                               size.d_view(m).x2min, size.d_view(m).x2max);
    const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                               size.d_view(m).x3min, size.d_view(m).x3max);
    const Real dx = x - center_x;
    const Real dy = y - center_y;
    const Real dz = z - center_z;
    const Real radius = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
    const Real rho = radius/mass;
    const ref_gh::RadialProfile alpha_profile =
        ref_gh::InterpolateTrumpetProfile(table, ref_gh::kCoeffAlpha, rho);
    const ref_gh::RadialProfile areal_profile =
        ref_gh::InterpolateTrumpetProfile(
            table, ref_gh::kCoeffArealRadius, rho);
    const ref_gh::RadialProfile shift_profile =
        ref_gh::InterpolateTrumpetProfile(table, ref_gh::kCoeffShiftQ, rho);
    const Real alpha_t = alpha_profile.value;
    const Real psi2_t = ref_gh::ArealRadiusToPsi2(areal_profile, rho).value;
    const Real shift_t = shift_profile.value/mass;
    const Real psi_w = 1.0 + 0.5/rho;
    const Real psi2_w = psi_w*psi_w;
    const Real alpha_w = 1.0/psi2_w;
    const Real r_core = r_core0*mass*Kokkos::exp(
        -start_time/(tau_core*mass));
    const Real core_coordinate =
        (radius/r_core - 1.0)/kappa_core;
    const Real core_blend = core_coordinate <= 0.0 ? 0.0
        : (core_coordinate >= 1.0 ? 1.0
           : core_coordinate*core_coordinate*core_coordinate
               *(10.0 + core_coordinate*(-15.0 + 6.0*core_coordinate)));
    const Real transition_coordinate = start_time/(tau_transition*mass);
    const Real activation = transition_coordinate <= 0.0 ? 0.0
        : (transition_coordinate >= 1.0 ? 1.0
           : transition_coordinate*transition_coordinate*transition_coordinate
               *(10.0 + transition_coordinate
                   *(-15.0 + 6.0*transition_coordinate)));
    const Real blend = activation*core_blend;
    const Real base_log_alpha =
        (1.0 - blend)*Kokkos::log(alpha_w) + blend*Kokkos::log(alpha_t);
    const Real base_log_psi2 =
        (1.0 - blend)*Kokkos::log(psi2_w) + blend*Kokkos::log(psi2_t);
    const Real outer_coordinate =
        (rho - outer_start)/(outer_end - outer_start);
    const Real outer_smooth = outer_coordinate <= 0.0 ? 0.0
        : (outer_coordinate >= 1.0 ? 1.0
           : outer_coordinate*outer_coordinate*outer_coordinate
               *(10.0 + outer_coordinate*(-15.0 + 6.0*outer_coordinate)));
    const Real window = core_blend*(1.0 - outer_smooth);
    const Real log_rho = Kokkos::log(rho);
    const Real alpha_ref = Kokkos::exp(
        base_log_alpha + planted_delta_p*window*log_rho);
    const Real psi2_ref = Kokkos::exp(
        base_log_psi2 - planted_delta_q*window*log_rho);
    const Real shift_ref = blend*shift_t;
    const Real spatial_ratio2 = (psi2_t/psi2_ref)*(psi2_t/psi2_ref);
    const Real shift_difference = shift_t - shift_ref;
    state(m, ref_gh::PsiIndex(0, 0), k, j, i) =
        (-alpha_t*alpha_t
         + psi2_t*psi2_t*shift_difference*shift_difference*radius*radius)
        /(alpha_ref*alpha_ref);
    const Real displacements[3] = {dx, dy, dz};
    for (int I = 0; I < 3; ++I) {
      state(m, ref_gh::PsiIndex(0, I + 1), k, j, i) =
          psi2_t*psi2_t*shift_difference*displacements[I]
          /(alpha_ref*psi2_ref);
      state(m, ref_gh::PsiIndex(I + 1, I + 1), k, j, i) = spatial_ratio2;
    }
  });

  if (estimator_calibration) {
    pack->prefgh->MeasureControllerAtTime(pmy_mesh_->time);
    const auto &diagnostics = pack->prefgh->controller_diagnostics;
    if (global_variable::my_rank == 0) {
      std::cout << "reference-GH planted estimator initialized: e_G="
                << diagnostics.e_G << " e_alpha=" << diagnostics.e_alpha
                << " shell-valid=" << diagnostics.fitting_shell_valid
                << std::endl;
    }
    return;
  }

  pack->prefgh->UpdateDiagnostics();
  const auto adm_vars = pack->padm->adm;
  Real state_error = 0.0;
  Real spatial_error = 0.0;
  Real lapse_error = 0.0;
  Real shift_error = 0.0;
  Real minimum_radius = std::numeric_limits<Real>::max();
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Kokkos::parallel_reduce(
      "ref_gh wormhole initial match",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, Real &local_state_error,
                    Real &local_spatial_error, Real &local_lapse_error,
                    Real &local_shift_error, Real &local_minimum_radius) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is; work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js; work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        const Real dx = x - center[0];
        const Real dy = y - center[1];
        const Real dz = z - center[2];
        const Real radius = Kokkos::sqrt(dx*dx + dy*dy + dz*dz);
        if (radius < local_minimum_radius) local_minimum_radius = radius;
        for (int n = 0; n < ref_gh::nvar; ++n) {
          Real expected = 0.0;
          if (n == ref_gh::PsiIndex(0, 0)) expected = -1.0;
          if (n == ref_gh::PsiIndex(1, 1) || n == ref_gh::PsiIndex(2, 2)
              || n == ref_gh::PsiIndex(3, 3)) expected = 1.0;
          const Real error = Kokkos::abs(state(m, n, k, j, i) - expected);
          if (error > local_state_error) local_state_error = error;
        }
        for (int I = 0; I < 3; ++I) {
          for (int J = 0; J < 3; ++J) {
            const Real expected = I == J ? 1.0 : 0.0;
            const Real error = Kokkos::abs(
                state(m, ref_gh::PsiIndex(I + 1, J + 1), k, j, i) - expected);
            if (error > local_spatial_error) local_spatial_error = error;
          }
          const Real shift = Kokkos::abs(adm_vars.beta_u(m, I, k, j, i));
          if (shift > local_shift_error) local_shift_error = shift;
        }
        const Real psi_w = 1.0 + 0.5*mass/radius;
        const Real alpha_w = 1.0/(psi_w*psi_w);
        const Real alpha_ratio_error = Kokkos::abs(
            adm_vars.alpha(m, k, j, i)/alpha_w - 1.0);
        if (alpha_ratio_error > local_lapse_error) {
          local_lapse_error = alpha_ratio_error;
        }
      }, Kokkos::Max<Real>(state_error), Kokkos::Max<Real>(spatial_error),
      Kokkos::Max<Real>(lapse_error), Kokkos::Max<Real>(shift_error),
      Kokkos::Min<Real>(minimum_radius));
#if MPI_PARALLEL_ENABLED
  Real maxima[4] = {state_error, spatial_error, lapse_error, shift_error};
  MPI_Allreduce(MPI_IN_PLACE, maxima, 4, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  state_error = maxima[0]; spatial_error = maxima[1];
  lapse_error = maxima[2]; shift_error = maxima[3];
  MPI_Allreduce(MPI_IN_PLACE, &minimum_radius, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  initial_match = {state_error, spatial_error, lapse_error, shift_error,
                   minimum_radius};
  if (!(minimum_radius > 0.0) || state_error > 1.0e-13
      || spatial_error > 1.0e-13 || lapse_error > 1.0e-12
      || shift_error > 1.0e-13) {
    std::cout << "### FATAL ERROR: Ref-GH wormhole/reference initial match failed: "
              << "state=" << state_error << " G=" << spatial_error
              << " lapse=" << lapse_error << " shift=" << shift_error
              << " rmin=" << minimum_radius << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pack->prefgh->opt.reference_controlled) {
    pack->prefgh->MeasureControllerAtTime(0.0);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "reference-GH puncture vertex audit passed; minimum cell radius="
              << minimum_radius << ", wormhole match state=" << state_error
              << ", G=" << spatial_error << ", lapse=" << lapse_error
              << ", shift=" << shift_error << std::endl;
  }
}
