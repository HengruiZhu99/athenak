//========================================================================================
//! \file ref_gh.cpp
//! \brief Construction and storage for the separate 50-field reference-frame GH module.
//========================================================================================
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "coordinates/adm.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "ref_gh/ref_gh.hpp"
#include "ref_gh/reference_cache.hpp"
#include "ref_gh/reference_controlled_schwarzschild.hpp"
#include "ref_gh/reference_provider_cache.hpp"
#include "ref_gh/reference_trumpet_schwarzschild.hpp"

namespace ref_gh {

char const * const RefGh::StateNames[RefGh::nref_gh] = {
  "ref_gh_Psi00", "ref_gh_Psi01", "ref_gh_Psi02", "ref_gh_Psi03",
  "ref_gh_Psi11", "ref_gh_Psi12", "ref_gh_Psi13", "ref_gh_Psi22",
  "ref_gh_Psi23", "ref_gh_Psi33",
  "ref_gh_Pi00", "ref_gh_Pi01", "ref_gh_Pi02", "ref_gh_Pi03",
  "ref_gh_Pi11", "ref_gh_Pi12", "ref_gh_Pi13", "ref_gh_Pi22",
  "ref_gh_Pi23", "ref_gh_Pi33",
  "ref_gh_Phi100", "ref_gh_Phi101", "ref_gh_Phi102", "ref_gh_Phi103",
  "ref_gh_Phi111", "ref_gh_Phi112", "ref_gh_Phi113", "ref_gh_Phi122",
  "ref_gh_Phi123", "ref_gh_Phi133",
  "ref_gh_Phi200", "ref_gh_Phi201", "ref_gh_Phi202", "ref_gh_Phi203",
  "ref_gh_Phi211", "ref_gh_Phi212", "ref_gh_Phi213", "ref_gh_Phi222",
  "ref_gh_Phi223", "ref_gh_Phi233",
  "ref_gh_Phi300", "ref_gh_Phi301", "ref_gh_Phi302", "ref_gh_Phi303",
  "ref_gh_Phi311", "ref_gh_Phi312", "ref_gh_Phi313", "ref_gh_Phi322",
  "ref_gh_Phi323", "ref_gh_Phi333"
};

char const * const RefGh::ConstraintNames[RefGh::ncon] = {
  "ref_gh_C0", "ref_gh_C1", "ref_gh_C2", "ref_gh_C3",
  "ref_gh_reduction", "ref_gh_curl",
  "ref_gh_Q", "ref_gh_Delta", "ref_gh_frame_Ricci",
  "ref_gh_coordinate_Ricci", "ref_gh_source_curvature",
  "ref_gh_source_QQ", "ref_gh_source_DeltaDelta",
  "ref_gh_source_damping", "ref_gh_source_frame_correction",
  "ref_gh_metric_condition"
};

RefGh::RefGh(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("u0 ref_gh", 1, 1, 1, 1, 1),
    u1("u1 ref_gh", 1, 1, 1, 1, 1),
    u_rhs("u_rhs ref_gh", 1, 1, 1, 1, 1),
    u_con("u_con ref_gh", 1, 1, 1, 1, 1),
    coarse_u0("coarse u0 ref_gh", 1, 1, 1, 1, 1),
    reference_provider("ref_gh reference provider", 1, 1, 1, 1, 1),
    reference_workspace("ref_gh reference workspace", 1, 1, 1, 1, 1),
    reference_evolution("ref_gh reference evolution", 1, 1, 1, 1, 1),
    reference_diagnostic("ref_gh reference diagnostic", 1, 1, 1, 1, 1),
    reference_table("ref_gh reference table", 1, 1),
    reference_cache_time(NAN), reference_diagnostic_time(NAN),
    controller_generation(0), reference_cache_generation(0),
    reference_diagnostic_generation(0),
    controller{0.0, 0.0, 0.0, 0.0},
    controller_base{0.0, 0.0, 0.0, 0.0},
    controller_rhs{0.0, 0.0, 0.0, 0.0},
    controller_diagnostics{},
    reference_cache_oracle_validated(false),
    reference_diagnostic_oracle_validated(false),
    dtnew(0.0), max_char_speed(0.0), pmy_pack(ppack), pinput(pin) {
  opt.fd_order = pin->GetOrAddInteger("ref_gh", "fd_order", 4);
  opt.extrap_order = pin->GetOrAddInteger("ref_gh", "extrap_order", 2);
  const std::string reference_name =
      pin->GetOrAddString("ref_gh", "reference", "minkowski");
  if (reference_name == "minkowski") {
    opt.reference_kind = 0;
  } else if (reference_name == "trumpet") {
    opt.reference_kind = 1;
  } else if (reference_name == "time_dependent_lapse_test") {
    opt.reference_kind = 2;
  } else if (reference_name == "time_dependent_spatial_test") {
    opt.reference_kind = 3;
  } else if (reference_name == "wormhole") {
    opt.reference_kind = 4;
  } else if (reference_name == "controlled_transition") {
    opt.reference_kind = 5;
  } else {
    std::cout << "### FATAL ERROR: ref_gh reference must be minkowski, trumpet, "
                 "time_dependent_lapse_test, time_dependent_spatial_test, "
                 "wormhole, or controlled_transition."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.reference_time_dependent =
      GetReferenceProviderMetadata(opt.reference_kind).time_dependent;
  opt.reference_controlled = opt.reference_kind == 5;
  opt.controller_enabled =
      pin->GetOrAddBoolean("ref_gh", "controller_enabled", false);
  const std::string source_name =
      pin->GetOrAddString("ref_gh", "source", "covariant");
  if (source_name == "covariant") {
    opt.source_kind = 0;
  } else if (source_name == "coordinate_oracle") {
    opt.source_kind = 1;
  } else {
    std::cout << "### FATAL ERROR: ref_gh source must be covariant or coordinate_oracle."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.debug_task_fences =
      pin->GetOrAddBoolean("ref_gh", "debug_task_fences", false);
  opt.validate_reference_cache =
      pin->GetOrAddBoolean("ref_gh", "validate_reference_cache", false);
  opt.gamma0 = pin->GetOrAddReal("ref_gh", "gamma0", 1.0);
  opt.diss = pin->GetOrAddReal("ref_gh", "diss", 0.02);
  opt.fail_closed_dt = pin->GetOrAddReal("ref_gh", "fail_closed_dt", 0.0);
  opt.reference_mass = pin->GetOrAddReal("ref_gh", "reference_mass", 1.0);
  opt.reference_center[0] = pin->GetOrAddReal("ref_gh", "reference_x", 0.0);
  opt.reference_center[1] = pin->GetOrAddReal("ref_gh", "reference_y", 0.0);
  opt.reference_center[2] = pin->GetOrAddReal("ref_gh", "reference_z", 0.0);
  opt.r_core0 = pin->GetOrAddReal("ref_gh", "r_core0", 0.30);
  opt.tau_core = pin->GetOrAddReal("ref_gh", "tau_core", 1.5);
  opt.kappa_core = pin->GetOrAddReal("ref_gh", "kappa_core", 1.0);
  opt.tau_transition = pin->GetOrAddReal("ref_gh", "tau_transition", 4.0);
  opt.r_fit_min = pin->GetOrAddReal("ref_gh", "r_fit_min", 0.15);
  opt.r_fit_max = pin->GetOrAddReal("ref_gh", "r_fit_max", 0.40);
  opt.regularization_outer_start =
      pin->GetOrAddReal("ref_gh", "regularization_outer_start", 0.50);
  opt.regularization_outer_end =
      pin->GetOrAddReal("ref_gh", "regularization_outer_end", 0.60);
  opt.controller_zeta = pin->GetOrAddReal("ref_gh", "controller_zeta", 1.0);
  opt.controller_omega_q =
      pin->GetOrAddReal("ref_gh", "controller_omega_q", 0.25);
  opt.controller_omega_p =
      pin->GetOrAddReal("ref_gh", "controller_omega_p", 0.25);
  opt.controller_acceleration_limit =
      pin->GetOrAddReal("ref_gh", "controller_acceleration_limit", 0.05);
  opt.controller_delta_bound =
      pin->GetOrAddReal("ref_gh", "controller_delta_bound", 0.25);
  opt.controller_rate_bound =
      pin->GetOrAddReal("ref_gh", "controller_rate_bound", 0.10);
  controller.delta_q =
      pin->GetOrAddReal("ref_gh", "controller_delta_q", 0.0);
  controller.delta_q_dot =
      pin->GetOrAddReal("ref_gh", "controller_delta_q_dot", 0.0);
  controller.delta_p =
      pin->GetOrAddReal("ref_gh", "controller_delta_p", 0.0);
  controller.delta_p_dot =
      pin->GetOrAddReal("ref_gh", "controller_delta_p_dot", 0.0);
  const Real stored_generation =
      pin->GetOrAddReal("ref_gh", "controller_generation", 0.0);
  if (!(stored_generation >= 0.0) || !std::isfinite(stored_generation)) {
    std::cout << "### FATAL ERROR: invalid stored Ref-GH controller generation."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  controller_generation = static_cast<std::uint64_t>(stored_generation);
  controller_base = controller;
  const int derivative_radius = opt.fd_order/2;
  if ((opt.fd_order != 2 && opt.fd_order != 4 && opt.fd_order != 6)
      || ppack->pmesh->mb_indcs.ng < 2*derivative_radius) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh fd_order must be 2, 4, or 6, with at least "
              << "fd_order ghost cells for its compatible two-pass update." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.gamma0 <= 0.0 || opt.diss < 0.0 || opt.fail_closed_dt < 0.0
      || opt.reference_mass <= 0.0
      || opt.r_core0 <= 0.0 || opt.tau_core <= 0.0
      || opt.kappa_core <= 0.0 || opt.tau_transition <= 0.0
      || opt.r_fit_min <= 0.0 || opt.r_fit_max <= opt.r_fit_min
      || opt.regularization_outer_start <= opt.r_fit_max
      || opt.regularization_outer_end <= opt.regularization_outer_start
      || opt.controller_zeta <= 0.0 || opt.controller_omega_q <= 0.0
      || opt.controller_omega_p <= 0.0
      || opt.controller_acceleration_limit <= 0.0
      || opt.controller_delta_bound <= 0.0 || opt.controller_rate_bound <= 0.0
      || opt.extrap_order < 2 || opt.extrap_order > 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh requires gamma0>0, diss>=0, fail_closed_dt>=0, "
              << "valid positive reference/controller scales, and extrap_order "
                 "in [2,4]." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppack->pmesh->multilevel && opt.fd_order == 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh fd_order=6 lacks matching AthenaK AMR transfer."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->GetString("time", "evolution") != "static" && !ppack->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "ref_gh evolution requires a three-dimensional mesh."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = ppack->pmesh->mb_indcs;
  const int n1 = indcs.nx1 + 2*indcs.ng;
  const int n2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  const int n3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;
  Kokkos::realloc(u0, nmb, nref_gh, n3, n2, n1);
  Kokkos::realloc(u1, nmb, nref_gh, n3, n2, n1);
  Kokkos::realloc(u_rhs, nmb, nref_gh, n3, n2, n1);
  Kokkos::realloc(u_con, nmb, ncon, n3, n2, n1);
  Kokkos::realloc(reference_provider, nmb, kReferenceProviderSize, n3, n2, n1);
  Kokkos::realloc(reference_workspace, nmb, kReferenceWorkspaceSize, n3, n2, n1);
  Kokkos::realloc(reference_evolution, nmb, kReferenceEvolutionSize, n3, n2, n1);
  Kokkos::realloc(reference_diagnostic, nmb, kReferenceDiagnosticSize, n3, n2, n1);
  if (ppack->pmesh->multilevel) {
    const int cn1 = indcs.cnx1 + 2*indcs.ng;
    const int cn2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    const int cn3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(coarse_u0, nmb, nref_gh, cn3, cn2, cn1);
  }
  if (opt.reference_kind == 1 || opt.reference_kind == 5) {
    Kokkos::realloc(reference_table, kTrumpetProfiles, kTrumpetTableSize);
    auto host_table = Kokkos::create_mirror_view(reference_table);
    for (int i = 0; i < kTrumpetTableSize; ++i) {
      host_table(kProfileAlpha, i) = kTrumpetAlpha[i];
      host_table(kProfileAlphaDy, i) = kTrumpetAlphaDy[i];
      host_table(kProfileAlphaDyy, i) = kTrumpetAlphaDyy[i];
      host_table(kProfileArealRadius, i) = kTrumpetArealRadius[i];
      host_table(kProfileArealRadiusDy, i) = kTrumpetArealRadiusDy[i];
      host_table(kProfileArealRadiusDyy, i) = kTrumpetArealRadiusDyy[i];
      host_table(kProfileShiftQ, i) = kTrumpetShiftQ[i];
      host_table(kProfileShiftQDy, i) = kTrumpetShiftQDy[i];
      host_table(kProfileShiftQDyy, i) = kTrumpetShiftQDyy[i];
      host_table(kCoeffAlpha, i) = kTrumpetAlphaA0[i];
      host_table(kCoeffAlpha + 1, i) = kTrumpetAlphaA1[i];
      host_table(kCoeffAlpha + 2, i) = kTrumpetAlphaA2[i];
      host_table(kCoeffAlpha + 3, i) = kTrumpetAlphaA3[i];
      host_table(kCoeffAlpha + 4, i) = kTrumpetAlphaA4[i];
      host_table(kCoeffAlpha + 5, i) = kTrumpetAlphaA5[i];
      host_table(kCoeffArealRadius, i) = kTrumpetArealRadiusA0[i];
      host_table(kCoeffArealRadius + 1, i) = kTrumpetArealRadiusA1[i];
      host_table(kCoeffArealRadius + 2, i) = kTrumpetArealRadiusA2[i];
      host_table(kCoeffArealRadius + 3, i) = kTrumpetArealRadiusA3[i];
      host_table(kCoeffArealRadius + 4, i) = kTrumpetArealRadiusA4[i];
      host_table(kCoeffArealRadius + 5, i) = kTrumpetArealRadiusA5[i];
      host_table(kCoeffShiftQ, i) = kTrumpetShiftQA0[i];
      host_table(kCoeffShiftQ + 1, i) = kTrumpetShiftQA1[i];
      host_table(kCoeffShiftQ + 2, i) = kTrumpetShiftQA2[i];
      host_table(kCoeffShiftQ + 3, i) = kTrumpetShiftQA3[i];
      host_table(kCoeffShiftQ + 4, i) = kTrumpetShiftQA4[i];
      host_table(kCoeffShiftQ + 5, i) = kTrumpetShiftQA5[i];
    }
    Kokkos::deep_copy(reference_table, host_table);
  }
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(nref_gh);
  if (ppack->padm != nullptr) ppack->padm->SetADMVariables = &RefGh::SetADMVariables;
}

RefGh::~RefGh() { delete pbval_u; }

}  // namespace ref_gh
