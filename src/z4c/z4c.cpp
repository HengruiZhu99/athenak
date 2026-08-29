//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c.cpp
//! \brief implementation of Z4c class constructor and assorted other functions

#include <math.h>
#include <sys/stat.h>  // mkdir

#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <memory>    // make_unique, unique_ptr
#include <type_traits>
#include <vector>    // vector
#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/vertex_amr.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/horizon_dump.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_vertex_topology.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/z4c_symmetry.hpp"
#include "z4c/state_admissibility.hpp"
#include "coordinates/adm.hpp"
#include "utils/cart_grid.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

namespace {

int NativeCoarseGhostWidth(MeshBlockPack *ppack) {
  const auto &indcs = ppack->pmesh->mb_indcs;
  if (ppack->z4c_symmetry.grid_centering != Z4cGridCentering::vertex) {
    return indcs.ng;
  }
  return vertex_amr::RequiredCoarseGhostWidthForTransferOrder(
      ppack->z4c_symmetry.vertex_prolongation_order, indcs.ng);
}

}  // namespace

const char *Z4cStateCheckpointName(const Z4cStateCheckpoint checkpoint) {
  switch (checkpoint) {
    case Z4cStateCheckpoint::pre_rhs: return "PRE_RHS";
    case Z4cStateCheckpoint::post_rk_update: return "POST_RK_UPDATE";
    case Z4cStateCheckpoint::post_restriction: return "POST_RESTRICTION";
    case Z4cStateCheckpoint::post_receive: return "POST_RECEIVE";
    case Z4cStateCheckpoint::post_physical_bc: return "POST_PHYSICAL_BC";
    case Z4cStateCheckpoint::post_prolongation: return "POST_PROLONGATION";
    case Z4cStateCheckpoint::pre_algconstr: return "PRE_ALGCONSTR";
    case Z4cStateCheckpoint::post_algconstr: return "POST_ALGCONSTR";
    case Z4cStateCheckpoint::post_amr_transfer: return "POST_AMR_TRANSFER";
    case Z4cStateCheckpoint::final_accepted_state: return "FINAL_ACCEPTED_STATE";
  }
  return "UNKNOWN";
}

char const * const Z4c::Z4c_names[Z4c::nz4c] = {
  "z4c_chi",
  "z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz",
  "z4c_Khat",
  "z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy", "z4c_Ayz", "z4c_Azz",
  "z4c_Gamx", "z4c_Gamy", "z4c_Gamz",
  "z4c_Theta",
  "z4c_alpha",
  "z4c_betax", "z4c_betay", "z4c_betaz",
  "z4c_Bx", "z4c_By", "z4c_Bz",
};

char const * const Z4c::Constraint_names[Z4c::ncon] = {
  "con_C",
  "con_H",
  "con_M",
  "con_Z",
  "con_Mx", "con_My", "con_Mz",
};

template <typename Centering>
void Z4c::AllocateNativeStorage(const int nmb) {
  constexpr bool is_cell = std::is_same_v<Centering, CellCenteredZ4c>;
  constexpr bool is_vertex = std::is_same_v<Centering, VertexCenteredZ4c>;
  static_assert(is_cell || is_vertex, "unknown Z4c centering tag");
  constexpr Z4cGridCentering expected =
      is_vertex ? Z4cGridCentering::vertex : Z4cGridCentering::cell;
  if (layout.centering != expected) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": native Z4c storage centering dispatch mismatch" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Kokkos::realloc(u_con, nmb, ncon, layout.n3, layout.n2, layout.n1);
  Kokkos::realloc(u0, nmb, nz4c, layout.n3, layout.n2, layout.n1);
  Kokkos::realloc(u1, nmb, nz4c, layout.n3, layout.n2, layout.n1);
  Kokkos::realloc(u_rhs, nmb, nz4c, layout.n3, layout.n2, layout.n1);
  Kokkos::realloc(u_telegraph_mu, nmb, 1, layout.n3, layout.n2, layout.n1);
  Kokkos::deep_copy(u_telegraph_mu, 0.0);
  Kokkos::realloc(u_weyl, nmb, 2, layout.n3, layout.n2, layout.n1);
  if constexpr (is_vertex) {
    Kokkos::realloc(u_adm_native, nmb, ::adm::ADM::nadm, layout.n3,
                    layout.n2, layout.n1);
  }
  if (pmy_pack->pmesh->multilevel) {
    Kokkos::realloc(coarse_u0, nmb, nz4c, layout.cn3, layout.cn2, layout.cn1);
    Kokkos::realloc(coarse_u_weyl, nmb, 2, layout.cn3, layout.cn2, layout.cn1);
  }
}

void Z4c::ValidateNativeStorageExtents() const {
  const auto matches = [this](const auto &view, const int variables) {
    return view.extent_int(1) == variables && view.extent_int(2) == layout.n3 &&
           view.extent_int(3) == layout.n2 && view.extent_int(4) == layout.n1;
  };
  if (!matches(u_con, ncon) || !matches(u0, nz4c) || !matches(u1, nz4c) ||
      !matches(u_rhs, nz4c) || !matches(u_telegraph_mu, 1) ||
      !matches(u_weyl, 2)) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": a native Z4c array does not match the immutable "
              << ToString(layout.centering) << " layout" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (layout.centering == Z4cGridCentering::vertex &&
      !matches(u_adm_native, ::adm::ADM::nadm)) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": native VC ADM cache does not match the immutable layout"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->multilevel) {
    const auto coarse_matches = [this](const auto &view, const int variables) {
      return view.extent_int(1) == variables && view.extent_int(2) == layout.cn3 &&
             view.extent_int(3) == layout.cn2 && view.extent_int(4) == layout.cn1;
    };
    if (!coarse_matches(coarse_u0, nz4c) ||
        !coarse_matches(coarse_u_weyl, 2)) {
      std::cerr << "### FATAL ERROR in " << __FILE__
                << ": a coarse native Z4c array does not match the immutable "
                << ToString(layout.centering) << " layout" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

/*char const * const Z4c::Matter_names[Z4c::nmat] = {
  "mat_rho",
  "mat_Sx", "mat_Sy", "mat_Sz",
  "mat_Sxx", "mat_Sxy", "mat_Sxz", "mat_Syy", "mat_Syz", "mat_Szz",
};*/

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Z4c::Z4c(MeshBlockPack *ppack, ParameterInput *pin) :
  layout(MakeZ4cGridLayout(ppack->z4c_symmetry.grid_centering,
                           ppack->pmesh->mb_indcs,
                           NativeCoarseGhostWidth(ppack))),
  u_con("u_con",1,1,1,1,1),
  //u_mat("u_mat",1,1,1,1,1),
  u0("u0 z4c",1,1,1,1,1),
  u1("u1 z4c",1,1,1,1,1),
  u_rhs("u_rhs z4c",1,1,1,1,1),
  chi_provenance_terms("chi provenance terms",1,1,1,1,1),
  u_telegraph_mu("u_telegraph_mu",1,1,1,1,1),
  coarse_u0("coarse u0 z4c",1,1,1,1,1),
  u_weyl("u_weyl",1,1,1,1,1),
  coarse_u_weyl("coarse_u_weyl",1,1,1,1,1),
  u_adm_native("u_adm native z4c",1,1,1,1,1),
  pamr(new Z4c_AMR(pin)),
  pmy_pack(ppack) {
  dtnew = std::numeric_limits<Real>::max();
  dt_spatial = std::numeric_limits<Real>::max();
  dt_source = std::numeric_limits<Real>::max();
  max_source_rate = 0.0;
  max_coordinate_speed = 0.0;
  negative_real_stability_radius = 0.0;
  // (1) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  // int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  Kokkos::Profiling::pushRegion("Tensor fields");
  if (layout.centering == Z4cGridCentering::cell) {
    AllocateNativeStorage<CellCenteredZ4c>(nmb);
  } else if (layout.centering == Z4cGridCentering::vertex) {
    AllocateNativeStorage<VertexCenteredZ4c>(nmb);
  } else {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": invalid native Z4c storage centering" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  con.C.InitWithShallowSlice(u_con, I_CON_C);
  con.H.InitWithShallowSlice(u_con, I_CON_H);
  con.M.InitWithShallowSlice(u_con, I_CON_M);
  con.Z.InitWithShallowSlice(u_con, I_CON_Z);
  con.M_d.InitWithShallowSlice(u_con, I_CON_MX, I_CON_MZ);

  z4c.alpha.InitWithShallowSlice (u0, I_Z4C_ALPHA);
  z4c.beta_u.InitWithShallowSlice(u0, I_Z4C_BETAX, I_Z4C_BETAZ);
  z4c.vB_d.InitWithShallowSlice(u0, I_Z4C_BX, I_Z4C_BZ);
  z4c.chi.InitWithShallowSlice   (u0, I_Z4C_CHI);
  z4c.vKhat.InitWithShallowSlice  (u0, I_Z4C_KHAT);
  z4c.vTheta.InitWithShallowSlice (u0, I_Z4C_THETA);
  z4c.vGam_u.InitWithShallowSlice (u0, I_Z4C_GAMX, I_Z4C_GAMZ);
  z4c.g_dd.InitWithShallowSlice  (u0, I_Z4C_GXX, I_Z4C_GZZ);
  z4c.vA_dd.InitWithShallowSlice  (u0, I_Z4C_AXX, I_Z4C_AZZ);

  rhs.alpha.InitWithShallowSlice (u_rhs, I_Z4C_ALPHA);
  rhs.beta_u.InitWithShallowSlice(u_rhs, I_Z4C_BETAX, I_Z4C_BETAZ);
  rhs.vB_d.InitWithShallowSlice  (u_rhs, I_Z4C_BX, I_Z4C_BZ);
  rhs.chi.InitWithShallowSlice   (u_rhs, I_Z4C_CHI);
  rhs.vKhat.InitWithShallowSlice  (u_rhs, I_Z4C_KHAT);
  rhs.vTheta.InitWithShallowSlice (u_rhs, I_Z4C_THETA);
  rhs.vGam_u.InitWithShallowSlice (u_rhs, I_Z4C_GAMX, I_Z4C_GAMZ);
  rhs.g_dd.InitWithShallowSlice  (u_rhs, I_Z4C_GXX, I_Z4C_GZZ);
  rhs.vA_dd.InitWithShallowSlice  (u_rhs, I_Z4C_AXX, I_Z4C_AZZ);

  weyl.rpsi4.InitWithShallowSlice (u_weyl, 0);
  weyl.ipsi4.InitWithShallowSlice (u_weyl, 1);

  if (layout.centering == Z4cGridCentering::vertex) {
    adm.g_dd.InitWithShallowSlice(u_adm_native, ::adm::ADM::I_ADM_GXX,
                                 ::adm::ADM::I_ADM_GZZ);
    adm.vK_dd.InitWithShallowSlice(u_adm_native, ::adm::ADM::I_ADM_KXX,
                                  ::adm::ADM::I_ADM_KZZ);
    adm.psi4.InitWithShallowSlice(u_adm_native, ::adm::ADM::I_ADM_PSI4);
    adm.alpha.InitWithShallowSlice(u_adm_native, ::adm::ADM::I_ADM_ALPHA);
    adm.beta_u.InitWithShallowSlice(u_adm_native, ::adm::ADM::I_ADM_BETAX,
                                   ::adm::ADM::I_ADM_BETAZ);
  }

  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opt.chi_min_floor = pin->GetOrAddReal("z4c", "chi_min_floor", 1e-12);
  opt.floor_chi = pin->GetOrAddBoolean("z4c", "floor_chi", false);
  opt.diss = pin->GetOrAddReal("z4c", "diss", 0.0);
  // Preserve legacy parameter/output bytes unless the new runtime family is
  // explicitly opted into by the input deck.  Athena command-line overrides
  // are applied before construction, so an explicitly declared false value is
  // still sufficient to select lean_runtime=true at launch.
  const bool runtime_parameter_declared =
      pin->DoesParameterExist("z4c", "lean_runtime");
  const char *runtime_environment =
      std::getenv("ATHENA_Z4C_LEAN_RUNTIME");
  const bool runtime_environment_declared = runtime_environment != nullptr;
  bool runtime_environment_value = false;
  if (runtime_environment_declared) {
    const std::string value(runtime_environment);
    if (value == "1" || value == "true" || value == "on") {
      runtime_environment_value = true;
    } else if (value == "0" || value == "false" || value == "off") {
      runtime_environment_value = false;
    } else {
      std::cerr << "### FATAL ERROR in " << __FILE__
                << ": ATHENA_Z4C_LEAN_RUNTIME=" << value
                << "; expected 0/1, false/true, or off/on" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  const bool runtime_parameter_value = runtime_parameter_declared
      ? pin->GetBoolean("z4c", "lean_runtime") : false;
  if (runtime_parameter_declared && runtime_environment_declared &&
      runtime_parameter_value != runtime_environment_value) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": <z4c>/lean_runtime and ATHENA_Z4C_LEAN_RUNTIME disagree"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.lean_runtime = runtime_parameter_declared
      ? runtime_parameter_value : runtime_environment_value;
  const std::string admissibility_checks = runtime_parameter_declared
      ? pin->GetOrAddString("z4c", "admissibility_checks",
                            opt.lean_runtime ? "consume_and_accept"
                                             : "exhaustive")
      : (opt.lean_runtime ? "consume_and_accept" : "exhaustive");
  if (admissibility_checks == "exhaustive") {
    opt.admissibility_checks = Z4cAdmissibilityChecks::exhaustive;
  } else if (admissibility_checks == "consume_and_accept") {
    opt.admissibility_checks = Z4cAdmissibilityChecks::consume_and_accept;
  } else {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": unknown <z4c>/admissibility_checks="
              << admissibility_checks
              << "; expected exhaustive or consume_and_accept" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.vertex_axis_regularity_audit = runtime_parameter_declared
      ? pin->GetOrAddBoolean("z4c", "vertex_axis_regularity_audit",
                             !opt.lean_runtime)
      : !opt.lean_runtime;
  opt.vc_single_rank_device_sync = runtime_parameter_declared
      ? pin->GetOrAddBoolean("z4c", "vc_single_rank_device_sync",
                             opt.lean_runtime)
      : opt.lean_runtime;
  opt.vc_sync_postcondition = runtime_parameter_declared
      ? pin->GetOrAddBoolean("z4c", "vc_sync_postcondition",
                             !opt.lean_runtime)
      : !opt.lean_runtime;
  opt.timestep_structural_shortcuts = runtime_parameter_declared
      ? pin->GetOrAddBoolean("z4c", "timestep_structural_shortcuts",
                             opt.lean_runtime)
      : opt.lean_runtime;
  opt.timestep_contract_diagnostic = runtime_parameter_declared
      ? pin->GetOrAddBoolean("z4c", "timestep_contract_diagnostic",
                             !opt.lean_runtime)
      : !opt.lean_runtime;
  // Do not materialize a VC-only default in legacy CC input/output bytes.
  opt.vertex_axis_correction_tolerance =
      layout.centering == Z4cGridCentering::vertex
          ? pin->GetOrAddReal("z4c", "vertex_axis_correction_tolerance", 1.0e-8)
          : 1.0e-8;
  if (!(opt.vertex_axis_correction_tolerance > 0.0) ||
      !std::isfinite(opt.vertex_axis_correction_tolerance)) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": <z4c>/vertex_axis_correction_tolerance must be finite and positive"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string amr_transfer =
      pin->GetOrAddString("z4c", "amr_transfer", "high_order");
  if (amr_transfer == "high_order") {
    opt.amr_transfer = Z4cAMRTransfer::high_order;
  } else if (amr_transfer == "limited_o2") {
    opt.amr_transfer = Z4cAMRTransfer::limited_o2;
  } else {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": unknown <z4c>/amr_transfer=" << amr_transfer
              << "; expected high_order or limited_o2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.eps_floor = pin->GetOrAddReal("z4c", "eps_floor", 1e-12);
  opt.timestep_source_safety =
      pin->GetOrAddReal("z4c", "timestep_source_safety", 0.8);
  if (!(opt.timestep_source_safety > 0.0) || opt.timestep_source_safety > 1.0 ||
      !std::isfinite(opt.timestep_source_safety)) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": <z4c>/timestep_source_safety must be finite and in (0,1]"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.damp_kappa1 = pin->GetOrAddReal("z4c", "damp_kappa1", 0.0);
  opt.damp_kappa2 = pin->GetOrAddReal("z4c", "damp_kappa2", 0.0);
  opt.damp_kappa1_max_K =
      pin->GetOrAddBoolean("z4c", "damp_kappa1_max_K", false);
  opt.history_kretschmann =
      pin->GetOrAddBoolean("z4c", "history_kretschmann", false);
  opt.rhs_stage_diagnostics =
      pin->GetOrAddBoolean("z4c", "rhs_stage_diagnostics", false);
  opt.rhs_stage_diagnostics_start_time =
      pin->GetOrAddReal("z4c", "rhs_stage_diagnostics_start_time", 0.0);
  opt.rhs_stage_diagnostics_rho_max =
      pin->GetOrAddReal("z4c", "rhs_stage_diagnostics_rho_max", 0.5);
  opt.rhs_stage_diagnostics_abs_z_max =
      pin->GetOrAddReal("z4c", "rhs_stage_diagnostics_abs_z_max", 0.5);
  if (opt.rhs_stage_diagnostics_rho_max <= 0.0 ||
      opt.rhs_stage_diagnostics_abs_z_max <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "Z4c RHS stage diagnostic extents must be positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  try {
    AMRJumpDiagnosticContext jump_context;
    jump_context.cartoon =
        pmy_pack->z4c_symmetry.mode == Z4cSymmetryMode::cartoon_so2;
    jump_context.adaptive = pmy_pack->pmesh->adaptive;
    jump_context.multilevel = pmy_pack->pmesh->multilevel;
    // User-facing AMR levels and the history maximum-level column are measured
    // relative to the root grid, whereas LogicalLocation::level includes the
    // root-grid Morton depth.
    jump_context.root_level = 0;
    jump_context.maximum_level =
        pmy_pack->pmesh->max_level - pmy_pack->pmesh->root_level;
    jump_context.nranks = global_variable::nranks;
    opt.amr_jump_diagnostic =
        ReadAMRJumpDiagnosticConfig(pin, jump_context);
  } catch (const std::invalid_argument &error) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": invalid Z4c AMR jump diagnostic configuration: "
              << error.what() << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.chi_parent_provenance = ReadChiParentProvenanceConfig(pin);
  if (opt.chi_parent_provenance.enabled) {
    Kokkos::realloc(chi_provenance_terms, nmb, n_chi_provenance_terms,
                    layout.n3, layout.n2, layout.n1);
    Kokkos::deep_copy(chi_provenance_terms, 0.0);
  }
  // Gauge conditions (default to moving puncture gauge)
  opt.lapse_harmonicf = pin->GetOrAddReal("z4c", "lapse_harmonicf", 1.0);
  opt.lapse_harmonic = pin->GetOrAddReal("z4c", "lapse_harmonic", 0.0);
  opt.lapse_oplog = pin->GetOrAddReal("z4c", "lapse_oplog", 2.0);
  opt.lapse_advect = pin->GetOrAddReal("z4c", "lapse_advect", 1.0);
  // Keep the legacy input/output bytes unchanged unless this prospective gauge is
  // explicitly requested.
  opt.lapse_shock_avoiding =
      pin->DoesParameterExist("z4c", "lapse_shock_avoiding") &&
      pin->GetBoolean("z4c", "lapse_shock_avoiding");
  opt.lapse_shock_avoiding_kappa =
      pin->DoesParameterExist("z4c", "lapse_shock_avoiding_kappa")
          ? pin->GetReal("z4c", "lapse_shock_avoiding_kappa")
          : 1.0;
  opt.slow_start_lapse = pin->GetOrAddBoolean("z4c", "slow_start_lapse", false);
  opt.ssl_damping_amp = pin->GetOrAddReal("z4c", "ssl_damping_amp", 0.6);
  opt.ssl_damping_time = pin->GetOrAddReal("z4c", "ssl_damping_time", 20.0);
  opt.ssl_damping_index = pin->GetOrAddInteger("z4c", "ssl_damping_index", 1);
  opt.sss_damping_amp = pin->GetOrAddReal("z4c", "sss_damping_amp", 0.);
  opt.sss_damping_time = pin->GetOrAddReal("z4c", "sss_damping_time", 10.0);
  opt.telegraph_lapse = pin->GetOrAddBoolean("z4c", "telegraph_lapse", false);
  opt.telegraph_max_K = pin->GetOrAddBoolean("z4c", "telegraph_max_K", false);
  const std::string telegraph_damping_prescription =
      pin->DoesParameterExist("z4c", "telegraph_damping_prescription")
          ? pin->GetString("z4c", "telegraph_damping_prescription")
          : (opt.telegraph_max_K ? "max_domain_abs_K" : "fixed");
  if (telegraph_damping_prescription == "fixed") {
    opt.telegraph_damping_prescription = TelegraphDampingPrescription::fixed;
  } else if (telegraph_damping_prescription == "max_domain_abs_K") {
    opt.telegraph_damping_prescription =
        TelegraphDampingPrescription::max_domain_abs_K;
  } else if (telegraph_damping_prescription == "local_abs_K") {
    opt.telegraph_damping_prescription = TelegraphDampingPrescription::local_abs_K;
  } else if (telegraph_damping_prescription ==
             "local_extrinsic_curvature_norm") {
    opt.telegraph_damping_prescription =
        TelegraphDampingPrescription::local_extrinsic_curvature_norm;
  } else if (telegraph_damping_prescription == "local_chi_gradient_norm") {
    opt.telegraph_damping_prescription =
        TelegraphDampingPrescription::local_chi_gradient_norm;
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "unknown <z4c>/telegraph_damping_prescription='"
              << telegraph_damping_prescription << "'; expected fixed, "
              << "max_domain_abs_K, local_abs_K, "
              << "local_extrinsic_curvature_norm, or local_chi_gradient_norm"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.telegraph_tau = pin->GetOrAddReal("z4c", "telegraph_tau", 0.1);
  opt.telegraph_kappa = pin->GetOrAddReal("z4c", "telegraph_kappa", 0.1);

  opt.shift_ggamma = pin->GetOrAddReal("z4c", "shift_Gamma", 1.0);
  opt.shift_advect = pin->GetOrAddReal("z4c", "shift_advect", 1.0);
  opt.shift_alpha2ggamma = pin->GetOrAddReal("z4c", "shift_alpha2Gamma", 0.0);
  opt.shift_hh = pin->GetOrAddReal("z4c", "shift_H", 0.0);
  opt.shift_eta = pin->GetOrAddReal("z4c", "shift_eta", 2.0);
  opt.shift_eta_max_K = pin->GetOrAddBoolean("z4c", "shift_eta_max_K", false);
  const std::string shift_mode =
      pin->GetOrAddString("z4c", "shift_mode", "gamma_driver");
  if (shift_mode == "gamma_driver") {
    opt.shift_mode = Z4cShiftMode::gamma_driver;
  } else if (shift_mode == "prescribed_zero") {
    opt.shift_mode = Z4cShiftMode::prescribed_zero;
  } else {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": unknown <z4c>/shift_mode=" << shift_mode
              << "; expected gamma_driver or prescribed_zero" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::string shift_advection_order =
      pin->GetOrAddString("z4c", "shift_advection_order", "spatial");
  if (shift_advection_order == "spatial") {
    opt.shift_advection_order = Z4cShiftAdvectionOrder::spatial;
  } else if (shift_advection_order == "2") {
    opt.shift_advection_order = Z4cShiftAdvectionOrder::o2;
  } else {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": unknown <z4c>/shift_advection_order="
              << shift_advection_order << "; expected spatial or 2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.shift_invariant_diagnostic = pin->GetOrAddBoolean(
      "z4c", "shift_invariant_diagnostic", false);
  if (opt.shift_mode == Z4cShiftMode::prescribed_zero &&
      opt.shift_advection_order != Z4cShiftAdvectionOrder::spatial) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": prescribed_zero has no shift transport; "
                 "shift_advection_order must remain spatial" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (opt.telegraph_tau <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<z4c>/telegraph_tau must be positive, but is "
              << opt.telegraph_tau << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.telegraph_kappa < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<z4c>/telegraph_kappa must be nonnegative, but is "
              << opt.telegraph_kappa << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.lapse_shock_avoiding) {
    if (opt.lapse_shock_avoiding_kappa <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "<z4c>/lapse_shock_avoiding_kappa must be positive, but is "
                << opt.lapse_shock_avoiding_kappa << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (opt.telegraph_lapse || opt.slow_start_lapse || opt.lapse_oplog != 0.0 ||
        opt.lapse_harmonic != 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl
                << "shock-avoiding lapse requires telegraph_lapse=false, "
                << "slow_start_lapse=false, lapse_oplog=0, and lapse_harmonic=0"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  opt.use_z4c = pin->GetOrAddBoolean("z4c", "use_z4c", true);

  opt.user_Sbc = pin->GetOrAddBoolean("z4c", "user_Sbc", false);

  const std::string boundary_rhs =
      pin->GetOrAddString("z4c", "boundary_rhs", "sommerfeld");
  if (boundary_rhs == "sommerfeld") {
    opt.boundary_rhs_mode = Z4cBoundaryRHSMode::sommerfeld;
  } else if (boundary_rhs == "full_constraint_bjorhus") {
    opt.boundary_rhs_mode = Z4cBoundaryRHSMode::full_constraint_bjorhus;
    if (!opt.use_z4c) {
      std::cerr << "### FATAL ERROR in " << __FILE__
                << ": <z4c>/boundary_rhs=full_constraint_bjorhus requires "
                   "<z4c>/use_z4c=true"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": unknown <z4c>/boundary_rhs=" << boundary_rhs
              << "; expected sommerfeld or full_constraint_bjorhus"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  opt.excise_chi = pin->GetOrAddReal("z4c", "excise_chi", 0.0625);

  opt.extrap_order = fmax(2,fmin(indcs.ng,fmin(4,
      pin->GetOrAddInteger("z4c", "extrap_order", 2))));

  int const default_spatial_order = 2 * (indcs.ng - 1);
  int const requested_spatial_order = pin->GetOrAddInteger("z4c", "spatial_order",
                                                           default_spatial_order);
  opt.spatial_order = EffectiveZ4cSpatialOrder(requested_spatial_order, indcs.ng);
  if (opt.spatial_order != 2 && opt.spatial_order != 4 && opt.spatial_order != 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<z4c>/spatial_order must be 2, 4, or 6, but is "
              << opt.spatial_order << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.fd_stencil = opt.spatial_order/2 + 1;
  if (indcs.ng < opt.fd_stencil) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<z4c>/spatial_order=" << opt.spatial_order
              << " requires at least " << opt.fd_stencil
              << " ghost cells, but <mesh>/nghost=" << indcs.ng << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.vertex_prolongation_order = 0;
  if (layout.centering == Z4cGridCentering::vertex) {
    // Materialize this option only for native VC.  Cell-centered parameter and
    // restart bytes therefore remain unchanged.
    pin->GetOrAddString("z4c", "vertex_prolongation_order", "auto");
    opt.vertex_prolongation_order =
        pmy_pack->z4c_symmetry.vertex_prolongation_order;
    if (!vertex_amr::IsSupportedTransferOrder(
            opt.vertex_prolongation_order)) {
      std::cerr << "### FATAL ERROR: invalid validated native-VC prolongation "
                   "order="
                << opt.vertex_prolongation_order << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (layout.centering == Z4cGridCentering::vertex &&
      pmy_pack->pmesh->multilevel &&
      (!vertex_amr::SupportsSingleHopCoarseHalo(layout.cnx1,
                                                layout.coarse_ng) ||
       (layout.nx2 > 1 && !vertex_amr::SupportsSingleHopCoarseHalo(
                              layout.cnx2, layout.coarse_ng)) ||
       (layout.nx3 > 1 && !vertex_amr::SupportsSingleHopCoarseHalo(
                              layout.cnx3, layout.coarse_ng)))) {
    std::cerr << "### FATAL ERROR: native VC centered interpolation requires each "
                 "coarse MeshBlock interval count to be at least coarse_nghost="
              << layout.coarse_ng << "; increase the MeshBlock size" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.roll_kappa = pin->GetOrAddBoolean("z4c", "roll_kappa", false);
  opt.kappa_roll_start_time = pin->GetOrAddReal("z4c", "kappa_roll_start_time", 0.0);
  opt.roll_window = pin->GetOrAddReal("z4c", "roll_window", 20.0);
  opt.target_kappa1 = pin->GetOrAddReal("z4c", "target_kappa1", 0.0);

  diss = opt.diss*pow(2., -2.*opt.fd_stencil)*(opt.fd_stencil % 2 == 0 ? -1. : 1.);
  }

  ValidateNativeStorageExtents();
  if (layout.centering == Z4cGridCentering::vertex) {
    vertex_topology_plan = std::make_unique<Z4cVertexTopologyPlan>();
    vertex_topology_plan->ConfigureRuntime(
        opt.vc_single_rank_device_sync, opt.vc_sync_postcondition, nz4c);
    RebuildVertexTopologyPlan();
  }
  Kokkos::Profiling::popRegion();

  // allocate boundary buffers for conserved (cell-centered) variables
  Kokkos::Profiling::pushRegion("Buffers");
  if (layout.centering == Z4cGridCentering::vertex) {
    const VertexBoundaryLayout boundary_layout{
        layout.ng, layout.coarse_ng, layout.is, layout.ie, layout.js, layout.je,
        layout.ks, layout.ke, 0, layout.n1 - 1, 0, layout.n2 - 1,
        0, layout.n3 - 1, layout.cis, layout.cie, layout.cjs, layout.cje,
        layout.cks, layout.cke, 0, layout.cn1 - 1, 0, layout.cn2 - 1,
        0, layout.cn3 - 1, layout.nx2 <= 1, layout.nx3 <= 1};
    pbval_u_vc = new MeshBoundaryValuesVC(ppack, pin, boundary_layout);
    pbval_u_vc->InitializeBuffers(nz4c);
    pbval_weyl_vc = new MeshBoundaryValuesVC(ppack, pin, boundary_layout);
    pbval_weyl_vc->InitializeBuffers(2);
  } else {
    pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
    pbval_u->InitializeBuffers((nz4c));
    pbval_weyl = new MeshBoundaryValuesCC(ppack, pin, true);
    pbval_weyl->InitializeBuffers((2));
  }
  Kokkos::Profiling::popRegion();

  // wave extraction spheres
  // TODO(@hzhu): Read radii from input file
  auto &grids = spherical_grids;
  // set nrad_wave_extraction = 0 to turn off wave extraction
  nrad = pin->GetOrAddReal("z4c", "nrad_wave_extraction", 0);
  int nlev = pin->GetOrAddReal("z4c", "extraction_nlev", 10);
  for (int i=0; i<nrad; i++) {
    Real rad = pin->GetOrAddReal("z4c", "extraction_radius_"+std::to_string(i), 10);
    grids.push_back(std::make_unique<SphericalGrid>(ppack, nlev, rad));
  }
  // TODO(@dur566): Why is the size of psi_out hardcoded?
  psi_out = new Real[nrad*77*2];
  if (nrad > 0) {
    mkdir("waveforms",0775);
  }
  waveform_dt = pin->GetOrAddReal("z4c", "waveform_dt", 1);
  last_output_time = 0;
  // CCE
  cce_dump_dt = pin->GetOrAddReal("cce", "cce_dt", 1);
  int ncce = pin->GetOrAddInteger("cce", "num_radii", 0);
  if (ncce > 0) {
    mkdir("cce",0775);
  }
  cce_dump_last_output_time = -100;

  // Construct the compact object trackers
  int n = 0;
  while (true) {
    if (pin->DoesParameterExist("z4c", "co_" + std::to_string(n) + "_type")) {
      ptracker.push_back(std::make_unique<CompactObjectTracker>(pmy_pack->pmesh, pin, n));
      n++;
    } else {
      break;
    }
  }
  // Construct the apparent horizon finders
  n = 0;
  while (n < pin->GetOrAddInteger("fastflow", "num_horizons", 0)) {
    pfastflow.push_back(std::make_unique<FastFlow>(pmy_pack, pin, n));
    n++;
  }
  // Construct the Cartesian data grid for dumping horizon data
  n = 0;
  while (true) {
    if (pin->GetOrAddBoolean("z4c", "dump_horizon_" + std::to_string(n),false)) {
      // phorizon_dump.emplace_back(pmy_pack, pin, n,false);
      phorizon_dump.push_back(std::make_unique<HorizonDump>(pmy_pack, pin, n, 0));
      std::string foldername = "horizon_"+std::to_string(n);
      mkdir(foldername.c_str(),0775);
      n++;
    } else {
      break;
    }
  }
  if (opt.amr_jump_diagnostic.enabled) {
    if (layout.centering == Z4cGridCentering::vertex) {
      std::cerr << "### FATAL ERROR: the CC AMR-jump diagnostic is not valid for "
                   "native VC storage" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    amr_jump_diagnostic = std::make_unique<AMRJumpDiagnosticRuntime>(
        pmy_pack, opt.amr_jump_diagnostic);
  }
  if (opt.chi_parent_provenance.enabled) {
    if (layout.centering == Z4cGridCentering::vertex) {
      std::cerr << "### FATAL ERROR: the CC chi-parent provenance diagnostic is not "
                   "valid for native VC storage" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    chi_parent_provenance = std::make_unique<ChiParentProvenanceRuntime>(
        pmy_pack, opt.chi_parent_provenance);
  }
}

void Z4c::RebuildVertexTopologyPlan() {
  if (layout.centering != Z4cGridCentering::vertex) return;
  if (vertex_topology_plan == nullptr) {
    std::cerr << "### FATAL ERROR: missing native VC topology plan" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  vertex_topology_plan->Rebuild(pmy_pack, layout);
}

//----------------------------------------------------------------------------------------
//! \fn void Z4c::CheckStateAdmissibility
//! \brief fail closed before an invalid conformal state is consumed or projected.
//
// The scan is intentionally separate from projection: invalid inputs are never
// normalized in place.  The deterministic key is (global GID, local active
// cell ordinal), so the selected failure does not depend on thread or rank
// scheduling.
void Z4c::CheckStateAdmissibility(Driver *driver, const int stage,
                                  const Z4cStateCheckpoint checkpoint,
                                  const bool include_ghosts) {
  if (opt.admissibility_checks ==
      Z4cAdmissibilityChecks::consume_and_accept) {
    const bool required = checkpoint == Z4cStateCheckpoint::pre_rhs ||
                          checkpoint == Z4cStateCheckpoint::pre_algconstr ||
                          checkpoint ==
                              Z4cStateCheckpoint::final_accepted_state;
    if (!required) return;
  }
  const int is = include_ghosts ? 0 : layout.is;
  const int ie = include_ghosts ? layout.n1 - 1 : layout.ie;
  const int js = include_ghosts ? 0 : layout.js;
  const int je = include_ghosts ? layout.n2 - 1 : layout.je;
  const int ks = include_ghosts ? 0 : layout.ks;
  const int ke = include_ghosts ? layout.n3 - 1 : layout.ke;
  const int nmb = pmy_pack->nmb_thispack;
  const int nx1 = ie - is + 1;
  const int nx2 = je - js + 1;
  const auto state = u0;
  const auto gids = pmy_pack->pmb->mb_gid.d_view;
  // Device kernels must not reach through the host-resident Z4c object.  Copy
  // the gauge-dependent policy to a scalar before entering the lambda.
  const bool require_positive_lapse = !opt.lapse_shock_avoiding;
  constexpr unsigned long long kNoFailure =
      std::numeric_limits<unsigned long long>::max();
  Kokkos::View<unsigned long long *> first_key("z4c first inadmissible key", 1);
  Kokkos::deep_copy(first_key, kNoFailure);
  par_for("z4c state admissibility", DevExeSpace(), 0, nmb - 1,
          ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real values[nz4c];
        for (int variable = 0; variable < nz4c; ++variable) {
          values[variable] = state(m, variable, k, j, i);
        }
        if (EvaluateZ4cState(values, nz4c, require_positive_lapse).reason ==
            Z4cStateFailureReason::valid) return;
        const unsigned long long ordinal =
            (static_cast<unsigned long long>(k - ks) * nx2 + (j - js)) * nx1 +
            static_cast<unsigned long long>(i - is);
        const unsigned long long key =
            (static_cast<unsigned long long>(gids(m)) << 32) | ordinal;
        Kokkos::atomic_min(&first_key(0), key);
      });
  Kokkos::fence();
  const auto host_key = Kokkos::create_mirror_view_and_copy(HostMemSpace(), first_key);
  unsigned long long selected_key = host_key(0);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &selected_key, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  if (selected_key == kNoFailure) return;

  pmy_pack->pmb->mb_gid.sync_host();
  pmy_pack->pmb->mb_lev.sync_host();
  pmy_pack->pmb->mb_size.sync_host();
  const int selected_gid = static_cast<int>(selected_key >> 32);
  const unsigned long long ordinal = selected_key & 0xffffffffULL;
  int m = -1;
  for (int candidate = 0; candidate < nmb; ++candidate) {
    if (pmy_pack->pmb->mb_gid.h_view(candidate) == selected_gid) {
      m = candidate;
      break;
    }
  }
  if (m >= 0) {
    const int k = ks + static_cast<int>(ordinal / (nx1 * nx2));
    const int remainder = static_cast<int>(ordinal % (nx1 * nx2));
    const int j = js + remainder / nx1;
    const int i = is + remainder % nx1;
    // A component slice of the pack-major state is strided/noncontiguous.  In
    // particular, SYCL cannot deep-copy that subview directly to HostMemSpace.
    // Pack the selected point explicitly so every backend sees one contiguous
    // device allocation and one supported device-to-host copy.
    Kokkos::View<Real *> packed_values("z4c selected inadmissible state", nz4c);
    par_for("pack selected inadmissible z4c state", DevExeSpace(), 0, nz4c - 1,
        KOKKOS_LAMBDA(const int variable) {
          packed_values(variable) = state(m, variable, k, j, i);
        });
    Kokkos::fence();
    const auto values =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), packed_values);
    const auto admissibility =
        EvaluateZ4cState(values.data(), nz4c, require_positive_lapse);
    const auto &size = pmy_pack->pmb->mb_size.h_view(m);
    const Real offset = layout.centering == Z4cGridCentering::vertex ? 0.0 : 0.5;
    const Real rho = size.x1min + (static_cast<Real>(i - layout.is) + offset) * size.dx1;
    const Real z = size.x2min + (static_cast<Real>(j - layout.js) + offset) * size.dx2;
    const Real edge_distance = std::min(
        std::min(static_cast<Real>(i - layout.is), static_cast<Real>(layout.ie - i)) * size.dx1,
        std::min(static_cast<Real>(j - layout.js), static_cast<Real>(layout.je - j)) * size.dx2);
    const auto &location = pmy_pack->pmesh->lloc_eachmb[selected_gid];
    std::ofstream output("z4c_state_failure.json", std::ios::trunc);
    output << std::setprecision(17);
    const auto write_real = [&output](const Real value) {
      if (std::isfinite(value)) {
        output << value;
      } else if (std::isnan(value)) {
        output << "\"nan\"";
      } else if (value > 0.0) {
        output << "\"+inf\"";
      } else {
        output << "\"-inf\"";
      }
    };
    output
           << "{\"schema\":\"z4c_state_admissibility_v1\","
           << "\"time\":" << pmy_pack->pmesh->time << ",\"cycle\":"
           << pmy_pack->pmesh->ncycle << ",\"rk_stage\":" << stage
           << ",\"checkpoint\":\"" << Z4cStateCheckpointName(checkpoint) << "\","
           << "\"global_gid\":" << selected_gid << ",\"level\":"
           << pmy_pack->pmb->mb_lev.h_view(m) << ",\"relative_level\":"
           << location.level - pmy_pack->pmesh->root_level
           << ",\"logical_location\":[" << location.level << ',' << location.lx1
           << ',' << location.lx2 << ',' << location.lx3 << "],\"local_indices\":["
           << i << ',' << j << ',' << k << "],\"rho\":" << rho << ",\"z\":" << z
           << ",\"include_ghosts\":" << (include_ghosts ? "true" : "false")
           << ",\"axis_distance\":" << rho << ",\"block_edge_distance\":"
           << edge_distance << ",\"coarse_fine_interface_distance\":null,"
           << "\"reason\":\"" << Z4cStateFailureReasonName(admissibility.reason)
           << "\",\"first_nonfinite_component\":"
           << admissibility.first_nonfinite_component << ",\"chi\":";
    write_real(values[I_Z4C_CHI]);
    output << ",\"alpha\":";
    write_real(values[I_Z4C_ALPHA]);
    output << ",\"det_gtilde\":";
    write_real(admissibility.metric.determinant);
    output << ",\"spd_pivots\":[";
    write_real(admissibility.metric.pivot0);
    output << ',';
    write_real(admissibility.metric.pivot1);
    output << ',';
    write_real(admissibility.metric.pivot2);
    output << "],\"gtilde\":[";
    for (int variable = I_Z4C_GXX; variable <= I_Z4C_GZZ; ++variable) {
      if (variable != I_Z4C_GXX) output << ',';
      write_real(values(variable));
    }
    output << "],\"Khat\":";
    write_real(values[I_Z4C_KHAT]);
    output << ",\"Theta\":";
    write_real(values[I_Z4C_THETA]);
    output << ",\"beta\":[";
    for (int variable = I_Z4C_BETAX; variable <= I_Z4C_BETAZ; ++variable) {
      if (variable != I_Z4C_BETAX) output << ',';
      write_real(values(variable));
    }
    output << "],\"B\":[";
    for (int variable = I_Z4C_BX; variable <= I_Z4C_BZ; ++variable) {
      if (variable != I_Z4C_BX) output << ',';
      write_real(values(variable));
    }
    output << "],\"state25\":[";
    for (int variable = 0; variable < nz4c; ++variable) {
      if (variable != 0) output << ',';
      write_real(values(variable));
    }
    output << "]}\n";
    output.flush();
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#else
  std::exit(EXIT_FAILURE);
#endif
}

#if defined(ATHENA_Z4C_KERNEL_TESTS)
//----------------------------------------------------------------------------------------
//! \fn void Z4c::InjectStateAdmissibilityExtractionTestFailure
//! \brief Test-only hook that exercises the complete production failure extractor.

void Z4c::InjectStateAdmissibilityExtractionTestFailure(Driver *driver) {
  if (pmy_pack->nmb_thispack <= 0) return;
  const auto bounds = layout;
  const auto state = u0;
  par_for("inject selected inadmissible z4c state", DevExeSpace(), 0, 0,
      KOKKOS_LAMBDA(const int) {
        state(0, I_Z4C_CHI, bounds.ks, bounds.js, bounds.is) = -1.0;
      });
  Kokkos::fence();
  CheckStateAdmissibility(driver, 0, Z4cStateCheckpoint::pre_rhs);
  std::cerr << "### FATAL ERROR: state-admissibility extraction test did not abort"
            << std::endl;
  std::exit(EXIT_FAILURE);
}
#endif

//----------------------------------------------------------------------------------------
//! \fn void Z4c::AlgConstr(AthenaArray<Real> & u)
//! \brief algebraic constraints projection
//
// This function operates on evolved active cells of the MeshBlock.
void Z4c::AlgConstr(MeshBlockPack *pmbp, Driver *driver, const int stage) {
  // Algebraic constraints are defined for evolved active cells.  Boundary
  // storage is owned by the communication/physical-BC passes and can be
  // deliberately unpopulated between those passes; projecting it previously
  // hid that distinction through the detg->1 fallback.
  CheckStateAdmissibility(driver, stage, Z4cStateCheckpoint::pre_algconstr);
  // capture variables for the kernel
  const auto bounds = pmbp->pz4c->layout;

  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  par_for("Alg constr loop",DevExeSpace(),
  0,nmb-1,bounds.ks,bounds.ke,bounds.js,bounds.je,bounds.is,bounds.ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real metric[6] = {z4c.g_dd(m,0,0,k,j,i), z4c.g_dd(m,0,1,k,j,i),
                      z4c.g_dd(m,0,2,k,j,i), z4c.g_dd(m,1,1,k,j,i),
                      z4c.g_dd(m,1,2,k,j,i), z4c.g_dd(m,2,2,k,j,i)};
    Real atracefree[6] = {z4c.vA_dd(m,0,0,k,j,i), z4c.vA_dd(m,0,1,k,j,i),
                          z4c.vA_dd(m,0,2,k,j,i), z4c.vA_dd(m,1,1,k,j,i),
                          z4c.vA_dd(m,1,2,k,j,i), z4c.vA_dd(m,2,2,k,j,i)};
    if (!ProjectAdmissibleConformalState(metric, atracefree)) {
      Kokkos::abort("invalid Z4c conformal state reached algebraic projection");
    }
    z4c.g_dd(m,0,0,k,j,i) = metric[0]; z4c.g_dd(m,0,1,k,j,i) = metric[1];
    z4c.g_dd(m,0,2,k,j,i) = metric[2]; z4c.g_dd(m,1,1,k,j,i) = metric[3];
    z4c.g_dd(m,1,2,k,j,i) = metric[4]; z4c.g_dd(m,2,2,k,j,i) = metric[5];
    z4c.vA_dd(m,0,0,k,j,i) = atracefree[0]; z4c.vA_dd(m,0,1,k,j,i) = atracefree[1];
    z4c.vA_dd(m,0,2,k,j,i) = atracefree[2]; z4c.vA_dd(m,1,1,k,j,i) = atracefree[3];
    z4c.vA_dd(m,1,2,k,j,i) = atracefree[4]; z4c.vA_dd(m,2,2,k,j,i) = atracefree[5];
  });
  Kokkos::fence();
  CheckStateAdmissibility(driver, stage, Z4cStateCheckpoint::post_algconstr);
}

//----------------------------------------------------------------------------------------
// destructor
Z4c::~Z4c() {
  delete[] psi_out;
  delete pbval_weyl;
  delete pbval_weyl_vc;
  delete pbval_u;
  delete pbval_u_vc;
  delete pamr;
}

} // namespace z4c
