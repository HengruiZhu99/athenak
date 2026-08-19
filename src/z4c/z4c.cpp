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
#include <memory>    // make_unique, unique_ptr
#include <vector>    // vector
#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "bvals/bvals.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/compact_object_tracker.hpp"
#include "z4c/horizon_dump.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "z4c/z4c_symmetry.hpp"
#include "z4c/stored_domain_bounds.hpp"
#include "z4c/state_admissibility.hpp"
#include "coordinates/adm.hpp"
#include "utils/cart_grid.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {

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

/*char const * const Z4c::Matter_names[Z4c::nmat] = {
  "mat_rho",
  "mat_Sx", "mat_Sy", "mat_Sz",
  "mat_Sxx", "mat_Sxy", "mat_Sxz", "mat_Syy", "mat_Syz", "mat_Szz",
};*/

//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Z4c::Z4c(MeshBlockPack *ppack, ParameterInput *pin) :
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
  pamr(new Z4c_AMR(pin)),
  pmy_pack(ppack) {
  // (1) read time-evolution option [already error checked in driver constructor]
  // Then initialize memory and algorithms for reconstruction and Riemann solvers
  std::string evolution_t = pin->GetString("time","evolution");

  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  // int nmb = ppack->nmb_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  {
  const auto bounds = MakeStoredDomainBounds(indcs);
  Kokkos::Profiling::pushRegion("Tensor fields");
  Kokkos::realloc(u_con, nmb, (ncon), bounds.n3, bounds.n2, bounds.n1);
  // Matter storage is currently disabled.
  Kokkos::realloc(u0,    nmb, (nz4c), bounds.n3, bounds.n2, bounds.n1);
  Kokkos::realloc(u1,    nmb, (nz4c), bounds.n3, bounds.n2, bounds.n1);
  Kokkos::realloc(u_rhs, nmb, (nz4c), bounds.n3, bounds.n2, bounds.n1);
  Kokkos::realloc(u_telegraph_mu, nmb, 1, bounds.n3, bounds.n2, bounds.n1);
  Kokkos::deep_copy(u_telegraph_mu, 0.0);
  Kokkos::realloc(u_weyl,    nmb, (2), bounds.n3, bounds.n2, bounds.n1);

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

  opt.chi_psi_power = pin->GetOrAddReal("z4c", "chi_psi_power", -4.0);
  opt.chi_div_floor = pin->GetOrAddReal("z4c", "chi_div_floor", -1000.0);
  opt.chi_min_floor = pin->GetOrAddReal("z4c", "chi_min_floor", 1e-12);
  opt.floor_chi = pin->GetOrAddBoolean("z4c", "floor_chi", false);
  opt.diss = pin->GetOrAddReal("z4c", "diss", 0.0);
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
                    bounds.n3, bounds.n2, bounds.n1);
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
  if (opt.history_kretschmann && opt.fd_stencil != 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "<z4c>/history_kretschmann=true requires "
              << "<z4c>/spatial_order=6 and at least four ghost cells"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  opt.roll_kappa = pin->GetOrAddBoolean("z4c", "roll_kappa", false);
  opt.kappa_roll_start_time = pin->GetOrAddReal("z4c", "kappa_roll_start_time", 0.0);
  opt.roll_window = pin->GetOrAddReal("z4c", "roll_window", 20.0);
  opt.target_kappa1 = pin->GetOrAddReal("z4c", "target_kappa1", 0.0);

  diss = opt.diss*pow(2., -2.*opt.fd_stencil)*(opt.fd_stencil % 2 == 0 ? -1. : 1.);
  }

  // allocate memory for conserved variables on coarse mesh
  if (ppack->pmesh->multilevel) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    const auto coarse_bounds = MakeCoarseStoredDomainBounds(indcs);
    Kokkos::realloc(coarse_u0, nmb, (nz4c), coarse_bounds.n3,
                    coarse_bounds.n2, coarse_bounds.n1);
    Kokkos::realloc(coarse_u_weyl, nmb, (2), coarse_bounds.n3,
                    coarse_bounds.n2, coarse_bounds.n1);
  }
  Kokkos::Profiling::popRegion();

  // allocate boundary buffers for conserved (cell-centered) variables
  Kokkos::Profiling::pushRegion("Buffers");
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers((nz4c));
  pbval_weyl = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_weyl->InitializeBuffers((2));
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
    amr_jump_diagnostic = std::make_unique<AMRJumpDiagnosticRuntime>(
        pmy_pack, opt.amr_jump_diagnostic);
  }
  if (opt.chi_parent_provenance.enabled) {
    chi_parent_provenance = std::make_unique<ChiParentProvenanceRuntime>(
        pmy_pack, opt.chi_parent_provenance);
  }
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
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto stored = MakeStoredDomainBounds(indcs);
  const int is = include_ghosts ? stored.is : indcs.is;
  const int ie = include_ghosts ? stored.ie : indcs.ie;
  const int js = include_ghosts ? stored.js : indcs.js;
  const int je = include_ghosts ? stored.je : indcs.je;
  const int ks = include_ghosts ? stored.ks : indcs.ks;
  const int ke = include_ghosts ? stored.ke : indcs.ke;
  const int nmb = pmy_pack->nmb_thispack;
  const int nx1 = ie - is + 1;
  const int nx2 = je - js + 1;
  const auto state = u0;
  const auto gids = pmy_pack->pmb->mb_gid.d_view;
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
        if (EvaluateZ4cState(values, nz4c).reason ==
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
    const auto device_values = Kokkos::subview(state, m, Kokkos::ALL(), k, j, i);
    const auto values = Kokkos::create_mirror_view_and_copy(HostMemSpace(), device_values);
    const auto admissibility = EvaluateZ4cState(values.data(), nz4c);
    const auto &size = pmy_pack->pmb->mb_size.h_view(m);
    const Real rho = size.x1min + (static_cast<Real>(i - indcs.is) + 0.5) * size.dx1;
    const Real z = size.x2min + (static_cast<Real>(j - indcs.js) + 0.5) * size.dx2;
    const Real edge_distance = std::min(
        std::min(static_cast<Real>(i - indcs.is), static_cast<Real>(indcs.ie - i)) * size.dx1,
        std::min(static_cast<Real>(j - indcs.js), static_cast<Real>(indcs.je - j)) * size.dx2);
    std::ofstream output("z4c_state_failure.json", std::ios::trunc);
    output << std::setprecision(17)
           << "{\"schema\":\"z4c_state_admissibility_v1\","
           << "\"time\":" << pmy_pack->pmesh->time << ",\"cycle\":"
           << pmy_pack->pmesh->ncycle << ",\"rk_stage\":" << stage
           << ",\"checkpoint\":\"" << Z4cStateCheckpointName(checkpoint) << "\","
           << "\"global_gid\":" << selected_gid << ",\"level\":"
           << pmy_pack->pmb->mb_lev.h_view(m) << ",\"local_indices\":["
           << i << ',' << j << ',' << k << "],\"rho\":" << rho << ",\"z\":" << z
           << ",\"include_ghosts\":" << (include_ghosts ? "true" : "false")
           << ",\"axis_distance\":" << rho << ",\"block_edge_distance\":"
           << edge_distance << ",\"coarse_fine_interface_distance\":null,"
           << "\"reason\":\"" << Z4cStateFailureReasonName(admissibility.reason)
           << "\",\"first_nonfinite_component\":"
           << admissibility.first_nonfinite_component << ",\"chi\":" << values[I_Z4C_CHI]
           << ",\"alpha\":" << values[I_Z4C_ALPHA] << ",\"det_gtilde\":"
           << admissibility.metric.determinant << ",\"spd_pivots\":["
           << admissibility.metric.pivot0 << ',' << admissibility.metric.pivot1 << ','
           << admissibility.metric.pivot2 << "],\"gtilde\":[" << values[I_Z4C_GXX]
           << ',' << values[I_Z4C_GXY] << ',' << values[I_Z4C_GXZ] << ','
           << values[I_Z4C_GYY] << ',' << values[I_Z4C_GYZ] << ',' << values[I_Z4C_GZZ]
           << "],\"Khat\":" << values[I_Z4C_KHAT] << ",\"Theta\":"
           << values[I_Z4C_THETA] << ",\"beta\":[" << values[I_Z4C_BETAX] << ','
           << values[I_Z4C_BETAY] << ',' << values[I_Z4C_BETAZ] << "],\"B\":["
           << values[I_Z4C_BX] << ',' << values[I_Z4C_BY] << ',' << values[I_Z4C_BZ]
           << "],\"state25\":[";
    for (int variable = 0; variable < nz4c; ++variable) {
      if (variable != 0) output << ',';
      output << values(variable);
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

//----------------------------------------------------------------------------------------
//! \fn void Z4c::AlgConstr(AthenaArray<Real> & u)
//! \brief algebraic constraints projection
//
// This function operates on all grid points of the MeshBlock
void Z4c::AlgConstr(MeshBlockPack *pmbp, Driver *driver, const int stage) {
  // Algebraic constraints are defined for evolved active cells.  Boundary
  // storage is owned by the communication/physical-BC passes and can be
  // deliberately unpopulated between those passes; projecting it previously
  // hid that distinction through the detg->1 fallback.
  CheckStateAdmissibility(driver, stage, Z4cStateCheckpoint::pre_algconstr);
  // capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;

  int nmb = pmbp->nmb_thispack;

  auto &z4c = pmbp->pz4c->z4c;
  par_for("Alg constr loop",DevExeSpace(),
  0,nmb-1,indcs.ks,indcs.ke,indcs.js,indcs.je,indcs.is,indcs.ie,
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
  delete pbval_u;
  delete pamr;
}

} // namespace z4c
