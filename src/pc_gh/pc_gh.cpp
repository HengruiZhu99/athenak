//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh.cpp
//! \brief allocation and option validation for the PC-GH module

#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "geodesic-grid/spherical_grid.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "pc_gh/pc_gh.hpp"
#include "utils/horizon_dump.hpp"
#include "utils/compact_object_tracker.hpp"

namespace pc_gh {

char const * const PcGh::PcGhNames[PcGh::npcgh] = {
  "pcgh_w",
  "pcgh_gtxx", "pcgh_gtxy", "pcgh_gtxz", "pcgh_gtyy", "pcgh_gtyz", "pcgh_gtzz",
  "pcgh_K",
  "pcgh_Atxx", "pcgh_Atxy", "pcgh_Atxz", "pcgh_Atyy", "pcgh_Atyz", "pcgh_Atzz",
  "pcgh_Zx", "pcgh_Zy", "pcgh_Zz",
  "pcgh_Cperp", "pcgh_rho",
  "pcgh_betax", "pcgh_betay", "pcgh_betaz",
  "pcgh_p1", "pcgh_p2", "pcgh_p3",
  "pcgh_Q1xx", "pcgh_Q1xy", "pcgh_Q1xz", "pcgh_Q1yy", "pcgh_Q1yz", "pcgh_Q1zz",
  "pcgh_Q2xx", "pcgh_Q2xy", "pcgh_Q2xz", "pcgh_Q2yy", "pcgh_Q2yz", "pcgh_Q2zz",
  "pcgh_Q3xx", "pcgh_Q3xy", "pcgh_Q3xz", "pcgh_Q3yy", "pcgh_Q3yz", "pcgh_Q3zz",
  "pcgh_L1", "pcgh_L2", "pcgh_L3",
  "pcgh_B11", "pcgh_B12", "pcgh_B13",
  "pcgh_B21", "pcgh_B22", "pcgh_B23",
  "pcgh_B31", "pcgh_B32", "pcgh_B33",
};

char const * const PcGh::ConstraintNames[PcGh::ncon] = {
  "pcgh_Cperp", "pcgh_Zx", "pcgh_Zy", "pcgh_Zz",
  "pcgh_H", "pcgh_alphaMx", "pcgh_alphaMy", "pcgh_alphaMz",
  "pcgh_red_w", "pcgh_red_Q", "pcgh_red_alpha", "pcgh_red_B",
  "pcgh_curl_p", "pcgh_curl_Q", "pcgh_curl_L", "pcgh_curl_B",
  "pcgh_detg", "pcgh_trA", "pcgh_trQ", "pcgh_projection",
  "pcgh_minor1", "pcgh_minor2", "pcgh_min_eigenvalue",
  "pcgh_physical_valid", "pcgh_p_norm", "pcgh_L_norm",
  "pcgh_rhs_primary", "pcgh_rhs_gradient",
};

PcGh::PcGh(MeshBlockPack *ppack, ParameterInput *pin)
    : u0("u0 pc_gh", 1, 1, 1, 1, 1),
      u1("u1 pc_gh", 1, 1, 1, 1, 1),
      u_rhs("u_rhs pc_gh", 1, 1, 1, 1, 1),
      u_con("u_con pc_gh", 1, 1, 1, 1, 1),
      transfer_reduction_before("PC-GH transfer reduction before", 1, 1, 1, 1, 1),
      transfer_reduction_after("PC-GH transfer reduction after", 1, 1, 1, 1, 1),
      gauge_a0_table("Gauge A0 table", 1, 1),
      gauge_a0_npoints(0),
      gauge_a0_log_r_min(0.0),
      gauge_a0_inv_dlog_r(0.0),
      coarse_u0("coarse u0 pc_gh", 1, 1, 1, 1, 1),
      u_weyl("u_weyl pc_gh", 1, 1, 1, 1, 1),
      coarse_u_weyl("coarse u_weyl pc_gh", 1, 1, 1, 1, 1),
      pbval_u(nullptr),
      pbval_weyl(nullptr),
      psi_out(nullptr),
      waveform_dt(1.0),
      last_waveform_time(0.0),
      nrad(0),
      dtnew(std::numeric_limits<float>::max()),
      pmy_pack(ppack) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  int const ncells1 = indcs.nx1 + 2*indcs.ng;
  int const ncells2 = (indcs.nx2 > 1) ? indcs.nx2 + 2*indcs.ng : 1;
  int const ncells3 = (indcs.nx3 > 1) ? indcs.nx3 + 2*indcs.ng : 1;

  Kokkos::realloc(u0, nmb, npcgh, ncells3, ncells2, ncells1);
  Kokkos::realloc(u1, nmb, npcgh, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_rhs, nmb, npcgh, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_con, nmb, ncon, ncells3, ncells2, ncells1);
  Kokkos::realloc(transfer_reduction_before, nmb, 8, ncells3, ncells2, ncells1);
  Kokkos::realloc(transfer_reduction_after, nmb, 8, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_weyl, nmb, 2, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(u_con, 0.0);
  BindVariables(u0, u);
  BindVariables(u_rhs, rhs);

  if (ppack->pmesh->multilevel) {
    int const nccells1 = indcs.cnx1 + 2*indcs.ng;
    int const nccells2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    int const nccells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(coarse_u0, nmb, npcgh, nccells3, nccells2, nccells1);
    Kokkos::realloc(coarse_u_weyl, nmb, 2, nccells3, nccells2, nccells1);
  }

  int const default_order = 2*(indcs.ng - 1);
  int const requested_order = pin->GetOrAddInteger("pc_gh", "spatial_order",
                                                    default_order);
  opt.spatial_order = (requested_order > 0) ? requested_order : default_order;
  if (opt.spatial_order != 2 && opt.spatial_order != 4 && opt.spatial_order != 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "<pc_gh>/spatial_order must be 2, 4, or 6, but is "
              << opt.spatial_order << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.fd_stencil = opt.spatial_order/2 + 1;
  if (indcs.ng < opt.fd_stencil) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "<pc_gh>/spatial_order=" << opt.spatial_order << " requires at least "
              << opt.fd_stencil << " ghost cells, but <mesh>/nghost=" << indcs.ng
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppack->pmesh->multilevel && opt.fd_stencil == 3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "PC-GH high-order AMR transfer for fd_stencil=3 is not implemented"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.extrap_order = std::max(2, std::min(indcs.ng, std::min(4,
      pin->GetOrAddInteger("pc_gh", "extrap_order", 2))));
  opt.gauge = pin->GetOrAddString("pc_gh", "gauge", "harmonic");
  if (opt.gauge != "harmonic" && opt.gauge != "a0" && opt.gauge != "z4c_mp"
      && opt.gauge != "z4c_mp_hyperbolic") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "PC-GH gauge must be harmonic, a0, z4c_mp, or "
              << "z4c_mp_hyperbolic, but is " << opt.gauge << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.gauge == "z4c_mp" && global_variable::my_rank == 0) {
    std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "The direct PC-GH z4c_mp gauge has defective characteristic "
              << "surfaces. Use z4c_mp_hyperbolic for black-hole evolution."
              << std::endl;
  }
  opt.gauge_a0_table_file = pin->GetOrAddString(
      "pc_gh", "gauge_a0_table", "inputs/pc_gh/gauge_a0_m1.dat");
  opt.gauge_mass = pin->GetOrAddReal("pc_gh", "gauge_mass", 1.0);
  opt.gauge_center[0] = pin->GetOrAddReal("pc_gh", "gauge_center_x", 0.0);
  opt.gauge_center[1] = pin->GetOrAddReal("pc_gh", "gauge_center_y", 0.0);
  opt.gauge_center[2] = pin->GetOrAddReal("pc_gh", "gauge_center_z", 0.0);
  opt.shift_eta = pin->GetOrAddReal("pc_gh", "shift_eta", 2.0);
  if (!(std::isfinite(opt.shift_eta) && opt.shift_eta >= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "<pc_gh>/shift_eta must be finite and nonnegative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.shift_switch_z0 = pin->GetOrAddReal("pc_gh", "shift_switch_z0", 0.1);
  opt.shift_switch_z1 = pin->GetOrAddReal("pc_gh", "shift_switch_z1", 0.5);
  if (opt.gauge == "z4c_mp_hyperbolic"
      && !(std::isfinite(opt.shift_switch_z0) && std::isfinite(opt.shift_switch_z1)
        && opt.shift_switch_z0 > 0.0
        && opt.shift_switch_z0 < opt.shift_switch_z1
        && opt.shift_switch_z1 < 4.0/7.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "<pc_gh>/shift_switch_z0,z1 must obey 0 < z0 < z1 < 4/7"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.gauge == "a0") {
    if (!(opt.gauge_mass > 0.0)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
                << "<pc_gh>/gauge_mass must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    LoadGaugeA0Table();
    ValidateGaugeA0Domain();
  }
  opt.kappa = pin->GetOrAddReal("pc_gh", "kappa", 0.0);
  Real const ko_amplitude = pin->GetOrAddReal("pc_gh", "dissipation", 0.0);
  if (!(std::isfinite(ko_amplitude) && ko_amplitude >= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "<pc_gh>/dissipation must be finite and nonnegative" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Diss<FD_STENCIL> is the unnormalized alternating 2p-th difference.  Convert the
  // user-facing nonnegative KO amplitude to the sign and 2^(-2p) normalization that
  // makes every nonconstant Fourier mode nonpositive for each supported stencil.
  opt.dissipation = ko_amplitude*std::pow(2.0, -2.0*opt.fd_stencil)
      *((opt.fd_stencil % 2 == 0) ? -1.0 : 1.0);
  opt.project_gauge_constraints = pin->GetOrAddBoolean(
      "pc_gh", "project_gauge_constraints", false);
  opt.project_reduction_constraints = pin->GetOrAddBoolean(
      "pc_gh", "project_reduction_constraints", false);

  opt.constraint_excise_chi = pin->GetOrAddReal(
      "pc_gh", "constraint_excise_chi", 0.0625);
  opt.constraint_exterior_horizon = pin->GetOrAddBoolean(
      "pc_gh", "constraint_exterior_horizon", false);
  opt.constraint_horizon_radius = pin->GetOrAddReal(
      "pc_gh", "constraint_horizon_radius", 0.5*opt.gauge_mass);
  opt.constraint_horizon_buffer = pin->GetOrAddReal(
      "pc_gh", "constraint_horizon_buffer", 0.0);
  opt.constraint_dcycle = pin->GetOrAddInteger("pc_gh", "constraint_dcycle", 1);
  opt.physical_output_inner_radius = pin->GetOrAddReal(
      "pc_gh", "physical_output_inner_radius", 0.25*opt.gauge_mass);
  opt.initial_data_division_floor = pin->GetOrAddReal(
      "pc_gh", "initial_data_division_floor", 1.0e-14);
  opt.reconstruct_adm_output = pin->GetOrAddBoolean(
      "pc_gh", "reconstruct_adm_output", false);
  opt.boundedness_output = pin->GetOrAddBoolean(
      "pc_gh", "boundedness_output", true);
  opt.boundedness_dcycle = pin->GetOrAddInteger(
      "pc_gh", "boundedness_dcycle", 1);
  opt.boundedness_file = pin->GetOrAddString(
      "pc_gh", "boundedness_file",
      pin->GetString("job", "basename") + ".pcgh-boundedness.dat");
  if (!(std::isfinite(opt.constraint_excise_chi)
        && opt.constraint_excise_chi >= 0.0
        && std::isfinite(opt.constraint_horizon_radius)
        && opt.constraint_horizon_radius > 0.0
        && std::isfinite(opt.constraint_horizon_buffer)
        && opt.constraint_horizon_buffer >= 0.0
        && opt.constraint_dcycle > 0
        && std::isfinite(opt.physical_output_inner_radius)
        && opt.physical_output_inner_radius >= 0.0
        && std::isfinite(opt.initial_data_division_floor)
        && opt.initial_data_division_floor > 0.0
        && opt.boundedness_dcycle > 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "PC-GH constraint mask thresholds must be finite and nonnegative, "
              << "with a positive constraint_horizon_radius and nonnegative "
              << "physical_output_inner_radius, and a positive finite "
              << "initial_data_division_floor" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(npcgh);
  pbval_weyl = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_weyl->InitializeBuffers(2);

  nrad = pin->GetOrAddInteger("pc_gh", "nrad_wave_extraction", 0);
  int const extraction_nlev = pin->GetOrAddInteger("pc_gh", "extraction_nlev", 10);
  for (int n = 0; n < nrad; ++n) {
    Real const radius = pin->GetOrAddReal(
        "pc_gh", "extraction_radius_" + std::to_string(n), 10.0);
    spherical_grids.push_back(std::make_unique<SphericalGrid>(ppack, extraction_nlev,
                                                               radius));
  }
  psi_out = new Real[std::max(1, nrad*77*2)]{};
  waveform_dt = pin->GetOrAddReal("pc_gh", "waveform_dt", 1.0);
  if (nrad > 0) mkdir("waveforms", 0775);

  int tracker_index = 0;
  while (pin->DoesParameterExist(
      "pc_gh", "co_" + std::to_string(tracker_index) + "_type")) {
    ptracker.push_back(std::make_unique<CompactObjectTracker>(
        pmy_pack->pmesh, pin, tracker_index, "pc_gh"));
    ++tracker_index;
  }

  int horizon_index = 0;
  while (pin->GetOrAddBoolean(
      "pc_gh", "dump_horizon_" + std::to_string(horizon_index), false)) {
    phorizon_dump.push_back(std::make_unique<HorizonDump>(
        pmy_pack, pin, horizon_index, 0, "pc_gh", true));
    ++horizon_index;
  }
}

void PcGh::LoadGaugeA0Table() {
  std::ifstream input(opt.gauge_a0_table_file);
  if (!input.is_open()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Unable to open Gauge A0 table '" << opt.gauge_a0_table_file << "'"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<std::array<Real, na0 + 1>> rows;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    std::istringstream stream(line);
    stream >> std::ws;
    if (stream.eof() || stream.peek() == '#') continue;
    std::array<Real, na0 + 1> row{};
    for (int column = 0; column < na0 + 1; ++column) {
      if (!(stream >> row[column])) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << '\n' << "Malformed Gauge A0 table row " << line_number
                  << " in '" << opt.gauge_a0_table_file << "'" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    Real extra;
    if (stream >> extra) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << '\n' << "Too many columns in Gauge A0 table row " << line_number
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rows.push_back(row);
  }
  if (rows.size() < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Gauge A0 table must contain at least two data rows" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real const spacing = (rows.back()[0] - rows.front()[0])
                       /static_cast<Real>(rows.size() - 1);
  if (!(spacing > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Gauge A0 log-radius nodes must increase" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t point = 0; point < rows.size(); ++point) {
    Real const expected = rows[0][0] + static_cast<Real>(point)*spacing;
    Real const tolerance = 2048.0*std::numeric_limits<Real>::epsilon()
                           *std::max(1.0, std::abs(expected));
    if (std::abs(rows[point][0] - expected) > tolerance) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << '\n' << "Gauge A0 table log-radius nodes are not uniform at row "
                << point << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  gauge_a0_npoints = static_cast<int>(rows.size());
  gauge_a0_log_r_min = rows.front()[0];
  gauge_a0_inv_dlog_r = 1.0/spacing;
  Kokkos::realloc(gauge_a0_table, na0, gauge_a0_npoints);
  auto host_table = Kokkos::create_mirror_view(gauge_a0_table);
  for (int point = 0; point < gauge_a0_npoints; ++point) {
    for (int field = 0; field < na0; ++field) {
      host_table(field, point) = rows[point][field + 1];
    }
  }
  Kokkos::deep_copy(gauge_a0_table, host_table);
}

void PcGh::ValidateGaugeA0Domain() {
  if (!pmy_pack->pmesh->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "Gauge A0 stationary trumpet requires a three-dimensional mesh"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size = pmy_pack->pmb->mb_size;
  size.template sync<HostMemSpace>();
  Real const log_r_max = gauge_a0_log_r_min
      + static_cast<Real>(gauge_a0_npoints - 1)/gauge_a0_inv_dlog_r;
  for (int m = 0; m < pmy_pack->nmb_thispack; ++m) {
    for (int k = indcs.ks - indcs.ng; k <= indcs.ke + indcs.ng; ++k) {
      Real const z = CellCenterX(k - indcs.ks, indcs.nx3,
          size.h_view(m).x3min, size.h_view(m).x3max) - opt.gauge_center[2];
      for (int j = indcs.js - indcs.ng; j <= indcs.je + indcs.ng; ++j) {
        Real const y = CellCenterX(j - indcs.js, indcs.nx2,
            size.h_view(m).x2min, size.h_view(m).x2max) - opt.gauge_center[1];
        for (int i = indcs.is - indcs.ng; i <= indcs.ie + indcs.ng; ++i) {
          Real const x = CellCenterX(i - indcs.is, indcs.nx1,
              size.h_view(m).x1min, size.h_view(m).x1max) - opt.gauge_center[0];
          Real const radius = std::sqrt(x*x + y*y + z*z);
          if (!(radius > 0.0)) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << '\n' << "Gauge A0 may not be evaluated at its exact center"
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }
          Real const log_radius = std::log(radius/opt.gauge_mass);
          if (log_radius < gauge_a0_log_r_min
              || log_radius >= log_r_max) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                      << '\n' << "Gauge A0 cell/ghost radius lies outside the open table "
                      << "domain at (" << x << ',' << y << ',' << z << ")" << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
      }
    }
  }
}

void PcGh::BindVariables(DvceArray5D<Real> state, Variables &vars) {
  vars.w.InitWithShallowSlice(state, I_W);
  vars.gtilde.InitWithShallowSlice(state, I_GTXX, I_GTZZ);
  vars.K.InitWithShallowSlice(state, I_K);
  vars.Atilde.InitWithShallowSlice(state, I_ATXX, I_ATZZ);
  vars.Z.InitWithShallowSlice(state, I_ZX, I_ZZ);
  vars.Cperp.InitWithShallowSlice(state, I_CPERP);
  vars.rho.InitWithShallowSlice(state, I_RHO);
  vars.beta.InitWithShallowSlice(state, I_BETAX, I_BETAZ);
  vars.p.InitWithShallowSlice(state, I_P1, I_P3);
  vars.L.InitWithShallowSlice(state, I_L1, I_L3);
}

void PcGh::ValidateState(const char *stage, bool check_rhs, bool check_constraints) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const ncell = pmy_pack->nmb_thispack*nx1*nx2*nx3;
  int const nstate = ncell*npcgh;
  auto state = u0;
  int first_bad_state = nstate;
  Kokkos::parallel_reduce("PC-GH strict state validation",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nstate),
  KOKKOS_LAMBDA(int flat, int &bad) {
    int const v = flat % npcgh;
    int const cell = flat/npcgh;
    int const i0 = cell % nx1;
    int const j0 = (cell/nx1) % nx2;
    int const k0 = (cell/(nx1*nx2)) % nx3;
    int const m = cell/(nx1*nx2*nx3);
    Real const value = state(m, v, indcs.ks + k0, indcs.js + j0, indcs.is + i0);
    bool const invalid = !std::isfinite(value)
        || ((v == I_W || v == I_RHO) && value < 0.0);
    if (invalid && flat < bad) bad = flat;
  }, Kokkos::Min<int>(first_bad_state));

  int first_bad_metric = ncell;
  Kokkos::parallel_reduce("PC-GH strict conformal-metric validation",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(int cell, int &bad) {
    int const i0 = cell % nx1;
    int const j0 = (cell/nx1) % nx2;
    int const k0 = (cell/(nx1*nx2)) % nx3;
    int const m = cell/(nx1*nx2*nx3);
    int const i = indcs.is + i0;
    int const j = indcs.js + j0;
    int const k = indcs.ks + k0;
    Real const g00 = state(m, I_GTXX, k, j, i);
    Real const g01 = state(m, I_GTXY, k, j, i);
    Real const g02 = state(m, I_GTXZ, k, j, i);
    Real const g11 = state(m, I_GTYY, k, j, i);
    Real const g12 = state(m, I_GTYZ, k, j, i);
    Real const g22 = state(m, I_GTZZ, k, j, i);
    Real const minor01 = g00*g11 - g01*g01;
    Real const minor02 = g00*g22 - g02*g02;
    Real const minor12 = g11*g22 - g12*g12;
    Real const determinant = adm::SpatialDet(g00, g01, g02, g11, g12, g22);
    bool const invalid = !(std::isfinite(determinant) && g00 > 0.0 && g11 > 0.0
        && g22 > 0.0 && minor01 > 0.0 && minor02 > 0.0 && minor12 > 0.0
        && determinant > 0.0);
    if (invalid && cell < bad) bad = cell;
  }, Kokkos::Min<int>(first_bad_metric));

  int first_bad_rhs = nstate;
  if (check_rhs) {
    auto state_rhs = u_rhs;
    Kokkos::parallel_reduce("PC-GH strict RHS validation",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nstate),
    KOKKOS_LAMBDA(int flat, int &bad) {
      int const v = flat % npcgh;
      int const cell = flat/npcgh;
      int const i0 = cell % nx1;
      int const j0 = (cell/nx1) % nx2;
      int const k0 = (cell/(nx1*nx2)) % nx3;
      int const m = cell/(nx1*nx2*nx3);
      Real const value = state_rhs(
          m, v, indcs.ks + k0, indcs.js + j0, indcs.is + i0);
      if (!std::isfinite(value) && flat < bad) bad = flat;
    }, Kokkos::Min<int>(first_bad_rhs));
  }

  int const nconstraints = ncell*ncon;
  int first_bad_constraint = nconstraints;
  if (check_constraints) {
    auto constraints = u_con;
    Kokkos::parallel_reduce("PC-GH strict constraint validation",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, nconstraints),
    KOKKOS_LAMBDA(int flat, int &bad) {
      int const v = flat % ncon;
      int const cell = flat/ncon;
      int const i0 = cell % nx1;
      int const j0 = (cell/nx1) % nx2;
      int const k0 = (cell/(nx1*nx2)) % nx3;
      int const m = cell/(nx1*nx2*nx3);
      Real const value = constraints(
          m, v, indcs.ks + k0, indcs.js + j0, indcs.is + i0);
      bool const invalid = !std::isfinite(value)
          || ((v == I_CON_MINOR1 || v == I_CON_MINOR2 || v == I_CON_MINEIG)
              && value <= 0.0);
      if (invalid && flat < bad) bad = flat;
    }, Kokkos::Min<int>(first_bad_constraint));
  }

  int local_invalid = (first_bad_state < nstate || first_bad_metric < ncell
      || first_bad_rhs < nstate || first_bad_constraint < nconstraints) ? 1 : 0;
  int global_invalid = local_invalid;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_invalid, &global_invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
  if (global_invalid == 0) return;
  if (local_invalid != 0) {
    std::cout << "### FATAL ERROR: PC-GH strict diagnostic failed at t="
              << pmy_pack->pmesh->time << " during " << stage
              << " on rank " << global_variable::my_rank;
    if (first_bad_state < nstate) {
      int const flat = first_bad_state;
      int const cell = flat/npcgh;
      int const i0 = cell % nx1;
      int const j0 = (cell/nx1) % nx2;
      int const k0 = (cell/(nx1*nx2)) % nx3;
      int const m = cell/(nx1*nx2*nx3);
      int const v = flat % npcgh;
      auto host_state = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u0);
      std::cout << ": state " << PcGhNames[v] << '='
                << host_state(m, v, indcs.ks + k0, indcs.js + j0, indcs.is + i0)
                << " at (m,k,j,i)=(" << m << ',' << indcs.ks + k0 << ','
                << indcs.js + j0 << ',' << indcs.is + i0 << ')';
    } else if (first_bad_metric < ncell) {
      int const cell = first_bad_metric;
      int const i0 = cell % nx1;
      int const j0 = (cell/nx1) % nx2;
      int const k0 = (cell/(nx1*nx2)) % nx3;
      int const m = cell/(nx1*nx2*nx3);
      auto host_state = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u0);
      int const i = indcs.is + i0;
      int const j = indcs.js + j0;
      int const k = indcs.ks + k0;
      auto &block_size = pmy_pack->pmb->mb_size;
      Real const x = CellCenterX(i0, nx1, block_size.h_view(m).x1min,
                                 block_size.h_view(m).x1max);
      Real const y = CellCenterX(j0, nx2, block_size.h_view(m).x2min,
                                 block_size.h_view(m).x2max);
      Real const z = CellCenterX(k0, nx3, block_size.h_view(m).x3min,
                                 block_size.h_view(m).x3max);
      Real const determinant = adm::SpatialDet(
          host_state(m, I_GTXX, k, j, i), host_state(m, I_GTXY, k, j, i),
          host_state(m, I_GTXZ, k, j, i), host_state(m, I_GTYY, k, j, i),
          host_state(m, I_GTYZ, k, j, i), host_state(m, I_GTZZ, k, j, i));
      std::cout << ": conformal metric lost positive definiteness, det="
                << determinant << " at (m,k,j,i)=(" << m << ',' << k << ','
                << j << ',' << i << "), (x,y,z)=(" << x << ',' << y << ','
                << z << ')';
    } else if (first_bad_rhs < nstate) {
      int const flat = first_bad_rhs;
      int const cell = flat/npcgh;
      int const i0 = cell % nx1;
      int const j0 = (cell/nx1) % nx2;
      int const k0 = (cell/(nx1*nx2)) % nx3;
      int const m = cell/(nx1*nx2*nx3);
      int const v = flat % npcgh;
      auto host_rhs = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_rhs);
      std::cout << ": RHS " << PcGhNames[v] << '='
                << host_rhs(m, v, indcs.ks + k0, indcs.js + j0, indcs.is + i0)
                << " at (m,k,j,i)=(" << m << ',' << indcs.ks + k0 << ','
                << indcs.js + j0 << ',' << indcs.is + i0 << ')';
    } else {
      int const flat = first_bad_constraint;
      int const cell = flat/ncon;
      int const i0 = cell % nx1;
      int const j0 = (cell/nx1) % nx2;
      int const k0 = (cell/(nx1*nx2)) % nx3;
      int const m = cell/(nx1*nx2*nx3);
      int const v = flat % ncon;
      auto host_constraints = Kokkos::create_mirror_view_and_copy(HostMemSpace(), u_con);
      std::cout << ": diagnostic " << ConstraintNames[v] << '='
                << host_constraints(m, v, indcs.ks + k0, indcs.js + j0,
                                    indcs.is + i0)
                << " at (m,k,j,i)=(" << m << ',' << indcs.ks + k0 << ','
                << indcs.js + j0 << ',' << indcs.is + i0 << ')';
    }
    std::cout << std::endl;
  }
  std::exit(EXIT_FAILURE);
}

PcGh::~PcGh() {
  delete[] psi_out;
  delete pbval_u;
  delete pbval_weyl;
}

}  // namespace pc_gh
