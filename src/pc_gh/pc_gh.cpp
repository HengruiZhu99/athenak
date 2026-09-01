//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh.cpp
//! \brief allocation and option validation for the PC-GH module

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "parameter_input.hpp"
#include "pc_gh/pc_gh.hpp"

namespace pc_gh {

char const * const PcGh::PcGhNames[PcGh::npcgh] = {
  "pcgh_chi",
  "pcgh_gtxx", "pcgh_gtxy", "pcgh_gtxz", "pcgh_gtyy", "pcgh_gtyz", "pcgh_gtzz",
  "pcgh_K",
  "pcgh_Atxx", "pcgh_Atxy", "pcgh_Atxz", "pcgh_Atyy", "pcgh_Atyz", "pcgh_Atzz",
  "pcgh_Lamx", "pcgh_Lamy", "pcgh_Lamz",
  "pcgh_pi", "pcgh_A",
  "pcgh_betax", "pcgh_betay", "pcgh_betaz",
  "pcgh_X1", "pcgh_X2", "pcgh_X3",
  "pcgh_Q1xx", "pcgh_Q1xy", "pcgh_Q1xz", "pcgh_Q1yy", "pcgh_Q1yz", "pcgh_Q1zz",
  "pcgh_Q2xx", "pcgh_Q2xy", "pcgh_Q2xz", "pcgh_Q2yy", "pcgh_Q2yz", "pcgh_Q2zz",
  "pcgh_Q3xx", "pcgh_Q3xy", "pcgh_Q3xz", "pcgh_Q3yy", "pcgh_Q3yz", "pcgh_Q3zz",
  "pcgh_Y1", "pcgh_Y2", "pcgh_Y3",
  "pcgh_B11", "pcgh_B12", "pcgh_B13",
  "pcgh_B21", "pcgh_B22", "pcgh_B23",
  "pcgh_B31", "pcgh_B32", "pcgh_B33",
};

char const * const PcGh::ConstraintNames[PcGh::ncon] = {
  "pcgh_Cperp", "pcgh_Zx", "pcgh_Zy", "pcgh_Zz",
  "pcgh_H", "pcgh_Mhatx", "pcgh_Mhaty", "pcgh_Mhatz",
  "pcgh_red_X", "pcgh_red_Q", "pcgh_red_Y", "pcgh_red_B",
  "pcgh_curl_X", "pcgh_curl_Q", "pcgh_curl_Y", "pcgh_curl_B",
  "pcgh_detg", "pcgh_trA", "pcgh_trQ", "pcgh_projection",
  "pcgh_rminus", "pcgh_rplus", "pcgh_W", "pcgh_L",
  "pcgh_rhs_primary", "pcgh_rhs_gradient",
};

PcGh::PcGh(MeshBlockPack *ppack, ParameterInput *pin)
    : u0("u0 pc_gh", 1, 1, 1, 1, 1),
      u1("u1 pc_gh", 1, 1, 1, 1, 1),
      u_rhs("u_rhs pc_gh", 1, 1, 1, 1, 1),
      u_con("u_con pc_gh", 1, 1, 1, 1, 1),
      gauge_a0_table("Gauge A0 table", 1, 1),
      gauge_a0_npoints(0),
      gauge_a0_log_r_min(0.0),
      gauge_a0_inv_dlog_r(0.0),
      coarse_u0("coarse u0 pc_gh", 1, 1, 1, 1, 1),
      pbval_u(nullptr),
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
  Kokkos::deep_copy(u_con, 0.0);
  BindVariables(u0, u);
  BindVariables(u_rhs, rhs);

  if (ppack->pmesh->multilevel) {
    int const nccells1 = indcs.cnx1 + 2*indcs.ng;
    int const nccells2 = (indcs.cnx2 > 1) ? indcs.cnx2 + 2*indcs.ng : 1;
    int const nccells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
    Kokkos::realloc(coarse_u0, nmb, npcgh, nccells3, nccells2, nccells1);
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
  opt.gauge = pin->GetOrAddString("pc_gh", "gauge", "harmonic");
  if (opt.gauge != "harmonic" && opt.gauge != "a0") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "PC-GH gauge must be harmonic or a0, but is " << opt.gauge
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.gauge_a0_table_file = pin->GetOrAddString(
      "pc_gh", "gauge_a0_table", "inputs/pc_gh/gauge_a0_m1.dat");
  opt.gauge_mass = pin->GetOrAddReal("pc_gh", "gauge_mass", 1.0);
  opt.gauge_center[0] = pin->GetOrAddReal("pc_gh", "gauge_center_x", 0.0);
  opt.gauge_center[1] = pin->GetOrAddReal("pc_gh", "gauge_center_y", 0.0);
  opt.gauge_center[2] = pin->GetOrAddReal("pc_gh", "gauge_center_z", 0.0);
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
  opt.dissipation = pin->GetOrAddReal("pc_gh", "dissipation", 0.0);

  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(npcgh);
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
  vars.chi.InitWithShallowSlice(state, I_CHI);
  vars.gtilde.InitWithShallowSlice(state, I_GTXX, I_GTZZ);
  vars.K.InitWithShallowSlice(state, I_K);
  vars.Atilde.InitWithShallowSlice(state, I_ATXX, I_ATZZ);
  vars.Lambda.InitWithShallowSlice(state, I_LAMX, I_LAMZ);
  vars.pi.InitWithShallowSlice(state, I_PI);
  vars.A.InitWithShallowSlice(state, I_A);
  vars.beta.InitWithShallowSlice(state, I_BETAX, I_BETAZ);
  vars.X.InitWithShallowSlice(state, I_X1, I_X3);
  vars.Y.InitWithShallowSlice(state, I_Y1, I_Y3);
}

PcGh::~PcGh() {
  delete pbval_u;
}

}  // namespace pc_gh
