//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pc_gh.cpp
//! \brief allocation and option validation for the PC-GH module

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
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

PcGh::PcGh(MeshBlockPack *ppack, ParameterInput *pin)
    : u0("u0 pc_gh", 1, 1, 1, 1, 1),
      u1("u1 pc_gh", 1, 1, 1, 1, 1),
      u_rhs("u_rhs pc_gh", 1, 1, 1, 1, 1),
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
  if (opt.gauge != "harmonic") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << '\n'
              << "PC-GH gauge must currently be harmonic, but is " << opt.gauge
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  opt.kappa = pin->GetOrAddReal("pc_gh", "kappa", 0.0);
  opt.dissipation = pin->GetOrAddReal("pc_gh", "dissipation", 0.0);

  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(npcgh);
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
