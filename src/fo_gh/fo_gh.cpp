//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file fo_gh.cpp
//! \brief Construction and storage binding for regularized first-order GH.

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "bvals/bvals.hpp"
#include "fo_gh/fo_gh.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"

namespace fo_gh {

char const * const FoGh::StateNames[FoGh::nfo_gh] = {
  "fo_gh_gtxx", "fo_gh_gtxy", "fo_gh_gtxz", "fo_gh_gtyy", "fo_gh_gtyz",
  "fo_gh_gtzz", "fo_gh_chi", "fo_gh_alpha", "fo_gh_betax", "fo_gh_betay",
  "fo_gh_betaz", "fo_gh_Atxx", "fo_gh_Atxy", "fo_gh_Atxz", "fo_gh_Atyy",
  "fo_gh_Atyz", "fo_gh_Atzz", "fo_gh_K", "fo_gh_Lambdax", "fo_gh_Lambday",
  "fo_gh_Lambdaz", "fo_gh_pi",
  "fo_gh_Qxxx", "fo_gh_Qxxy", "fo_gh_Qxxz", "fo_gh_Qxyy", "fo_gh_Qxyz",
  "fo_gh_Qxzz", "fo_gh_Qyxx", "fo_gh_Qyxy", "fo_gh_Qyxz", "fo_gh_Qyyy",
  "fo_gh_Qyyz", "fo_gh_Qyzz", "fo_gh_Qzxx", "fo_gh_Qzxy", "fo_gh_Qzxz",
  "fo_gh_Qzyy", "fo_gh_Qzyz", "fo_gh_Qzzz",
  "fo_gh_Xx", "fo_gh_Xy", "fo_gh_Xz", "fo_gh_ax", "fo_gh_ay", "fo_gh_az",
  "fo_gh_Bxx", "fo_gh_Bxy", "fo_gh_Bxz", "fo_gh_Byx", "fo_gh_Byy", "fo_gh_Byz",
  "fo_gh_Bzx", "fo_gh_Bzy", "fo_gh_Bzz", "fo_gh_h_perp", "fo_gh_hx", "fo_gh_hy",
  "fo_gh_hz", "fo_gh_vartheta_perp", "fo_gh_varthetax", "fo_gh_varthetay",
  "fo_gh_varthetaz"
};

FoGh::FoGh(MeshBlockPack *ppack, ParameterInput *pin) :
    u0("u0 fo_gh", 1, 1, 1, 1, 1),
    u1("u1 fo_gh", 1, 1, 1, 1, 1),
    u_rhs("u_rhs fo_gh", 1, 1, 1, 1, 1),
    coarse_u0("coarse u0 fo_gh", 1, 1, 1, 1, 1),
    dtnew(0.0),
    pmy_pack(ppack) {
  opt.kappa = pin->GetOrAddReal("fo_gh", "kappa", 1.0);
  opt.fd_order = pin->GetOrAddInteger("fo_gh", "fd_order", 4);
  opt.extrap_order = pin->GetOrAddInteger("fo_gh", "extrap_order", 2);
  opt.mu_H = pin->GetOrAddReal("fo_gh", "mu_H", 1.0);
  opt.eta_H = pin->GetOrAddReal("fo_gh", "eta_H", 1.0);
  opt.eta_beta = pin->GetOrAddReal("fo_gh", "eta_beta", 2.0);
  opt.diss = pin->GetOrAddReal("fo_gh", "diss", 0.0);
  if (opt.kappa <= 0.0 || opt.mu_H <= 0.0 || opt.eta_H <= 0.0 ||
      opt.eta_beta < 0.0 || opt.diss < 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "FO-GH requires kappa, mu_H, and eta_H > 0 and "
              << "eta_beta and diss >= 0." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int derivative_radius = opt.fd_order/2;
  if ((opt.fd_order != 2 && opt.fd_order != 4 && opt.fd_order != 6) ||
      ppack->pmesh->mb_indcs.ng < 2*derivative_radius) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "FO-GH fd_order must be 2, 4, or 6, with at least "
              << "fd_order ghost cells for its two-pass compatible derivative."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (opt.extrap_order < 2 || opt.extrap_order > 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "FO-GH extrap_order must be 2, 3, or 4."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppack->pmesh->multilevel && opt.fd_order == 6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "FO-GH fd_order=6 is not supported on multilevel meshes: "
              << "AthenaK has no matching sixth-order AMR prolongation operator."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool evolves = pin->GetString("time", "evolution") != "static";
  if (evolves && !(ppack->pmesh->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "FO-GH currently requires a three-dimensional mesh."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const int nmb = std::max(ppack->nmb_thispack, ppack->pmesh->nmb_maxperrank);
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = (indcs.nx2 > 1 ? indcs.nx2 + 2*indcs.ng : 1);
  const int ncells3 = (indcs.nx3 > 1 ? indcs.nx3 + 2*indcs.ng : 1);
  Kokkos::realloc(u0, nmb, nfo_gh, ncells3, ncells2, ncells1);
  Kokkos::realloc(u1, nmb, nfo_gh, ncells3, ncells2, ncells1);
  Kokkos::realloc(u_rhs, nmb, nfo_gh, ncells3, ncells2, ncells1);
  BindVariables(u0, u);
  BindVariables(u_rhs, rhs);

  if (ppack->pmesh->multilevel) {
    const int nccells1 = indcs.cnx1 + 2*indcs.ng;
    const int nccells2 = (indcs.cnx2 > 1 ? indcs.cnx2 + 2*indcs.ng : 1);
    const int nccells3 = (indcs.cnx3 > 1 ? indcs.cnx3 + 2*indcs.ng : 1);
    Kokkos::realloc(coarse_u0, nmb, nfo_gh, nccells3, nccells2, nccells1);
  }
  pbval_u = new MeshBoundaryValuesCC(ppack, pin, true);
  pbval_u->InitializeBuffers(nfo_gh);
}

FoGh::~FoGh() {
  delete pbval_u;
}

void FoGh::BindVariables(DvceArray5D<Real> data, Variables &vars) {
  vars.gtilde.InitWithShallowSlice(data, I_TGXX, I_TGZZ);
  vars.chi.InitWithShallowSlice(data, I_CHI);
  vars.alpha.InitWithShallowSlice(data, I_ALPHA);
  vars.beta.InitWithShallowSlice(data, I_BETAX, I_BETAZ);
  vars.Atilde.InitWithShallowSlice(data, I_TAXX, I_TAZZ);
  vars.K.InitWithShallowSlice(data, I_K);
  vars.Lambda.InitWithShallowSlice(data, I_LAMBDAX, I_LAMBDAZ);
  vars.pi.InitWithShallowSlice(data, I_PI);
  for (int k = 0; k < 3; ++k) {
    vars.Q[k].InitWithShallowSlice(data, I_QXXX + 6*k, I_QXXX + 6*k + 5);
  }
  vars.X.InitWithShallowSlice(data, I_XX, I_XZ);
  vars.a.InitWithShallowSlice(data, I_AX, I_AZ);
  vars.B.InitWithShallowSlice(data, I_BXX, I_BZZ);
  vars.h_perp.InitWithShallowSlice(data, I_H_PERP);
  vars.h.InitWithShallowSlice(data, I_HX, I_HZ);
  vars.vartheta_perp.InitWithShallowSlice(data, I_VARTHETA_PERP);
  vars.vartheta.InitWithShallowSlice(data, I_VARTHETAX, I_VARTHETAZ);
}

} // namespace fo_gh
