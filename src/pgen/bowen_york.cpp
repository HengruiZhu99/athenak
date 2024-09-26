//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) ...
// Licensed under the 3-clause BSD License
//========================================================================================
//! \file z4c_bowen_york.cpp
//  \brief Problem generator for a single boosted black hole using Bowen-York initial data

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

void ADMBowenYorkBoosted(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single boosted black hole using Bowen-York initial data
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Bowen-York test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMBowenYorkBoosted(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMConstraints<2>(pmbp);
            break;
    case 3: pmbp->pz4c->ADMConstraints<3>(pmbp);
            break;
    case 4: pmbp->pz4c->ADMConstraints<4>(pmbp);
            break;
  }
  std::cout<<"Bowen-York initial data initialized."<<std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMBowenYorkBoosted(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single boosted black hole using Bowen-York initial data

void ADMBowenYorkBoosted(MeshBlockPack *pmbp, ParameterInput *pin) {
  // Capture variables for the kernel
  auto &indcs = pmbp->pmesh->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  // For GLOOPS
  int isg = is-indcs.ng; int ieg = ie+indcs.ng;
  int jsg = js-indcs.ng; int jeg = je+indcs.ng;
  int ksg = ks-indcs.ng; int keg = ke+indcs.ng;
  int nmb = pmbp->nmb_thispack;
  Real m0 = pin->GetOrAddReal("problem", "punc_rest_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);
  Real P_x = pin->GetOrAddReal("problem", "punc_momentum_x1", 0.0);
  Real P_y = pin->GetOrAddReal("problem", "punc_momentum_x2", 0.0);
  Real P_z = pin->GetOrAddReal("problem", "punc_momentum_x3", 0.0);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen bowen-york puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Compute cell-centered coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real y = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real z = CellCenterX(k-ks, nx3, x3min, x3max);

    // Shift coordinates to black hole center
    x -= center_x1;
    y -= center_x2;
    z -= center_x3;

    Real r = std::sqrt(x*x + y*y + z*z);

    // Avoid division by zero at the puncture
    if (r == 0.0) r = 1e-12;

    // Compute conformal factor psi
    Real psi = 1.0 + m0 / (2.0 * r);

    // Compute psi^2 and psi^4
    Real psi2 = psi * psi;
    Real psi4 = psi2 * psi2;

    // Set the physical metric gamma_{ij} = psi^4 * delta_{ij}
    adm.g_dd(m,0,0,k,j,i) = psi4;
    adm.g_dd(m,1,1,k,j,i) = psi4;
    adm.g_dd(m,2,2,k,j,i) = psi4;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,2,k,j,i) = 0.0;

    // Set the lapse function alpha = 1
    adm.alpha(m,k,j,i) = 1.0;

    // Set the shift vector beta^i = 0
    adm.beta_u(m,0,k,j,i) = 0.0;
    adm.beta_u(m,1,k,j,i) = 0.0;
    adm.beta_u(m,2,k,j,i) = 0.0;

    // Compute unit radial vector n^i = x^i / r
    Real nx = x / r;
    Real ny = y / r;
    Real nz = z / r;

    // Compute n_i P^i
    Real n_dot_P = nx * P_x + ny * P_y + nz * P_z;

    // Compute the conformal extrinsic curvature \tilde{A}_{ij}
    Real factor = 3.0 / (2.0 * r * r);

    // Compute components of \tilde{A}_{ij}
    adm.vK_dd(m,0,0,k,j,i) = factor * (2.0 * nx * P_x - (1.0 - nx * nx) * n_dot_P);
    adm.vK_dd(m,1,1,k,j,i) = factor * (2.0 * ny * P_y - (1.0 - ny * ny) * n_dot_P);
    adm.vK_dd(m,2,2,k,j,i) = factor * (2.0 * nz * P_z - (1.0 - nz * nz) * n_dot_P);
    adm.vK_dd(m,0,1,k,j,i) = factor * (nx * P_y + ny * P_x - (-nx * ny) * n_dot_P);
    adm.vK_dd(m,0,2,k,j,i) = factor * (nx * P_z + nz * P_x - (-nx * nz) * n_dot_P);
    adm.vK_dd(m,1,2,k,j,i) = factor * (ny * P_z + nz * P_y - (-ny * nz) * n_dot_P);

    // The physical extrinsic curvature K_{ij} = \psi^{-2} * \tilde{A}_{ij}
    adm.vK_dd(m,0,0,k,j,i) /= psi2;
    adm.vK_dd(m,1,1,k,j,i) /= psi2;
    adm.vK_dd(m,2,2,k,j,i) /= psi2;
    adm.vK_dd(m,0,1,k,j,i) /= psi2;
    adm.vK_dd(m,0,2,k,j,i) /= psi2;
    adm.vK_dd(m,1,2,k,j,i) /= psi2;
  });
}

// Refinement condition
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
