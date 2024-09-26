//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) ...
// Licensed under the 3-clause BSD License
//========================================================================================
//! \file z4c_boosted_schwarzschild_paper.cpp
//  \brief Problem generator for an analytically boosted Schwarzschild black hole in isotropic coordinates, following specific paper equations.

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

void ADMBoostedSchwarzschildPaper(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for an analytically boosted Schwarzschild black hole following the paper equations.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Boosted Schwarzschild test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMBoostedSchwarzschildPaper(pmbp, pin);
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
  std::cout<<"Boosted Schwarzschild initial data initialized following paper equations."<<std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMBoostedSchwarzschildPaper(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to an analytically boosted Schwarzschild black hole following the given paper equations

void ADMBoostedSchwarzschildPaper(MeshBlockPack *pmbp, ParameterInput *pin) {
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

  // Problem parameters
  Real m0 = pin->GetOrAddReal("problem", "punc_rest_mass", 1.0);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.0);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.0);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.0);
  Real vx = pin->GetOrAddReal("problem", "punc_velocity_x1", 0.0);
  Real vy = pin->GetOrAddReal("problem", "punc_velocity_x2", 0.0);
  Real vz = pin->GetOrAddReal("problem", "punc_velocity_x3", 0.0);

  // Compute the magnitude of the momentum and the velocity
  Real v2 = vx*vx + vy*vy + vz*vz;
  Real v = std::sqrt(v2);
  Real gamma = 1.0 / std::sqrt(1.0 - v2);

  // Normalize the velocity vector to get the boost direction
  Real n_vx = vx / v;
  Real n_vy = vy / v;
  Real n_vz = vz / v;

  // Rotate coordinates to align boost along y-axis
  // Compute rotation matrix components
  Real boost_dir[3] = {n_vx, n_vy, n_vz};
  Real rot_matrix[3][3];
  // Assuming the boost is along P^i, we need to rotate the coordinate system such that P^i aligns with y-axis
  // For simplicity, we'll assume the boost is along y-axis in this implementation
  // If P^i is not along y-axis, additional rotation is needed

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen boosted schwarzschild paper",
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

    // Coordinates in the boosted frame
    // Since the Lorentz boost is along y-axis, and at t = 0, the time coordinate remains unchanged
    Real x0 = x;
    Real y0 = gamma * y;
    Real z0 = z;

    // Compute isotropic radial coordinate in boosted frame
    Real r = std::sqrt(x0 * x0 + y0 * y0 + z0 * z0);

    // Avoid division by zero at the puncture
    if (r == 0.0) r = 1e-12;

    // Compute conformal factor psi_B
    Real psi_B = 1.0 + m0 / (2.0 * r);

    // Compute lapse function alpha_0
    Real alpha0 = (1.0 - m0 / (2.0 * r)) / (1.0 + m0 / (2.0 * r));

    // Compute B
    Real temp1 = m0 + 2.0 * r;
    Real temp2 = m0 - 2.0 * r;
    Real temp1_6 = std::pow(temp1, 6);
    Real temp2_2 = temp2 * temp2;
    Real r4 = r * r * r * r;
    Real B_sq = temp1_6 - 16.0 * temp2_2 * r4 * v2;
    Real B = std::sqrt(B_sq);

    // Compute shift vector beta^i
    // Only beta^y is non-zero before rotation
    Real numerator_beta_y = -m0 * v * (m0 * m0 + 6.0 * m0 * r + 16.0 * r * r) *
          (m0 * m0 * m0 + 6.0 * m0 * m0 * r + 8.0 * m0 * r * r + 16.0 * r * r * r);
    Real beta_y = numerator_beta_y / (B_sq);

    // Rotate beta^i back to original orientation
    // Since the boost direction is along P^i, we need to rotate beta^i accordingly
    Real beta_u[3];
    // For this example, we assume boost is along y-axis, so beta_x and beta_z are zero
    beta_u[0] = 0.0;
    beta_u[1] = beta_y;
    beta_u[2] = 0.0;

    // Set the shift vector components
    adm.beta_u(m,0,k,j,i) = beta_u[0];
    adm.beta_u(m,1,k,j,i) = beta_u[1];
    adm.beta_u(m,2,k,j,i) = beta_u[2];

    // Set the spatial metric gamma_{ij}
    // The conformal spatial line element is
    // dl^2 = dx^2 + gamma^2 [1 - (16 (m - 2r)^2 r^4 v^2)/(m + 2r)^6 ] dy^2 + dz^2
    // Let's compute the factor in front of dy^2
    Real factor_dy = gamma * gamma * (1.0 - (16.0 * temp2_2 * r4 * v2) / temp1_6);
    adm.g_dd(m,0,0,k,j,i) = psi_B * psi_B * psi_B * psi_B; // gamma_xx
    adm.g_dd(m,1,1,k,j,i) = psi_B * psi_B * psi_B * psi_B * factor_dy; // gamma_yy
    adm.g_dd(m,2,2,k,j,i) = psi_B * psi_B * psi_B * psi_B; // gamma_zz
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,2,k,j,i) = 0.0;

    // Set the lapse function alpha
    adm.alpha(m,k,j,i) = alpha0;

    // Compute extrinsic curvature components K_{ij}
    // Mean curvature K
    Real numerator_K = 32.0 * gamma * m0 * v * ((std::pow(m0 + 2.0 * r, 7)) - 32.0 * temp2_2 * (m0 - r) * r4 * v2) * r * r * y;
    Real denominator_K = std::pow(m0 + 2.0 * r, 3) * B * B * B;
    Real K = numerator_K / denominator_K;

    // Compute conformal trace-free extrinsic curvature components tilde{A}_{ij}
    // Auxiliary quantities C and D
    Real C = std::pow(temp1, 6) - 8.0 * std::pow(temp2, 2) * std::pow(r, 4) * v2;
    Real D = std::pow(temp1, 12) - 32.0 * std::pow(temp2, 2) * std::pow(r, 4) * std::pow(temp1, 6) * v2
              + 256.0 * std::pow(temp2, 4) * std::pow(r, 8) * std::pow(v2, 2);

    // Compute tilde{A}_{xx} and tilde{A}_{zz}
    Real tilde_A_xx = (gamma * m0 * v * (m0 - 4.0 * r) * std::pow(m0 + 2.0 * r, 3) * B * C * y) / (3.0 * D * std::pow(r, 4));
    Real tilde_A_zz = tilde_A_xx;

    // Compute tilde{A}_{xy}
    Real tilde_A_xy = - (gamma * m0 * v * (m0 - 4.0 * r) * std::pow(m0 + 2.0 * r, 3) * x) / (2.0 * B * std::pow(r, 4));

    // Compute tilde{A}_{yy}
    Real tilde_A_yy = - (2.0 * std::pow(gamma, 3) * m0 * v * (m0 - 4.0 * r) * C * y) /
                      (3.0 * std::pow(m0 + 2.0 * r, 3) * B * std::pow(r, 4));

    // Compute tilde{A}_{yz}
    Real tilde_A_yz = - (gamma * m0 * v * (m0 - 4.0 * r) * std::pow(m0 + 2.0 * r, 3) * z) / (2.0 * B * std::pow(r, 4));

    // Compute tilde{A}_{xz}
    Real tilde_A_xz = 0.0; // As per provided equations

    // Compute A_{ij} = psi^{-2} * tilde{A}_{ij}
    Real psi2 = psi_B * psi_B;
    Real A_xx = tilde_A_xx / psi2;
    Real A_yy = tilde_A_yy / psi2;
    Real A_zz = tilde_A_zz / psi2;
    Real A_xy = tilde_A_xy / psi2;
    Real A_yz = tilde_A_yz / psi2;
    Real A_xz = tilde_A_xz / psi2;

    // Compute K_{ij} = A_{ij} + (1/3) * gamma_{ij} * K
    adm.vK_dd(m, 0, 0, k, j, i) = A_xx + (1.0 / 3.0) * adm.g_dd(m, 0, 0, k, j, i) * K;
    adm.vK_dd(m, 1, 1, k, j, i) = A_yy + (1.0 / 3.0) * adm.g_dd(m, 1, 1, k, j, i) * K;
    adm.vK_dd(m, 2, 2, k, j, i) = A_zz + (1.0 / 3.0) * adm.g_dd(m, 2, 2, k, j, i) * K;
    adm.vK_dd(m, 0, 1, k, j, i) = A_xy;
    adm.vK_dd(m, 1, 2, k, j, i) = A_yz;
    adm.vK_dd(m, 0, 2, k, j, i) = A_xz;

    // Rotate tilde{A}_{ij} back to original orientation if necessary
    // For this example, we assume the boost is along y-axis, so no rotation is applied

  });
}

// Refinement condition
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
