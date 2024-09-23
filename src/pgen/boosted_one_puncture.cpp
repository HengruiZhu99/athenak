//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) ...
// Licensed under the 3-clause BSD License
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

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

void ADMOnePunctureBoosted(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single boosted puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMOnePunctureBoosted(pmbp, pin);
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
  std::cout<<"OnePuncture initialized."<<std::endl;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePunctureBoosted(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single boosted puncture (no spin), based on the given equations

void ADMOnePunctureBoosted(MeshBlockPack *pmbp, ParameterInput *pin) {
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
  Real m0 = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);
  Real vx1 = pin->GetOrAddReal("problem", "punc_velocity_x1", 0.5); // Example velocity
  Real vx2 = pin->GetOrAddReal("problem", "punc_velocity_x2", 0.0);
  Real vx3 = pin->GetOrAddReal("problem", "punc_velocity_x3", 0.0);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Compute cell-centered coordinates
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Shift coordinates to black hole center
    x1v -= center_x1;
    x2v -= center_x2;
    x3v -= center_x3;

    // Velocity magnitude
    Real vel = std::sqrt(vx1*vx1 + vx2*vx2 + vx3*vx3);

    // Lorentz factor
    Real Gamma = 1.0 / std::sqrt(1.0 - vel * vel);

    // Coordinates in comoving frame (x0)
    Real x0[4]; // x0[0] = t0, x0[1] = x0, x0[2] = y0, x0[3] = z0

    // Lorentz transformation along x-direction
    x0[1] = Gamma * (x1v - vel * 0.0); // At t = 0
    x0[0] = -vel * x0[1]; // t0 = -v * x0
    x0[2] = x2v;
    x0[3] = x3v;

    // Radial coordinate in comoving frame
    Real r0 = std::sqrt(x0[1]*x0[1] + x0[2]*x0[2] + x0[3]*x0[3]);

    // Compute psi0 and its derivative
    Real psi0 = 1.0 + m0 / (2.0 * r0);
    Real psi0_prime = -m0 / (2.0 * r0 * r0);

    // Compute A and its derivative
    Real A = 1.0 - m0 / (2.0 * r0);
    Real A_prime = m0 / (2.0 * r0 * r0);

    // Compute alpha0 and its derivative
    Real alpha0 = A / psi0;
    Real alpha0_prime = (A_prime * psi0 - A * psi0_prime) / (psi0 * psi0);

    // Compute psi0^4 and alpha0^2
    Real psi0_4 = psi0 * psi0 * psi0 * psi0;
    Real alpha0_2 = alpha0 * alpha0;

    // Compute B0^2 and B0
    Real B0_squared = Gamma * Gamma * (1.0 - vel * vel * alpha0_2 / psi0_4);
    Real B0 = std::sqrt(B0_squared);

    // Compute the lapse function alpha
    Real alpha = alpha0 / B0;

    // Compute the shift vector beta^i (only beta^x is non-zero)
    Real num_beta = alpha0_2 - psi0_4;
    Real den_beta = psi0_4 - alpha0_2 * vel * vel;
    Real beta_x = (num_beta / den_beta) * vel;
    Real beta_y = 0.0;
    Real beta_z = 0.0;

    // Compute s and its derivative
    Real s = den_beta; // s = psi0^4 - alpha0^2 * v^2
    Real psi0_prime_4 = 4.0 * psi0 * psi0 * psi0 * psi0_prime;
    Real alpha0_prime_2 = 2.0 * alpha0 * alpha0_prime;
    Real s_prime = psi0_prime_4 - alpha0_prime_2 * vel * vel;
    Real ln_s_prime = s_prime / s;

    // Compute extrinsic curvature components
    Real x = x1v;
    Real y = x2v;
    Real z = x3v;
    Real r = r0;

    // K_xx
    Real prefactor_xx = Gamma * Gamma * B0 * x * vel / r;
    Real expr_xx = 2.0 * alpha0_prime - (alpha0 / 2.0) * ln_s_prime;
    Real K_xx = prefactor_xx * expr_xx;

    // K_yy and K_zz
    Real num_yy = 2.0 * Gamma * Gamma * x * vel * alpha0 * psi0_prime;
    Real den_yy = psi0 * B0 * r;
    Real K_yy = num_yy / den_yy;
    Real K_zz = K_yy;

    // K_xy and K_xz
    Real prefactor_xy = B0 * vel / r;
    Real expr_xy = alpha0_prime - (alpha0 / 2.0) * ln_s_prime;
    Real K_xy = prefactor_xy * y * expr_xy;
    Real K_xz = prefactor_xy * z * expr_xy;

    // K_yz is zero due to symmetry
    Real K_yz = 0.0;

    // Set the ADM variables in the data structures

    // Lapse function
    adm.alpha(m,k,j,i) = alpha;

    // Shift vector components
    adm.beta_u(m,0,k,j,i) = beta_x;
    adm.beta_u(m,1,k,j,i) = beta_y;
    adm.beta_u(m,2,k,j,i) = beta_z;

    // Spatial metric components gamma_{ij}
    adm.g_dd(m,0,0,k,j,i) = psi0_4 * B0_squared;
    adm.g_dd(m,1,1,k,j,i) = psi0_4;
    adm.g_dd(m,2,2,k,j,i) = psi0_4;
    adm.g_dd(m,0,1,k,j,i) = 0.0;
    adm.g_dd(m,0,2,k,j,i) = 0.0;
    adm.g_dd(m,1,2,k,j,i) = 0.0;

    // Extrinsic curvature components K_{ij}
    adm.vK_dd(m,0,0,k,j,i) = K_xx;
    adm.vK_dd(m,1,1,k,j,i) = K_yy;
    adm.vK_dd(m,2,2,k,j,i) = K_zz;
    adm.vK_dd(m,0,1,k,j,i) = K_xy;
    adm.vK_dd(m,0,2,k,j,i) = K_xz;
    adm.vK_dd(m,1,2,k,j,i) = K_yz;
  });
}

// Refinement condition
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
