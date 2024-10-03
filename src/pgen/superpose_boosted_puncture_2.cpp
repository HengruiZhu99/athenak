//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

void ComputeADMVariables(MeshBlockPack *pmbp, AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &Ag_dd, AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &Bg_dd,
    AthenaTensor<Real, TensorSymm::NONE, 3, 2> &KA_ud, AthenaTensor<Real, TensorSymm::NONE, 3, 2> &KB_ud);
void AddBoostedPuncture(MeshBlockPack *pmbp, ParameterInput *pin, int punc_num, AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
    AthenaTensor<Real, TensorSymm::NONE, 3, 2> &K_ud);
void RefinementCondition(MeshBlockPack* pmbp);

KOKKOS_INLINE_FUNCTION
AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> 
inverse(AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> matrix);

KOKKOS_INLINE_FUNCTION
void LorentzBoost(Real vx1, Real vx2, Real vx3, AthenaScratchTensor<Real, TensorSymm::SYM2, 4, 2> &lambda);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  int nmb = pmbp->nmb_thispack;
  auto &indcs = pmy_mesh_->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = indcs.nx2 + 2*(indcs.ng);
  int ncells3 = indcs.nx3 + 2*(indcs.ng);

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Arrays to store metric and extrinsic curvature for individual black holes
  DvceArray5D<Real> gA_dd("gA_dd", nmb, 6, ncells3, ncells2, ncells1);
  DvceArray5D<Real> gB_dd("gB_dd", nmb, 6, ncells3, ncells2, ncells1); 
  DvceArray5D<Real> kA_ud("kA_ud", nmb, 9, ncells3, ncells2, ncells1);
  DvceArray5D<Real> kB_ud("kB_ud", nmb, 9, ncells3, ncells2, ncells1);


  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> Ag_dd;
  Ag_dd.InitWithShallowSlice(gA_dd, 0, 5);
  AthenaTensor<Real, TensorSymm::SYM2, 3, 2> Bg_dd;
  Bg_dd.InitWithShallowSlice(gB_dd, 0, 5);

  AthenaTensor<Real, TensorSymm::NONE, 3, 2> KA_ud;
  KA_ud.InitWithShallowSlice(kA_ud, 0, 8);
  AthenaTensor<Real, TensorSymm::NONE, 3, 2> KB_ud;
  KB_ud.InitWithShallowSlice(kB_ud, 0, 8);

  AddBoostedPuncture(pmbp,pin,1,Ag_dd,KA_ud);
  AddBoostedPuncture(pmbp,pin,2,Bg_dd,KB_ud);
  ComputeADMVariables(pmbp,Ag_dd,Bg_dd,KA_ud,KB_ud);

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
  std::cout<<"Superposed punctures initialized."<<std::endl;

  return;
}

void ComputeADMVariables(MeshBlockPack *pmbp, AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &Ag_dd, AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &Bg_dd,
    AthenaTensor<Real, TensorSymm::NONE, 3, 2> &KA_ud, AthenaTensor<Real, TensorSymm::NONE, 3, 2> &KB_ud) {
  // capture variables for the kernel
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
  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    // Superposed metric
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = Ag_dd(m,a,b,k,j,i) + Bg_dd(m,a,b,k,j,i);
      adm.g_dd(m,a,b,k,j,i) -= (a==b) ? 1.0 : 0.0;
    }
    // Extrinsic Curvature
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.vK_dd(m,a,b,k,j,i) = 0;
      for(int c = 0; c < 3; ++c) {
        adm.vK_dd(m,a,b,k,j,i) += 0.5*(adm.g_dd(m,a,c,k,j,i)*(KA_ud(m,c,b,k,j,i)+KB_ud(m,c,b,k,j,i))
                                  + adm.g_dd(m,b,c,k,j,i)*(KA_ud(m,c,a,k,j,i)+KB_ud(m,c,a,k,j,i)));
      }
    }
  });
}

void AddBoostedPuncture(MeshBlockPack *pmbp, ParameterInput *pin, int punc_num, AthenaTensor<Real, TensorSymm::SYM2, 3, 2> &g_dd,
    AthenaTensor<Real, TensorSymm::NONE, 3, 2> &K_ud) {
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

  Real m0 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_rest_mass", 0.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_center_x3", 0.);
  Real vel = pin->GetOrAddReal("problem", "punc" + std::to_string(punc_num) + "_velocity_x1", 0.); // Example velocity
  std::cout << center_x1 << std::endl;
  // Lorentz factor
  Real Gamma = 1.0 / std::sqrt(1.0 - vel * vel);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> K_dd;
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> g_uu;

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


    // Coordinates in comoving frame (x0)
    // Lorentz transformation along x-direction
    // TODO (@hzhu): Add rotation here.
    Real x0 = Gamma * x; // At t = 0
    Real y0 = y;
    Real z0 = z;

    // Radial coordinate in comoving frame
    Real r0 = std::sqrt(x0*x0 + y0*y0 + z0*z0);

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
    // adm.alpha(m,k,j,i) = alpha0 / B0;

    // Compute the shift vector beta^i (only beta^x is non-zero)
    Real num_beta = alpha0_2 - psi0_4;
    Real den_beta = psi0_4 - alpha0_2 * vel * vel;
    // Shift vector components
    //adm.beta_u(m,0,k,j,i) += (num_beta / den_beta) * vel;
    //adm.beta_u(m,1,k,j,i) = 0;
    //adm.beta_u(m,2,k,j,i) = 0;

    // Spatial metric components gamma_{ij}
    g_dd(m,0,0,k,j,i) = psi0_4 * B0_squared;
    g_dd(m,1,1,k,j,i) = psi0_4;
    g_dd(m,2,2,k,j,i) = psi0_4;
    g_dd(m,0,1,k,j,i) = 0.0;
    g_dd(m,0,2,k,j,i) = 0.0;
    g_dd(m,1,2,k,j,i) = 0.0;

    // compute metric inverse
    Real detg = adm::SpatialDet(g_dd(m,0,0,k,j,i), g_dd(m,0,1,k,j,i),
                                g_dd(m,0,2,k,j,i), g_dd(m,1,1,k,j,i),
                                g_dd(m,1,2,k,j,i), g_dd(m,2,2,k,j,i));
    
    adm::SpatialInv(1.0/detg,
            g_dd(m,0,0,k,j,i), g_dd(m,0,1,k,j,i),
            g_dd(m,0,2,k,j,i), g_dd(m,1,1,k,j,i),
            g_dd(m,1,2,k,j,i), g_dd(m,2,2,k,j,i),
            &g_uu(0,0), &g_uu(0,1), &g_uu(0,2),
            &g_uu(1,1), &g_uu(1,2), &g_uu(2,2));

    // Compute extrinsic curvature components
    // Compute s and its derivative
    Real s = den_beta; // s = psi0^4 - alpha0^2 * v^2
    Real psi0_prime_4 = 4.0 * psi0 * psi0 * psi0 * psi0_prime;
    Real alpha0_prime_2 = 2.0 * alpha0 * alpha0_prime;
    Real s_prime = psi0_prime_4 - alpha0_prime_2 * vel * vel;
    Real ln_s_prime = s_prime / s;

    // K_xx
    Real prefactor_xx = Gamma * Gamma * B0 * x * vel / r0;
    Real expr_xx = 2.0 * alpha0_prime - (alpha0 / 2.0) * ln_s_prime;
    K_dd(0,0) = prefactor_xx * expr_xx;

    // K_yy and K_zz
    Real num_yy = 2.0 * Gamma * Gamma * x * vel * alpha0 * psi0_prime;
    Real den_yy = psi0 * B0 * r0;
    K_dd(1,1)  = num_yy / den_yy;
    K_dd(2,2)  = num_yy / den_yy;

    // K_xy and K_xz
    Real prefactor_xy = B0 * vel / r0;
    Real expr_xy = alpha0_prime - (alpha0 / 2.0) * ln_s_prime;
    K_dd(0,1) = prefactor_xy * y * expr_xy;
    K_dd(0,2) = prefactor_xy * z * expr_xy;

    // K_yz is zero due to symmetry
    K_dd(1,2) = 0.0;

    for(int a = 0; a < 3; ++a) {
      for(int b = 0; b < 3; ++b) {
        K_ud(m,a,b,k,j,i) = 0.0;
        for(int c = 0; c < 3; ++c) {
          K_ud(m,a,b,k,j,i) += g_uu(a,c) * K_dd(c,b);
        }
      }
    }
  });
}

// Refinement condition
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}