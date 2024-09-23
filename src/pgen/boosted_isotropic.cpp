//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture.cpp
//  \brief Problem generator for a single puncture placed at the origin of the domain

#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>   // endl
#include <limits>     // numeric_limits::max()
#include <memory>
#include <string>     // c_str(), string
#include <vector>
#include <math.h>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_amr.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"

void BoostedPunctureIsotropic(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
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

  BoostedPunctureIsotropic(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  // pmbp->pz4c->Z4cToADM(pmbp);
  // pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
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
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single boosted puncture (no spin), based on 1909.02997

void BoostedPunctureIsotropic(MeshBlockPack *pmbp, ParameterInput *pin) {
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
  Real m0 = pin->GetOrAddReal("problem", "punc_rest_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);
  Real vx1 = pin->GetOrAddReal("problem", "punc_velocity_x1", 0.);
  Real vx2 = pin->GetOrAddReal("problem", "punc_velocity_x2", 0.);
  Real vx3 = pin->GetOrAddReal("problem", "punc_velocity_x3", 0.);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;
  z4c::Z4c::Z4c_vars &z4c = pmbp->pz4c->z4c;
  auto &opt = pmbp->pz4c->opt;

  par_for("pgen boosted puncture isotropic",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    x -= center_x1;
    y -= center_x2;
    z -= center_x3;

    // (TO DO) add rotation to the coordinate here later on to acount for boost off the axis

    // velocity magnitude for now assuming only vx1 is non-zero! Do a rotation later
    Real v = std::sqrt(std::pow(vx1,2) + std::pow(vx2,2) + std::pow(vx3,2));

    // boost factor
    Real gamma = 1/std::sqrt(1-std::pow(v,2));

    Real x0 = gamma * x;
    Real y0 = y;
    Real z0 = z;



    // radial coordinate in comoving frame
    Real r = std::sqrt(std::pow(x0,2) + std::pow(y0,2) + std::pow(z0,2));

    // conformal factor Equation 13 in arxiv:1410.8607
    Real psi = 1.0 + 0.5*m0/r;

    // adm metric in the boosted frame
    adm.g_dd(m,0,0,k,j,i) = gamma * gamma * (1.0 - 16.0 * std::pow(m0 - 2 * r, 2) * std::pow(r, 4) * std::pow(v, 2) / std::pow(m0 + 2 * r, 6));
    adm.g_dd(m,1,1,k,j,i) = 1;
    adm.g_dd(m,2,2,k,j,i) = 1;

    adm.g_dd(m,0,0,k,j,i) *= std::pow(psi,4);
    adm.g_dd(m,1,1,k,j,i) *= std::pow(psi,4);
    adm.g_dd(m,2,2,k,j,i) *= std::pow(psi,4);


    // shift in boosted frame
    Real B = std::sqrt(std::pow(m0 + 2 * r, 6) - 16.0 * std::pow(m0 - 2 * r, 2) * std::pow(r, 4) * std::pow(v, 2));
    adm.beta_u(m,0,k,j,i) = - m0 * v * (std::pow(m0, 2) + 6.0 * m0 * r + 16.0 * std::pow(r, 2)) * 
                (std::pow(m0, 3) + 6.0 * std::pow(m0, 2) * r + 8.0 * m0 * std::pow(r, 2) + 16.0 * std::pow(r, 3)) / 
                pow(B, 2);

    // trace of extrinsic curvature
    Real K = 32.0 * gamma * m0 * v * 
           ((std::pow(m0 + 2.0 * r, 7) - 32.0 * std::pow(m0 - 2.0 * r, 2) * (m0 - r) * std::pow(r, 4) * std::pow(v, 2)) * 
           std::pow(r, 2) * x) / (std::pow(m0 + 2.0 * r, 3) * std::pow(B, 3));

    // some auxiliary numbers
    Real C = std::pow(m0 + 2.0 * r, 6) - 8.0 * std::pow(m0 - 2.0 * r, 2) * std::pow(r, 4) * std::pow(v, 2);
    Real D = std::pow(m0 + 2.0 * r, 12) - 
           32.0 * std::pow(m0 - 2.0 * r, 2) * std::pow(r, 4) * std::pow(m0 + 2.0 * r, 6) * std::pow(v, 2) +
           256.0 * std::pow(m0 - 2.0 * r, 4) * std::pow(r, 8) * std::pow(v, 4);

    // non-vanishing component of the trace-free conformal extrinsic curvature
    // may need additional factors of psi^2 here. 
    // may need to also lorentz contract x
    AthenaScratchTensor<Real, TensorSymm::SYM2, 3, 2> A_tilda_dd;
    A_tilda_dd(0,0) = -2.0 * std::pow(gamma, 3) * m0 * v * (m0 - 4.0 * r) * C * x / (3.0 * std::pow(m0 + 2.0 * r, 3) * B * std::pow(r, 4));
    A_tilda_dd(1,1) = gamma * m0 * v * (m0 - 4.0 * r) * std::pow(m0 + 2.0 * r, 3) * B * C * x / (3.0 * D * std::pow(r, 4));
    A_tilda_dd(2,2) = A_tilda_dd(1,1);

    A_tilda_dd(0,1) = -gamma * m0 * v * (m0 - 4.0 * r) * std::pow(m0 + 2.0 * r, 3) * y / (2.0 * B * std::pow(r, 4));
    A_tilda_dd(0,2) = -gamma * m0 * v * (m0 - 4.0 * r) * std::pow(m0 + 2.0 * r, 3) * z / (2.0 * B * std::pow(r, 4));
    A_tilda_dd(1,2) = 0;

    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      adm.vK_dd(m,a,b,k,j,i) = 1.0/3.0*adm.g_dd(m,a,b,k,j,i)*K + std::pow(psi,-2)*A_tilda_dd(a,b);
    }
  });
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
