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


static void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin);
static void RefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::Z4cOnePuncture(ParameterInput *pin, const bool restart) {
  user_ref_func  = RefinementCondition;
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &indcs = pmy_mesh_->mb_indcs;

  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  ADMOnePuncture(pmbp, pin);
  pmbp->pz4c->GaugePreCollapsedLapse(pmbp, pin);
  switch (indcs.ng) {
    case 2: pmbp->pz4c->ADMToZ4c<2>(pmbp, pin);
            break;
    case 3: pmbp->pz4c->ADMToZ4c<3>(pmbp, pin);
            break;
    case 4: pmbp->pz4c->ADMToZ4c<4>(pmbp, pin);
            break;
  }
  pmbp->pz4c->Z4cToADM(pmbp);
  // Apply the perturbation after the pre-collapsed lapse has been initialized.
  // A compact C-infinity bump vanishes exactly outside radius width, so an
  // off-center pulse does not change the puncture's collapsed lapse.
  const Real amplitude = pin->GetOrAddReal("problem", "lapse_pulse_amplitude", 0.0);
  const Real width = pin->GetOrAddReal("problem", "lapse_pulse_width", 1.0);
  const Real xc = pin->GetOrAddReal("problem", "lapse_pulse_x", 4.0);
  const Real yc = pin->GetOrAddReal("problem", "lapse_pulse_y", 0.0);
  const Real zc = pin->GetOrAddReal("problem", "lapse_pulse_z", 0.0);
  if (!std::isfinite(amplitude) || amplitude<0 || !std::isfinite(width) || width<=0 ||
      !std::isfinite(xc) || !std::isfinite(yc) || !std::isfinite(zc)) {
    std::cerr << "Lapse pulse requires finite amplitude>=0, width>0 and center." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (amplitude>0) {
    auto &size = pmbp->pmb->mb_size;
    auto &z4c = pmbp->pz4c->z4c;
    const int ng=indcs.ng, is=indcs.is, js=indcs.js, ks=indcs.ks;
    const int nx=indcs.nx1, ny=indcs.nx2, nz=indcs.nx3;
    par_for("stationary puncture lapse pulse", DevExeSpace(),0,pmbp->nmb_thispack-1,
      ks-ng,indcs.ke+ng,js-ng,indcs.je+ng,is-ng,indcs.ie+ng,
      KOKKOS_LAMBDA(const int m,const int k,const int j,const int i) {
        const Real x=CellCenterX(i-is,nx,size.d_view(m).x1min,size.d_view(m).x1max)-xc;
        const Real y=CellCenterX(j-js,ny,size.d_view(m).x2min,size.d_view(m).x2max)-yc;
        const Real z=CellCenterX(k-ks,nz,size.d_view(m).x3min,size.d_view(m).x3max)-zc;
        const Real q=(x*x+y*y+z*z)/(width*width);
        if (q<1) z4c.alpha(m,k,j,i) += amplitude*exp(-q/(1-q));
      });
    pmbp->pz4c->Z4cToADM(pmbp);
  }
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

#ifdef ATHENA_CUSTOM_ONE_PUNCTURE
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  Z4cOnePuncture(pin,restart);
}
#endif

//----------------------------------------------------------------------------------------
//! \fn void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin)
//! \brief Initialize ADM vars to single puncture (no spin)

void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin) {
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
  Real ADM_mass = pin->GetOrAddReal("problem", "punc_ADM_mass", 1.);
  Real center_x1 = pin->GetOrAddReal("problem", "punc_center_x1", 0.);
  Real center_x2 = pin->GetOrAddReal("problem", "punc_center_x2", 0.);
  Real center_x3 = pin->GetOrAddReal("problem", "punc_center_x3", 0.);

  adm::ADM::ADM_vars &adm = pmbp->padm->adm;

  par_for("pgen one puncture",
  DevExeSpace(),0,nmb-1,ksg,keg,jsg,jeg,isg,ieg,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

    x1v -= center_x1;
    x2v -= center_x2;
    x3v -= center_x3;

    Real r = std::sqrt(std::pow(x3v,2) + std::pow(x2v,2) + std::pow(x1v,2));

    // Minkowski spacetime
    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) = (a == b ? 1. : 0.);
    }
    // admK_dd is automatically set to 0 when is initialized as Kokkos View

    // ADMOnePuncture
    adm.psi4(m,k,j,i) = std::pow(1.0 + 0.5*ADM_mass/r,4); // adm.psi4

    for(int a = 0; a < 3; ++a)
    for(int b = a; b < 3; ++b) {
      adm.g_dd(m,a,b,k,j,i) *= adm.psi4(m,k,j,i);
    }
  });
}

// how decide the refinement
void RefinementCondition(MeshBlockPack* pmbp) {
  pmbp->pz4c->pamr->Refine(pmbp);
}
