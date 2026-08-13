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
#include "pgen/z4c/z4c_one_puncture_gauge_diagnostics.hpp"
#include "z4c/fastflow.hpp"


void ADMOnePuncture(MeshBlockPack *pmbp, ParameterInput *pin);
void RefinementCondition(MeshBlockPack* pmbp);
void FinalizeOnePuncture(ParameterInput *pin, Mesh *pm);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem_()
//! \brief Problem Generator for single puncture
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_ref_func = RefinementCondition;
  pgen_final_func = FinalizeOnePuncture;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pz4c == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "One Puncture test can only be run in Z4c, but no <z4c> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }
  if (pin->GetOrAddBoolean("problem", "user_hist", false)) {
    user_hist_func = &z4c_puncture_gauge_diagnostics::GaugeDiagnostics;
    z4c_puncture_gauge_diagnostics::center[0] =
        pin->GetOrAddReal("problem", "punc_center_x1", 0.0);
    z4c_puncture_gauge_diagnostics::center[1] =
        pin->GetOrAddReal("problem", "punc_center_x2", 0.0);
    z4c_puncture_gauge_diagnostics::center[2] =
        pin->GetOrAddReal("problem", "punc_center_x3", 0.0);
    z4c_puncture_gauge_diagnostics::output_path = pin->GetOrAddString(
        "problem", "gauge_diagnostics_file", "z4c_gauge_source_diagnostics.csv");
  }
  z4c_puncture_gauge_diagnostics::profile_name =
      z4c_puncture_gauge_diagnostics::ProfileName(pmbp->pz4c->opt.shift_gauge_profile);
  if (restart) {
    // An exact terminal landing may persist a tiny, nonphysical remainder timestep.
    // Opt-in qualification restarts discard only that proposal; Driver::Initialize
    // immediately recomputes the ordinary spatial-CFL step from the restored state.
    if (pin->GetOrAddBoolean("problem", "reset_dt_from_cfl_on_restart", false)) {
      pmy_mesh_->dt = std::numeric_limits<float>::max();
    }
    return;
  }

  auto &indcs = pmy_mesh_->mb_indcs;

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
//! \fn void FinalizeOnePuncture(ParameterInput *pin, Mesh *pm)
//! \brief Optionally observe the accepted terminal slice with FastFlow.
//!
//! The regular FastFlow task runs before Driver advances Mesh::time.  This explicit
//! qualification hook instead observes the already accepted terminal state at its exact
//! accepted time.  It is opt-in and does not modify the evolution state.
void FinalizeOnePuncture(ParameterInput *pin, Mesh *pm) {
  if (!pin->GetOrAddBoolean("problem", "final_horizon", false)) return;
  auto *pz4c = pm->pmb_pack->pz4c;
  const Real accepted_time = pm->time;
  const int accepted_cycle = pm->ncycle;
  for (auto &horizon : pz4c->pfastflow) {
    switch (pm->mb_indcs.ng) {
      case 2: horizon->MetricDerivatives<2>(accepted_time); break;
      case 3: horizon->MetricDerivatives<3>(accepted_time); break;
      case 4: horizon->MetricDerivatives<4>(accepted_time); break;
    }
  }
  for (auto &horizon : pz4c->pfastflow) {
    horizon->Find(accepted_cycle, accepted_time);
    horizon->Write(accepted_cycle, accepted_time);
  }
}

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
