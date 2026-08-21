//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file adm.cpp
//  \brief implementation of ADM class
#include <algorithm>

#include "coordinates/adm.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "z4c/z4c.hpp"
#include "z4c/stored_domain_bounds.hpp"

namespace adm {
char const * const ADM::ADM_names[ADM::nadm] = {
  "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
  "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
  "adm_psi4",
  "adm_alpha", "adm_betax", "adm_betay", "adm_betaz",
};

//----------------------------------------------------------------------------------------
// constructor: initializes data structures and parameters
ADM::ADM(MeshBlockPack *ppack, ParameterInput *pin):
    u_adm("u_adm",1,1,1,1,1),
    SetADMVariables(&ADM::SetADMVariablesToKerrSchild),
    pmy_pack(ppack) {
  is_dynamic = pin->GetOrAddBoolean("adm" , "dynamic", false);

  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto bounds = z4c::MakeStoredDomainBounds(indcs);

  if (pmy_pack->pz4c == nullptr) {
    Kokkos::realloc(u_adm, nmb, nadm, bounds.n3, bounds.n2, bounds.n1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_ALPHA);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_BETAX, I_ADM_BETAZ);
  } else if (pmy_pack->pz4c->layout.centering ==
             z4c::Z4cGridCentering::cell) {
    // Lapse and shift are stored in the Z4c class
    z4c::Z4c * pz4c = pmy_pack->pz4c;
    Kokkos::realloc(u_adm, nmb, nadm - 4, bounds.n3, bounds.n2, bounds.n1);
    adm.alpha.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_ALPHA);
    adm.beta_u.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_BETAX, pz4c->I_Z4C_BETAZ);
  } else {
    // VC Z4c owns a differently shaped native ADM cache.  This object remains
    // an independently allocated CC adapter for legacy cell-centred consumers.
    Kokkos::realloc(u_adm, nmb, nadm, bounds.n3, bounds.n2, bounds.n1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_ALPHA);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_BETAX, I_ADM_BETAZ);
  }
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_PSI4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_GXX, I_ADM_GZZ);
  adm.vK_dd.InitWithShallowSlice(u_adm, I_ADM_KXX, I_ADM_KZZ);
}

//----------------------------------------------------------------------------------------
// destructor
ADM::~ADM() {}

//----------------------------------------------------------------------------------------
void ADM::SetADMVariablesToKerrSchild(MeshBlockPack *pmbp) {
  Real a = pmbp->pcoord->coord_data.bh_spin;
  bool minkowski = pmbp->pcoord->coord_data.is_minkowski;
  auto &adm = pmbp->padm->adm;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmbp->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  const auto bounds = z4c::MakeStoredDomainBounds(indcs);
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,bounds.ks,bounds.ke,
          bounds.js,bounds.je,bounds.is,bounds.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    ComputeADMDecomposition(x1v, x2v, x3v, minkowski, a,
      &adm.alpha(m,k,j,i),
      &adm.beta_u(m,0,k,j,i), &adm.beta_u(m,1,k,j,i), &adm.beta_u(m,2,k,j,i),
      &adm.psi4(m,k,j,i),
      &adm.g_dd(m,0,0,k,j,i), &adm.g_dd(m,0,1,k,j,i), &adm.g_dd(m,0,2,k,j,i),
      &adm.g_dd(m,1,1,k,j,i), &adm.g_dd(m,1,2,k,j,i), &adm.g_dd(m,2,2,k,j,i),
      &adm.vK_dd(m,0,0,k,j,i), &adm.vK_dd(m,0,1,k,j,i), &adm.vK_dd(m,0,2,k,j,i),
      &adm.vK_dd(m,1,1,k,j,i), &adm.vK_dd(m,1,2,k,j,i), &adm.vK_dd(m,2,2,k,j,i));
  });
}


} // namespace adm
