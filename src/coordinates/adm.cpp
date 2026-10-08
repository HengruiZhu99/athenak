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
  separate_z4c_gauge = pin->GetOrAddBoolean("adm", "separate_z4c_gauge", false);
  gauge_is_shared = pmy_pack->pz4c != nullptr;

  int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;

  if (pmy_pack->pz4c == nullptr) {
    Kokkos::realloc(u_adm, nmb, nadm, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(u_adm, I_ADM_ALPHA);
    adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_BETAX, I_ADM_BETAZ);
  } else {
    // Lapse and shift are stored in the Z4c class
    z4c::Z4c * pz4c = pmy_pack->pz4c;
    Kokkos::realloc(u_adm, nmb, nadm - 4, ncells3, ncells2, ncells1);
    adm.alpha.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_ALPHA);
    adm.beta_u.InitWithShallowSlice(pz4c->u0, pz4c->I_Z4C_BETAX, pz4c->I_Z4C_BETAZ);
  }
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_PSI4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_GXX, I_ADM_GZZ);
  adm.vK_dd.InitWithShallowSlice(u_adm, I_ADM_KXX, I_ADM_KZZ);
}

// Detach after initial data import, preserving the current aliased gauge. Many
// importers populate Z4c gauge directly, so constructor-time separation would
// lose their initial lapse/shift. Subsequent physical conversions own their gauge.
void ADM::EnsureSeparateGaugeStorage() {
  if (!separate_z4c_gauge || !gauge_is_shared) return;
  const int nmb = u_adm.extent_int(0), nk = u_adm.extent_int(2);
  const int nj = u_adm.extent_int(3), ni = u_adm.extent_int(4);
  const auto previous = u_adm;
  const auto evolved = pmy_pack->pz4c->u0;
  DvceArray5D<Real> expanded("independent ADM gauge", nmb, nadm, nk, nj, ni);
  par_for("detach ADM gauge", DevExeSpace(), 0,nmb-1,0,nk-1,0,nj-1,0,ni-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int v = 0; v < I_ADM_ALPHA; ++v) expanded(m,v,k,j,i) = previous(m,v,k,j,i);
    expanded(m,I_ADM_ALPHA,k,j,i) = evolved(m,z4c::Z4c::I_Z4C_ALPHA,k,j,i);
    for (int a = 0; a < 3; ++a) {
      expanded(m,I_ADM_BETAX+a,k,j,i) = evolved(m,z4c::Z4c::I_Z4C_BETAX+a,k,j,i);
    }
  });
  Kokkos::fence();
  u_adm = expanded;
  adm.alpha.InitWithShallowSlice(u_adm, I_ADM_ALPHA);
  adm.beta_u.InitWithShallowSlice(u_adm, I_ADM_BETAX, I_ADM_BETAZ);
  adm.psi4.InitWithShallowSlice(u_adm, I_ADM_PSI4);
  adm.g_dd.InitWithShallowSlice(u_adm, I_ADM_GXX, I_ADM_GZZ);
  adm.vK_dd.InitWithShallowSlice(u_adm, I_ADM_KXX, I_ADM_KZZ);
  gauge_is_shared = false;
}

// Identity gauge map for the current Cauchy runtime. A future conformal runtime
// must instead write alpha_phys=alpha_bar/Omega using its own conversion.
void ADM::SyncCauchyGaugeFromZ4c() {
  if (pmy_pack->pz4c != nullptr && pmy_pack->pz4c->hyperboloidal_patch) {
    pmy_pack->pz4c->HyperboloidalADM();
    return;
  }
  if (gauge_is_shared || pmy_pack->pz4c == nullptr) return;
  const int nmb = pmy_pack->nmb_thispack;
  const int nk = u_adm.extent_int(2), nj = u_adm.extent_int(3), ni = u_adm.extent_int(4);
  const auto dst = u_adm, src = pmy_pack->pz4c->u0;
  par_for("copy Cauchy ADM gauge", DevExeSpace(), 0,nmb-1,0,nk-1,0,nj-1,0,ni-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    dst(m,I_ADM_ALPHA,k,j,i) = src(m,z4c::Z4c::I_Z4C_ALPHA,k,j,i);
    for (int a = 0; a < 3; ++a) {
      dst(m,I_ADM_BETAX+a,k,j,i) = src(m,z4c::Z4c::I_Z4C_BETAX+a,k,j,i);
    }
  });
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
  int &ng = indcs.ng;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nmb = pmbp->nmb_thispack;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
  par_for("update_adm_vars", DevExeSpace(), 0,nmb-1,0,(n3-1),0,(n2-1),0,(n1-1),
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
