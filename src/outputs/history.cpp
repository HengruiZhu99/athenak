//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their evolution.

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "pc_gh/pc_gh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// Constructor: also calls BaseTypeOutput base class constructor

HistoryOutput::HistoryOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // cycle through physics modules and add HistoryData struct for each
  hist_data.clear();

  if (pm->pgen->user_hist && op.user_hist_only) {
    hist_data.emplace_back(PhysicsModule::UserDefined);
  } else {
    if (pm->pmb_pack->phydro != nullptr) {
      hist_data.emplace_back(PhysicsModule::HydroDynamics);
    }
    if (pm->pmb_pack->pmhd != nullptr) {
      hist_data.emplace_back(PhysicsModule::MagnetoHydroDynamics);
    }
    if (pm->pgen->user_hist) {
      hist_data.emplace_back(PhysicsModule::UserDefined);
    }
  }

  if (pm->pmb_pack->pz4c != nullptr) {
    hist_data.emplace_back(PhysicsModule::SpaceTimeDynamics);
  }
  if (pm->pmb_pack->ppcgh != nullptr) {
    hist_data.emplace_back(PhysicsModule::PcGhDynamics);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadOutputData()
//  \brief Wrapper function that cycles through hist_data vector and calls
//  appropriate LoadXXXData() function for that physics

void HistoryOutput::LoadOutputData(Mesh *pm) {
  for (auto &data : hist_data) {
    if (data.physics == PhysicsModule::HydroDynamics) {
      LoadHydroHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::MagnetoHydroDynamics) {
      LoadMHDHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::SpaceTimeDynamics) {
      LoadZ4cHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::PcGhDynamics) {
      LoadPcGhHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::UserDefined) {
      (pm->pgen->user_hist_func)(&data, pm);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadHydroHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadHydroHistoryData(HistoryData *pdata, Mesh *pm) {
  auto &eos_data = pm->pmb_pack->phydro->peos->eos_data;
  int &nhydro_ = pm->pmb_pack->phydro->nhydro;

  // set number of and names of history variables for hydro
  if (eos_data.is_ideal) {
    pdata->nhist = 8;
  } else {
    pdata->nhist = 7;
  }
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  if (eos_data.is_ideal) {
    pdata->label[IEN] = "tot-E";
  }
  pdata->label[nhydro_  ] = "1-KE";
  pdata->label[nhydro_+1] = "2-KE";
  pdata->label[nhydro_+2] = "3-KE";

  // capture class variables for kernel
  auto &u0_ = pm->pmb_pack->phydro->u0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // Hydro conserved variables:
    array_sum::GlobalSum hvars;
    hvars.the_array[IDN] = vol*u0_(m,IDN,k,j,i);
    hvars.the_array[IM1] = vol*u0_(m,IM1,k,j,i);
    hvars.the_array[IM2] = vol*u0_(m,IM2,k,j,i);
    hvars.the_array[IM3] = vol*u0_(m,IM3,k,j,i);
    if (eos_data.is_ideal) {
      hvars.the_array[IEN] = vol*u0_(m,IEN,k,j,i);
    }

    // Hydro KE
    hvars.the_array[nhydro_  ] = vol*0.5*SQR(u0_(m,IM1,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nhydro_+1] = vol*0.5*SQR(u0_(m,IM2,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nhydro_+2] = vol*0.5*SQR(u0_(m,IM3,k,j,i))/u0_(m,IDN,k,j,i);

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  // store data into hdata array
  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadZ4cHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadZ4cHistoryData(HistoryData *pdata, Mesh *pm) {
  // set number of and names of history variables for z4c
  pdata->nhist = 9;
  pdata->label[0] = "C-norm2";
  pdata->label[1] = "H-norm2";
  pdata->label[2] = "M-norm2";
  pdata->label[3] = "Z-norm2";
  pdata->label[4] = "Mx-norm2";
  pdata->label[5] = "My-norm2";
  pdata->label[6] = "Mz-norm2";
  pdata->label[7] = "Theta-norm2";
  pdata->label[8] = "Volume";

  // capture class variabels for kernel
  auto &u0_ = pm->pmb_pack->pz4c->u0;
  auto &u_con_ = pm->pmb_pack->pz4c->u_con;
  const int &I_Z4c_Theta_ =  pm->pmb_pack->pz4c->I_Z4C_THETA;
  auto &z4c = pm->pmb_pack->pz4c->z4c;
  auto &adm = pm->pmb_pack->padm->adm;

  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;
  auto &opt = pm->pmb_pack->pz4c->opt;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real detg = adm::SpatialDet(adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
                                adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
                                adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3
               * std::sqrt(std::abs(detg));

    // Excise the punctures based on chi
    array_sum::GlobalSum hvars;
    if (z4c.chi(m,k,j,i)>=opt.excise_chi) {
      hvars.the_array[0] = vol*u_con_(m,0,k,j,i); // ||C||^2 (comes already squared)
      hvars.the_array[1] = vol*SQR(u_con_(m,1,k,j,i)); //||H||^2
      hvars.the_array[2] = vol*u_con_(m,2,k,j,i); // ||M||^2 (comes already squared)
      hvars.the_array[3] = vol*u_con_(m,3,k,j,i); // ||Z||^2 (comes already squared)
      hvars.the_array[4] = vol*SQR(u_con_(m,4,k,j,i));      // ||Mx||^2
      hvars.the_array[5] = vol*SQR(u_con_(m,5,k,j,i));      // ||My||^2
      hvars.the_array[6] = vol*SQR(u_con_(m,6,k,j,i));      // ||Mz||^2
      hvars.the_array[7] = vol*SQR(u0_(m,I_Z4c_Theta_,k,j,i)); // ||Theta||^2
      hvars.the_array[8] = vol;
    } else {
      hvars.the_array[0] = 0;
      hvars.the_array[1] = 0;
      hvars.the_array[2] = 0;
      hvars.the_array[3] = 0;
      hvars.the_array[4] = 0;
      hvars.the_array[5] = 0;
      hvars.the_array[6] = 0;
      hvars.the_array[7] = 0;
      hvars.the_array[8] = 0;
    }

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  // store data into hdata array
  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadPcGhHistoryData()
//! \brief Compute volume-weighted squared PC-GH diagnostic norms on this rank.

void HistoryOutput::LoadPcGhHistoryData(HistoryData *pdata, Mesh *pm) {
  // The canonical columns are the primary w^2-masked coordinate-volume sums.
  // Full-domain coordinate-volume sums remain available with the all-* prefix.
  pdata->nhist = 96;
  pdata->label[0] = "Cperp-n2";
  pdata->label[1] = "Z-norm2";
  pdata->label[2] = "H-norm2";
  pdata->label[3] = "Mhat-norm2";
  pdata->label[4] = "redw-norm2";
  pdata->label[5] = "redQ-norm2";
  pdata->label[6] = "reda-norm2";
  pdata->label[7] = "redB-norm2";
  pdata->label[8] = "curlp-n2";
  pdata->label[9] = "curlQ-n2";
  pdata->label[10] = "curlL-n2";
  pdata->label[11] = "curlB-n2";
  pdata->label[12] = "detg-norm2";
  pdata->label[13] = "trA-norm2";
  pdata->label[14] = "trQ-norm2";
  pdata->label[15] = "proj-norm2";
  pdata->label[16] = "p-norm2";
  pdata->label[17] = "L-norm2";
  pdata->label[18] = "rhs-norm2";
  pdata->label[19] = "Volume";

  char const * const full_labels[20] = {
    "all-Cp2", "all-Z2", "all-H2", "all-M2",
    "all-rw2", "all-rQ2", "all-ra2", "all-rB2",
    "all-cp2", "all-cQ2", "all-cL2", "all-cB2",
    "all-det2", "all-trA2", "all-trQ2", "all-prj2",
    "all-p2", "all-L2", "all-rhs2", "all-Vol",
  };
  for (int n = 0; n < 20; ++n) pdata->label[20 + n] = full_labels[n];

  char const * const local_names[14] = {
    "Cp2", "Z2", "H2", "M2", "rw2", "rQ2", "ra2", "rB2",
    "cp2", "cQ2", "cL2", "cB2", "alg2", "Vol",
  };
  char const * const region_names[4] = {"r05", "r1", "r2", "ah"};
  for (int region = 0; region < 4; ++region) {
    for (int quantity = 0; quantity < 14; ++quantity) {
      pdata->label[40 + 14*region + quantity] =
          std::string(region_names[region]) + "-" + local_names[quantity];
    }
  }

  auto &con = pm->pmb_pack->ppcgh->u_con;
  auto &state = pm->pmb_pack->ppcgh->u0;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int const is = indcs.is;
  int const js = indcs.js;
  int const ks = indcs.ks;
  int const nx1 = indcs.nx1;
  int const nx2 = indcs.nx2;
  int const nx3 = indcs.nx3;
  int const nkji = nx3*nx2*nx1;
  int const nji = nx2*nx1;
  int const nmkji = pm->pmb_pack->nmb_thispack*nkji;
  Real const center_x = pm->pmb_pack->ppcgh->opt.gauge_center[0];
  Real const center_y = pm->pmb_pack->ppcgh->opt.gauge_center[1];
  Real const center_z = pm->pmb_pack->ppcgh->opt.gauge_center[2];
  Real const mass = pm->pmb_pack->ppcgh->opt.gauge_mass;
  Real const excise_chi = pm->pmb_pack->ppcgh->opt.constraint_excise_chi;
  bool const exterior_horizon =
      pm->pmb_pack->ppcgh->opt.constraint_exterior_horizon;
  Real const horizon_cut = pm->pmb_pack->ppcgh->opt.constraint_horizon_radius
                           + pm->pmb_pack->ppcgh->opt.constraint_horizon_buffer;

  array_sum::GlobalSum sum_this_rank;
  Kokkos::parallel_reduce("PC-GH history sums",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(int idx, array_sum::GlobalSum &sum) {
    int const m = idx/nkji;
    int const k0 = (idx - m*nkji)/nji;
    int const j0 = (idx - m*nkji - k0*nji)/nx1;
    int const i0 = idx - m*nkji - k0*nji - j0*nx1;
    int const k = k0 + ks;
    int const j = j0 + js;
    int const i = i0 + is;
    Real const coordinate_volume = size.d_view(m).dx1*size.d_view(m).dx2
                                   *size.d_view(m).dx3;
    Real const x = CellCenterX(i0, nx1, size.d_view(m).x1min,
                               size.d_view(m).x1max) - center_x;
    Real const y = CellCenterX(j0, nx2, size.d_view(m).x2min,
                               size.d_view(m).x2max) - center_y;
    Real const z = CellCenterX(k0, nx3, size.d_view(m).x3min,
                               size.d_view(m).x3max) - center_z;
    Real const radius = std::sqrt(x*x + y*y + z*z);
    bool const include_primary = SQR(state(m, pc_gh::PcGh::I_W, k, j, i))
                                 >= excise_chi;
    bool const include_region[4] = {
      radius > 0.5*mass,
      radius > mass,
      radius > 2.0*mass,
      exterior_horizon && radius > horizon_cut,
    };

    Real values[19];
    values[0] = SQR(con(m, pc_gh::PcGh::I_CON_CPERP, k, j, i));
    values[1] = SQR(con(m, pc_gh::PcGh::I_CON_ZX, k, j, i))
                + SQR(con(m, pc_gh::PcGh::I_CON_ZY, k, j, i))
                + SQR(con(m, pc_gh::PcGh::I_CON_ZZ, k, j, i));
    values[2] = SQR(con(m, pc_gh::PcGh::I_CON_H, k, j, i));
    values[3] = SQR(con(m, pc_gh::PcGh::I_CON_MX, k, j, i))
                + SQR(con(m, pc_gh::PcGh::I_CON_MY, k, j, i))
                + SQR(con(m, pc_gh::PcGh::I_CON_MZ, k, j, i));
    values[4] = SQR(con(m, pc_gh::PcGh::I_CON_RED_W, k, j, i));
    values[5] = SQR(con(m, pc_gh::PcGh::I_CON_RED_Q, k, j, i));
    values[6] = SQR(con(m, pc_gh::PcGh::I_CON_RED_ALPHA, k, j, i));
    values[7] = SQR(con(m, pc_gh::PcGh::I_CON_RED_B, k, j, i));
    values[8] = SQR(con(m, pc_gh::PcGh::I_CON_CURL_P, k, j, i));
    values[9] = SQR(con(m, pc_gh::PcGh::I_CON_CURL_Q, k, j, i));
    values[10] = SQR(con(m, pc_gh::PcGh::I_CON_CURL_L, k, j, i));
    values[11] = SQR(con(m, pc_gh::PcGh::I_CON_CURL_B, k, j, i));
    values[12] = SQR(con(m, pc_gh::PcGh::I_CON_DETG, k, j, i));
    values[13] = SQR(con(m, pc_gh::PcGh::I_CON_TRA, k, j, i));
    values[14] = SQR(con(m, pc_gh::PcGh::I_CON_TRQ, k, j, i));
    values[15] = SQR(con(m, pc_gh::PcGh::I_CON_PROJECTION, k, j, i));
    values[16] = SQR(con(m, pc_gh::PcGh::I_CON_P, k, j, i));
    values[17] = SQR(con(m, pc_gh::PcGh::I_CON_L, k, j, i));
    values[18] = SQR(con(m, pc_gh::PcGh::I_CON_RHS_PRIMARY, k, j, i))
                 + SQR(con(m, pc_gh::PcGh::I_CON_RHS_GRADIENT, k, j, i));

    array_sum::GlobalSum h;
    for (int quantity = 0; quantity < 19; ++quantity) {
      h.the_array[quantity] = include_primary
          ? coordinate_volume*values[quantity] : 0.0;
      h.the_array[20 + quantity] = coordinate_volume*values[quantity];
    }
    h.the_array[19] = include_primary ? coordinate_volume : 0.0;
    h.the_array[39] = coordinate_volume;

    Real const localized_values[13] = {
      values[0], values[1], values[2], values[3],
      values[4], values[5], values[6], values[7],
      values[8], values[9], values[10], values[11],
      values[12] + values[13] + values[14] + values[15],
    };
    for (int region = 0; region < 4; ++region) {
      int const base = 40 + 14*region;
      for (int quantity = 0; quantity < 13; ++quantity) {
        h.the_array[base + quantity] = include_region[region]
            ? coordinate_volume*localized_values[quantity] : 0.0;
      }
      h.the_array[base + 13] = include_region[region] ? coordinate_volume : 0.0;
    }
    sum += h;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_rank));

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_rank.the_array[n];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::LoadMHDHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadMHDHistoryData(HistoryData *pdata, Mesh *pm) {
  auto &eos_data = pm->pmb_pack->pmhd->peos->eos_data;
  int &nmhd_ = pm->pmb_pack->pmhd->nmhd;

  // set number of and names of history variables for mhd
  if (eos_data.is_ideal) {
    pdata->nhist = 11;
  } else {
    pdata->nhist = 10;
  }
  pdata->label[IDN] = "mass";
  pdata->label[IM1] = "1-mom";
  pdata->label[IM2] = "2-mom";
  pdata->label[IM3] = "3-mom";
  if (eos_data.is_ideal) {
    pdata->label[IEN] = "tot-E";
  }
  pdata->label[nmhd_  ] = "1-KE";
  pdata->label[nmhd_+1] = "2-KE";
  pdata->label[nmhd_+2] = "3-KE";
  pdata->label[nmhd_+3] = "1-ME";
  pdata->label[nmhd_+4] = "2-ME";
  pdata->label[nmhd_+5] = "3-ME";

  // capture class variabels for kernel
  auto &u0_ = pm->pmb_pack->pmhd->u0;
  auto &bx1f = pm->pmb_pack->pmhd->b0.x1f;
  auto &bx2f = pm->pmb_pack->pmhd->b0.x2f;
  auto &bx3f = pm->pmb_pack->pmhd->b0.x3f;
  auto &size = pm->pmb_pack->pmb->mb_size;
  int &nhist_ = pdata->nhist;

  // loop over all MeshBlocks in this pack
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;
  int is = indcs.is; int nx1 = indcs.nx1;
  int js = indcs.js; int nx2 = indcs.nx2;
  int ks = indcs.ks; int nx3 = indcs.nx3;
  const int nmkji = (pm->pmb_pack->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  array_sum::GlobalSum sum_this_mb;
  Kokkos::parallel_reduce("HistSums",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, array_sum::GlobalSum &mb_sum) {
    // compute n,k,j,i indices of thread
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;

    Real vol = size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;

    // MHD conserved variables:
    array_sum::GlobalSum hvars;
    hvars.the_array[IDN] = vol*u0_(m,IDN,k,j,i);
    hvars.the_array[IM1] = vol*u0_(m,IM1,k,j,i);
    hvars.the_array[IM2] = vol*u0_(m,IM2,k,j,i);
    hvars.the_array[IM3] = vol*u0_(m,IM3,k,j,i);
    if (eos_data.is_ideal) {
      hvars.the_array[IEN] = vol*u0_(m,IEN,k,j,i);
    }

    // MHD KE
    hvars.the_array[nmhd_  ] = vol*0.5*SQR(u0_(m,IM1,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+1] = vol*0.5*SQR(u0_(m,IM2,k,j,i))/u0_(m,IDN,k,j,i);
    hvars.the_array[nmhd_+2] = vol*0.5*SQR(u0_(m,IM3,k,j,i))/u0_(m,IDN,k,j,i);

    // MHD ME
    hvars.the_array[nmhd_+3] = vol*0.25*(SQR(bx1f(m,k,j,i+1)) + SQR(bx1f(m,k,j,i)));
    hvars.the_array[nmhd_+4] = vol*0.25*(SQR(bx2f(m,k,j+1,i)) + SQR(bx2f(m,k,j,i)));
    hvars.the_array[nmhd_+5] = vol*0.25*(SQR(bx3f(m,k+1,j,i)) + SQR(bx3f(m,k,j,i)));

    // fill rest of the_array with zeros, if nhist < NHISTORY_VARIABLES
    for (int n=nhist_; n<NHISTORY_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));
  Kokkos::fence();

  // store data into hdata array
  for (int n=0; n<pdata->nhist; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void HistoryOutput::WriteOutputFile()
//  \brief Cycles through hist_data vector and writes history file for each component

void HistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  for (auto &data : hist_data) {
    // first, perform in-place sum over all MPI ranks
#if MPI_PARALLEL_ENABLED
    if (global_variable::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, &(data.hdata[0]), data.nhist, MPI_ATHENA_REAL,
         MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(&(data.hdata[0]), &(data.hdata[0]), data.nhist,
         MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    }
#endif

    // only the master rank writes the file
    if (global_variable::my_rank == 0) {
      // create filename: "file_basename" + ".physics" + ".hst"
      // There is no file number or id in history output filenames.
      std::string fname;
      fname.assign(out_params.file_basename);
      switch (data.physics) {
        case PhysicsModule::HydroDynamics:
          fname.append(".hydro");
          break;
        case PhysicsModule::MagnetoHydroDynamics:
          fname.append(".mhd");
          break;
        case PhysicsModule::PcGhDynamics:
          fname.append(".pcgh");
          break;
        case PhysicsModule::SpaceTimeDynamics:
          fname.append(".z4c");
        case PhysicsModule::UserDefined:
          fname.append(".user");
          break;
        default:
          break;
      }
      fname.append(".hst");

      // open file for output
      FILE *pfile;
      if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
        exit(EXIT_FAILURE);
      }

      // Write header, if it has not been written already
      if (!(data.header_written)) {
        int iout = 1;
        std::fprintf(pfile,"# Athena++ history data\n");
        std::fprintf(pfile,"#  [%d]=time      ", iout++);
        std::fprintf(pfile,"[%d]=dt       ", iout++);
        for (int n=0; n<data.nhist; ++n) {
          std::fprintf(pfile,"[%d]=%.10s    ", iout++, data.label[n].c_str());
        }
        std::fprintf(pfile,"\n");                              // terminate line
        data.header_written = true;
      }

      // write history variables
      std::fprintf(pfile, out_params.data_format.c_str(), pm->time);
      std::fprintf(pfile, out_params.data_format.c_str(), pm->dt);
      for (int n=0; n<data.nhist; ++n)
        std::fprintf(pfile, out_params.data_format.c_str(), data.hdata[n]);
      std::fprintf(pfile,"\n"); // terminate line
      std::fclose(pfile);
    }
  } // End loop over hist_data vector

  // increment counters, clean up
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
