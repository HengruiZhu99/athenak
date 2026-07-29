//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their evolution.

#include <cmath>
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
#include "scalar_field/scalar_field.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "outputs.hpp"

namespace {

template <int NGHOST>
void LoadScalarHistoryWithStencil(HistoryData *pdata, Mesh *pm) {
  auto &scalar = *(pm->pmb_pack->pscalar);
  auto &state = scalar.u0;
  auto &adm_vars = pm->pmb_pack->padm->adm;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &indcs = pm->pmb_pack->pmesh->mb_indcs;

  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const int ndim = 1 + static_cast<int>(pm->multi_d) +
                   static_cast<int>(pm->three_d);
  const int ncomponents = scalar.ncomponents;
  const int nhist = pdata->nhist;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  const int nmkji = nmb*nkji;
  const scalar_field::PotentialData potential = scalar.potential;
  const bool use_excision = scalar.excision;
  auto &excision_mask = pm->pmb_pack->pcoord->excision_floor;

  array_sum::GlobalSum scalar_sum;
  Kokkos::parallel_reduce(
      "scalar history sums",
      Kokkos::RangePolicy<DevExeSpace>(0, nmkji),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &sum) {
        const int m = idx/nkji;
        const int k_offset = (idx - m*nkji)/nji;
        const int j_offset = (idx - m*nkji - k_offset*nji)/nx1;
        const int i_offset =
            idx - m*nkji - k_offset*nji - j_offset*nx1;
        const int k = k_offset + ks;
        const int j = j_offset + js;
        const int i = i_offset + is;

        if (use_excision && excision_mask(m, k, j, i)) {
          array_sum::GlobalSum point;
          for (int n = 0; n < NHISTORY_VARIABLES; ++n) {
            point.the_array[n] = 0.0;
          }
          sum += point;
          return;
        }

        const Real inverse_dx[3] = {
          Real(1.0)/size.d_view(m).dx1,
          Real(1.0)/size.d_view(m).dx2,
          Real(1.0)/size.d_view(m).dx3
        };
        Real phi[2] = {0.0, 0.0};
        Real pi[2] = {0.0, 0.0};
        Real gradient[2][3] = {
          {0.0, 0.0, 0.0},
          {0.0, 0.0, 0.0}
        };
        for (int component = 0; component < ncomponents; ++component) {
          const int iphi = 2*component;
          phi[component] = state(m, iphi, k, j, i);
          pi[component] = state(m, iphi + 1, k, j, i);
          for (int direction = 0; direction < ndim; ++direction) {
            gradient[component][direction] =
                Dx<NGHOST>(
                    direction, inverse_dx, state, m, iphi, k, j, i);
          }
        }

        const Real metric[6] = {
          adm_vars.g_dd(m, 0, 0, k, j, i),
          adm_vars.g_dd(m, 0, 1, k, j, i),
          adm_vars.g_dd(m, 0, 2, k, j, i),
          adm_vars.g_dd(m, 1, 1, k, j, i),
          adm_vars.g_dd(m, 1, 2, k, j, i),
          adm_vars.g_dd(m, 2, 2, k, j, i)
        };
        const scalar_field::MatterPoint matter =
            scalar_field::ComputeMatter(
                ncomponents, phi, pi, gradient, metric, potential);
        Real inverse_metric[6];
        Real determinant;
        scalar_field::InvertMetric(metric, inverse_metric, &determinant);
        const Real proper_volume =
            size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3 *
            sqrt(fabs(determinant));

        array_sum::GlobalSum point;
        point.the_array[0] = proper_volume*matter.energy;
        if (ncomponents == 2) {
          point.the_array[1] = proper_volume*matter.charge;
        }
        for (int n = nhist; n < NHISTORY_VARIABLES; ++n) {
          point.the_array[n] = 0.0;
        }
        sum += point;
      },
      Kokkos::Sum<array_sum::GlobalSum>(scalar_sum));

  for (int n = 0; n < pdata->nhist; ++n) {
    pdata->hdata[n] = scalar_sum.the_array[n];
  }
}

}  // namespace

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
    if (pm->pmb_pack->pscalar != nullptr) {
      hist_data.emplace_back(PhysicsModule::ScalarFieldDynamics);
    }
    if (pm->pgen->user_hist) {
      hist_data.emplace_back(PhysicsModule::UserDefined);
    }
  }

  if (pm->pmb_pack->pz4c != nullptr) {
    hist_data.emplace_back(PhysicsModule::SpaceTimeDynamics);
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
    } else if (data.physics == PhysicsModule::ScalarFieldDynamics) {
      LoadScalarHistoryData(&data, pm);
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
  int &nscalars_ = pm->pmb_pack->phydro->nscalars;

  // set number of and names of history variables for hydro
  if (eos_data.is_ideal) {
    pdata->nhist = 8;
  } else {
    pdata->nhist = 7;
  }
  if (nscalars_>0) {
    pdata->nhist += nscalars_;
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
  for (int s=0; s<nscalars_; ++s) {
    std::ostringstream labelSS;
    labelSS << "scal-" << s;
    pdata->label[nhydro_+3+s] = labelSS.str();
  }

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

    // Scalar masses
    for (int s=0; s<nscalars_; ++s) {
      hvars.the_array[nhydro_+3+s] = vol*u0_(m,nhydro_+s,k,j,i);
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
//! \fn void HistoryOutput::LoadScalarHistoryData()
//  \brief Integrate scalar Eulerian energy and complex Noether charge with proper volume.

void HistoryOutput::LoadScalarHistoryData(HistoryData *pdata, Mesh *pm) {
  const bool is_complex = pm->pmb_pack->pscalar->ncomponents == 2;
  pdata->nhist = is_complex ? 2 : 1;
  pdata->label[0] = "sf-energy";
  if (is_complex) {
    pdata->label[1] = "sf-charge";
  }

  const int fd_stencil = pm->pmb_pack->pscalar->fd_stencil;
  if (fd_stencil == 2) {
    LoadScalarHistoryWithStencil<2>(pdata, pm);
  } else if (fd_stencil == 3) {
    LoadScalarHistoryWithStencil<3>(pdata, pm);
  } else if (fd_stencil == 4) {
    LoadScalarHistoryWithStencil<4>(pdata, pm);
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
              << __LINE__ << std::endl
              << "Scalar history received an unsupported finite-difference stencil."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
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
//! \fn void HistoryOutput::LoadMHDHistoryData()
//  \brief Compute and store history data over all MeshBlocks on this rank
//  Data is stored in a Real array defined in derived class.

void HistoryOutput::LoadMHDHistoryData(HistoryData *pdata, Mesh *pm) {
  auto &eos_data = pm->pmb_pack->pmhd->peos->eos_data;
  int &nmhd_ = pm->pmb_pack->pmhd->nmhd;
  int &nscalars_ = pm->pmb_pack->pmhd->nscalars;

  // set number of and names of history variables for mhd
  if (eos_data.is_ideal) {
    pdata->nhist = 11;
  } else {
    pdata->nhist = 10;
  }
  if (nscalars_>0) {
    pdata->nhist += nscalars_;
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

  for (int s=0; s<nscalars_; ++s) {
    std::ostringstream labelSS;
    labelSS << "scal-" << s;
    pdata->label[nmhd_+6+s] = labelSS.str();
  }

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

    // Scalar masses
    for (int s=0; s<nscalars_; ++s) {
      hvars.the_array[nmhd_+6+s] = vol*u0_(m,nmhd_+s,k,j,i);
    }

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
        case PhysicsModule::ScalarFieldDynamics:
          fname.append(".scalar");
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
