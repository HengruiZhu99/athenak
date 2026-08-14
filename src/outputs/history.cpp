//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file history.cpp
//  \brief writes history output data, volume-averaged quantities that are output
//         frequently in time to trace their evolution.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "z4c/cartoon_meridional_sampler.hpp"
#include "z4c/curvature_diagnostics.hpp"
#include "z4c/fastflow.hpp"
#include "z4c/z4c.hpp"
#include "z4c/z4c_symmetry.hpp"
#include "coordinates/adm.hpp"
#include "outputs.hpp"

namespace {

constexpr int kCartoonConstraintFamilies = 4;
constexpr int kCartoonAxisLayers = 5;
constexpr int kCartoonRegionStride = kCartoonConstraintFamilies + 1;
constexpr int kCartoonAxisSumBase = 0;
constexpr int kCartoonOffAxisSumBase = kCartoonAxisSumBase + kCartoonRegionStride;
constexpr int kCartoonLayerSumBase = kCartoonOffAxisSumBase + kCartoonRegionStride;
constexpr int kCartoonDiagnosticSums =
    kCartoonLayerSumBase + kCartoonAxisLayers * kCartoonRegionStride;

struct ConstraintMaximum {
  Real value = 0.0;
  Real rho = 0.0;
  Real z = 0.0;
};

template <int FAMILY, typename ConstraintView, typename ChiView,
          typename MeshBlockSizeDualView>
ConstraintMaximum CartoonConstraintMaximum(
    const ConstraintView &constraints, const ChiView &chi,
    MeshBlockSizeDualView &size, const Real excise_chi,
    const int nmb, const int nx1, const int nx2, const int nx3,
    const int is, const int js, const int ks) {
  static_assert(FAMILY >= 0 && FAMILY < kCartoonConstraintFamilies);
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const int nmkji = nmb * nkji;
  typename Kokkos::MaxLoc<Real, int>::value_type local_maximum;
  Kokkos::parallel_reduce(
      "Cartoon constraint Linf location",
      Kokkos::RangePolicy<DevExeSpace>(0, nmkji),
      KOKKOS_LAMBDA(
          const int idx,
          typename Kokkos::MaxLoc<Real, int>::value_type &maximum) {
        const int m = idx / nkji;
        const int k0 = (idx - m * nkji) / nji;
        const int j0 = (idx - m * nkji - k0 * nji) / nx1;
        const int i0 = idx - m * nkji - k0 * nji - j0 * nx1;
        const int i = i0 + is;
        const int j = j0 + js;
        const int k = k0 + ks;
        if (chi(m, k, j, i) < excise_chi) return;
        const Real raw = constraints(m, FAMILY, k, j, i);
        Real magnitude = 0.0;
        if constexpr (FAMILY == 1) {
          magnitude = Kokkos::fabs(raw);
        } else {
          magnitude = (Kokkos::isfinite(raw) && raw >= 0.0)
                          ? Kokkos::sqrt(raw)
                          : std::numeric_limits<Real>::infinity();
        }
        if (!Kokkos::isfinite(magnitude)) {
          magnitude = std::numeric_limits<Real>::infinity();
        }
        if (magnitude > maximum.val) {
          maximum.val = magnitude;
          maximum.loc = idx;
        }
      },
      Kokkos::MaxLoc<Real, int>(local_maximum));

  const bool local_valid = local_maximum.loc >= 0 && local_maximum.loc < nmkji;
  Real global_maximum = local_valid ? local_maximum.val : -1.0;
  int owner_rank = global_variable::my_rank;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &global_maximum, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
  int candidate_rank =
      (local_valid && local_maximum.val == global_maximum)
          ? global_variable::my_rank
          : std::numeric_limits<int>::max();
  MPI_Allreduce(&candidate_rank, &owner_rank, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
#endif
  ConstraintMaximum result;
  if (global_maximum < 0.0 || owner_rank == std::numeric_limits<int>::max()) {
    return result;
  }
  result.value = global_maximum;
  Real position[2] = {0.0, 0.0};
  if (global_variable::my_rank == owner_rank) {
    const int m = local_maximum.loc / nkji;
    const int k0 = (local_maximum.loc - m * nkji) / nji;
    const int j0 = (local_maximum.loc - m * nkji - k0 * nji) / nx1;
    const int i0 = local_maximum.loc - m * nkji - k0 * nji - j0 * nx1;
    size.sync_host();
    position[0] = size.h_view(m).x1min + (i0 + 0.5) * size.h_view(m).dx1;
    position[1] = size.h_view(m).x2min + (j0 + 0.5) * size.h_view(m).dx2;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(position, 2, MPI_ATHENA_REAL, owner_rank, MPI_COMM_WORLD);
#endif
  result.rho = position[0];
  result.z = position[1];
  return result;
}

template <typename Symmetry, int NGHOST>
Real Z4cHistoryMaxKretschmann(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nmkji = pm->pmb_pack->nmb_thispack * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  auto &adm = pm->pmb_pack->padm->adm;
  auto &size = pm->pmb_pack->pmb->mb_size;
  Real maximum = 0.0;
  Kokkos::parallel_reduce(
      "Z4cHistoryMaxKretschmann",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &rank_maximum) {
        const int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        const int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;
        const Real inverse_spacing[3] = {
            1.0 / size.d_view(m).dx1,
            1.0 / size.d_view(m).dx2,
            1.0 / size.d_view(m).dx3};
        auto derivatives = z4c::MakeCellCenteredDerivativeProvider<Symmetry, NGHOST>(
            inverse_spacing, size.d_view, nx1, is, m, k, j, i);
        const auto diagnostic = ComputeZ4cCurvatureDiagnostics<NGHOST, false>(
            derivatives, adm.g_dd, adm.vK_dd, m, k, j, i);
        rank_maximum = diagnostic.valid
                           ? fmax(rank_maximum, fabs(diagnostic.kretschmann))
                           : std::numeric_limits<Real>::infinity();
      },
      Kokkos::Max<Real>(maximum));
  return maximum;
}

Real DispatchZ4cHistoryMaxKretschmann(Mesh *pm) {
  const auto &config = pm->pmb_pack->z4c_symmetry;
  const bool cartoon = config.mode == z4c::Z4cSymmetryMode::cartoon_so2;
  switch (config.stencil_width) {
    case 2:
      return cartoon ? Z4cHistoryMaxKretschmann<z4c::CartoonSO2, 2>(pm)
                     : Z4cHistoryMaxKretschmann<z4c::Cartesian3D, 2>(pm);
    case 3:
      return cartoon ? Z4cHistoryMaxKretschmann<z4c::CartoonSO2, 3>(pm)
                     : Z4cHistoryMaxKretschmann<z4c::Cartesian3D, 3>(pm);
    case 4:
      return cartoon ? Z4cHistoryMaxKretschmann<z4c::CartoonSO2, 4>(pm)
                     : Z4cHistoryMaxKretschmann<z4c::Cartesian3D, 4>(pm);
    default:
      return std::numeric_limits<Real>::infinity();
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

    // Fill the fixed-width legacy reducer, not the larger HistoryData storage.
    for (int n=nhist_; n<NREDUCTION_VARIABLES; ++n) {
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
  auto &opt = pm->pmb_pack->pz4c->opt;
  const bool cartoon =
      pm->pmb_pack->z4c_symmetry.mode == z4c::Z4cSymmetryMode::cartoon_so2;
  // set number of and names of history variables for z4c
  const int kretschmann_index = opt.history_kretschmann ? 11 : -1;
  const int max_refinement_level_index = opt.history_kretschmann ? 12 : 11;
  const int max_meshblocks_per_rank_index = max_refinement_level_index + 1;
  const int horizon_status_index = max_meshblocks_per_rank_index + 1;
  const int horizon_last_search_cycle_index = horizon_status_index + 1;
  const int cycle_index = horizon_last_search_cycle_index + 1;
  const int central_lapse_index = cartoon ? cycle_index + 1 : -1;
  const int central_proper_time_index = cartoon ? central_lapse_index + 1 : -1;
  const int central_kretschmann_index = cartoon ? central_proper_time_index + 1 : -1;
  const int cartoon_axis_index = cartoon ? central_kretschmann_index + 1 : -1;
  const int cartoon_off_axis_index =
      cartoon ? cartoon_axis_index + kCartoonRegionStride : -1;
  const int cartoon_layer_index =
      cartoon ? cartoon_off_axis_index + kCartoonRegionStride : -1;
  const int cartoon_linf_index =
      cartoon ? cartoon_layer_index +
                    kCartoonAxisLayers * kCartoonRegionStride : -1;
  const int base_history_count =
      cartoon ? cartoon_linf_index + 3 * kCartoonConstraintFamilies
              : cycle_index + 1;
  const int telegraph_mu_min_index = opt.telegraph_lapse ? base_history_count : -1;
  const int telegraph_mu_max_index = opt.telegraph_lapse ? base_history_count + 1 : -1;
  pdata->nhist = base_history_count + (opt.telegraph_lapse ? 2 : 0);
  if (pdata->nhist > NHISTORY_VARIABLES) {
    std::cerr << "### FATAL ERROR in " << __FILE__
              << ": Cartoon history inventory exceeds NHISTORY_VARIABLES"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pdata->label[0] = "C-norm2";
  pdata->label[1] = "H-norm2";
  pdata->label[2] = "M-norm2";
  pdata->label[3] = "Z-norm2";
  pdata->label[4] = "Mx-norm2";
  pdata->label[5] = "My-norm2";
  pdata->label[6] = "Mz-norm2";
  pdata->label[7] = "Theta-norm2";
  pdata->label[8] = "Volume";
  pdata->label[9] = "max_abs_K";
  pdata->label[10] = "nmb_total";
  pdata->reduction[9] = HistoryData::Reduction::max;
  pdata->reduction[10] = HistoryData::Reduction::max;
  if (opt.history_kretschmann) {
    // History headers retain ten characters, so keep this stable short label.
    pdata->label[kretschmann_index] = "maxAbsKret";
    pdata->reduction[kretschmann_index] = HistoryData::Reduction::max;
  }
  pdata->label[max_refinement_level_index] = "maxRefLev";
  pdata->label[max_meshblocks_per_rank_index] = "maxNmbRank";
  pdata->label[horizon_status_index] = "ahStatus";
  pdata->label[horizon_last_search_cycle_index] = "ahLastCyc";
  pdata->label[cycle_index] = "cycle";
  pdata->reduction[max_refinement_level_index] = HistoryData::Reduction::max;
  pdata->reduction[max_meshblocks_per_rank_index] = HistoryData::Reduction::max;
  pdata->reduction[horizon_status_index] = HistoryData::Reduction::max;
  pdata->reduction[horizon_last_search_cycle_index] = HistoryData::Reduction::max;
  pdata->reduction[cycle_index] = HistoryData::Reduction::max;
  if (opt.telegraph_lapse) {
    pdata->label[telegraph_mu_min_index] = "muMin";
    pdata->label[telegraph_mu_max_index] = "muMax";
    pdata->reduction[telegraph_mu_min_index] = HistoryData::Reduction::min;
    pdata->reduction[telegraph_mu_max_index] = HistoryData::Reduction::max;
  }
  if (cartoon) {
    pdata->label[central_lapse_index] = "axisLapse";
    pdata->label[central_proper_time_index] = "axisTau";
    pdata->label[central_kretschmann_index] = "axisKret";
    pdata->reduction[central_lapse_index] = HistoryData::Reduction::max;
    pdata->reduction[central_proper_time_index] = HistoryData::Reduction::max;
    pdata->reduction[central_kretschmann_index] = HistoryData::Reduction::max;

    constexpr std::array<const char *, kCartoonConstraintFamilies> names = {
        "C", "H", "M", "Z"};
    for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
      pdata->label[cartoon_axis_index + family] =
          std::string("ax-") + names[family] + "2";
      pdata->label[cartoon_off_axis_index + family] =
          std::string("off-") + names[family] + "2";
    }
    pdata->label[cartoon_axis_index + kCartoonConstraintFamilies] = "ax-N";
    pdata->label[cartoon_off_axis_index + kCartoonConstraintFamilies] = "off-Vol";
    for (int layer = 0; layer < kCartoonAxisLayers; ++layer) {
      const int base = cartoon_layer_index + layer * kCartoonRegionStride;
      for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
        pdata->label[base + family] =
            std::string("L") + std::to_string(layer) + "-" +
            names[family] + "2";
      }
      pdata->label[base + kCartoonConstraintFamilies] =
          std::string("L") + std::to_string(layer) + "-N";
    }
    for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
      const int base = cartoon_linf_index + 3 * family;
      pdata->label[base] = std::string(names[family]) + "-Linf";
      pdata->label[base + 1] = std::string(names[family]) + "-rho";
      pdata->label[base + 2] = std::string(names[family]) + "-z";
      // These values are made globally identical below, so the ordinary MPI
      // history reduction remains deterministic without multiplying locations.
      pdata->reduction[base] = HistoryData::Reduction::max;
      pdata->reduction[base + 1] = HistoryData::Reduction::max;
      pdata->reduction[base + 2] = HistoryData::Reduction::max;
    }
  }

  // capture class variabels for kernel
  auto &u0_ = pm->pmb_pack->pz4c->u0;
  auto &u_con_ = pm->pmb_pack->pz4c->u_con;
  auto &u_telegraph_mu_ = pm->pmb_pack->pz4c->u_telegraph_mu;
  const int &I_Z4c_Theta_ =  pm->pmb_pack->pz4c->I_Z4C_THETA;
  auto &z4c = pm->pmb_pack->pz4c->z4c;
  auto &adm = pm->pmb_pack->padm->adm;

  auto &size = pm->pmb_pack->pmb->mb_size;
  const z4c::Z4cSymmetryMode symmetry_mode = pm->pmb_pack->z4c_symmetry.mode;
  constexpr int nsum = 9;

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

    const Real rho = size.d_view(m).x1min +
                     (i - is + 0.5) * size.d_view(m).dx1;
    const Real vol = z4c::Z4cDiagnosticCellMeasure(
        symmetry_mode, rho, size.d_view(m).dx1, size.d_view(m).dx2,
        size.d_view(m).dx3, detg);

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

    // max|K| is reduced separately below.
    for (int n=nsum; n<NREDUCTION_VARIABLES; ++n) {
      hvars.the_array[n] = 0.0;
    }

    // sum into parallel reduce
    mb_sum += hvars;
  }, Kokkos::Sum<array_sum::GlobalSum>(sum_this_mb));

  // store data into hdata array
  for (int n=0; n<nsum; ++n) {
    pdata->hdata[n] = sum_this_mb.the_array[n];
  }
  if (cartoon) {
    DvceArray1D<Real> diagnostic_sums(
        "Cartoon constraint region sums", kCartoonDiagnosticSums);
    Kokkos::deep_copy(diagnostic_sums, 0.0);
    const Real excise_chi = opt.excise_chi;
    Kokkos::parallel_for(
        "Cartoon constraint axis off-axis and layer sums",
        Kokkos::RangePolicy<DevExeSpace>(0, nmkji),
        KOKKOS_LAMBDA(const int idx) {
          const int m = idx / nkji;
          const int k0 = (idx - m * nkji) / nji;
          const int j0 = (idx - m * nkji - k0 * nji) / nx1;
          const int i0 = idx - m * nkji - k0 * nji - j0 * nx1;
          const int i = i0 + is;
          const int j = j0 + js;
          const int k = k0 + ks;
          if (z4c.chi(m, k, j, i) < excise_chi) return;

          const Real dx1 = size.d_view(m).dx1;
          const Real rho = size.d_view(m).x1min + (i0 + 0.5) * dx1;
          const int radial_layer = static_cast<int>(Kokkos::floor(rho / dx1));
          const Real detg = adm::SpatialDet(
              adm.g_dd(m,0,0,k,j,i), adm.g_dd(m,0,1,k,j,i),
              adm.g_dd(m,0,2,k,j,i), adm.g_dd(m,1,1,k,j,i),
              adm.g_dd(m,1,2,k,j,i), adm.g_dd(m,2,2,k,j,i));
          const Real physical_volume = z4c::Z4cDiagnosticCellMeasure(
              symmetry_mode, rho, dx1, size.d_view(m).dx2,
              size.d_view(m).dx3, detg);
          const Real squared[kCartoonConstraintFamilies] = {
              u_con_(m,0,k,j,i), SQR(u_con_(m,1,k,j,i)),
              u_con_(m,2,k,j,i), u_con_(m,3,k,j,i)};

          if (radial_layer >= 0 && radial_layer < kCartoonAxisLayers) {
            for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
              Kokkos::atomic_add(
                  &diagnostic_sums(kCartoonAxisSumBase + family),
                  squared[family]);
              Kokkos::atomic_add(
                  &diagnostic_sums(
                      kCartoonLayerSumBase +
                      radial_layer * kCartoonRegionStride + family),
                  squared[family]);
            }
            Kokkos::atomic_add(
                &diagnostic_sums(
                    kCartoonAxisSumBase + kCartoonConstraintFamilies),
                1.0);
            Kokkos::atomic_add(
                &diagnostic_sums(
                    kCartoonLayerSumBase +
                    radial_layer * kCartoonRegionStride +
                    kCartoonConstraintFamilies),
                1.0);
          } else {
            for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
              Kokkos::atomic_add(
                  &diagnostic_sums(kCartoonOffAxisSumBase + family),
                  physical_volume * squared[family]);
            }
            Kokkos::atomic_add(
                &diagnostic_sums(
                    kCartoonOffAxisSumBase + kCartoonConstraintFamilies),
                physical_volume);
          }
        });
    auto host_sums =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), diagnostic_sums);
    for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
      pdata->hdata[cartoon_axis_index + family] =
          host_sums(kCartoonAxisSumBase + family);
      pdata->hdata[cartoon_off_axis_index + family] =
          host_sums(kCartoonOffAxisSumBase + family);
    }
    pdata->hdata[cartoon_axis_index + kCartoonConstraintFamilies] =
        host_sums(kCartoonAxisSumBase + kCartoonConstraintFamilies);
    pdata->hdata[cartoon_off_axis_index + kCartoonConstraintFamilies] =
        host_sums(kCartoonOffAxisSumBase + kCartoonConstraintFamilies);
    for (int layer = 0; layer < kCartoonAxisLayers; ++layer) {
      for (int entry = 0; entry < kCartoonRegionStride; ++entry) {
        pdata->hdata[
            cartoon_layer_index + layer * kCartoonRegionStride + entry] =
            host_sums(
                kCartoonLayerSumBase + layer * kCartoonRegionStride + entry);
      }
    }

    const std::array<ConstraintMaximum, kCartoonConstraintFamilies> maxima = {
        CartoonConstraintMaximum<0>(u_con_, z4c.chi, size, excise_chi,
                                    pm->pmb_pack->nmb_thispack,
                                    nx1, nx2, nx3, is, js, ks),
        CartoonConstraintMaximum<1>(u_con_, z4c.chi, size, excise_chi,
                                    pm->pmb_pack->nmb_thispack,
                                    nx1, nx2, nx3, is, js, ks),
        CartoonConstraintMaximum<2>(u_con_, z4c.chi, size, excise_chi,
                                    pm->pmb_pack->nmb_thispack,
                                    nx1, nx2, nx3, is, js, ks),
        CartoonConstraintMaximum<3>(u_con_, z4c.chi, size, excise_chi,
                                    pm->pmb_pack->nmb_thispack,
                                    nx1, nx2, nx3, is, js, ks)};
    for (int family = 0; family < kCartoonConstraintFamilies; ++family) {
      const int base = cartoon_linf_index + 3 * family;
      pdata->hdata[base] = maxima[family].value;
      pdata->hdata[base + 1] = maxima[family].rho;
      pdata->hdata[base + 2] = maxima[family].z;
    }
  }
  Real max_abs_K = 0.0;
  Kokkos::parallel_reduce(
      "Z4cHistoryMaxAbsK",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int &idx, Real &rank_max_abs_K) {
        int m = idx / nkji;
        int k = (idx - m * nkji) / nji;
        int j = (idx - m * nkji - k * nji) / nx1;
        int i = (idx - m * nkji - k * nji - j * nx1) + is;
        k += ks;
        j += js;
        const Real K = z4c.vKhat(m, k, j, i) + 2.0 * z4c.vTheta(m, k, j, i);
        rank_max_abs_K = fmax(rank_max_abs_K, fabs(K));
      },
      Kokkos::Max<Real>(max_abs_K));
  pdata->hdata[9] = max_abs_K;
  pdata->hdata[10] = static_cast<Real>(pm->nmb_total);
  if (opt.telegraph_lapse) {
    Real telegraph_mu_min = std::numeric_limits<Real>::max();
    Real telegraph_mu_max = 0.0;
    Kokkos::parallel_reduce(
        "Z4cHistoryTelegraphMuMin",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, Real &rank_min) {
          const int m = idx / nkji;
          const int k0 = (idx - m * nkji) / nji;
          const int j0 = (idx - m * nkji - k0 * nji) / nx1;
          const int i0 = idx - m * nkji - k0 * nji - j0 * nx1;
          rank_min = fmin(rank_min,
                          u_telegraph_mu_(m, 0, k0 + ks, j0 + js, i0 + is));
        },
        Kokkos::Min<Real>(telegraph_mu_min));
    Kokkos::parallel_reduce(
        "Z4cHistoryTelegraphMuMax",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int &idx, Real &rank_max) {
          const int m = idx / nkji;
          const int k0 = (idx - m * nkji) / nji;
          const int j0 = (idx - m * nkji - k0 * nji) / nx1;
          const int i0 = idx - m * nkji - k0 * nji - j0 * nx1;
          rank_max = fmax(rank_max,
                          u_telegraph_mu_(m, 0, k0 + ks, j0 + js, i0 + is));
        },
        Kokkos::Max<Real>(telegraph_mu_max));
    pdata->hdata[telegraph_mu_min_index] = telegraph_mu_min;
    pdata->hdata[telegraph_mu_max_index] = telegraph_mu_max;
  }
  if (opt.history_kretschmann) {
    pdata->hdata[kretschmann_index] = DispatchZ4cHistoryMaxKretschmann(pm);
  }
  int max_refinement_level = 0;
  const int first_local_gid = pm->gids_eachrank[global_variable::my_rank];
  for (int m = 0; m < pm->pmb_pack->nmb_thispack; ++m) {
    max_refinement_level = std::max(
        max_refinement_level,
        pm->lloc_eachmb[first_local_gid + m].level - pm->root_level);
  }
  pdata->hdata[max_refinement_level_index] =
      static_cast<Real>(max_refinement_level);
  pdata->hdata[max_meshblocks_per_rank_index] =
      static_cast<Real>(pm->pmb_pack->nmb_thispack);
  pdata->hdata[horizon_status_index] = 0.0;
  pdata->hdata[horizon_last_search_cycle_index] = -1.0;
  if (!pm->pmb_pack->pz4c->pfastflow.empty()) {
    pdata->hdata[horizon_status_index] =
        pm->pmb_pack->pz4c->pfastflow[0]->ah_found ? 1.0 : 0.0;
    pdata->hdata[horizon_last_search_cycle_index] = static_cast<Real>(
        pm->pmb_pack->pz4c->pfastflow[0]->last_search_cycle);
  }
  pdata->hdata[cycle_index] = static_cast<Real>(pm->ncycle);
  if (cartoon) {
    const auto &central = pm->pmb_pack->z4c_restart_state.central;
    const auto validation = z4c::ValidateZ4cCentralRestartState(central);
    if (!validation.valid || !central.initialized) {
      std::cerr << "### FATAL ERROR in " << __FILE__
                << ": invalid Cartoon central history state: "
                << validation.error << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pdata->hdata[central_lapse_index] = central.previous_lapse;
    pdata->hdata[central_proper_time_index] = central.proper_time;
    pdata->hdata[central_kretschmann_index] = central.abs_kretschmann;
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

    // Fill the fixed-width legacy reducer, not the larger HistoryData storage.
    for (int n=nhist_; n<NREDUCTION_VARIABLES; ++n) {
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
    // First reduce over all MPI ranks. Most history variables are extensive sums,
    // while diagnostics such as max|K| require a different operation.
#if MPI_PARALLEL_ENABLED
    for (int n = 0; n < data.nhist; ++n) {
      MPI_Op op = MPI_SUM;
      if (data.reduction[n] == HistoryData::Reduction::max) {
        op = MPI_MAX;
      } else if (data.reduction[n] == HistoryData::Reduction::min) {
        op = MPI_MIN;
      }
      if (global_variable::my_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &(data.hdata[n]), 1, MPI_ATHENA_REAL,
                   op, 0, MPI_COMM_WORLD);
      } else {
        Real unused_receive = 0.0;
        MPI_Reduce(&(data.hdata[n]), &unused_receive, 1, MPI_ATHENA_REAL,
                   op, 0, MPI_COMM_WORLD);
      }
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
