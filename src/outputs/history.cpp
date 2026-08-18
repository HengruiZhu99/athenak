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
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "fo_gh/fo_gh.hpp"
#include "z4c/z4c.hpp"
#include "coordinates/adm.hpp"
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

  if (pm->pmb_pack->pz4c != nullptr || pm->pmb_pack->pfogh != nullptr) {
    hist_data.emplace_back(PhysicsModule::SpaceTimeDynamics);
    if (pin->GetOrAddBoolean("problem", "common_adm_history", false)) {
      const int common_fd_order =
          pin->GetOrAddInteger("problem", "common_adm_fd_order", 4);
      if (common_fd_order != 2 && common_fd_order != 4 && common_fd_order != 6) {
        std::cout << "### FATAL ERROR: problem/common_adm_fd_order must be 2, 4, or 6"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      for (int chunk = 0; chunk < 6; ++chunk) {
        hist_data.emplace_back(PhysicsModule::CommonADMConstraints, chunk);
        hist_data.back().fd_order = common_fd_order;
      }
    }
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
      if (pm->pmb_pack->pfogh != nullptr) {
        LoadFoGhHistoryData(&data, pm);
      } else {
        LoadZ4cHistoryData(&data, pm);
      }
    } else if (data.physics == PhysicsModule::CommonADMConstraints) {
      LoadCommonADMHistoryData(&data, pm);
    } else if (data.physics == PhysicsModule::UserDefined) {
      (pm->pgen->user_hist_func)(&data, pm);
    }
  }
}

// Two fixed physical regions are stored per file so the standard 20-entry reduction
// buffer is not enlarged.  No lapse or chi mask is applied to these common diagnostics.
void HistoryOutput::LoadCommonADMHistoryData(HistoryData *pdata, Mesh *pm) {
  if (pdata->instance == 0) {
    switch (pdata->fd_order) {
      case 2: pm->pmb_pack->padm->ComputeVacuumConstraints<2>(pm->pmb_pack); break;
      case 4: pm->pmb_pack->padm->ComputeVacuumConstraints<3>(pm->pmb_pack); break;
      case 6: pm->pmb_pack->padm->ComputeVacuumConstraints<4>(pm->pmb_pack); break;
    }
  }

  static const char *prefix[12] = {
    "all", "lt1", "lt2", "r2to4", "r4to8", "gt8",
    "if64", "if32", "if16", "if8", "if4", "if2"
  };
  pdata->nhist = 14;
  for (int local_region = 0; local_region < 2; ++local_region) {
    const int base = 7*local_region;
    const int region = 2*pdata->instance + local_region;
    pdata->label[base + 0] = std::string(prefix[region]) + "H1";
    pdata->label[base + 1] = std::string(prefix[region]) + "H2";
    pdata->label[base + 2] = std::string(prefix[region]) + "Hi";
    pdata->label[base + 3] = std::string(prefix[region]) + "M1";
    pdata->label[base + 4] = std::string(prefix[region]) + "M2";
    pdata->label[base + 5] = std::string(prefix[region]) + "Mi";
    pdata->label[base + 6] = std::string(prefix[region]) + "V";
    pdata->use_max[base + 2] = true;
    pdata->use_max[base + 5] = true;
  }
  if (pdata->instance == 0) {
    pdata->nhist = 17;
    pdata->label[14] = "idxmax";
    pdata->label[15] = "charmax";
    pdata->label[16] = "effcfl";
    pdata->use_max[14] = true;
    pdata->use_max[15] = true;
    pdata->use_max[16] = true;
  }

  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const auto common = pm->pmb_pack->padm->u_common;
  const auto adm_vars = pm->pmb_pack->padm->adm;
  const int first_region = 2*pdata->instance;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce(
      "common ADM fixed-region sums", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pm->pmb_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &total) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        const Real r = std::sqrt(x*x + y*y + z*z);
        const Real cube_r = fmax(Kokkos::abs(x),
                                fmax(Kokkos::abs(y), Kokkos::abs(z)));
        const Real detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i), adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i), adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i), adm_vars.g_dd(m, 2, 2, k, j, i));
        const Real volume = size.d_view(m).dx1*size.d_view(m).dx2
                            *size.d_view(m).dx3*std::sqrt(Kokkos::abs(detg));
        const Real h = Kokkos::abs(common(m, adm::ADM::I_COMMON_H, k, j, i));
        const Real m2 = fmax(0.0, common(m, adm::ADM::I_COMMON_M2, k, j, i));
        const Real momentum = std::sqrt(m2);
        for (int local_region = 0; local_region < 2; ++local_region) {
          const int region = first_region + local_region;
          bool include = false;
          if (region == 0) include = true;
          if (region == 1) include = r < 1.0;
          if (region == 2) include = r < 2.0;
          if (region == 3) include = r >= 2.0 && r < 4.0;
          if (region == 4) include = r >= 4.0 && r < 8.0;
          if (region == 5) include = r >= 8.0;
          if (region >= 6) {
            const Real interface_radius = 64.0/std::pow(2.0, region - 6);
            const Real half_width = interface_radius/8.0;
            include = Kokkos::abs(cube_r - interface_radius) < half_width;
          }
          if (include) {
            const int base = 7*local_region;
            total.the_array[base + 0] += volume*h;
            total.the_array[base + 1] += volume*h*h;
            total.the_array[base + 3] += volume*momentum;
            total.the_array[base + 4] += volume*m2;
            total.the_array[base + 6] += volume;
          }
        }
      }, Kokkos::Sum<array_sum::GlobalSum>(sums));

  Real maxima[4] = {0.0, 0.0, 0.0, 0.0};
  for (int local_region = 0; local_region < 2; ++local_region) {
    const int region = first_region + local_region;
    Kokkos::parallel_reduce(
        "common ADM fixed-region H Linf", Kokkos::RangePolicy<>(DevExeSpace(),
        0, pm->pmb_pack->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is;
          work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js;
          work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min, size.d_view(m).x1max);
          const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                     size.d_view(m).x2min, size.d_view(m).x2max);
          const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                     size.d_view(m).x3min, size.d_view(m).x3max);
          const Real r = std::sqrt(x*x + y*y + z*z);
          const Real cube_r = fmax(Kokkos::abs(x),
                                  fmax(Kokkos::abs(y), Kokkos::abs(z)));
          bool include = region == 0 || (region == 1 && r < 1.0)
              || (region == 2 && r < 2.0)
              || (region == 3 && r >= 2.0 && r < 4.0)
              || (region == 4 && r >= 4.0 && r < 8.0)
              || (region == 5 && r >= 8.0);
          if (region >= 6) {
            const Real interface_radius = 64.0/std::pow(2.0, region - 6);
            include = Kokkos::abs(cube_r - interface_radius) < interface_radius/8.0;
          }
          if (include) maximum = fmax(maximum, Kokkos::abs(
              common(m, adm::ADM::I_COMMON_H, k, j, i)));
        }, Kokkos::Max<Real>(maxima[2*local_region]));
    Kokkos::parallel_reduce(
        "common ADM fixed-region M Linf", Kokkos::RangePolicy<>(DevExeSpace(),
        0, pm->pmb_pack->nmb_thispack*ncells),
        KOKKOS_LAMBDA(const int idx, Real &maximum) {
          int work = idx;
          const int i = work % indcs.nx1 + indcs.is;
          work /= indcs.nx1;
          const int j = work % indcs.nx2 + indcs.js;
          work /= indcs.nx2;
          const int k = work % indcs.nx3 + indcs.ks;
          const int m = work/indcs.nx3;
          const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                     size.d_view(m).x1min, size.d_view(m).x1max);
          const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                     size.d_view(m).x2min, size.d_view(m).x2max);
          const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                     size.d_view(m).x3min, size.d_view(m).x3max);
          const Real r = std::sqrt(x*x + y*y + z*z);
          const Real cube_r = fmax(Kokkos::abs(x),
                                  fmax(Kokkos::abs(y), Kokkos::abs(z)));
          bool include = region == 0 || (region == 1 && r < 1.0)
              || (region == 2 && r < 2.0)
              || (region == 3 && r >= 2.0 && r < 4.0)
              || (region == 4 && r >= 4.0 && r < 8.0)
              || (region == 5 && r >= 8.0);
          if (region >= 6) {
            const Real interface_radius = 64.0/std::pow(2.0, region - 6);
            include = Kokkos::abs(cube_r - interface_radius) < interface_radius/8.0;
          }
          if (include) maximum = fmax(maximum, std::sqrt(fmax(0.0,
              common(m, adm::ADM::I_COMMON_M2, k, j, i))));
        }, Kokkos::Max<Real>(maxima[2*local_region + 1]));
  }
  for (int local_region = 0; local_region < 2; ++local_region) {
    const int base = 7*local_region;
    if (sums.the_array[base + 6] == 0.0) {
      maxima[2*local_region] = 0.0;
      maxima[2*local_region + 1] = 0.0;
    }
    pdata->hdata[base + 0] = sums.the_array[base + 0];
    pdata->hdata[base + 1] = sums.the_array[base + 1];
    pdata->hdata[base + 2] = maxima[2*local_region];
    pdata->hdata[base + 3] = sums.the_array[base + 3];
    pdata->hdata[base + 4] = sums.the_array[base + 4];
    pdata->hdata[base + 5] = maxima[2*local_region + 1];
    pdata->hdata[base + 6] = sums.the_array[base + 6];
  }
  if (pdata->instance == 0) {
    Real inverse_dx = 0.0;
    Kokkos::parallel_reduce(
        "common ADM inverse minimum spacing", Kokkos::RangePolicy<>(DevExeSpace(),
        0, pm->pmb_pack->nmb_thispack),
        KOKKOS_LAMBDA(const int m, Real &maximum) {
          maximum = fmax(maximum, 1.0/size.d_view(m).dx1);
          maximum = fmax(maximum, 1.0/size.d_view(m).dx2);
          maximum = fmax(maximum, 1.0/size.d_view(m).dx3);
        }, Kokkos::Max<Real>(inverse_dx));
    pdata->hdata[14] = inverse_dx;
    if (pm->pmb_pack->pfogh != nullptr) {
      pdata->hdata[15] = pm->pmb_pack->pfogh->max_char_speed;
      pdata->hdata[16] = pm->pmb_pack->pfogh->dtnew > 0.0
          ? pm->dt/pm->pmb_pack->pfogh->dtnew : 0.0;
    } else {
      pdata->hdata[15] = 1.0;
      pdata->hdata[16] = pm->pmb_pack->pz4c->dtnew > 0.0
          ? pm->dt/pm->pmb_pack->pz4c->dtnew : 0.0;
    }
  }
}

void HistoryOutput::LoadFoGhHistoryData(HistoryData *pdata, Mesh *pm) {
  enum FoGhHistoryIndex {
    HIST_H, HIST_M, HIST_CP, HIST_C, HIST_RQ, HIST_RX, HIST_RA, HIST_RB,
    HIST_CURL, HIST_DET, HIST_TRA, HIST_H_MINUS_F, HIST_R_ALPHA, HIST_R_BETA,
    HIST_NEAR_H, HIST_NEAR_M, HIST_NEAR_GH, HIST_NEAR_R,
    HIST_VOLUME, HIST_NEAR_VOLUME, NHIST_FO_GH
  };
  static_assert(NHIST_FO_GH <= NHISTORY_VARIABLES,
                "FO-GH history exceeds NHISTORY_VARIABLES");
  pdata->nhist = NHIST_FO_GH;
  const char *labels[NHIST_FO_GH] = {
    "H-L2sq", "M-L2sq", "Cp-L2sq", "c-L2sq",
    "RQ-L2sq", "RX-L2sq", "Ra-L2sq", "RB-L2sq",
    "Curl-L2sq", "detgt-L2sq", "trAt-L2sq", "h-f-L2sq",
    "Ralpha-L2sq", "Rbeta-L2sq", "Hnear-L2sq", "Mnear-L2sq",
    "GHnear-L2sq", "Rnear-L2sq", "Volume", "NearVolume"
  };
  for (int n = 0; n < NHIST_FO_GH; ++n) pdata->label[n] = labels[n];

  auto &indcs = pm->mb_indcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  const auto constraints = pm->pmb_pack->pfogh->u_con;
  const auto vars = pm->pmb_pack->pfogh->u;
  const auto adm_vars = pm->pmb_pack->padm->adm;
  const Real excise_lapse = pm->pmb_pack->pfogh->opt.excise_lapse;
  const Real eta_beta = pm->pmb_pack->pfogh->opt.eta_beta;
  const Real diagnostic_radius = pm->pmb_pack->pfogh->opt.diagnostic_radius;
  const int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  array_sum::GlobalSum sums;
  Kokkos::parallel_reduce(
      "fo_gh history", Kokkos::RangePolicy<>(DevExeSpace(),
      0, pm->pmb_pack->nmb_thispack*ncells),
      KOKKOS_LAMBDA(const int idx, array_sum::GlobalSum &total) {
        int work = idx;
        const int i = work % indcs.nx1 + indcs.is;
        work /= indcs.nx1;
        const int j = work % indcs.nx2 + indcs.js;
        work /= indcs.nx2;
        const int k = work % indcs.nx3 + indcs.ks;
        const int m = work/indcs.nx3;
        const Real detg = adm::SpatialDet(
            adm_vars.g_dd(m, 0, 0, k, j, i), adm_vars.g_dd(m, 0, 1, k, j, i),
            adm_vars.g_dd(m, 0, 2, k, j, i), adm_vars.g_dd(m, 1, 1, k, j, i),
            adm_vars.g_dd(m, 1, 2, k, j, i), adm_vars.g_dd(m, 2, 2, k, j, i));
        const bool include = vars.alpha(m, k, j, i) >= excise_lapse;
        const Real volume = include
            ? size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3
                *std::sqrt(std::abs(detg))
            : 0.0;
        array_sum::GlobalSum local;
        local.the_array[HIST_H] = volume*SQR(
            constraints(m, fo_gh::FoGh::I_CON_H, k, j, i));
        fo_gh::RegularPointState point;
        fo_gh::LoadPoint(vars, m, k, j, i, point);
        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse;
        const Real determinant = fo_gh::Invert3(point.gtilde, inverse);
        Real momentum2 = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            momentum2 += point.chi*inverse(a, b)
                         *constraints(m, fo_gh::FoGh::I_CON_MX + a, k, j, i)
                         *constraints(m, fo_gh::FoGh::I_CON_MX + b, k, j, i);
          }
        }
        local.the_array[HIST_M] = volume*momentum2;
        local.the_array[HIST_CP] = volume*SQR(
            constraints(m, fo_gh::FoGh::I_CON_GH_PERP, k, j, i));

        Real c2 = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            c2 += point.gtilde(a, b)
                  *constraints(m, fo_gh::FoGh::I_CON_GHX + a, k, j, i)
                  *constraints(m, fo_gh::FoGh::I_CON_GHX + b, k, j, i);
          }
        }
        local.the_array[HIST_C] = volume*c2;
        const Real rq2 = SQR(constraints(m, fo_gh::FoGh::I_CON_RQ, k, j, i));
        const Real rx2 = SQR(constraints(m, fo_gh::FoGh::I_CON_RX, k, j, i));
        const Real ra2 = SQR(constraints(m, fo_gh::FoGh::I_CON_RA, k, j, i));
        const Real rb2 = SQR(constraints(m, fo_gh::FoGh::I_CON_RB, k, j, i));
        local.the_array[HIST_RQ] = volume*rq2;
        local.the_array[HIST_RX] = volume*rx2;
        local.the_array[HIST_RA] = volume*ra2;
        local.the_array[HIST_RB] = volume*rb2;
        Real curl2 = 0.0;
        for (int n = fo_gh::FoGh::I_CON_CURL_Q;
             n <= fo_gh::FoGh::I_CON_CURL_B; ++n) {
          curl2 += SQR(constraints(m, n, k, j, i));
        }
        local.the_array[HIST_CURL] = volume*curl2;

        Real trace_A = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) trace_A += inverse(a, b)*point.Atilde(a, b);
        }
        local.the_array[HIST_DET] = volume*SQR(determinant - 1.0);
        local.the_array[HIST_TRA] = volume*SQR(trace_A);

        Real f_perp = 0.0;
        AthenaPointTensor<Real, TensorSymm::NONE, 3, 1> f;
        fo_gh::GaugeTargets(point, eta_beta, f_perp, f);
        const Real perp_residual = point.h_perp - f_perp;
        Real beta_residual2 = 0.0;
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            beta_residual2 += point.gtilde(a, b)*(point.h(a) - f(a))
                              *(point.h(b) - f(b));
          }
        }
        local.the_array[HIST_H_MINUS_F] =
            volume*(perp_residual*perp_residual + beta_residual2);
        local.the_array[HIST_R_ALPHA] =
            volume*SQR(point.alpha*perp_residual);
        local.the_array[HIST_R_BETA] = volume*beta_residual2;

        const Real x = CellCenterX(i - indcs.is, indcs.nx1,
                                   size.d_view(m).x1min, size.d_view(m).x1max);
        const Real y = CellCenterX(j - indcs.js, indcs.nx2,
                                   size.d_view(m).x2min, size.d_view(m).x2max);
        const Real z = CellCenterX(k - indcs.ks, indcs.nx3,
                                   size.d_view(m).x3min, size.d_view(m).x3max);
        const Real near_volume = (x*x + y*y + z*z < SQR(diagnostic_radius))
                                 ? volume : 0.0;
        local.the_array[HIST_NEAR_H] = near_volume*SQR(
            constraints(m, fo_gh::FoGh::I_CON_H, k, j, i));
        local.the_array[HIST_NEAR_M] = near_volume*momentum2;
        local.the_array[HIST_NEAR_GH] = near_volume*(SQR(
            constraints(m, fo_gh::FoGh::I_CON_GH_PERP, k, j, i)) + c2);
        local.the_array[HIST_NEAR_R] = near_volume*(rq2 + rx2 + ra2 + rb2 + curl2);
        local.the_array[HIST_VOLUME] = volume;
        local.the_array[HIST_NEAR_VOLUME] = near_volume;
        for (int n = NHIST_FO_GH; n < NHISTORY_VARIABLES; ++n) {
          local.the_array[n] = 0.0;
        }
        total += local;
      }, Kokkos::Sum<array_sum::GlobalSum>(sums));
  for (int n = 0; n < NHIST_FO_GH; ++n) {
    pdata->hdata[n] = sums.the_array[n];
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
    for (int n = 0; n < data.nhist; ++n) {
      const MPI_Op op = data.use_max[n] ? MPI_MAX : MPI_SUM;
      if (global_variable::my_rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, &(data.hdata[n]), 1, MPI_ATHENA_REAL,
                   op, 0, MPI_COMM_WORLD);
      } else {
        MPI_Reduce(&(data.hdata[n]), &(data.hdata[n]), 1, MPI_ATHENA_REAL,
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
          fname.append(pm->pmb_pack->pfogh != nullptr ? ".fo_gh" : ".z4c");
          break;
        case PhysicsModule::CommonADMConstraints:
          fname.append(".adm_common");
          fname.append(std::to_string(data.instance));
          break;
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
