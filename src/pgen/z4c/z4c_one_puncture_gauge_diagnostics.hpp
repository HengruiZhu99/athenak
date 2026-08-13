#ifndef PGEN_Z4C_Z4C_ONE_PUNCTURE_GAUGE_DIAGNOSTICS_HPP_
#define PGEN_Z4C_Z4C_ONE_PUNCTURE_GAUGE_DIAGNOSTICS_HPP_

//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file z4c_one_puncture_gauge_diagnostics.hpp
//! \brief Accepted-state regional source diagnostics for the gauge qualification lane.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "z4c/z4c.hpp"

namespace z4c_puncture_gauge_diagnostics {

inline Real center[3] = {0.0, 0.0, 0.0};
inline std::string output_path = "z4c_gauge_source_diagnostics.csv";
inline std::string profile_name = "standard";

inline const char *ProfileName(const z4c::ShiftGaugeProfile profile) {
  if (profile == z4c::ShiftGaugeProfile::candidate_a) return "candidate_a";
  if (profile == z4c::ShiftGaugeProfile::candidate_c) return "candidate_c";
  return "standard";
}

KOKKOS_INLINE_FUNCTION
int RegionIndex(const Real radius) {
  if (radius <= 0.25) return 1;
  if (radius <= 1.0) return 2;
  if (radius >= 2.0 && radius <= 8.0) return 3;
  return -1;
}

template <int NGHOST>
void LoadGaugeDiagnostics(HistoryData *pdata, Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  auto &state = pmbp->pz4c->z4c;
  auto const options = pmbp->pz4c->opt;
  const int is = indcs.is;
  const int js = indcs.js;
  const int ks = indcs.ks;
  const int nx1 = indcs.nx1;
  const int nx2 = indcs.nx2;
  const int nx3 = indcs.nx3;
  const int cells_per_block = nx1 * nx2 * nx3;
  const int total_cells = pmbp->nmb_thispack * cells_per_block;
  const Real center_x = center[0];
  const Real center_y = center[1];
  const Real center_z = center[2];

  array_sum::GlobalSum coordinate_sums;
  array_sum::GlobalSum proper_sums;
  Kokkos::parallel_reduce(
      "one_puncture_gauge_source_sums",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, array_sum::GlobalSum &coordinate,
                    array_sum::GlobalSum &proper) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int local_k = local / (nx1 * nx2);
        const int local_j = (local - local_k * nx1 * nx2) / nx1;
        const int local_i = local - local_k * nx1 * nx2 - local_j * nx1;
        const int i = is + local_i;
        const int j = js + local_j;
        const int k = ks + local_k;
        const Real alpha = state.alpha(m,k,j,i);
        const Real chi = state.chi(m,k,j,i);

        const Real detg = adm::SpatialDet(
            state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
            state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
            state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i));
        if (!(Kokkos::isfinite(alpha) && Kokkos::isfinite(chi) &&
              Kokkos::isfinite(detg) && alpha > 0.0 && chi > 0.0 && detg > 0.0)) {
          return;
        }

        AthenaPointTensor<Real, TensorSymm::SYM2, 3, 2> inverse_metric;
        adm::SpatialInv(
            1.0 / detg, state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
            state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
            state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i),
            &inverse_metric(0,0), &inverse_metric(0,1), &inverse_metric(0,2),
            &inverse_metric(1,1), &inverse_metric(1,2), &inverse_metric(2,2));

        const Real inverse_spacing[3] = {
            1.0 / size.d_view(m).dx1,
            1.0 / size.d_view(m).dx2,
            1.0 / size.d_view(m).dx3};
        Real lapse_gradient[3];
        Real chi_gradient[3];
        for (int direction = 0; direction < 3; ++direction) {
          lapse_gradient[direction] =
              Dx<NGHOST>(direction, inverse_spacing, state.alpha, m,k,j,i);
          chi_gradient[direction] =
              Dx<NGHOST>(direction, inverse_spacing, state.chi, m,k,j,i);
        }

        Real gamma_square = 0.0;
        Real chi_gradient_square = 0.0;
        Real lapse_gradient_square = 0.0;
        Real damping_square = 0.0;
        for (int component = 0; component < 3; ++component) {
          Real inverse_metric_row[3];
          for (int direction = 0; direction < 3; ++direction) {
            inverse_metric_row[direction] = inverse_metric(component, direction);
          }

          z4c::ShiftGaugeForces forces;
          if (options.shift_gauge_profile == z4c::ShiftGaugeProfile::standard) {
            const Real chi_guarded =
                (chi > options.chi_div_floor) ? chi : options.chi_div_floor;
            forces.gamma =
                (options.shift_ggamma + options.shift_alpha2ggamma * alpha * alpha) *
                state.vGam_u(m,component,k,j,i);
            for (int direction = 0; direction < 3; ++direction) {
              forces.chi_gradient += options.shift_hh * 0.5 * alpha * alpha *
                  chi_guarded * inverse_metric_row[direction] * chi_gradient[direction];
              forces.lapse_gradient -= options.shift_hh * alpha * chi_guarded *
                  inverse_metric_row[direction] * lapse_gradient[direction];
            }
          } else {
            forces = z4c::EvaluateModifiedShiftGaugeForces(
                options.shift_gauge_profile, alpha, chi,
                state.vGam_u(m,component,k,j,i), inverse_metric_row,
                lapse_gradient, chi_gradient);
          }
          const Real damping = options.shift_eta * state.beta_u(m,component,k,j,i);
          gamma_square += forces.gamma * forces.gamma;
          chi_gradient_square += forces.chi_gradient * forces.chi_gradient;
          lapse_gradient_square += forces.lapse_gradient * forces.lapse_gradient;
          damping_square += damping * damping;
        }

        const Real x = CellCenterX(local_i, nx1, size.d_view(m).x1min,
                                   size.d_view(m).x1max);
        const Real y = CellCenterX(local_j, nx2, size.d_view(m).x2min,
                                   size.d_view(m).x2max);
        const Real z = CellCenterX(local_k, nx3, size.d_view(m).x3min,
                                   size.d_view(m).x3max);
        const Real dx = x - center_x;
        const Real dy = y - center_y;
        const Real dz = z - center_z;
        const Real radius = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
        const Real coordinate_volume =
            size.d_view(m).dx1 * size.d_view(m).dx2 * size.d_view(m).dx3;
        const Real proper_volume = coordinate_volume * Kokkos::sqrt(detg / (chi*chi*chi));

        const Real values[4] = {gamma_square, chi_gradient_square,
                                lapse_gradient_square, damping_square};
        const int selected_region = RegionIndex(radius);
        for (int region_slot = 0; region_slot < 2; ++region_slot) {
          const int region = (region_slot == 0) ? 0 : selected_region;
          if (region < 0) continue;
          for (int quantity = 0; quantity < 4; ++quantity) {
            coordinate.the_array[5 * region + quantity] +=
                coordinate_volume * values[quantity];
            proper.the_array[5 * region + quantity] += proper_volume * values[quantity];
          }
          coordinate.the_array[5 * region + 4] += coordinate_volume;
          proper.the_array[5 * region + 4] += proper_volume;
        }
      }, Kokkos::Sum<array_sum::GlobalSum>(coordinate_sums),
      Kokkos::Sum<array_sum::GlobalSum>(proper_sums));

  Real minimum_alpha = std::numeric_limits<Real>::max();
  Real minimum_chi = std::numeric_limits<Real>::max();
  Real minimum_metric_minor = std::numeric_limits<Real>::max();
  Real maximum_beta = 0.0;
  Real maximum_gamma = 0.0;
  long long invalid_points = 0;
  Kokkos::parallel_reduce(
      "one_puncture_minimum_alpha", Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, Real &minimum) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int k = ks + local / (nx1 * nx2);
        const int j = js + (local / nx1) % nx2;
        const int i = is + local % nx1;
        const Real value = state.alpha(m,k,j,i);
        if (Kokkos::isfinite(value) && value < minimum) minimum = value;
      }, Kokkos::Min<Real>(minimum_alpha));
  Kokkos::parallel_reduce(
      "one_puncture_minimum_chi", Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, Real &minimum) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int k = ks + local / (nx1 * nx2);
        const int j = js + (local / nx1) % nx2;
        const int i = is + local % nx1;
        const Real value = state.chi(m,k,j,i);
        if (Kokkos::isfinite(value) && value < minimum) minimum = value;
      }, Kokkos::Min<Real>(minimum_chi));
  Kokkos::parallel_reduce(
      "one_puncture_minimum_metric_minor",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, Real &minimum) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int k = ks + local / (nx1 * nx2);
        const int j = js + (local / nx1) % nx2;
        const int i = is + local % nx1;
        const Real chi = state.chi(m,k,j,i);
        const Real gxx = state.g_dd(m,0,0,k,j,i);
        const Real gxy = state.g_dd(m,0,1,k,j,i);
        const Real gyy = state.g_dd(m,1,1,k,j,i);
        const Real detg = adm::SpatialDet(
            gxx, gxy, state.g_dd(m,0,2,k,j,i), gyy,
            state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i));
        if (!(Kokkos::isfinite(chi) && Kokkos::isfinite(detg) && chi > 0.0)) return;
        const Real minor1 = gxx / chi;
        const Real minor2 = (gxx * gyy - gxy * gxy) / (chi * chi);
        const Real minor3 = detg / (chi * chi * chi);
        const Real value = Kokkos::fmin(minor1, Kokkos::fmin(minor2, minor3));
        if (Kokkos::isfinite(value) && value < minimum) minimum = value;
      }, Kokkos::Min<Real>(minimum_metric_minor));
  Kokkos::parallel_reduce(
      "one_puncture_maximum_beta", Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, Real &maximum) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int k = ks + local / (nx1 * nx2);
        const int j = js + (local / nx1) % nx2;
        const int i = is + local % nx1;
        Real square = 0.0;
        for (int component = 0; component < 3; ++component) {
          const Real value = state.beta_u(m,component,k,j,i);
          square += value * value;
        }
        const Real value = Kokkos::sqrt(square);
        if (Kokkos::isfinite(value) && value > maximum) maximum = value;
      }, Kokkos::Max<Real>(maximum_beta));
  Kokkos::parallel_reduce(
      "one_puncture_maximum_gamma", Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, Real &maximum) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int k = ks + local / (nx1 * nx2);
        const int j = js + (local / nx1) % nx2;
        const int i = is + local % nx1;
        Real square = 0.0;
        for (int component = 0; component < 3; ++component) {
          const Real value = state.vGam_u(m,component,k,j,i);
          square += value * value;
        }
        const Real value = Kokkos::sqrt(square);
        if (Kokkos::isfinite(value) && value > maximum) maximum = value;
      }, Kokkos::Max<Real>(maximum_gamma));
  Kokkos::parallel_reduce(
      "one_puncture_invalid_points", Kokkos::RangePolicy<>(DevExeSpace(), 0, total_cells),
      KOKKOS_LAMBDA(const int linear_index, long long &count) {
        const int m = linear_index / cells_per_block;
        const int local = linear_index - m * cells_per_block;
        const int k = ks + local / (nx1 * nx2);
        const int j = js + (local / nx1) % nx2;
        const int i = is + local % nx1;
        const Real alpha = state.alpha(m,k,j,i);
        const Real chi = state.chi(m,k,j,i);
        const Real detg = adm::SpatialDet(
            state.g_dd(m,0,0,k,j,i), state.g_dd(m,0,1,k,j,i),
            state.g_dd(m,0,2,k,j,i), state.g_dd(m,1,1,k,j,i),
            state.g_dd(m,1,2,k,j,i), state.g_dd(m,2,2,k,j,i));
        if (!(Kokkos::isfinite(alpha) && Kokkos::isfinite(chi) &&
              Kokkos::isfinite(detg) && alpha > 0.0 && chi > 0.0 && detg > 0.0)) {
          ++count;
        }
      }, invalid_points);

#if MPI_PARALLEL_ENABLED
  array_sum::GlobalSum global_coordinate_sums;
  array_sum::GlobalSum global_proper_sums;
  MPI_Allreduce(coordinate_sums.the_array, global_coordinate_sums.the_array,
                NREDUCTION_VARIABLES, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(proper_sums.the_array, global_proper_sums.the_array,
                NREDUCTION_VARIABLES, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  coordinate_sums = global_coordinate_sums;
  proper_sums = global_proper_sums;
  Real minima[3] = {minimum_alpha, minimum_chi, minimum_metric_minor};
  Real global_minima[3];
  MPI_Allreduce(minima, global_minima, 3, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  minimum_alpha = global_minima[0];
  minimum_chi = global_minima[1];
  minimum_metric_minor = global_minima[2];
  Real maxima[2] = {maximum_beta, maximum_gamma};
  Real global_maxima[2];
  MPI_Allreduce(maxima, global_maxima, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  maximum_beta = global_maxima[0];
  maximum_gamma = global_maxima[1];
  long long global_invalid_points = 0;
  MPI_Allreduce(&invalid_points, &global_invalid_points, 1, MPI_LONG_LONG,
                MPI_SUM, MPI_COMM_WORLD);
  invalid_points = global_invalid_points;
#endif

  const auto rms = [](const array_sum::GlobalSum &sum, const int region,
                      const int quantity) {
    const Real volume = sum.the_array[5 * region + 4];
    return (volume > 0.0) ? std::sqrt(sum.the_array[5 * region + quantity] / volume)
                          : std::numeric_limits<Real>::quiet_NaN();
  };

  pdata->nhist = 20;
  const char *labels[20] = {
      "min_alpha", "min_chi", "min_minor", "max_beta", "max_Gamma", "invalid",
      "g_G", "g_dchi", "g_da", "g_eta", "c_G", "c_dchi", "c_da", "c_eta",
      "h_G", "h_eta", "e_G", "e_dchi", "e_da", "e_eta"};
  for (int entry = 0; entry < pdata->nhist; ++entry) pdata->label[entry] = labels[entry];
  Real values[20] = {
      minimum_alpha, minimum_chi, minimum_metric_minor, maximum_beta, maximum_gamma,
      static_cast<Real>(invalid_points),
      rms(coordinate_sums, 0, 0), rms(coordinate_sums, 0, 1),
      rms(coordinate_sums, 0, 2), rms(coordinate_sums, 0, 3),
      rms(coordinate_sums, 1, 0), rms(coordinate_sums, 1, 1),
      rms(coordinate_sums, 1, 2), rms(coordinate_sums, 1, 3),
      rms(coordinate_sums, 2, 0), rms(coordinate_sums, 2, 3),
      rms(coordinate_sums, 3, 0), rms(coordinate_sums, 3, 1),
      rms(coordinate_sums, 3, 2), rms(coordinate_sums, 3, 3)};
  for (int entry = 0; entry < pdata->nhist; ++entry) {
    pdata->hdata[entry] = (global_variable::my_rank == 0) ? values[entry] : 0.0;
  }

  if (global_variable::my_rank == 0) {
    FILE *output = std::fopen(output_path.c_str(), "a+");
    if (output == nullptr) {
      std::fprintf(stderr, "Unable to open gauge diagnostic file %s\n", output_path.c_str());
      std::exit(EXIT_FAILURE);
    }
    std::fseek(output, 0, SEEK_END);
    if (std::ftell(output) == 0) {
      std::fprintf(output,
          "# time,dt,profile,min_alpha,min_chi,min_metric_minor,max_beta,max_Gamma,"
          "invalid_points");
      const char *regions[4] = {"global", "core_r_le_0p25", "ah_0p25_to_1", "ext_2_to_8"};
      const char *measures[2] = {"coordinate", "proper"};
      for (int measure = 0; measure < 2; ++measure) {
        for (int region = 0; region < 4; ++region) {
          std::fprintf(output,
              ",%s_%s_volume,%s_%s_gamma_rms,%s_%s_chi_gradient_rms,"
              "%s_%s_lapse_gradient_rms,%s_%s_damping_rms",
              measures[measure], regions[region], measures[measure], regions[region],
              measures[measure], regions[region], measures[measure], regions[region],
              measures[measure], regions[region]);
        }
      }
      std::fprintf(output, "\n");
    }
    std::fprintf(output, "%.17g,%.17g,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%lld",
                 pm->time, pm->dt, profile_name.c_str(), minimum_alpha, minimum_chi,
                 minimum_metric_minor, maximum_beta, maximum_gamma, invalid_points);
    const array_sum::GlobalSum sums[2] = {coordinate_sums, proper_sums};
    for (int measure = 0; measure < 2; ++measure) {
      for (int region = 0; region < 4; ++region) {
        std::fprintf(output, ",%.17g,%.17g,%.17g,%.17g,%.17g",
                     sums[measure].the_array[5 * region + 4], rms(sums[measure], region, 0),
                     rms(sums[measure], region, 1), rms(sums[measure], region, 2),
                     rms(sums[measure], region, 3));
      }
    }
    std::fprintf(output, "\n");
    std::fclose(output);
  }
}

inline void GaugeDiagnostics(HistoryData *pdata, Mesh *pm) {
  switch (pm->mb_indcs.ng) {
    case 2: LoadGaugeDiagnostics<2>(pdata, pm); break;
    case 3: LoadGaugeDiagnostics<3>(pdata, pm); break;
    case 4: LoadGaugeDiagnostics<4>(pdata, pm); break;
    default:
      std::fprintf(stderr, "Unsupported nghost=%d for puncture gauge diagnostics\n",
                   pm->mb_indcs.ng);
      std::exit(EXIT_FAILURE);
  }
}

}  // namespace z4c_puncture_gauge_diagnostics

#endif  // PGEN_Z4C_Z4C_ONE_PUNCTURE_GAUGE_DIAGNOSTICS_HPP_
