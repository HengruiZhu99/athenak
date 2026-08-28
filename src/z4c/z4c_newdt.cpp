//========================================================================================
// AthenaXXX astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
//========================================================================================
//! \file z4c_newdt.cpp
//! \brief Z4c spatial-characteristic and explicit-source timestep contracts.

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

#include "athena.hpp"
#include "coordinates/adm.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c/cartoon_derivatives.hpp"
#include "z4c/timestep_contract.hpp"
#include "z4c/z4c.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace z4c {
namespace {

ExplicitRKMethod MethodFromDriver(const Driver *driver) {
  ExplicitRKMethod method;
  method.stages = driver->nexp_stages;
  for (int stage = 0; stage < method.stages; ++stage) {
    method.gam0[stage] = driver->gam0[stage];
    method.gam1[stage] = driver->gam1[stage];
    method.beta[stage] = driver->beta[stage];
    method.delta[stage] = driver->delta[stage];
  }
  return method;
}

template <typename Centering, typename Symmetry, int NGHOST>
TaskStatus ComputeZ4cTimestepContracts(Z4c *self, MeshBlockPack *pack, Driver *driver) {
  auto *mesh = pack->pmesh;
  const auto &layout = self->layout;
  const int nmb = pack->nmb_thispack;
  const int nx1 = layout.ie - layout.is + 1;
  const int nx2 = layout.je - layout.js + 1;
  const int nx3 = layout.ke - layout.ks + 1;
  const int nmkji = nmb * nx3 * nx2 * nx1;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  const bool active_x2 = mesh->multi_d;
  const bool active_x3 = mesh->three_d;
  auto &size = pack->pmb->mb_size;
  auto &state = self->z4c;
  auto &telegraph_mu = self->u_telegraph_mu;
  const auto opt = self->opt;

  const bool use_max_K_scale =
      (opt.telegraph_lapse &&
       opt.telegraph_damping_prescription != TelegraphDampingPrescription::fixed) ||
      opt.shift_eta_max_K || opt.damp_kappa1_max_K;
  Real max_abs_K = 1.0;
  if (use_max_K_scale) {
    max_abs_K = 0.0;
    Kokkos::parallel_reduce(
        "z4c timestep max abs K", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
        KOKKOS_LAMBDA(const int idx, Real &result) {
          const int m = idx / nkji;
          const int k = (idx - m * nkji) / nji + layout.ks;
          const int j = (idx - m * nkji - (k - layout.ks) * nji) / nx1 + layout.js;
          const int i = idx - m * nkji - (k - layout.ks) * nji -
                        (j - layout.js) * nx1 + layout.is;
          const Real K = state.vKhat(m, k, j, i) + 2.0 * state.vTheta(m, k, j, i);
          if (!Kokkos::isfinite(K)) {
            result = std::numeric_limits<Real>::infinity();
          } else {
            result = fmax(result, Kokkos::fabs(K));
          }
        }, Kokkos::Max<Real>(max_abs_K));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &max_abs_K, 1, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
  }

  Real kappa1_effective = opt.damp_kappa1;
  if (opt.roll_kappa && mesh->time >= opt.kappa_roll_start_time) {
    const Real s = (mesh->time - opt.kappa_roll_start_time) / opt.roll_window;
    const Real stitch = std::exp(-2.30258509299 * s * s);
    kappa1_effective = opt.target_kappa1 +
                       (opt.damp_kappa1 - opt.target_kappa1) * stitch;
  }
  const Real kappa1_eff =
      kappa1_effective * (opt.damp_kappa1_max_K ? max_abs_K : 1.0);
  const Real shift_eta_eff =
      opt.shift_eta * (opt.shift_eta_max_K ? max_abs_K : 1.0);
  const Real slow_start_factor =
      opt.slow_start_lapse
          ? opt.ssl_damping_amp *
                std::exp(-0.5 * std::pow(mesh->time / opt.ssl_damping_time, 2))
          : 0.0;
  const Real shift_gamma =
      (1.0 - opt.sss_damping_amp *
                 std::exp(-0.5 * std::pow(mesh->time / opt.sss_damping_time, 2))) *
      opt.shift_ggamma;

  // Recompute the current local telegraph profile before taking the source maximum. This
  // covers t=0, where a previous RHS profile does not yet exist, with the same helper the
  // RHS uses for every prescription.
  Real local_max_source_rate = 0.0;
  Kokkos::parallel_reduce(
      "z4c timestep source rate", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
      KOKKOS_LAMBDA(const int idx, Real &result) {
        const int m = idx / nkji;
        const int k = (idx - m * nkji) / nji + layout.ks;
        const int j = (idx - m * nkji - (k - layout.ks) * nji) / nx1 + layout.js;
        const int i = idx - m * nkji - (k - layout.ks) * nji -
                      (j - layout.js) * nx1 + layout.is;
        const Real alpha = state.alpha(m, k, j, i);
        const Real chi = state.chi(m, k, j, i);
        const Real K = state.vKhat(m, k, j, i) + 2.0 * state.vTheta(m, k, j, i);
        const Real detg = adm::SpatialDet(
            state.g_dd(m, 0, 0, k, j, i), state.g_dd(m, 0, 1, k, j, i),
            state.g_dd(m, 0, 2, k, j, i), state.g_dd(m, 1, 1, k, j, i),
            state.g_dd(m, 1, 2, k, j, i), state.g_dd(m, 2, 2, k, j, i));
        if (!Kokkos::isfinite(alpha) || !Kokkos::isfinite(chi) ||
            (!opt.lapse_shock_avoiding && !(alpha > 0.0)) ||
            !(chi > 0.0) || !Kokkos::isfinite(detg) || !(detg > 0.0)) {
          result = std::numeric_limits<Real>::infinity();
          return;
        }
        Real g_uu[6];
        adm::SpatialInv(1.0 / detg,
                        state.g_dd(m, 0, 0, k, j, i), state.g_dd(m, 0, 1, k, j, i),
                        state.g_dd(m, 0, 2, k, j, i), state.g_dd(m, 1, 1, k, j, i),
                        state.g_dd(m, 1, 2, k, j, i), state.g_dd(m, 2, 2, k, j, i),
                        &g_uu[0], &g_uu[1], &g_uu[2], &g_uu[3], &g_uu[4], &g_uu[5]);
        Real rate = 0.0;
        if (opt.telegraph_lapse) {
          Real mu = 1.0;
          if (opt.telegraph_damping_prescription ==
              TelegraphDampingPrescription::max_domain_abs_K) {
            mu = max_abs_K;
          } else if (opt.telegraph_damping_prescription ==
                     TelegraphDampingPrescription::local_abs_K) {
            mu = LocalAbsKTelegraphMu(K);
          } else if (opt.telegraph_damping_prescription ==
                     TelegraphDampingPrescription::local_extrinsic_curvature_norm) {
            mu = LocalExtrinsicCurvatureNormTelegraphMu(
                K, g_uu[0], g_uu[1], g_uu[2], g_uu[3], g_uu[4], g_uu[5],
                state.vA_dd(m, 0, 0, k, j, i), state.vA_dd(m, 0, 1, k, j, i),
                state.vA_dd(m, 0, 2, k, j, i), state.vA_dd(m, 1, 1, k, j, i),
                state.vA_dd(m, 1, 2, k, j, i), state.vA_dd(m, 2, 2, k, j, i));
          } else if (opt.telegraph_damping_prescription ==
                     TelegraphDampingPrescription::local_chi_gradient_norm) {
            const Real inverse_spacing[] = {1.0 / size.d_view(m).dx1,
                                             1.0 / size.d_view(m).dx2,
                                             1.0 / size.d_view(m).dx3};
            auto derivatives = MakeZ4cDerivativeProvider<Centering, Symmetry, NGHOST>(
                inverse_spacing, size.d_view, layout.nx1, layout.is, m, k, j, i,
                layout.nx3 == 1);
            mu = LocalChiGradientNormTelegraphMu(
                chi, opt.chi_psi_power, g_uu[0], g_uu[1], g_uu[2], g_uu[3], g_uu[4],
                g_uu[5], derivatives.ScalarFirst(0, state.chi),
                derivatives.ScalarFirst(1, state.chi),
                derivatives.ScalarFirst(2, state.chi));
          }
          telegraph_mu(m, 0, k, j, i) = mu;
          const auto coefficients = ScaleInvariantTelegraphCoefficients(
              mu, max_abs_K, opt.telegraph_tau, opt.telegraph_kappa);
          if (!Kokkos::isfinite(coefficients.damping) || coefficients.damping < 0.0) {
            result = std::numeric_limits<Real>::infinity();
            return;
          }
          rate = fmax(rate, coefficients.damping);
        }
        if (opt.shift_mode != Z4cShiftMode::prescribed_zero && shift_eta_eff > 0.0) {
          rate = fmax(rate, shift_eta_eff);
        }
        if (kappa1_eff > 0.0) {
          rate = fmax(rate, 2.0 * alpha * kappa1_eff);  // Gamma damping
          if (opt.use_z4c && (2.0 + opt.damp_kappa2) > 0.0) {
            rate = fmax(rate, alpha * (2.0 + opt.damp_kappa2) * kappa1_eff);
          }
        }
        if (opt.slow_start_lapse) {
          const Real W = chi > opt.chi_min_floor ? Kokkos::sqrt(chi) :
                                                    Kokkos::sqrt(opt.chi_min_floor);
          const Real lapse_rate = slow_start_factor * Kokkos::pow(W, opt.ssl_damping_index);
          if (!Kokkos::isfinite(lapse_rate) || lapse_rate < 0.0) {
            result = std::numeric_limits<Real>::infinity();
            return;
          }
          rate = fmax(rate, lapse_rate);
        }
        if (!Kokkos::isfinite(rate) || rate < 0.0) {
          result = std::numeric_limits<Real>::infinity();
          return;
        }
        result = fmax(result, rate);
      }, Kokkos::Max<Real>(local_max_source_rate));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &local_max_source_rate, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif

  auto ComputeCoordinateSpeed = KOKKOS_LAMBDA(const int idx, Real &result,
                                                const bool minimum) {
    const int m = idx / nkji;
    const int k = (idx - m * nkji) / nji + layout.ks;
    const int j = (idx - m * nkji - (k - layout.ks) * nji) / nx1 + layout.js;
    const int i = idx - m * nkji - (k - layout.ks) * nji -
                  (j - layout.js) * nx1 + layout.is;
    const Real alpha = state.alpha(m, k, j, i);
    const Real chi = state.chi(m, k, j, i);
    const Real detg = adm::SpatialDet(
        state.g_dd(m, 0, 0, k, j, i), state.g_dd(m, 0, 1, k, j, i),
        state.g_dd(m, 0, 2, k, j, i), state.g_dd(m, 1, 1, k, j, i),
        state.g_dd(m, 1, 2, k, j, i), state.g_dd(m, 2, 2, k, j, i));
    if (!Kokkos::isfinite(alpha) || !Kokkos::isfinite(chi) ||
        (!opt.lapse_shock_avoiding && !(alpha > 0.0)) ||
        !(chi > 0.0) || !Kokkos::isfinite(detg) || !(detg > 0.0)) {
      result = minimum ? 0.0 : std::numeric_limits<Real>::infinity();
      return;
    }
    Real g_uu[6];
    adm::SpatialInv(1.0 / detg,
                    state.g_dd(m, 0, 0, k, j, i), state.g_dd(m, 0, 1, k, j, i),
                    state.g_dd(m, 0, 2, k, j, i), state.g_dd(m, 1, 1, k, j, i),
                    state.g_dd(m, 1, 2, k, j, i), state.g_dd(m, 2, 2, k, j, i),
                    &g_uu[0], &g_uu[1], &g_uu[2], &g_uu[3], &g_uu[4], &g_uu[5]);
    const Real physical_factor = Kokkos::pow(chi, -4.0 / opt.chi_psi_power);
    const Real lapse_f = opt.lapse_oplog * opt.lapse_harmonicf +
                         opt.lapse_harmonic * alpha;
    const Real gamma_driver_coefficient =
        opt.shift_mode == Z4cShiftMode::prescribed_zero
            ? 0.0
            : shift_gamma + opt.shift_alpha2ggamma * alpha * alpha;
    if (!Kokkos::isfinite(physical_factor) || !(physical_factor > 0.0) ||
        !Kokkos::isfinite(lapse_f) || lapse_f < 0.0 ||
        !Kokkos::isfinite(gamma_driver_coefficient) || gamma_driver_coefficient < 0.0) {
      result = minimum ? 0.0 : std::numeric_limits<Real>::infinity();
      return;
    }
    const auto telegraph = ScaleInvariantTelegraphCoefficients(
        1.0, max_abs_K, opt.telegraph_tau, opt.telegraph_kappa);
    const Real diagonal[3] = {g_uu[0], g_uu[3], g_uu[5]};
    const Real spacing[3] = {size.d_view(m).dx1, size.d_view(m).dx2, size.d_view(m).dx3};
    for (int direction = 0; direction < 3; ++direction) {
      if ((direction == 1 && !active_x2) || (direction == 2 && !active_x3)) continue;
      const Real conformal_inverse = diagonal[direction];
      const Real physical_inverse = physical_factor * conformal_inverse;
      if (!Kokkos::isfinite(conformal_inverse) || !(conformal_inverse > 0.0) ||
          !Kokkos::isfinite(physical_inverse) || !(physical_inverse > 0.0) ||
          !Kokkos::isfinite(spacing[direction]) || !(spacing[direction] > 0.0)) {
        result = minimum ? 0.0 : std::numeric_limits<Real>::infinity();
        return;
      }
      Real lapse_speed = alpha * Kokkos::sqrt(lapse_f * physical_inverse);
      if (opt.lapse_shock_avoiding) {
        lapse_speed = Kokkos::sqrt(
            (alpha * alpha + opt.lapse_shock_avoiding_kappa) * physical_inverse);
      }
      const Real telegraph_speed = opt.telegraph_lapse
          ? Kokkos::sqrt(chi * telegraph.gradient * conformal_inverse) : 0.0;
      const Real gamma_speed = opt.shift_mode != Z4cShiftMode::prescribed_zero
          ? Kokkos::sqrt((4.0 / 3.0) * gamma_driver_coefficient * conformal_inverse)
          : 0.0;
      const Real coordinate_speed = CoordinateCharacteristicSpeed(
          state.beta_u(m, direction, k, j, i),
          Kokkos::fabs(alpha) * Kokkos::sqrt(physical_inverse),
          lapse_speed, telegraph_speed, gamma_speed);
      if (!Kokkos::isfinite(coordinate_speed) || !(coordinate_speed > 0.0)) {
        result = minimum ? 0.0 : std::numeric_limits<Real>::infinity();
        return;
      }
      result = minimum ? fmin(result, spacing[direction] / coordinate_speed)
                       : fmax(result, coordinate_speed);
    }
  };

  Real local_dt_spatial = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("z4c timestep spatial", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
                          KOKKOS_LAMBDA(const int idx, Real &result) {
                            ComputeCoordinateSpeed(idx, result, true);
                          }, Kokkos::Min<Real>(local_dt_spatial));
  Real local_max_speed = 0.0;
  Kokkos::parallel_reduce("z4c timestep speed", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
                          KOKKOS_LAMBDA(const int idx, Real &result) {
                            ComputeCoordinateSpeed(idx, result, false);
                          }, Kokkos::Max<Real>(local_max_speed));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &local_dt_spatial, 1, MPI_ATHENA_REAL, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &local_max_speed, 1, MPI_ATHENA_REAL, MPI_MAX,
                MPI_COMM_WORLD);
#endif

  self->negative_real_stability_radius = ExplicitRKNegativeRealStabilityRadius(MethodFromDriver(driver));
  self->max_source_rate = local_max_source_rate;
  self->max_coordinate_speed = local_max_speed;
  self->dt_spatial = local_dt_spatial;
  self->dt_source = SourceTimestepCeiling(opt.timestep_source_safety,
                                           self->negative_real_stability_radius,
                                           self->max_source_rate);
  self->dtnew = self->dt_spatial;
  if (!std::isfinite(max_abs_K) || !std::isfinite(self->negative_real_stability_radius) ||
      !(self->negative_real_stability_radius > 0.0) ||
      !std::isfinite(self->max_source_rate) || !(self->max_source_rate >= 0.0) ||
      !std::isfinite(self->max_coordinate_speed) || !(self->max_coordinate_speed > 0.0) ||
      !std::isfinite(self->dt_spatial) || !(self->dt_spatial > 0.0) ||
      !std::isfinite(self->dt_source) || !(self->dt_source > 0.0)) {
    std::cerr << "### FATAL ERROR in Z4c timestep contract: nonfinite or nonpositive "
              << "rate/speed/limit at cycle " << mesh->ncycle << " time " << mesh->time
              << " max_abs_K=" << max_abs_K << " max_source_rate=" << self->max_source_rate
              << " max_coordinate_speed=" << self->max_coordinate_speed
              << " dt_spatial=" << self->dt_spatial << " dt_source=" << self->dt_source
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

template <int NGHOST>
TaskStatus DispatchZ4cTimestepContracts(Z4c *self, MeshBlockPack *pack,
                                        Driver *driver) {
  const bool vertex = self->layout.centering == Z4cGridCentering::vertex;
  const bool cartoon = pack->z4c_symmetry.mode == Z4cSymmetryMode::cartoon_so2;
  if (cartoon) {
    return vertex
               ? ComputeZ4cTimestepContracts<VertexCenteredZ4c, CartoonSO2, NGHOST>(
                     self, pack, driver)
               : ComputeZ4cTimestepContracts<CellCenteredZ4c, CartoonSO2, NGHOST>(
                     self, pack, driver);
  }
  return vertex
             ? ComputeZ4cTimestepContracts<VertexCenteredZ4c, Cartesian3D, NGHOST>(
                   self, pack, driver)
             : ComputeZ4cTimestepContracts<CellCenteredZ4c, Cartesian3D, NGHOST>(
                   self, pack, driver);
}

}  // namespace

TaskStatus Z4c::NewTimeStep(Driver *driver, int stage) {
  if (stage != driver->nexp_stages) return TaskStatus::complete;
  switch (opt.fd_stencil) {
    case 2:
      return DispatchZ4cTimestepContracts<2>(this, pmy_pack, driver);
    case 3:
      return DispatchZ4cTimestepContracts<3>(this, pmy_pack, driver);
    case 4:
      return DispatchZ4cTimestepContracts<4>(this, pmy_pack, driver);
    default:
      std::cerr << "### FATAL ERROR in Z4c timestep contract: unsupported fd stencil "
                << opt.fd_stencil << std::endl;
      std::exit(EXIT_FAILURE);
  }
}

void Z4c::WriteTimestepContractRecord(const Real final_dt) const {
  if (global_variable::my_rank != 0) return;
  std::ofstream output("z4c_timestep_contract.csv", std::ios::app);
  if (!output) {
    std::cerr << "### FATAL ERROR: failed to write z4c timestep contract record" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (output.tellp() == 0) {
    output << "cycle,time,dt_spatial,dt_source,dt_final,max_source_rate,"
              "max_coordinate_speed,negative_real_stability_radius,limiter\n";
  }
  const Real spatial_limited = pmy_pack->pmesh->cfl_no * dt_spatial;
  const char *limiter = final_dt <= dt_source && dt_source <= spatial_limited
                            ? "z4c_source"
                            : (final_dt <= spatial_limited ? "z4c_spatial_or_other"
                                                           : "other_or_tlim");
  output << pmy_pack->pmesh->ncycle << ',' << std::setprecision(17)
         << pmy_pack->pmesh->time << ',' << dt_spatial << ',' << dt_source << ','
         << final_dt << ',' << max_source_rate << ',' << max_coordinate_speed << ','
         << negative_real_stability_radius << ',' << limiter << '\n';
  if (!output) {
    std::cerr << "### FATAL ERROR: failed while writing z4c timestep contract record"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace z4c
