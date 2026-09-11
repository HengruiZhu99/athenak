#ifndef Z4C_SPATIAL_TIMESTEP_POINT_HPP_
#define Z4C_SPATIAL_TIMESTEP_POINT_HPP_
#include "coordinates/adm.hpp"
#include "z4c/timestep_contract.hpp"
#include "z4c/z4c.hpp"
namespace z4c {
// Shared by the native mesh and auxiliary hierarchy. No reduction or time
// ownership here: callers must supply a common-time global gauge coefficient.
template<class State>
KOKKOS_INLINE_FUNCTION
Real SpatialTimestepPoint(const State &state,const RegionSize &cell_size,
    const Z4c::Options &opt,int m,int k,int j,int i,Real max_abs_K,
    Real shift_gamma,bool active_x2,bool active_x3,bool minimum) {
  Real result=minimum ? std::numeric_limits<Real>::max() : Real(0);
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
      return result;
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
      return result;
    }
    const auto telegraph = ScaleInvariantTelegraphCoefficients(
        1.0, max_abs_K, opt.telegraph_tau, opt.telegraph_kappa);
    const Real diagonal[3] = {g_uu[0], g_uu[3], g_uu[5]};
    const Real spacing[3] = {cell_size.dx1, cell_size.dx2, cell_size.dx3};
    for (int direction = 0; direction < 3; ++direction) {
      if ((direction == 1 && !active_x2) || (direction == 2 && !active_x3)) continue;
      const Real conformal_inverse = diagonal[direction];
      const Real physical_inverse = physical_factor * conformal_inverse;
      if (!Kokkos::isfinite(conformal_inverse) || !(conformal_inverse > 0.0) ||
          !Kokkos::isfinite(physical_inverse) || !(physical_inverse > 0.0) ||
          !Kokkos::isfinite(spacing[direction]) || !(spacing[direction] > 0.0)) {
        result = minimum ? 0.0 : std::numeric_limits<Real>::infinity();
        return result;
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
        return result;
      }
      result = minimum ? fmin(result, spacing[direction] / coordinate_speed)
                       : fmax(result, coordinate_speed);
    }
  return result;
}
}
#endif
