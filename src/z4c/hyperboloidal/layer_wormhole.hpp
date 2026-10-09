// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_LAYER_WORMHOLE_HPP_
#define Z4C_HYPERBOLOIDAL_LAYER_WORMHOLE_HPP_

#include "z4c/hyperboloidal/layer_reference.hpp"

namespace z4c {
namespace hyperboloidal {

// Schwarzschild wormhole INITIAL DATA, with a mass-corrected outer height.
// The supplied reference remains Minkowski and is never changed by this class.
// In isotropic physical radius R=r/Omega, psi=1+M/(2R), N=(1-m)/(1+m),
// m=M/(2R). If v is the Minkowski reference's boost, choose h'_M=psi^2*v/N
// only outside the throat neighborhood. Then gamma_bar^BH=psi^4*gamma_bar^ref
// exactly. The outer height has the required R+2M*log(R)+O(1/R) asymptotics.
// The Cauchy interior is the time-symmetric Einstein-Rosen wormhole; it has
// a positive pre-collapsed lapse, not the signed static Schwarzschild lapse.
// These are constraint-satisfying geometric data, not a stationary solution of
// the reference gauge. No RHS subtraction or analytic evolution is provided.
template <typename T>
class LayerWormhole {
 public:
  LayerWormhole(const LayerReference<T>& reference, T mass)
      : reference_(reference), mass_(mass) {
    reference_.Validate();
    if (!reference_.layer.enabled || !(mass > 0) || !std::isfinite(mass) ||
        !(mass < 2 * reference_.layer.r0)) {
      throw std::invalid_argument(
          "layer wormhole requires enabled layer and 0<M<2*layer_r0");
    }
  }

  T mass() const { return mass_; }
  const LayerReference<T>& reference() const { return reference_; }

  // Analytic jets required by ConformalRHS on 0<r<=S. The exact puncture has
  // chi=alpha=0 and is a coordinate limit, not a valid live PDE/ADM point.
  // The adapter must use a cell-centered grid excluding that exact point.
  KOKKOS_INLINE_FUNCTION Z4cJet<T> At(T x, T y, T z) const {
    const T r = Kokkos::sqrt(x * x + y * y + z * z);
    if (r == 0) {
      Z4cJet<T> origin{};
      for (int i = 0; i < 3; ++i) {
        origin.metric.g[i][i] = 1;
        origin.alpha.dd[i][i] = 8 / (mass_ * mass_);
      }
      return origin;
    }
    const auto axis_reference = reference_.At(r, T(0), T(0));
    const auto& ur = axis_reference.state;
    const T a = reference_.curvature_radius;
    const Radial2<T> one{1, 0, 0}, radius{r, 1, 0}, mass_half{mass_ / 2, 0, 0};
    const Radial2<T> omega{axis_reference.omega, axis_reference.domega[0],
                           axis_reference.omega_hessian[0][0]};
    const auto cutoff = SmoothCutoff(r, T(reference_.layer.r0), T(reference_.layer.r1));
    const Radial2<T> w{cutoff.value, cutoff.d, cutoff.dd};
    // r/(r+M*Omega/2) avoids a huge psi near the puncture; no numerical floor.
    const Radial2<T> inverse_psi = radius / (radius + mass_half * omega);
    const Radial2<T> lapse_static =
        (radius - mass_half * omega) / (radius + mass_half * omega);
    const Radial2<T> psi_inverse2 = inverse_psi * inverse_psi;
    const Radial2<T> alpha_ref{ur.alpha.value, ur.alpha.d[0], ur.alpha.dd[0][0]};
    const Radial2<T> chi_ref{ur.chi.value, ur.chi.d[0], ur.chi.dd[0][0]};
    const Radial2<T> beta_ref{ur.beta.value[0], ur.beta.d[0][0], ur.beta.dd[0][0][0]};
    const auto alpha = alpha_ref * ((one - w) * psi_inverse2 + w * lapse_static);
    const auto chi = chi_ref * psi_inverse2 * psi_inverse2;
    const auto beta = beta_ref * lapse_static * psi_inverse2;
    Z4cJet<T> axis = ur;
    axis.alpha = {alpha.value, {alpha.d, 0, 0}, {{alpha.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    axis.chi = {chi.value, {chi.d, 0, 0}, {{chi.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    axis.beta = {};
    axis.beta.value[0] = beta.value;
    axis.beta.d[0][0] = beta.d;
    axis.beta.dd[0][0][0] = beta.dd;
    axis.trace = {};
    axis.a = {};
    if (cutoff.value > 0 || cutoff.d != 0) {
      // This branch is strictly outside the throat. Do not evaluate the static
      // 1/(1-m^2) formula anywhere in the exactly Cauchy throat neighborhood.
      const T s = reference_.scri_radius;
      const T outer = (s - r) * (s + r) / (2 * a * s);
      const T outer_d = -r / (a * s), outer_dd = -1 / (a * s);
      const T omega_ddd =
          cutoff.ddd * (outer - 1) + 3 * cutoff.dd * outer_d + 3 * cutoff.d * outer_dd;
      const Radial2<T> ell{axis_reference.L, -r * omega.dd, -omega.dd - r * omega_ddd};
      const Radial2<T> dw{cutoff.d, cutoff.dd, cutoff.ddd};
      const Radial2<T> m = mass_half * omega / radius;
      const Radial2<T> denominator = one - m * m;
      const auto eta = radius * omega * dw / ell;
      const Radial2<T> inv_a{-1 / a, 0, 0};
      const auto kr =
          inv_a * psi_inverse2 * (w + eta + Radial2<T>{2, 0, 0} * w * m / denominator);
      const auto kt = inv_a * psi_inverse2 * w * lapse_static;
      const auto trace = kr + Radial2<T>{2, 0, 0} * kt;
      axis.trace = {
          trace.value, {trace.d, 0, 0}, {{trace.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
      // Factored (kr-kt)/Omega: regular even at exact scri, no subtraction of
      // nearly equal curvature eigenvalues and no division by Omega.
      const auto shear =
          inv_a * psi_inverse2 *
          (radius * dw / ell + w * Radial2<T>{mass_, 0, 0} * (Radial2<T>{2, 0, 0} - m) /
                                   (radius * denominator));
      const Radial2<T> radial_metric{ur.metric.g[0][0], ur.metric.dg[0][0][0],
                                     ur.metric.ddg[0][0][0][0]};
      const Radial2<T> tangential_metric{ur.metric.g[1][1], ur.metric.dg[0][1][1],
                                         ur.metric.ddg[0][0][1][1]};
      const auto ar = Radial2<T>{T(2) / 3, 0, 0} * radial_metric * shear;
      const auto at = Radial2<T>{-T(1) / 3, 0, 0} * tangential_metric * shear;
      axis.a.k[0][0] = ar.value;
      axis.a.dk[0][0][0] = ar.d;
      for (int i = 1; i < 3; ++i) {
        axis.a.k[i][i] = at.value;
        axis.a.dk[0][i][i] = at.d;
      }
    }
    const T n[3] = {x / r, y / r, z / r};
    auto result = CartesianRadialJet(axis, r, n);
    // The spatial mass factor changes chi alone: use the reference metric and
    // connection jets verbatim, including the nonzero transition Lambda.
    const auto reference_point = reference_.At(x, y, z);
    result.metric = reference_point.state.metric;
    result.lambda = reference_point.state.lambda;
    return result;
  }

 private:
  LayerReference<T> reference_;
  T mass_;
};

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_LAYER_WORMHOLE_HPP_
