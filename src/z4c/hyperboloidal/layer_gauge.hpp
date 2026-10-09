// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_LAYER_GAUGE_HPP_
#define Z4C_HYPERBOLOIDAL_LAYER_GAUGE_HPP_

#include <initializer_list>
#include "z4c/hyperboloidal/layer_reference.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"

namespace z4c {
namespace hyperboloidal {

struct LayerGaugeParameters {
  double r0 = 0.45, r1 = 0.85, q0 = 0.5;
  double lapse_inner = 0, lapse_outer = 1.5, shift_inner = 0, shift_outer = 1;
  bool preferred_source = true;
  void Validate(double s) const {
    if (!std::isfinite(r0) || !std::isfinite(r1) || !std::isfinite(q0) ||
        !(0 < r0 && r0 < r1 && r1 < s) || !(0 < q0 && q0 < 1)) {
      throw std::invalid_argument("gauge layer requires 0<r0<r1<S and 0<q0<1");
    }
    for (double v : {lapse_inner, lapse_outer, shift_inner, shift_outer}) {
      if (!std::isfinite(v) || v < 0) {
        throw std::invalid_argument("layer restoring rates must be finite >=0");
      }
    }
    if (lapse_outer < lapse_inner || shift_outer < shift_inner) {
      throw std::invalid_argument("layer restoring rates must increase outward");
    }
  }
};

template <typename T>
struct LayerGaugeCoefficients {
  T weight, f, alpha2f, mu, ea, ec, q, nu, eta;
};

template <typename T>
KOKKOS_INLINE_FUNCTION LayerGaugeCoefficients<T> LayerCoefficients(
    T r, T alpha, const LayerGaugeParameters& g) {
  const T w = SmoothCutoff(r, T(g.r0), T(g.r1)).value, e = 1 - w;
  return {w,
          1 + 2 * e / alpha,
          alpha * alpha + 2 * e * alpha,
          e * 3 * T(g.q0) / 4 + w,
          w,
          w / 2,
          T(g.q0) + (1 - T(g.q0)) * w,
          T(g.lapse_inner) + T(g.lapse_outer - g.lapse_inner) * w,
          T(g.shift_inner) + T(g.shift_outer - g.shift_inner) * w};
}

// Evolve the stored lapse, shift and stabilized physical trace. Log variables
// are gauge definitions, not an assumed O falloff for arbitrary live fields.
// This is an INTERIOR-ONLY assembly: pole.alpha/O is still singular off the
// compatible scri manifold. No Q formed by subtracting 3*wn from the full P.
template <typename T>
KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> InteriorLayerGauge(
    const LayerPoint<T>& p, const Z4cJet<T>& u, const LayerGaugeParameters& g) {
  GaugeRHSParts<T> out{};
  const T alpha = u.alpha.value, chi = u.chi.value;
  if (!(alpha > 0) || !(chi > 0) || !Kokkos::isfinite(alpha) || !Kokkos::isfinite(chi)) {
    return out;
  }
  const auto geo = Geometry(u.metric);
  if (!geo.valid)
    return out;
  const auto c = LayerCoefficients(p.radius, alpha, g);
  const T da = alpha - p.alpha;
  const T loga = Kokkos::log1p(da / p.alpha);
  T db[3], du[3], dv[3], ref_advection[3]{};
  T q_numerator = u.trace.value - p.k_physical, ref_log_advection = 0;
  for (int i = 0; i < 3; ++i) {
    db[i] = u.beta.value[i] - p.beta[i];
    du[i] = u.alpha.d[i] / alpha - p.dalpha[i] / p.alpha;
    dv[i] = u.chi.d[i] / chi - p.state.chi.d[i] / p.state.chi.value;
    // O*(Q-Qhat) expanded in physical trace and gauge deviations.
    q_numerator += 3 * (db[i] - p.beta[i] * da / p.alpha) * p.domega[i] / alpha;
    ref_log_advection += u.beta.value[i] * p.dalpha[i] / p.alpha;
    out.regular.alpha += u.beta.value[i] * u.alpha.d[i];
    for (int j = 0; j < 3; ++j) {
      ref_advection[i] += p.beta[j] * p.state.beta.d[j][i];
    }
  }
  out.regular.alpha -= alpha * ref_log_advection + alpha * c.nu * loga;
  out.pole.alpha = -c.alpha2f * q_numerator;
  for (int i = 0; i < 3; ++i) {
    out.regular.beta[i] =
        alpha * alpha * chi * c.mu * (u.lambda.value[i] - p.state.lambda.value[i]) -
        ref_advection[i] - c.eta * db[i];
    for (int j = 0; j < 3; ++j) {
      out.regular.beta[i] +=
          u.beta.value[j] * u.beta.d[j][i] +
          alpha * alpha * chi * geo.inverse[i][j] * (c.ec * dv[j] - c.ea * du[j]);
    }
  }
  if (g.preferred_source && c.weight > 0) {
    // Algebraic extension of the harmonic collar's F^a to the transition.
    // At W=1: Gamma4^a+2 Z4^a=F^a. F0 is computed without Q or Theta/O.
    const T f0 = (ref_log_advection + c.nu * loga) / (alpha * alpha) - p.k_bar / alpha;
    T norm = 0, contraction = 0, hessian4 = 0;
    for (int i = 0; i < 3; ++i) {
      norm += p.domega[i] * p.domega[i];
      T source = chi * p.state.lambda.value[i] + ref_advection[i] / (alpha * alpha) +
                 c.eta * db[i] / (alpha * alpha) - u.beta.value[i] * f0;
      for (int j = 0; j < 3; ++j) {
        source += chi * geo.inverse[i][j] *
                  (p.state.chi.d[j] / (2 * p.state.chi.value) - p.dalpha[j] / p.alpha);
        hessian4 += (chi * geo.inverse[i][j] -
                     u.beta.value[i] * u.beta.value[j] / (alpha * alpha)) *
                    p.omega_hessian[i][j];
      }
      contraction += p.domega[i] * source;
    }
    if (!(norm > 0))
      return out;  // host parameters must place projection outside r0
    const T delta = hessian4 - p.omega * p.w_omega - contraction;
    for (int i = 0; i < 3; ++i) {
      out.regular.beta[i] -= c.weight * alpha * alpha * p.domega[i] * delta / norm;
    }
  }
  out.valid = Kokkos::isfinite(out.regular.alpha) && Kokkos::isfinite(out.pole.alpha);
  for (int i = 0; i < 3; ++i)
    out.valid = out.valid && Kokkos::isfinite(out.regular.beta[i]);
  return out;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_LAYER_GAUGE_HPP_
