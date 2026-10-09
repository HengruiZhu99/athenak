// Scratch-only local stationary Schwarzschild slices, not BH initial data.
#ifndef SCRATCH_LOCAL_TRUMPET_HPP_
#define SCRATCH_LOCAL_TRUMPET_HPP_
#include "z4c/hyperboloidal/layer_gauge.hpp"

namespace inner_audit {
namespace hyp = z4c::hyperboloidal;
using R2 = hyp::Radial2<double>;
using Jet = hyp::Z4cJet<double>;
inline R2 C(double v) { return {v, 0, 0}; }
struct LocalTrumpet {
  double mass, radius0, exponent, epsilon = .05;
  double S0() const { return std::sqrt(2 * mass / radius0 - 1); }
  double K0() const { return (3 * mass - 2 * radius0) / (radius0 * radius0 * S0()); }
  double AlphaR0() const { return 2 * K0() / S0(); }
  double V() const { return 2 * K0() / exponent; }
  double Zeta() const { return exponent / (AlphaR0() * radius0); }
  Jet At(double x, double y, double z) const {
    const double r = std::hypot(std::hypot(x, y), z), p = exponent, R0 = radius0;
    if (!(r > 0 && mass > 0 && R0 > 0 && R0 < 1.5 * mass && p > 0))
      throw std::invalid_argument("local trumpet domain");
    // R=R0+epsilon*M*(r/M)^p. Keep delta separate for the lapse root.
    const double dv = epsilon * mass * std::pow(r / mass, p);
    const R2 d{dv, p * dv / r, p * (p - 1) * dv / (r * r)};
    const R2 rp{d.d, d.dd, p * (p - 1) * (p - 2) * dv / (r * r * r)};
    const R2 R = C(R0) + d, rr{r, 1, 0};
    const double c4 = R0 * R0 * R0 * (2 * mass - R0);
    const R2 aa = hyp::Power(R, 4.) - C(c4 / 4);
    // B=R^4-2MR^3+C4, explicitly factored at the lapse-zero root.
    const R2 bb = d * (C(R0 * R0 * (4 * R0 - 6 * mass)) +
        d * (C(6 * R0 * R0 - 6 * mass * R0) +
        d * (C(4 * R0 - 2 * mass) + d)));
    const R2 alpha = C(-2) * bb /
        (C(c4) + hyp::Power(C(c4 * c4) + C(4) * aa * bb, .5));
    const R2 F = C(1) - C(2 * mass) / R;
    const R2 s = hyp::Power(alpha * alpha - F, .5);
    const R2 beta = alpha * s / rp;
    const R2 gamma_r = rp * rp / (alpha * alpha);
    const R2 gamma_t = R * R / (rr * rr);
    const R2 chi = hyp::Power(gamma_r * gamma_t * gamma_t, -1. / 3);
    const R2 gr = chi * gamma_r, gt = chi * gamma_t;
    const R2 kr = R2{s.d, s.dd, 0} / rp, kt = s / R;
    const R2 K = kr + C(2) * kt;
    const R2 ar = gr * (kr - K / C(3)), at = gt * (kt - K / C(3));
    Jet axis{};
    axis.alpha = {alpha.value, {alpha.d, 0, 0},
        {{alpha.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    axis.chi = {chi.value, {chi.d, 0, 0},
        {{chi.dd, 0, 0}, {0, 0, 0}, {0, 0, 0}}};
    axis.trace.value = K.value;
    axis.trace.d[0] = K.d;
    axis.beta.value[0] = beta.value;
    axis.beta.d[0][0] = beta.d;
    axis.beta.dd[0][0][0] = beta.dd;
    for (int i = 0; i < 3; ++i) {
      const R2 g = i == 0 ? gr : gt, a = i == 0 ? ar : at;
      axis.metric.g[i][i] = g.value;
      axis.metric.dg[0][i][i] = g.d;
      axis.metric.ddg[0][0][i][i] = g.dd;
      axis.a.k[i][i] = a.value;
      axis.a.dk[0][i][i] = a.d;
    }
    const double n[3] = {x / r, y / r, z / r};
    Jet out = hyp::CartesianRadialJet(axis, r, n);
    const auto geo = hyp::Geometry(out.metric);
    for (int i = 0; i < 3; ++i) {
      out.lambda.value[i] = geo.contracted[i];
      for (int j = 0; j < 3; ++j) out.lambda.d[j][i] = geo.dcontracted[j][i];
    }
    return out;
  }
};
}  // namespace inner_audit
#endif
