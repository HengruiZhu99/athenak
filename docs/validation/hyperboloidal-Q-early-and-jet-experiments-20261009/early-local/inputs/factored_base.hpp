// Private algebraically equivalent gauge copy; no Q quotient is evaluated.
// Origin: production layer_gauge.hpp, transformations documented in DERIVATION.
namespace qnf { using namespace z4c::hyperboloidal;
template <typename T>
KOKKOS_INLINE_FUNCTION GaugeRHSParts<T> FactoredBaseGauge(
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
  const T relative_da = da / p.alpha;
  // Near a puncture, da may round to -alpha_hat despite alpha remaining
  // positive. Log differences retain that lapse; log1p handles small changes.
  const T loga = Kokkos::abs(relative_da) < T(0.5)
      ? Kokkos::log1p(relative_da) : Kokkos::log(alpha) - Kokkos::log(p.alpha);
  T db[3], alpha2du[3], alpha2dv[3], ref_advection[3]{};
  T Dmatch = 0, ref_log_advection = 0, bref_omega = 0, b_omega = 0;
  for (int i = 0; i < 3; ++i) {
    db[i] = u.beta.value[i] - p.beta[i];
    alpha2du[i] = alpha*u.alpha.d[i]-alpha*alpha*p.dalpha[i]/p.alpha;
    alpha2dv[i] = alpha*alpha*(u.chi.d[i]/chi-p.state.chi.d[i]/p.state.chi.value);
    // Dmatch = alpha/alpha_ref beta_ref.dOmega - beta.dOmega.
    Dmatch += (p.beta[i] * da / p.alpha - db[i]) * p.domega[i];
    bref_omega += p.beta[i]*p.domega[i];b_omega += u.beta.value[i]*p.domega[i];
    ref_log_advection += u.beta.value[i] * p.dalpha[i] / p.alpha;
    out.regular.alpha += u.beta.value[i] * u.alpha.d[i];
    for (int j = 0; j < 3; ++j) {
      ref_advection[i] += p.beta[j] * p.state.beta.d[j][i];
    }
  }
  if(alpha/p.alpha<=T(.5))Dmatch=alpha*bref_omega/p.alpha-b_omega;
  out.regular.alpha -= alpha * ref_log_advection + alpha * c.nu * loga;
  out.pole.alpha = -c.alpha2f * (u.trace.value-p.k_physical)
      + 3 * (alpha+2*(1-c.weight)) * Dmatch;
  if (g.physical_trace_lapse) {
    // Cartesian extension of arXiv:2408.08952 Eq. (9), retaining the
    // coupled-shift principal part. Physical P is the paper's tilde K.
    // Match the full analytic reference advection in a nonflat layer.
    out.regular.alpha = -alpha * c.nu * loga;
    out.pole.alpha = -c.alpha2f * (u.trace.value - p.k_physical)
        - c.weight * T(g.scri_lapse_damping) * (alpha + p.alpha) * da;
    for (int i = 0; i < 3; ++i) {
      out.regular.alpha += u.beta.value[i] * u.alpha.d[i]
          - p.beta[i] * p.dalpha[i];
      out.pole.alpha -= c.weight * (alpha * db[i] + p.beta[i] * da) * p.domega[i];
    }
  }
  for (int i = 0; i < 3; ++i) {
    out.regular.beta[i] =
        alpha * alpha * chi * c.mu * (u.lambda.value[i] - p.state.lambda.value[i]) -
        ref_advection[i] - c.eta * db[i];
    for (int j = 0; j < 3; ++j) {
      out.regular.beta[i] +=
          u.beta.value[j] * u.beta.d[j][i] +
          chi * geo.inverse[i][j] * (c.ec * alpha2dv[j] - c.ea * alpha2du[j]);
    }
  }
  if (g.preferred_source && c.weight > 0) {
    // Algebraic extension of the harmonic collar's F^a to the transition.
    // At W=1: Gamma4^a+2 Z4^a=F^a. F0 is computed without Q or Theta/O.
    // Evaluate alpha^2*Delta directly: no small-live-alpha division.
    const T alpha2f0=ref_log_advection+c.nu*loga-alpha*p.k_bar;
    T norm=0,contraction=0,hessian4=0;
    for(int i=0;i<3;++i){
      norm+=p.domega[i]*p.domega[i];
      T source=alpha*alpha*chi*p.state.lambda.value[i]+ref_advection[i]
          +c.eta*db[i]-u.beta.value[i]*alpha2f0;
      for(int j=0;j<3;++j){
        source+=alpha*alpha*chi*geo.inverse[i][j]*(p.state.chi.d[j]/(2*p.state.chi.value)-p.dalpha[j]/p.alpha);
        hessian4+=(alpha*alpha*chi*geo.inverse[i][j]-u.beta.value[i]*u.beta.value[j])*p.omega_hessian[i][j];
      }
      contraction+=p.domega[i]*source;
    }
    if(!(norm>0))return out;
    const T delta=hessian4-alpha*alpha*p.omega*p.w_omega-contraction;
    for(int i=0;i<3;++i)out.regular.beta[i]-=c.weight*p.domega[i]*delta/norm;
  }
  out.valid = Kokkos::isfinite(out.regular.alpha) && Kokkos::isfinite(out.pole.alpha);
  for (int i = 0; i < 3; ++i)
    out.valid = out.valid && Kokkos::isfinite(out.regular.beta[i]);
  return out;
}

}
