// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_SHIFT_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_SHIFT_HPP_

#include "z4c/hyperboloidal/reference_gauge.hpp"
#include "z4c/hyperboloidal/conformal_profile.hpp"
#include "z4c/hyperboloidal/cartesian_inner_adjustment.hpp"

#ifndef HYPERBOLIC_SMOOTH_GAUGE_BLEND
#define HYPERBOLIC_SMOOTH_GAUGE_BLEND 0
#endif

#ifndef HYPERBOLIC_INNER_TELEGRAPHER
#define HYPERBOLIC_INNER_TELEGRAPHER 0
#endif

#ifndef HYPERBOLIC_DIRECT_OUTER
#define HYPERBOLIC_DIRECT_OUTER 0
#endif

#ifndef HYPERBOLIC_RESTORE_INNER_DAMPING
#define HYPERBOLIC_RESTORE_INNER_DAMPING 0
#endif

#ifndef HYPERBOLIC_LATE_INNER_GAUGE
#define HYPERBOLIC_LATE_INNER_GAUGE 0
#endif

#ifndef HYPERBOLIC_SHORT_INNER_GAUGE
#define HYPERBOLIC_SHORT_INNER_GAUGE 0
#endif

#ifndef HYPERBOLIC_SHARED_RESTORATION
#define HYPERBOLIC_SHARED_RESTORATION 0
#endif

#ifndef HYPERBOLIC_WIDE_RESTORING_COLLAR
#define HYPERBOLIC_WIDE_RESTORING_COLLAR 0
#endif

#ifndef HYPERBOLIC_BROAD_SMOOTH_A
#define HYPERBOLIC_BROAD_SMOOTH_A 0
#endif

#ifndef HYPERBOLIC_ZERO_SHIFT_RESTORATION
#define HYPERBOLIC_ZERO_SHIFT_RESTORATION 0
#endif

#ifndef HYPERBOLIC_NORMALIZED_INNER_SHIFT
#define HYPERBOLIC_NORMALIZED_INNER_SHIFT 0
#endif

namespace z4c::hyperboloidal {
constexpr bool normalized_inner_shift=HYPERBOLIC_NORMALIZED_INNER_SHIFT;
static_assert(!normalized_inner_shift || HYPERBOLIC_BROAD_SMOOTH_A);
// Constant inner map x=k X, t=T for the native C6 4M--12M slice, M=1/3.
// beta_x=k beta_X and Lambda_x=Lambda_X/k, hence Gamma-driver coefficient k^2.
// This changes only the inner driver, through the existing complete-RHS blend.
constexpr double normalized_inner_spatial_scale=.306142283363079;

// Combined restored-A experiment: broader restoring collar and native C6 slice.
constexpr bool broad_smooth_a=HYPERBOLIC_BROAD_SMOOTH_A;
static_assert(!broad_smooth_a || (HYPERBOLIC_SHORT_INNER_GAUGE &&
    !HYPERBOLIC_SHARED_RESTORATION && !HYPERBOLIC_WIDE_RESTORING_COLLAR &&
    !HYPERBOLIC_ZERO_SHIFT_RESTORATION));
// Experimental removal of only the extra singular shift source. The lapse,
// inner shift damping, auxiliary damping and geometric damping are unchanged.
constexpr bool zero_shift_restoration=HYPERBOLIC_ZERO_SHIFT_RESTORATION;
static_assert(!zero_shift_restoration || (HYPERBOLIC_SHORT_INNER_GAUGE &&
    !HYPERBOLIC_SHARED_RESTORATION && !HYPERBOLIC_WIDE_RESTORING_COLLAR));
constexpr bool wide_restoring_collar=HYPERBOLIC_WIDE_RESTORING_COLLAR;
static_assert(!wide_restoring_collar || (HYPERBOLIC_SHORT_INNER_GAUGE && !HYPERBOLIC_SHARED_RESTORATION));
constexpr bool shared_restoration=HYPERBOLIC_SHARED_RESTORATION;
static_assert(!shared_restoration || HYPERBOLIC_SHORT_INNER_GAUGE);
constexpr bool short_inner_gauge=HYPERBOLIC_SHORT_INNER_GAUGE;
static_assert(!short_inner_gauge || HYPERBOLIC_RESTORE_INNER_DAMPING);
static_assert(!(short_inner_gauge && HYPERBOLIC_LATE_INNER_GAUGE));
constexpr bool late_inner_gauge=HYPERBOLIC_LATE_INNER_GAUGE;
static_assert(!late_inner_gauge || HYPERBOLIC_RESTORE_INNER_DAMPING);
constexpr bool restore_inner_damping=HYPERBOLIC_RESTORE_INNER_DAMPING;
static_assert(!restore_inner_damping || HYPERBOLIC_DIRECT_OUTER);

constexpr bool direct_outer = HYPERBOLIC_DIRECT_OUTER;
static_assert(!direct_outer || HYPERBOLIC_INNER_TELEGRAPHER);

constexpr bool inner_telegrapher = HYPERBOLIC_INNER_TELEGRAPHER;
static_assert(HYPERBOLIC_INNER_TELEGRAPHER == 0 || HYPERBOLIC_INNER_TELEGRAPHER == 1);
// telegrapher_lapse b99e27f, g1_bhahaha_L5: mass-scaled parameters.
// Our puncture mass is 1/3 in coordinate units. B stores a physical covector
// in the inner plateau; the same Cartesian slots interpolate to donor B outside.
struct InnerTelegrapher {
  static constexpr double mass=1./3., begin=late_inner_gauge?.4:.2, end=direct_outer&&!short_inner_gauge?.9:.56;
  static constexpr double tau=.1*mass, kappa=.2*mass;
  static constexpr double shift_gamma=normalized_inner_shift ? normalized_inner_spatial_scale*normalized_inner_spatial_scale : 1., shift_eta=2./mass, kappa1=.01/mass;
  static constexpr double chi_guard=1.e-5;
};

constexpr bool smooth_gauge_blend = HYPERBOLIC_SMOOTH_GAUGE_BLEND;
static_assert(HYPERBOLIC_SMOOTH_GAUGE_BLEND == 0 || HYPERBOLIC_SMOOTH_GAUGE_BLEND == 1);

// Integral of 630*t^4*(1-t)^4: a broad C4 transition, with exact constants
// outside [0,1]. Symmetry avoids cancellation near the upper endpoint.
template<class T> KOKKOS_INLINE_FUNCTION
T Smoothstep9(T t) {
  if (t<=0) return 0;
  if (t>=1) return 1;
  const T x=t>T(.5)?1-t:t;
  const T value=x*x*x*x*x*(126+x*(-420+x*(540+x*(-315+70*x))));
  return t>T(.5)?1-value:value;
}

template<class T> KOKKOS_INLINE_FUNCTION
T InnerTelegrapherWeight(T r, T begin, T end) {
  if constexpr(!inner_telegrapher) return 0;
  return Smoothstep9((end-r)/(end-begin));
}

template<class T> KOKKOS_INLINE_FUNCTION
T InnerTelegrapherWeight(T r) {
  return InnerTelegrapherWeight(r,T(InnerTelegrapher::begin),T(InnerTelegrapher::end));
}

// Point kernels for the donor c0ae7472 gauge, in GLOBAL CARTESIAN components.
// Not an admission of this gauge into the production task graph. In particular,
// ScriShift is a gauge identity, not the missing full geometric scri closure.
#ifndef HYPERBOLIC_CONFIGURED_FORMULATION
#define HYPERBOLIC_CONFIGURED_FORMULATION 0
#endif
inline constexpr bool configured_conformal_frame=HYPERBOLIC_CONFIGURED_FORMULATION;

template<class T> struct HyperbolicShiftParameters {
  ConformalProfile profile;
  bool enforce_cfl=false;
  // Numerical interface policy, not a change to any gauge RHS.
  bool conformal_frame_halos=false;
  // Optional field-dependent inner driver; outer gauge remains unchanged.
  bool separated_shift_speeds=false;
  // Prescribed stationary outer gauge only; geometric equations are untouched.
  bool stationary_outer_reference=false;
  T reference_start_M=4, reference_end_M=12;
  T inner_begin=InnerTelegrapher::begin, inner_end=InnerTelegrapher::end;
  T inner_gamma=InnerTelegrapher::shift_gamma;
  T bulk_mu = 4, ratio = 2, damping = 1, restoring = 4;
  T kappa = T(.005), tau = T(.02), epsilon = T(1.e-4);
  T inner_z4 = 8, kappa1 = 4, kappa2 = T(-.99);
  void Validate() const {
    profile.Validate();
    if (stationary_outer_reference && (!configured_conformal_frame || !direct_outer ||
        !inner_telegrapher || !profile.flat || !std::isfinite(reference_start_M) ||
        !std::isfinite(reference_end_M) || !(reference_start_M>2 && reference_end_M>reference_start_M)))
      throw std::invalid_argument("stationary reference requires configured flat direct outer gauge and valid slice");
    if (separated_shift_speeds && (!configured_conformal_frame || !direct_outer || !inner_telegrapher || ratio!=T(2)))
      throw std::invalid_argument("separated shift speeds require configured direct outer gauge with ratio 2");
    if (profile.flat && !configured_conformal_frame)
      throw std::invalid_argument("flat Omega requires the separate configured formulation target");
    if (!std::isfinite(inner_begin) || !std::isfinite(inner_end) ||
        !std::isfinite(inner_gamma) || !(inner_begin<inner_end && inner_gamma>0))
      throw std::invalid_argument("invalid inner gauge profile");
    for (T v:{bulk_mu,ratio,damping,restoring,kappa,tau,epsilon,inner_z4,kappa1})
      if (!std::isfinite(v) || v<0) throw std::invalid_argument("invalid hyperbolic gauge parameter");
    if (bulk_mu<=0 || ratio<=0 || tau<=0 || !std::isfinite(kappa2))
      throw std::invalid_argument("invalid hyperbolic gauge speed/relaxation parameter");
  }
};

// The optional coefficient separates the inner longitudinal-shift speed from
// the telegrapher lapse speed, including the chi guard. It is a source
// coefficient (not a conservative flux); no gradient of it is added to beta_t.
// For direct_outer with ratio=2 the squared-speed gap divided by Q is
// 4*s*g0/3 + (1-s)*chi*F + s*(1-s)*guard*kappa/tau > 0
// for positive lapse/chi and Q. This is not a full hyperbolicity proof.
template<class T> KOKKOS_INLINE_FUNCTION
T InnerShiftGamma(T alpha,T chi,T omega,const HyperbolicShiftParameters<T>&p) {
  if (!p.separated_shift_speeds) return p.inner_gamma;
  const T guard=fmax(omega*omega*chi,T(InnerTelegrapher::chi_guard));
  return p.inner_gamma+T(.75)*(2*omega*alpha*chi
      +guard*T(InnerTelegrapher::kappa/InnerTelegrapher::tau));
}

template<class T> KOKKOS_INLINE_FUNCTION
T SmoothCollar(T t) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  T a=t*t, b=(1-t)*(1-t); a*=a; b*=b; a*=a; b*=b;
  return a/(a+b);
}

template<class T> struct HyperbolicCoefficients {
  T omega, reference_alpha, w, dw_dr, collar, F, eta, mu, xi, inner, inner_gamma;
};
template<class T> KOKKOS_INLINE_FUNCTION
HyperbolicCoefficients<T> ShiftCoefficients(T r, T alpha, T chi,
    const HyperbolicShiftParameters<T> &p = {}) {
  HyperbolicCoefficients<T> c{};
  c.omega=p.profile.Radial(r).value; c.reference_alpha=(1+r*r)/2;
  c.inner=InnerTelegrapherWeight(r,p.inner_begin,p.inner_end); c.inner_gamma=InnerShiftGamma(alpha,chi,c.omega,p);
  const T y=(T(.8)-r)/T(.24);
  if constexpr(smooth_gauge_blend) {
    c.w=Smoothstep9(y);
    if (y>0 && y<1) {
      const T q=y*(1-y),q2=q*q;
      c.dw_dr=-630*q2*q2/T(.24);
    }
  } else {
    c.w=SmoothCollar(y);
    if (y>0 && y<1) {
      const T a=pow(y,8),b=pow(1-y,8);
      c.dw_dr=-8*pow(y,7)*pow(1-y,7)/((a+b)*(a+b)*T(.24));
    }
  }
  // A-width control: preserve the original profile and outer plateau, widen inward.
  c.collar=broad_smooth_a ? SmoothCollar((r-T(.6))/T(.3)) : wide_restoring_collar ? SmoothCollar((r-T(.7))/T(.2)) : SmoothCollar((r-T(.8))/T(.1));
  c.F=alpha*alpha+4*c.omega*c.omega*alpha;
  c.eta=(1+p.epsilon)*alpha/(alpha+p.epsilon*c.reference_alpha);
  // w*C == 0 exactly: the donor's radial-g term in mu_separated never
  // contributes to the blended coefficient. No direction or radial metric
  // component is necessary. This is identical to the coefficient-four donor.
  c.mu=(1-c.collar)*p.bulk_mu+c.collar*T(.75)*p.ratio*chi*c.F;
  if constexpr(direct_outer) c.mu=T(.75)*p.ratio*chi*c.F;
  c.xi=1+2*pow(1-r*r,4);
  return c;
}

// Frozen-coefficient principal-speed estimate, not a strong-hyperbolicity
// certificate. The mixed B/lapse factors must be multiplied AFTER blending.
template<class T> KOKKOS_INLINE_FUNCTION
T GaugeLapseSpeedSquared(T alpha,T chi,const HyperbolicCoefficients<T>&c,T Q,T E,
                        const HyperbolicShiftParameters<T>&p={}) {
  if constexpr(direct_outer) {
    const T s=c.inner,o=c.omega;
    if(s==0) return chi*c.F*Q;
    const T guard=fmax(o*o*chi,T(InnerTelegrapher::chi_guard));
    return (chi*((1-s)*c.F+s*2*o*alpha)
        +s*s*guard*T(InnerTelegrapher::kappa/InnerTelegrapher::tau))*Q;
  }
  if (c.inner==0) return chi*(c.F*Q+c.eta*(p.kappa/p.tau)*c.w*c.w*E);
  const T s=c.inner,o=c.omega,guard=fmax(o*o*chi,T(InnerTelegrapher::chi_guard));
  const T lapse_k=(1-s)*c.F+s*2*o*alpha;
  const T lapse_b=(1-s)*c.eta*chi*c.w*E+s*o*guard*Q;
  const T b_lapse=(1-s)*p.kappa*c.w/p.tau+s*T(InnerTelegrapher::kappa/InnerTelegrapher::tau)/o;
  return chi*lapse_k*Q+lapse_b*b_lapse;
}
template<class T> KOKKOS_INLINE_FUNCTION
T GaugeShiftSpeedSquared(const HyperbolicCoefficients<T>&c,T Q) {
  if(c.inner==0) return 4*c.mu*Q/3;
  return T(4./3.)*((1-c.inner)*c.mu+c.inner*c.inner_gamma)*Q;
}

// Exact extremal eigenvalue of a symmetric 3x3 metric. This supplies a
// direction-independent speed-gap diagnostic, not a change to mu_bulk=4.
template<class T> KOKKOS_INLINE_FUNCTION
T LargestMetricEigenvalue(const T g[3][3]) {
  const T q=(g[0][0]+g[1][1]+g[2][2])/3;
  T variance=0;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    const T v=g[i][j]-(i==j ? q : T(0));variance+=v*v;
  }
  if (variance==0) return q;
  const T p=sqrt(variance/6);
  T b[3][3];
  for (int i=0;i<3;++i) for (int j=0;j<3;++j)
    b[i][j]=(g[i][j]-(i==j ? q : T(0)))/p;
  const T det=b[0][0]*(b[1][1]*b[2][2]-b[1][2]*b[2][1])
      -b[0][1]*(b[1][0]*b[2][2]-b[1][2]*b[2][0])
      +b[0][2]*(b[1][0]*b[2][1]-b[1][1]*b[2][0]);
  const T arg=fmin(T(1),fmax(T(-1),det/2));  // roundoff in acos only
  return q+2*p*cos(acos(arg)/3);
}

template<class T> KOKKOS_INLINE_FUNCTION
T AllDirectionShiftLapseGap(const T g[3][3],T chi,
    const HyperbolicCoefficients<T> &c,const HyperbolicShiftParameters<T> &p={}) {
  if (c.inner==0 && direct_outer) return 4*c.mu/3-chi*c.F;
  if (c.inner==0) return 4*c.mu/3-chi*c.F-chi*c.eta*p.kappa*c.w*c.w/p.tau*LargestMetricEigenvalue(g);
  // Recover alpha from F=alpha^2+4 Omega^2 alpha without cancellation.
  const T q=2*c.omega*c.omega;
  const T alpha=c.F/(sqrt(q*q+c.F)+q);
  return GaugeShiftSpeedSquared(c,T(1))-GaugeLapseSpeedSquared(alpha,chi,c,T(1),LargestMetricEigenvalue(g),p);
}

template<class T> struct HyperbolicGaugeParts {
  T lapse_regular{}, lapse_pole{};
  T shift_regular[3]{}, shift_pole[3]{}, flux_rate[3]{};
  bool valid=false;
};

// Host-precomputed prescribed sources, never inferred from evolved geometry.
// All three corrections already include the complete-RHS outer blend weight.
template<class T> struct StationaryGaugeSource {
  T alpha{}, shift_ratio{}, lapse_offset{}, lapse_linear{}, shift_offset{};
};
template<class T> KOKKOS_INLINE_FUNCTION
void ApplyStationaryGaugeSource(const T x[3],T alpha,const StationaryGaugeSource<T>&s,
                               HyperbolicGaugeParts<T>&g) {
  g.lapse_regular+=s.lapse_offset+s.lapse_linear*(alpha-s.alpha);
  for(int i=0;i<3;++i)g.shift_regular[i]+=s.shift_offset*x[i];
  g.valid=g.valid&&Kokkos::isfinite(g.lapse_regular);
  for(int i=0;i<3;++i)g.valid=g.valid&&Kokkos::isfinite(g.shift_regular[i]);
}

// B is a vector with the fixed Euclidean reference raising operation.
// D is the shift auxiliary, and is never substituted for u.lambda.
// Componentwise shift advection is intentional, matching the donor.
template<class T> KOKKOS_INLINE_FUNCTION
HyperbolicGaugeParts<T> DonorHyperbolicGauge(const T x[3], const Z4cJet<T> &u,
    const VectorJet<T> &B, const T D[3],
    const HyperbolicShiftParameters<T> &p = {}) {
  HyperbolicGaugeParts<T> out{};
  T r2=0; for (int i=0;i<3;++i) r2+=x[i]*x[i];
  const T r=sqrt(r2),alpha=u.alpha.value,chi=u.chi.value;
  const auto c=ShiftCoefficients(r,alpha,chi,p);
  const T da=alpha-c.reference_alpha;
  out.lapse_regular=r2;
  out.lapse_pole=-c.F*(u.trace.value+3)-2*c.xi*alpha*da
      +16*pow(c.omega,4)*r2*da/c.reference_alpha;
  for (int i=0;i<3;++i) {
    const T dw=r>0 ? c.dw_dr*x[i]/r : T(0);
    out.lapse_regular+=u.beta.value[i]*u.alpha.d[i]
        +c.eta*chi*(c.w*B.d[i][i]+dw*B.value[i]);
    out.lapse_pole+=x[i]*(alpha*u.beta.value[i]+c.reference_alpha*x[i]);
    out.shift_regular[i]=-x[i]+c.mu*D[i];
    out.shift_pole[i]=zero_shift_restoration ? T(0) : -p.restoring*c.collar*(u.beta.value[i]+x[i]);
    out.flux_rate[i]=(p.kappa*c.w*(u.alpha.d[i]-x[i])-B.value[i])/p.tau;
    for (int j=0;j<3;++j) {
      out.shift_regular[i]+=u.beta.value[j]*u.beta.d[j][i];
      out.flux_rate[i]+=u.beta.value[j]*B.d[j][i];
    }
  }
  out.valid=alpha>0 && chi>0 && c.mu>0 && p.tau>0
      && Kokkos::isfinite(out.lapse_regular) && Kokkos::isfinite(out.lapse_pole);
  for (int i=0;i<3;++i) out.valid=out.valid
      && Kokkos::isfinite(out.shift_regular[i])
      && Kokkos::isfinite(out.shift_pole[i]) && Kokkos::isfinite(out.flux_rate[i]);
  return out;
}

// Direct outer branch: preserve reference sources and the original restoring
// collar; remove Euclidean B/lapse coupling and B-gradient forcing. Its mu is
// the outer 1.5 chi F throughout. Blend complete RHSs, never weighted fluxes.
template<class T> KOKKOS_INLINE_FUNCTION
HyperbolicGaugeParts<T> OuterHyperbolicGauge(const T x[3],const Z4cJet<T>&u,
    const VectorJet<T>&B,const T D[3],const HyperbolicShiftParameters<T>&p={}) {
  if constexpr(!direct_outer) return DonorHyperbolicGauge(x,u,B,D,p);
  const VectorJet<T> zero{};
  auto out=DonorHyperbolicGauge(x,u,zero,D,p);
  for(int i=0;i<3;++i) {
    // The complete outer RHS is multiplied by W=1-inner exactly once in
    // HyperbolicGauge. Remove only the separate C(r) activation, not the
    // restoring source. For r>=.9 this is identical to the original source,
    // including the scri pole and dependent-D limiting equation.
    if constexpr(shared_restoration)
      out.shift_pole[i]=-p.restoring*(u.beta.value[i]+x[i]);
    out.flux_rate[i]=-B.value[i]/p.tau;
    for(int j=0;j<3;++j)out.flux_rate[i]+=u.beta.value[j]*B.d[j][i];
    out.valid=out.valid&&Kokkos::isfinite(out.flux_rate[i]);
  }
  return out;
}

// Cauchy-type gauge on r<=.2 (normalized_shift matches the spatially scaled driver); outer plateau is .56 (IT1) or .9 (DO1). Blend complete
// local RHSs, not fluxes: no spurious derivative-of-weight source is added.
// For stationary Omega, dt(alpha_bar)=beta.grad(alpha_bar)
// -alpha_phys beta.grad(Omega)+Omega*(nonadvective physical lapse RHS).
template<class T> KOKKOS_INLINE_FUNCTION
HyperbolicGaugeParts<T> HyperbolicGauge(const T x[3],const Z4cJet<T>&u,
    const VectorJet<T>&B,const T D[3],const HyperbolicShiftParameters<T>&p={}) {
  if constexpr(!inner_telegrapher) return DonorHyperbolicGauge(x,u,B,D,p);
  T r2=0;for(int i=0;i<3;++i)r2+=x[i]*x[i];
  const T s=InnerTelegrapherWeight(sqrt(r2),p.inner_begin,p.inner_end);
  if (s==0) return OuterHyperbolicGauge(x,u,B,D,p); // no Omega division at scri
  HyperbolicGaugeParts<T> out{};
  // Do not evaluate inactive donor sources on the inner plateau.
  if(s<1) out=OuterHyperbolicGauge(x,u,B,D,p);
  else out.valid=u.alpha.value>0&&u.chi.value>0;
  const auto omega=p.profile.At(x);
  const T o=p.profile.flat ? omega.omega : (1-r2)/2,alpha=u.alpha.value/o;
  const auto g=Geometry(u.metric);
  if (!g.valid || !(o>0)) {out.valid=false;return out;}
  const T chi=fmax(o*o*u.chi.value,T(InnerTelegrapher::chi_guard));
  T lapse=-2*u.alpha.value*u.trace.value;
  for(int i=0;i<3;++i) {
    lapse+=u.beta.value[i]*((p.profile.flat ? u.alpha.d[i]-alpha*omega.gradient[i] : u.alpha.d[i]+alpha*x[i]));
    for(int j=0;j<3;++j)lapse+=o*chi*g.inverse[i][j]*B.d[i][j];
    T flux=(T(InnerTelegrapher::kappa)*((p.profile.flat ? u.alpha.d[i]-alpha*omega.gradient[i] : u.alpha.d[i]+alpha*x[i]))/o-B.value[i])/T(InnerTelegrapher::tau);
    T shift=InnerShiftGamma(u.alpha.value,u.chi.value,o,p)*u.lambda.value[i]-T(InnerTelegrapher::shift_eta)*u.beta.value[i];
    for(int j=0;j<3;++j) {flux+=u.beta.value[j]*B.d[j][i];shift+=u.beta.value[j]*u.beta.d[j][i];}
    out.flux_rate[i]=(1-s)*out.flux_rate[i]+s*flux;
    out.shift_regular[i]=(1-s)*out.shift_regular[i]+s*shift;
    out.shift_pole[i]*=1-s;
    out.valid=out.valid&&Kokkos::isfinite(out.flux_rate[i])&&Kokkos::isfinite(out.shift_regular[i]);
  }
  out.lapse_regular=(1-s)*out.lapse_regular+s*lapse;out.lapse_pole*=1-s;
  out.valid=out.valid&&Kokkos::isfinite(out.lapse_regular);
  return out;
}

// Finite products only. Apply to the SAME free ConformalRHS used in the donor.
// Geometry(u.metric).contracted is used for damping Z, not to overwrite Lambda.
template<class T> KOKKOS_INLINE_FUNCTION
bool HyperbolicDamping(const T x[3], const Z4cJet<T> &u,
    const OmegaJet<T> &omega, Z4cRHSParts<T> &parts,
    const HyperbolicShiftParameters<T> &p = {}) {
  const auto g=Geometry(u.metric);
  if (!parts.valid || !g.valid) return false;
  T r2=0; for (int i=0;i<3;++i) r2+=x[i]*x[i];
  const auto c=ShiftCoefficients(sqrt(r2),u.alpha.value,u.chi.value,p);
  const T delta=((direct_outer && !restore_inner_damping) || p.inner_z4==0) ? T(0) :
      c.w*(p.inner_z4*c.reference_alpha-p.kappa1*u.alpha.value);
  parts.pole.theta+=3*u.alpha.value*omega.normal*u.theta.value;
  parts.pole.trace+=delta*(1-p.kappa2)*u.theta.value;
  parts.pole.theta-=delta*(2+p.kappa2)*u.theta.value;
  for (int i=0;i<3;++i)
    parts.pole.lambda[i]-=delta*(u.lambda.value[i]-g.contracted[i]);
  if constexpr(inner_telegrapher) {
    const T s=p.profile.flat ? InnerTelegrapherWeight(sqrt(r2),p.inner_begin,p.inner_end) :
      (restore_inner_damping ? Smoothstep9((T(.56)-sqrt(r2))/T(.36)) : InnerTelegrapherWeight(sqrt(r2)));
    if(s>0) {
      const T old=p.kappa1*u.alpha.value+delta;
      const T target=T(InnerTelegrapher::kappa1)*u.alpha.value;
      parts.pole.trace+=s*(target-old*(1-p.kappa2))*u.theta.value;
      parts.pole.theta+=s*(old*(2+p.kappa2)-2*target)*u.theta.value;
      for(int i=0;i<3;++i)parts.pole.lambda[i]+=s*(old-2*target)*(u.lambda.value[i]-g.contracted[i]);
      // Keep the experimental off-constraint algebra out of legacy GPU kernels.
      if constexpr (configured_conformal_frame)
        if (p.profile.flat && !CartesianInnerAdjustment(u,omega,s,parts)) return false;
    }
  }
  return true;
}

// Use the freely evolved connection RHS, including its chosen numerical
// derivative/dissipation policy. No metric-built connection enters this rate.
template<class T> KOKKOS_INLINE_FUNCTION
bool HyperbolicAuxiliaryRate(const Z4cJet<T> &u,const VectorJet<T> &D,
    const T lambda_rhs[3],T rate[3],T damping=T(1)) {
  bool valid=true;
  for (int i=0;i<3;++i) {
    rate[i]=lambda_rhs[i]-damping*D.value[i];
    for (int j=0;j<3;++j)
      rate[i]+=u.beta.value[j]*(D.d[j][i]-u.lambda.d[j][i]);
    valid=valid && Kokkos::isfinite(rate[i]);
  }
  return valid;
}

template<class T> struct ScriShift {
  bool valid=false;
  T covariant[3]{}, contravariant[3]{}, beta[3]{}, normal_rate{};
};

// Call only on an actual Omega=0 trace. Fixed analytic outward n=-dOmega.
// Trial tangential components remain free; no condition on them is invented.
template<class T> KOKKOS_INLINE_FUNCTION
ScriShift<T> ReconstructScriShift(T omega,T alpha,const T metric[3][3],
    const T n[3],const T trial[3]) {
  ScriShift<T> out{};
  if (omega!=0 || !(alpha>0) || !Kokkos::isfinite(alpha)) return out;
  MetricJet<T> jet{};
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    if (!Kokkos::isfinite(metric[i][j]) || metric[i][j]!=metric[j][i]) return out;
    jet.g[i][j]=metric[i][j];
  }
  const auto g=Geometry(jet); if (!g.valid) return out;
  T norm2=0;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
    out.contravariant[i]+=g.inverse[i][j]*n[j];
    norm2+=n[i]*g.inverse[i][j]*n[j];
  }
  if (!(norm2>0) || !Kokkos::isfinite(norm2)) return out;
  T normal=0;
  for (int i=0;i<3;++i) {
    out.covariant[i]=n[i]/sqrt(norm2);
    out.contravariant[i]/=sqrt(norm2);
    normal+=out.covariant[i]*trial[i];
  }
  for (int i=0;i<3;++i) {
    out.beta[i]=trial[i]-(normal+alpha)*out.contravariant[i];
    if (!Kokkos::isfinite(out.beta[i])) return out;
  }
  out.valid=true; return out;
}

// beta_t = a + mu D, where a contains advection, analytic source, and the
// REGULAR LIMIT of the restoring term. The caller must supply that limit from
// the same derivative operator. This function never divides by Omega.
template<class T> KOKKOS_INLINE_FUNCTION
bool ReconstructScriAuxiliary(ScriShift<T> &b,T alpha_t,const T metric_t[3][3],
    const T a[3],T mu,T D[3]) {
  if (!b.valid || !(mu>0) || !Kokkos::isfinite(mu)) return false;
  T contraction=0,beta_normal=0,a_normal=0,D_normal=0;
  for (int i=0;i<3;++i) {
    beta_normal+=b.covariant[i]*b.beta[i];
    a_normal+=b.covariant[i]*a[i]; D_normal+=b.covariant[i]*D[i];
    for (int j=0;j<3;++j)
      contraction+=b.contravariant[i]*metric_t[i][j]*b.contravariant[j];
  }
  b.normal_rate=-alpha_t-T(.5)*beta_normal*contraction;
  const T required=(b.normal_rate-a_normal)/mu;
  if (!Kokkos::isfinite(required)) return false;
  for (int i=0;i<3;++i) {
    D[i]+=(required-D_normal)*b.contravariant[i];
    if (!Kokkos::isfinite(D[i])) return false;
  }
  return true;
}
}  // namespace z4c::hyperboloidal
#endif
