// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_SHIFT_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_SHIFT_HPP_

#include "z4c/hyperboloidal/reference_gauge.hpp"

#ifndef HYPERBOLIC_SMOOTH_GAUGE_BLEND
#define HYPERBOLIC_SMOOTH_GAUGE_BLEND 0
#endif

namespace z4c::hyperboloidal {

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

// Point kernels for the donor c0ae7472 gauge, in GLOBAL CARTESIAN components.
// Not an admission of this gauge into the production task graph. In particular,
// ScriShift is a gauge identity, not the missing full geometric scri closure.
template<class T> struct HyperbolicShiftParameters {
  T bulk_mu = 4, ratio = 2, damping = 1, restoring = 4;
  T kappa = T(.005), tau = T(.02), epsilon = T(1.e-4);
  T inner_z4 = 8, kappa1 = 4, kappa2 = T(-.99);
  void Validate() const {
    for (T v:{bulk_mu,ratio,damping,restoring,kappa,tau,epsilon,inner_z4,kappa1})
      if (!std::isfinite(v) || v<0) throw std::invalid_argument("invalid hyperbolic gauge parameter");
    if (bulk_mu<=0 || ratio<=0 || tau<=0 || !std::isfinite(kappa2))
      throw std::invalid_argument("invalid hyperbolic gauge speed/relaxation parameter");
  }
};

template<class T> KOKKOS_INLINE_FUNCTION
T SmoothCollar(T t) {
  if (t <= 0) return 0;
  if (t >= 1) return 1;
  T a=t*t, b=(1-t)*(1-t); a*=a; b*=b; a*=a; b*=b;
  return a/(a+b);
}

template<class T> struct HyperbolicCoefficients {
  T omega, reference_alpha, w, dw_dr, collar, F, eta, mu, xi;
};
template<class T> KOKKOS_INLINE_FUNCTION
HyperbolicCoefficients<T> ShiftCoefficients(T r, T alpha, T chi,
    const HyperbolicShiftParameters<T> &p = {}) {
  HyperbolicCoefficients<T> c{};
  c.omega=(1-r*r)/2; c.reference_alpha=(1+r*r)/2;
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
  c.collar=SmoothCollar((r-T(.8))/T(.1));
  c.F=alpha*alpha+4*c.omega*c.omega*alpha;
  c.eta=(1+p.epsilon)*alpha/(alpha+p.epsilon*c.reference_alpha);
  // w*C == 0 exactly: the donor's radial-g term in mu_separated never
  // contributes to the blended coefficient. No direction or radial metric
  // component is necessary. This is identical to the coefficient-four donor.
  c.mu=(1-c.collar)*p.bulk_mu+c.collar*T(.75)*p.ratio*chi*c.F;
  c.xi=1+2*pow(1-r*r,4);
  return c;
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
  return 4*c.mu/3-chi*c.F-chi*c.eta*p.kappa*c.w*c.w/p.tau*LargestMetricEigenvalue(g);
}

template<class T> struct HyperbolicGaugeParts {
  T lapse_regular{}, lapse_pole{};
  T shift_regular[3]{}, shift_pole[3]{}, flux_rate[3]{};
  bool valid=false;
};

// B is a vector with the fixed Euclidean reference raising operation.
// D is the shift auxiliary, and is never substituted for u.lambda.
// Componentwise shift advection is intentional, matching the donor.
template<class T> KOKKOS_INLINE_FUNCTION
HyperbolicGaugeParts<T> HyperbolicGauge(const T x[3], const Z4cJet<T> &u,
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
    out.shift_pole[i]=-p.restoring*c.collar*(u.beta.value[i]+x[i]);
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
  const T delta=p.inner_z4==0 ? T(0) :
      c.w*(p.inner_z4*c.reference_alpha-p.kappa1*u.alpha.value);
  parts.pole.theta+=3*u.alpha.value*omega.normal*u.theta.value;
  parts.pole.trace+=delta*(1-p.kappa2)*u.theta.value;
  parts.pole.theta-=delta*(2+p.kappa2)*u.theta.value;
  for (int i=0;i<3;++i)
    parts.pole.lambda[i]-=delta*(u.lambda.value[i]-g.contracted[i]);
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
