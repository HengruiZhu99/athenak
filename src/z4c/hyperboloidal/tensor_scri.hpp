// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_TENSOR_SCRI_HPP_
#define Z4C_HYPERBOLOIDAL_TENSOR_SCRI_HPP_

#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c {
namespace hyperboloidal {

// First-order directional AD. Only algebra is needed by the combined pole
// numerators, so no finite differences, third metric jets or Omega division enter.
template <typename T> struct ScriDual {
  T value{}, derivative{};
  KOKKOS_INLINE_FUNCTION ScriDual() = default;
  KOKKOS_INLINE_FUNCTION ScriDual(T v) : value(v) {}
  KOKKOS_INLINE_FUNCTION ScriDual(T v, T d) : value(v), derivative(d) {}
  KOKKOS_INLINE_FUNCTION ScriDual &operator+=(const ScriDual &b) {
    value += b.value; derivative += b.derivative; return *this;
  }
  KOKKOS_INLINE_FUNCTION ScriDual &operator-=(const ScriDual &b) {
    value -= b.value; derivative -= b.derivative; return *this;
  }
};
template <typename T> KOKKOS_INLINE_FUNCTION
ScriDual<T> operator+(ScriDual<T> a, const ScriDual<T> &b) {return a += b;}
template <typename T> KOKKOS_INLINE_FUNCTION
ScriDual<T> operator-(ScriDual<T> a, const ScriDual<T> &b) {return a -= b;}
template <typename T> KOKKOS_INLINE_FUNCTION
ScriDual<T> operator-(const ScriDual<T> &a) {return {-a.value, -a.derivative};}
template <typename T> KOKKOS_INLINE_FUNCTION
ScriDual<T> operator*(const ScriDual<T> &a, const ScriDual<T> &b) {
  return {a.value*b.value, a.derivative*b.value+a.value*b.derivative};
}
template <typename T> KOKKOS_INLINE_FUNCTION
ScriDual<T> operator/(const ScriDual<T> &a, const ScriDual<T> &b) {
  return {a.value/b.value, (a.derivative*b.value-a.value*b.derivative)/(b.value*b.value)};
}

template <typename T> struct TensorPole {
  Z4cRHS<T> geometry{};
  T lapse{};
};

// Algebraically identical to the COMPLETE combined pole terms in ConformalRHS
// and UnfactoredReferenceGauge. Keeping this small first-jet evaluator allows AD
// through poles without differentiating unused Ricci/regular terms. Native value
// and independent finite-difference derivative tests guard against divergence.
template <typename T>
KOKKOS_INLINE_FUNCTION
TensorPole<T> TensorPoleSources(const Z4cJet<T> &u, const OmegaJet<T> &o,
    T reference_alpha, const T reference_beta[3], T kappa1, T kappa2,
    T slicing, T restoring, bool appendix_c = true, bool multiplicative = false) {
  TensorPole<T> out{};
  const auto &g = u.metric.g;
  const T det = g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])
      -g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])
      +g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);
  T inverse[3][3]{}, gamma[3][3][3]{}, bar_gamma[3][3][3]{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    const int p = (i+1)%3, q = (i+2)%3, r = (j+1)%3, s = (j+2)%3;
    inverse[j][i] = (g[p][r]*g[q][s]-g[p][s]*g[q][r])/det;
  }
  const T alpha = u.alpha.value, chi = u.chi.value;
  T w{}, gradient2{}, contracted[3]{}, zup[3]{}, z[3]{}, aup[3][3]{};
  for (int i = 0; i < 3; ++i) w -= u.beta.value[i]*o.gradient[i]/alpha;
  for (int k = 0; k < 3; ++k)
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    for (int a = 0; a < 3; ++a) {
      const T dg = u.metric.dg[i][a][j]+u.metric.dg[j][a][i]-u.metric.dg[a][i][j];
      gamma[k][i][j] += T(.5)*inverse[k][a]*dg;
      const T dbar = dg/chi-(g[a][j]*u.chi.d[i]+g[a][i]*u.chi.d[j]-g[i][j]*u.chi.d[a])/(chi*chi);
      bar_gamma[k][i][j] += T(.5)*chi*inverse[k][a]*dbar;
    }
    contracted[k] += inverse[i][j]*gamma[k][i][j];
  }
  for (int i = 0; i < 3; ++i) zup[i] = (u.lambda.value[i]-contracted[i])/T(2);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    z[i] += g[i][j]*zup[j];
    gradient2 += chi*inverse[i][j]*o.gradient[i]*o.gradient[j];
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) aup[i][j] += inverse[i][a]*inverse[j][b]*u.a.k[a][b];
  }
  const T K = u.trace.value+T(2)*u.theta.value;
  auto &s = out.geometry;
  s.chi = T(2./3.)*alpha*chi*(K-T(3)*w);
  s.trace = alpha*(K*K/T(3)-T(3)*gradient2+kappa1*(T(1)-kappa2)*u.theta.value);
  s.theta = alpha*(K*K/T(3)-T(3)*gradient2
      -(kappa1*(T(2)+kappa2)+(appendix_c ? T(0) : T(3)*w))*u.theta.value);
  T tensor[3][3]{}, trace{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    T hessian = o.hessian[i][j];
    for (int k = 0; k < 3; ++k) hessian -= bar_gamma[k][i][j]*o.gradient[k];
    tensor[i][j] = T(2)*alpha*chi*(hessian+z[i]*o.gradient[j]+z[j]*o.gradient[i]);
    trace += inverse[i][j]*tensor[i][j];
  }
  for (int i = 0; i < 3; ++i) {
    s.lambda[i] = -alpha*(T(4./3.)*K+T(2)*kappa1)*zup[i];
    for (int j = 0; j < 3; ++j) {
      s.a[i][j] = tensor[i][j]-g[i][j]*trace/T(3)+alpha*u.a.k[i][j]*(K-w);
      s.lambda[i] -= T(2./3.)*alpha*inverse[i][j]*(T(2)*u.trace.d[j]+u.theta.d[j]);
      s.lambda[i] -= T(4)*alpha*aup[i][j]*o.gradient[j];
    }
  }
  const T delta_alpha = alpha-reference_alpha;
  out.lapse = -(alpha*alpha+T(4)*slicing*o.omega*o.omega*alpha)*(u.trace.value+T(3))
      -restoring*(multiplicative ? T(2)*alpha : alpha+reference_alpha)*delta_alpha;
  for (int i = 0; i < 3; ++i)
    out.lapse -= (alpha*u.beta.value[i]-reference_alpha*reference_beta[i])*o.gradient[i];
  return out;
}

// Stationary unit-sphere CMC compactification: Omega=(1-r^2)/2,
// alpha_ref=(1+r^2)/2, beta_ref=-x. Its third Omega derivatives vanish.
// Other compactifications must supply their third Omega derivatives explicitly;
// this adapter must not be silently used for them.
template <typename T>
KOKKOS_INLINE_FUNCTION
TensorPole<ScriDual<T>> TensorPoleNormalJet(const Z4cJet<T> &u,
    const OmegaJet<T> &o, const T x[3], const T direction[3], T kappa1, T kappa2,
    T slicing, T restoring, bool appendix_c = true, bool multiplicative = false) {
  using D = ScriDual<T>;
  Z4cJet<D> v{}; OmegaJet<D> omega{};
  auto scalar = [&](const ScalarJet<T> &a, ScalarJet<D> &b) {
    T derivative{};
    for (int d = 0; d < 3; ++d) derivative += direction[d]*a.d[d];
    b.value = D(a.value, derivative);
    for (int i = 0; i < 3; ++i) {
      derivative = T(0);
      for (int d = 0; d < 3; ++d) derivative += direction[d]*a.dd[d][i];
      b.d[i] = D(a.d[i], derivative);
    }
  };
  scalar(u.alpha, v.alpha); scalar(u.chi, v.chi);
  scalar(u.trace, v.trace); scalar(u.theta, v.theta);
  T alpha_ref = T(.5), dref{}, domega{};
  D beta_ref[3];
  for (int i = 0; i < 3; ++i) {
    alpha_ref += x[i]*x[i]/T(2); dref += x[i]*direction[i];
    domega += o.gradient[i]*direction[i];
    beta_ref[i] = D(-x[i], -direction[i]);
    T dbeta{}, dlambda{};
    for (int d = 0; d < 3; ++d) {
      dbeta += direction[d]*u.beta.d[d][i]; dlambda += direction[d]*u.lambda.d[d][i];
    }
    v.beta.value[i] = D(u.beta.value[i], dbeta);
    v.lambda.value[i] = D(u.lambda.value[i], dlambda);
    T gradient_derivative{};
    for (int d = 0; d < 3; ++d) gradient_derivative += direction[d]*o.hessian[d][i];
    omega.gradient[i] = D(o.gradient[i], gradient_derivative);
    for (int j = 0; j < 3; ++j) {
      T dg{}, da{};
      for (int d = 0; d < 3; ++d) {dg += direction[d]*u.metric.dg[d][i][j]; da += direction[d]*u.a.dk[d][i][j];}
      v.metric.g[i][j] = D(u.metric.g[i][j], dg); v.a.k[i][j] = D(u.a.k[i][j], da);
      omega.hessian[i][j] = D(o.hessian[i][j]);
      for (int a = 0; a < 3; ++a) {
        T ddg{};
        for (int d = 0; d < 3; ++d) ddg += direction[d]*u.metric.ddg[d][a][i][j];
        v.metric.dg[a][i][j] = D(u.metric.dg[a][i][j], ddg);
      }
    }
  }
  omega.omega = D(o.omega, domega);
  return TensorPoleSources(v, omega, D(alpha_ref, dref), beta_ref,
      D(kappa1), D(kappa2), D(slicing), D(restoring), appendix_c, multiplicative);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
Z4cRHS<T> TensorScriLimit(const Z4cRHS<T> &regular,
                         const TensorPole<ScriDual<T>> &pole, T omega_normal) {
  auto out = regular;
  out.chi += pole.geometry.chi.derivative/omega_normal;
  out.trace += pole.geometry.trace.derivative/omega_normal;
  out.theta += pole.geometry.theta.derivative/omega_normal;
  for (int i = 0; i < 3; ++i) {
    out.lambda[i] += pole.geometry.lambda[i].derivative/omega_normal;
    for (int j = 0; j < 3; ++j) out.a[i][j] += pole.geometry.a[i][j].derivative/omega_normal;
  }
  return out;
}
}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_TENSOR_SCRI_HPP_
