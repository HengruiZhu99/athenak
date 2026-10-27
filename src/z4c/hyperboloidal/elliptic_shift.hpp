// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_ELLIPTIC_SHIFT_HPP_
#define Z4C_HYPERBOLOIDAL_ELLIPTIC_SHIFT_HPP_

#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c {
namespace hyperboloidal {

// AF-regular support of the selected verified spherical telegrapher. The
// coordinate radius is tied to the Minkowski reference, not a moving chart.
template<typename T> struct TelegraphSupport {T value=T(1),radial_derivative=T(0);bool valid=true;};
template<typename T> KOKKOS_INLINE_FUNCTION
TelegraphSupport<T> ReferenceTelegraphSupport(T radius,T outer=T(.8),T width=T(.24)){
  TelegraphSupport<T> out{};
  out.valid=radius>=0&&outer>width&&width>0&&Kokkos::isfinite(radius+outer+width);
  if(!out.valid)return out;
  if(radius>=outer){out.value=0;return out;}
  if(radius<=outer-width)return out;
  const T y=(outer-radius)/width,a=Kokkos::pow(y,8),b=Kokkos::pow(1-y,8);
  out.value=a/(a+b);out.radial_derivative=-8*Kokkos::pow(y*(1-y),7)/((a+b)*(a+b)*width);
  return out;
}

// All components and derivatives are in the GLOBAL Cartesian reference chart.
// A patch map transforms derivative directions before calling this interface.
// The reference connection vanishes in this chart. Lambda never enters here.
template <typename T>
struct MetricRate {
  T h[3][3]{}, dh[3][3][3]{};
};

// Frechet derivative of g^{jk} Gamma^i_jk. This form also retains determinant
// errors: it does not silently assume that sampled metric jets have det(g)=1.
template <typename T>
KOKKOS_INLINE_FUNCTION
void MetricConnectionVariation(const MetricJet<T> &g,
    const SpatialGeometry<T> &geometry, const MetricRate<T> &rate, T out[3]) {
  T raised[3][3]{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b)
    raised[i][j] += geometry.inverse[i][a]*geometry.inverse[j][b]*rate.h[a][b];
  for (int i = 0; i < 3; ++i) {
    out[i] = 0;
    for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) {
      out[i] -= raised[j][k]*geometry.connection[i][j][k];
      for (int a = 0; a < 3; ++a) {
        const T derivative = rate.dh[j][a][k]+rate.dh[k][a][j]-rate.dh[a][j][k];
        const T metric_derivative = g.dg[j][a][k]+g.dg[k][a][j]-g.dg[a][j][k];
        out[i] += T(0.5)*geometry.inverse[j][k]
            *(geometry.inverse[i][a]*derivative-raised[i][a]*metric_derivative);
      }
    }
  }
}

template <typename T>
KOKKOS_INLINE_FUNCTION
MetricRate<T> ShiftMetricRate(const MetricJet<T> &g, const VectorJet<T> &beta) {
  MetricRate<T> h{};
  T div = 0, ddiv[3]{};
  for (int k = 0; k < 3; ++k) {
    div += beta.d[k][k];
    for (int d = 0; d < 3; ++d) ddiv[d] += beta.dd[d][k][k];
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    h.h[i][j] = -T(2./3.)*g.g[i][j]*div;
    for (int d = 0; d < 3; ++d)
      h.dh[d][i][j] = -T(2./3.)*(g.dg[d][i][j]*div+g.g[i][j]*ddiv[d]);
    for (int k = 0; k < 3; ++k) {
      h.h[i][j] += beta.value[k]*g.dg[k][i][j]
          +g.g[k][j]*beta.d[i][k]+g.g[i][k]*beta.d[j][k];
      for (int d = 0; d < 3; ++d)
        h.dh[d][i][j] += beta.d[d][k]*g.dg[k][i][j]
            +beta.value[k]*g.ddg[d][k][i][j]
            +g.dg[d][k][j]*beta.d[i][k]+g.g[k][j]*beta.dd[d][i][k]
            +g.dg[d][i][k]*beta.d[j][k]+g.g[i][k]*beta.dd[d][j][k];
    }
  }
  return h;
}

// Linear operator at a fixed evolution stage: L_g beta = DV_g[H_g(beta)].
// Principal symbol: (g^{jk} xi_j xi_k) I + (g^{ij} xi_j) xi_k/3.
template <typename T>
KOKKOS_INLINE_FUNCTION
void ApplyEllipticShift(const MetricJet<T> &g, const SpatialGeometry<T> &geometry,
                        const VectorJet<T> &beta, T out[3]) {
  MetricConnectionVariation(g, geometry, ShiftMetricRate(g, beta), out);
}

template <typename T>
KOKKOS_INLINE_FUNCTION
void EllipticShiftSource(const Z4cJet<T> &u, const SpatialGeometry<T> &geometry,
                         T eta, T source[3]) {
  MetricRate<T> curvature{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    curvature.h[i][j] = -2*u.alpha.value*u.a.k[i][j];
    for (int d = 0; d < 3; ++d)
      curvature.dh[d][i][j] = -2*(u.alpha.d[d]*u.a.k[i][j]
                                   +u.alpha.value*u.a.dk[d][i][j]);
  }
  T rate[3]{};
  MetricConnectionVariation(u.metric, geometry, curvature, rate);
  for (int i = 0; i < 3; ++i) source[i] = -eta*geometry.contracted[i]-rate[i];
}

// Fixed reference angular coordinates on a spherical scri. This is coordinate
// radial beta, which differs from metric-normal beta for a general metric.
// The clock alpha is supplied by the evolution boundary, never solved here.
template <typename T>
KOKKOS_INLINE_FUNCTION
bool FixedAngleScriShift(const SpatialGeometry<T> &geometry, T alpha, T chi,
                         const T x[3], T beta[3]) {
  T r2 = 0, norm2 = 0;
  for (int i = 0; i < 3; ++i) r2 += x[i]*x[i];
  if (!geometry.valid || !(r2 > 0 && alpha > 0 && chi > 0)) return false;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) norm2 += chi*geometry.inverse[i][j]*x[i]*x[j]/r2;
  if (!(norm2 > 0) || !Kokkos::isfinite(norm2)) return false;
  const T amplitude = -alpha*Kokkos::sqrt(norm2/r2);
  for (int i = 0; i < 3; ++i) beta[i] = amplitude*x[i];
  return true;
}

// Additional telegrapher coupling used by the successful spherical duration
// run. Apply this OUTSIDE div(window B); the B equation is unchanged. At the
// Minkowski reference alpha=A the strength is one, and it vanishes with alpha.
template <typename T>
KOKKOS_INLINE_FUNCTION
T ReferenceTelegraphAttenuation(T alpha, T reference_alpha, T epsilon) {
  return epsilon > 0 ? (1+epsilon)*alpha/(alpha+epsilon*reference_alpha) : T(1);
}

template <typename T>
struct TelegraphRate {
  T lapse{}, flux[3]{};
};

// Reference-Euclidean vector flux, transported by Cartesian component advection.
// This is a disclosed gauge prescription, not a physical covariant divergence.
// The caller owns the fixed taper and its Cartesian gradient.
template <typename T>
KOKKOS_INLINE_FUNCTION
TelegraphRate<T> ReferenceTelegraph(const Z4cJet<T> &u,
    const VectorJet<T> &flux, const T reference_lapse_gradient[3],
    T window, const T window_gradient[3], T tau, T kappa,
    T attenuation = T(1)) {
  TelegraphRate<T> out{};
  for (int i = 0; i < 3; ++i) {
    out.lapse += u.chi.value*(window*flux.d[i][i]+window_gradient[i]*flux.value[i]);
    out.flux[i] = (kappa*window*(u.alpha.d[i]-reference_lapse_gradient[i])
                   -flux.value[i])/tau;
    for (int j = 0; j < 3; ++j) out.flux[i] += u.beta.value[j]*flux.d[j][i];
  }
  out.lapse *= attenuation;
  return out;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_ELLIPTIC_SHIFT_HPP_
