// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Frozen function from f37942825 conformal_rhs.hpp, independently retained for
// numerical-equivalence tests. Never used in evolution or to subtract an RHS.
#ifndef TST_ELLIPTIC_SHIFT_3D_LEGACY_CONFORMAL_RHS_HPP_
#define TST_ELLIPTIC_SHIFT_3D_LEGACY_CONFORMAL_RHS_HPP_
#include "z4c/hyperboloidal/conformal_rhs.hpp"
namespace z4c { namespace hyperboloidal {
template <typename T>
KOKKOS_INLINE_FUNCTION
Z4cRHSParts<T> LegacyConformalRHS(const Z4cJet<T> &u, const OmegaJet<T> &o,
                          T kappa1, T kappa2) {
  Z4cRHSParts<T> out{};
  out.valid = u.chi.value > 0 && u.alpha.value > 0
      && Kokkos::isfinite(u.chi.value) && Kokkos::isfinite(u.alpha.value);
  if (!out.valid) return out;
  const auto gt = Geometry(u.metric);
  const auto b = PenroseMetric(u.metric, u.chi);
  const auto gb = Geometry(b);
  out.valid = gt.valid && gb.valid;
  if (!out.valid) return out;
  const T alpha = u.alpha.value, chi = u.chi.value, omega = o.omega;
  const T p = u.trace.value, theta = u.theta.value, w = o.normal;
  const T physical_k = p+2*theta;
  T divbeta = 0, lapalpha = 0, lapomega = 0, gradient2 = 0, cross = 0;
  T hessalpha[3][3]{}, hessomega[3][3]{};
  T z[3]{}, zup[3]{}, dz[3][3]{}, dzbar[3][3]{};
  T aup[3][3]{}, aa[3][3]{}, norm_a = 0, divz = 0;
  for (int i = 0; i < 3; ++i) {
    divbeta += u.beta.d[i][i];
    zup[i] = T(0.5)*(u.lambda.value[i]-gt.contracted[i]);
    for (int j = 0; j < 3; ++j) {
      z[i] += T(0.5)*u.metric.g[i][j]*(u.lambda.value[j]-gt.contracted[j]);
      for (int d = 0; d < 3; ++d) {
        dz[d][i] += T(0.5)*(u.metric.dg[d][i][j]
            *(u.lambda.value[j]-gt.contracted[j])
            +u.metric.g[i][j]*(u.lambda.d[d][j]-gt.dcontracted[d][j]));
      }
    }
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    hessalpha[i][j] = u.alpha.dd[i][j];
    hessomega[i][j] = o.hessian[i][j];
    dzbar[i][j] = dz[i][j];
    T dztilde = dz[i][j];
    for (int k = 0; k < 3; ++k) {
      hessalpha[i][j] -= gb.connection[k][i][j]*u.alpha.d[k];
      hessomega[i][j] -= gb.connection[k][i][j]*o.gradient[k];
      dzbar[i][j] -= gb.connection[k][i][j]*z[k];
      dztilde -= gt.connection[k][i][j]*z[k];
      for (int l = 0; l < 3; ++l) {
        aup[i][j] += gt.inverse[i][k]*gt.inverse[j][l]*u.a.k[k][l];
      }
      for (int l = 0; l < 3; ++l) {
        aa[i][j] += gt.inverse[k][l]*u.a.k[i][k]*u.a.k[l][j];
      }
    }
    divz += gt.inverse[i][j]*dztilde;
    norm_a += gt.inverse[i][j]*aa[i][j];
    lapalpha += gb.inverse[i][j]*hessalpha[i][j];
    lapomega += gb.inverse[i][j]*hessomega[i][j];
    gradient2 += gb.inverse[i][j]*o.gradient[i]*o.gradient[j];
    cross += gb.inverse[i][j]*u.alpha.d[i]*o.gradient[j];
  }
  auto &r = out.regular;
  auto &s = out.pole;
  r.chi = -T(2./3.)*chi*divbeta;
  s.chi = T(2./3.)*alpha*chi*(physical_k-3*w);
  r.trace = omega*(alpha*norm_a-lapalpha)+3*cross+alpha*lapomega;
  s.trace = alpha*(physical_k*physical_k/3-3*gradient2+kappa1*(1-kappa2)*theta);
  r.theta = omega*alpha*(gb.scalar-norm_a+2*chi*divz)/2+2*alpha*lapomega;
  s.theta = alpha*(physical_k*physical_k/3-3*gradient2
                  -(3*w+kappa1*(2+kappa2))*theta);
  for (int d = 0; d < 3; ++d) {
    r.chi += u.beta.value[d]*u.chi.d[d];
    r.trace += u.beta.value[d]*u.trace.d[d];
    r.theta += u.beta.value[d]*u.theta.d[d];
  }
  T tensor_regular[3][3]{}, tensor_pole[3][3]{}, tr = 0, ts = 0;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    tensor_regular[i][j] = chi*(-hessalpha[i][j]
        +alpha*(gb.ricci[i][j]+dzbar[i][j]+dzbar[j][i]));
    tensor_pole[i][j] = 2*alpha*chi*(hessomega[i][j]
        +z[i]*o.gradient[j]+z[j]*o.gradient[i]);
    tr += gt.inverse[i][j]*tensor_regular[i][j];
    ts += gt.inverse[i][j]*tensor_pole[i][j];
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    r.metric[i][j] = -2*alpha*u.a.k[i][j]-T(2./3.)*u.metric.g[i][j]*divbeta;
    r.a[i][j] = tensor_regular[i][j]-u.metric.g[i][j]*tr/3
        -2*alpha*aa[i][j]-T(2./3.)*u.a.k[i][j]*divbeta;
    s.a[i][j] = tensor_pole[i][j]-u.metric.g[i][j]*ts/3
        +alpha*u.a.k[i][j]*(physical_k-w);
    for (int d = 0; d < 3; ++d) {
      r.metric[i][j] += u.beta.value[d]*u.metric.dg[d][i][j]
          +u.metric.g[d][j]*u.beta.d[i][d]+u.metric.g[i][d]*u.beta.d[j][d];
      r.a[i][j] += u.beta.value[d]*u.a.dk[d][i][j]
          +u.a.k[d][j]*u.beta.d[i][d]+u.a.k[i][d]*u.beta.d[j][d];
    }
  }
  for (int i = 0; i < 3; ++i) {
    r.lambda[i] = T(2./3.)*u.lambda.value[i]*divbeta;
    s.lambda[i] = -alpha*(T(4./3.)*physical_k+2*kappa1)*zup[i];
    for (int j = 0; j < 3; ++j) {
      r.lambda[i] += u.beta.value[j]*u.lambda.d[j][i]
          -gt.contracted[j]*u.beta.d[j][i]
          -2*aup[i][j]*u.alpha.d[j]-3*alpha*aup[i][j]*u.chi.d[j]/chi;
      s.lambda[i] -= T(2./3.)*alpha*gt.inverse[i][j]
          *(2*u.trace.d[j]+u.theta.d[j])+4*alpha*aup[i][j]*o.gradient[j];
      for (int k = 0; k < 3; ++k) {
        r.lambda[i] += 2*alpha*gt.connection[i][j][k]*aup[j][k]
            +gt.inverse[j][k]*u.beta.dd[j][k][i]
            +gt.inverse[i][j]*u.beta.dd[j][k][k]/3;
      }
    }
  }
  // Reject any invalid result; no clipping, floors or silently zeroed equations.
  out.valid = Kokkos::isfinite(r.chi) && Kokkos::isfinite(s.chi)
      && Kokkos::isfinite(r.trace) && Kokkos::isfinite(s.trace)
      && Kokkos::isfinite(r.theta) && Kokkos::isfinite(s.theta);
  for (int i = 0; i < 3; ++i) {
    out.valid = out.valid && Kokkos::isfinite(r.lambda[i])
        && Kokkos::isfinite(s.lambda[i]);
    for (int j = 0; j < 3; ++j) {
      out.valid = out.valid && Kokkos::isfinite(r.metric[i][j])
          && Kokkos::isfinite(r.a[i][j]) && Kokkos::isfinite(s.a[i][j]);
    }
  }
  return out;
}

// Interior assembly only. At scri the caller needs the limit of pole/Omega,
// not its value with a small denominator substituted.
}}
#endif
