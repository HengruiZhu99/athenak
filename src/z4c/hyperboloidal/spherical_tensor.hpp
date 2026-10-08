// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_SPHERICAL_TENSOR_HPP_
#define Z4C_HYPERBOLOIDAL_SPHERICAL_TENSOR_HPP_

#include "z4c/hyperboloidal/cmc_reference.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace z4c {
namespace hyperboloidal {

enum RadialField { DCHI, DGRR, ARR, DK, THETA, LAMBDA, DALPHA, DBETA, NFIELDS };

// Cartesian jets at (r,0,0). The radial grid excludes r=0. Angular derivatives
// come from the tensor transformation, never from one-sided Cartesian stencils.
template <typename T>
KOKKOS_INLINE_FUNCTION
ScalarJet<T> RadialScalar(T f, T df, T ddf, T radius) {
  ScalarJet<T> out{};
  out.value = f;
  out.d[0] = df;
  out.dd[0][0] = ddf;
  out.dd[1][1] = out.dd[2][2] = df/radius;
  return out;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
VectorJet<T> RadialVector(T f, T df, T ddf, T radius) {
  VectorJet<T> out{};
  out.value[0] = f;
  out.d[0][0] = df;
  out.d[1][1] = out.d[2][2] = f/radius;
  out.dd[0][0][0] = ddf;
  for (int j = 1; j < 3; ++j) {
    out.dd[j][j][0] = df/radius-f/(radius*radius);
    out.dd[0][j][j] = out.dd[j][0][j] = out.dd[j][j][0];
  }
  return out;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
MetricJet<T> RadialTensor(T a, T da, T dda, T b, T db, T ddb, T radius) {
  MetricJet<T> out{};
  const T n[3] = {1, 0, 0};
  T dn[3][3]{}, ddn[3][3][3]{};
  for (int i = 0; i < 3; ++i)
  for (int d = 0; d < 3; ++d) {
    dn[d][i] = ((d == i ? 1 : 0)-n[d]*n[i])/radius;
    for (int e = 0; e < 3; ++e) {
      ddn[d][e][i] = (-(d == i ? n[e] : 0)-(e == i ? n[d] : 0)
          -(d == e ? n[i] : 0)+3*n[d]*n[e]*n[i])/(radius*radius);
    }
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    const T delta = i == j ? 1 : 0;
    out.g[i][j] = b*delta+(a-b)*n[i]*n[j];
    for (int d = 0; d < 3; ++d) {
      out.dg[d][i][j] = db*n[d]*delta+(da-db)*n[d]*n[i]*n[j]
          +(a-b)*(dn[d][i]*n[j]+n[i]*dn[d][j]);
      for (int e = 0; e < 3; ++e) {
        out.ddg[d][e][i][j] = (ddb*n[d]*n[e]+db*dn[d][e])*delta
            +((dda-ddb)*n[d]*n[e]+(da-db)*dn[d][e])*n[i]*n[j]
            +(da-db)*n[d]*(dn[e][i]*n[j]+n[i]*dn[e][j])
            +(da-db)*n[e]*(dn[d][i]*n[j]+n[i]*dn[d][j])
            +(a-b)*(ddn[d][e][i]*n[j]+dn[d][i]*dn[e][j]
                    +dn[e][i]*dn[d][j]+n[i]*ddn[d][e][j]);
      }
    }
  }
  return out;
}

// Stored fields are deviations from CMC Minkowski where appropriate. Analytic
// background derivatives are added after differencing deviations. This avoids
// a truncation-error forcing of the known reference and preserves its parity.
template <typename T>
KOKKOS_INLINE_FUNCTION
Z4cJet<T> SphericalJet(T radius, const T q[NFIELDS], const T d[NFIELDS],
                     const T dd[NFIELDS], const CMCReference<T> &ref) {
  Z4cJet<T> u{};
  const auto p = ref.At(radius, T(0), T(0));
  const T grr = 1+q[DGRR], gtt = 1/Kokkos::sqrt(grr);
  const T dgtt = -T(0.5)*gtt*d[DGRR]/grr;
  const T ddgtt = T(0.75)*gtt*d[DGRR]*d[DGRR]/(grr*grr)
      -T(0.5)*gtt*dd[DGRR]/grr;
  const T att = -T(0.5)*q[ARR]*gtt/grr;
  const T datt = -T(0.5)*gtt/grr*(d[ARR]-T(1.5)*q[ARR]*d[DGRR]/grr);
  u.metric = RadialTensor(grr, d[DGRR], dd[DGRR], gtt, dgtt, ddgtt, radius);
  const auto tensor_a = RadialTensor(q[ARR], d[ARR], T(0), att, datt, T(0), radius);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    u.a.k[i][j] = tensor_a.g[i][j];
    for (int k = 0; k < 3; ++k) u.a.dk[k][i][j] = tensor_a.dg[k][i][j];
  }
  u.chi = RadialScalar(1+q[DCHI], d[DCHI], dd[DCHI], radius);
  u.trace = RadialScalar(p.k_physical+q[DK], d[DK], dd[DK], radius);
  u.theta = RadialScalar(q[THETA], d[THETA], dd[THETA], radius);
  u.alpha = RadialScalar(p.alpha+q[DALPHA], p.dalpha[0]+d[DALPHA],
                         p.hessian_alpha+dd[DALPHA], radius);
  u.beta = RadialVector(p.beta[0]+q[DBETA], -1/ref.curvature_radius+d[DBETA],
                        dd[DBETA], radius);
  u.lambda = RadialVector(q[LAMBDA], d[LAMBDA], dd[LAMBDA], radius);
  return u;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_SPHERICAL_TENSOR_HPP_
