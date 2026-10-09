// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CARTESIAN_RADIAL_HPP_
#define Z4C_HYPERBOLOIDAL_CARTESIAN_RADIAL_HPP_

#include "z4c/hyperboloidal/spherical_tensor.hpp"

namespace z4c {
namespace hyperboloidal {

// Extend a spherical jet known at (r,0,0) to x=r*n. All derivative indices
// transform too. The caller supplies r>0 and a unit n; the puncture is excluded.
// This is a spherical initial-data/analytic reconstruction helper, not an
// assumption of spherical symmetry for subsequent Cartesian evolution.
template <typename T>
KOKKOS_INLINE_FUNCTION
Z4cJet<T> CartesianRadialJet(const Z4cJet<T> &axis, T r, const T n[3]) {
  T dn[3][3]{}, ddn[3][3][3]{};
  for (int d = 0; d < 3; ++d)
  for (int i = 0; i < 3; ++i) {
    dn[d][i] = ((d == i ? 1 : 0)-n[d]*n[i])/r;
    for (int e = 0; e < 3; ++e) {
      ddn[d][e][i] = (-(d == i ? n[e] : 0)-(e == i ? n[d] : 0)
          -(d == e ? n[i] : 0)+3*n[d]*n[e]*n[i])/(r*r);
    }
  }
  Z4cJet<T> out{};
  const ScalarJet<T> scalars[4] = {axis.chi,axis.alpha,axis.trace,axis.theta};
  ScalarJet<T> result[4]{};
  for (int f = 0; f < 4; ++f) {
    result[f].value = scalars[f].value;
    for (int d = 0; d < 3; ++d) {
      result[f].d[d] = scalars[f].d[0]*n[d];
      for (int e = 0; e < 3; ++e) {
        result[f].dd[d][e] = scalars[f].dd[0][0]*n[d]*n[e]+scalars[f].d[0]*dn[d][e];
      }
    }
  }
  out.chi = result[0]; out.alpha = result[1];
  out.trace = result[2]; out.theta = result[3];
  const VectorJet<T> vectors[2] = {axis.beta,axis.lambda};
  VectorJet<T> vr[2]{};
  for (int f = 0; f < 2; ++f) {
    const T v = vectors[f].value[0], dv = vectors[f].d[0][0];
    const T ddv = vectors[f].dd[0][0][0];
    for (int i = 0; i < 3; ++i) {
      vr[f].value[i] = v*n[i];
      for (int d = 0; d < 3; ++d) {
        vr[f].d[d][i] = dv*n[d]*n[i]+v*dn[d][i];
        for (int e = 0; e < 3; ++e) {
          vr[f].dd[d][e][i] = ddv*n[d]*n[e]*n[i]
              +dv*(dn[d][e]*n[i]+n[d]*dn[e][i]+n[e]*dn[d][i])+v*ddn[d][e][i];
        }
      }
    }
  }
  out.beta = vr[0]; out.lambda = vr[1];
  const T a = axis.metric.g[0][0], b = axis.metric.g[1][1];
  const T da = axis.metric.dg[0][0][0], db = axis.metric.dg[0][1][1];
  const T dda = axis.metric.ddg[0][0][0][0], ddb = axis.metric.ddg[0][0][1][1];
  const T ar = axis.a.k[0][0], at = axis.a.k[1][1];
  const T dar = axis.a.dk[0][0][0], dat = axis.a.dk[0][1][1];
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    const T delta = i == j ? 1 : 0;
    out.metric.g[i][j] = b*delta+(a-b)*n[i]*n[j];
    out.a.k[i][j] = at*delta+(ar-at)*n[i]*n[j];
    for (int d = 0; d < 3; ++d) {
      const T angular = dn[d][i]*n[j]+n[i]*dn[d][j];
      out.metric.dg[d][i][j] = db*n[d]*delta+(da-db)*n[d]*n[i]*n[j]+(a-b)*angular;
      out.a.dk[d][i][j] = dat*n[d]*delta+(dar-dat)*n[d]*n[i]*n[j]+(ar-at)*angular;
      for (int e = 0; e < 3; ++e) {
        out.metric.ddg[d][e][i][j] = (ddb*n[d]*n[e]+db*dn[d][e])*delta
            +((dda-ddb)*n[d]*n[e]+(da-db)*dn[d][e])*n[i]*n[j]
            +(da-db)*(n[d]*(dn[e][i]*n[j]+n[i]*dn[e][j])+n[e]*angular)
            +(a-b)*(ddn[d][e][i]*n[j]+dn[d][i]*dn[e][j]
                    +dn[e][i]*dn[d][j]+n[i]*ddn[d][e][j]);
      }
    }
  }
  return out;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CARTESIAN_RADIAL_HPP_
