// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CONFORMAL_CONSTRAINTS_HPP_
#define Z4C_HYPERBOLOIDAL_CONFORMAL_CONSTRAINTS_HPP_

#include <limits>
#include <Kokkos_Core.hpp>

namespace z4c {
namespace hyperboloidal {

// Cartesian spatial tensors of the Penrose-rescaled metric, not the unit-
// determinant Z4c metric. Derivative indices precede tensor indices. No symmetry
// compression: populate all entries, including both symmetric copies.
template <typename T>
struct MetricJet {
  T g[3][3], dg[3][3][3], ddg[3][3][3][3];
};
template <typename T>
struct CurvatureJet {
  T k[3][3], dk[3][3][3];
};
template <typename T>
struct OmegaJet {
  T omega, gradient[3], hessian[3][3];
  T normal, dnormal[3];  // w=n_bar(Omega), and its coordinate gradient
};
template <typename T>
struct SpatialGeometry {
  bool valid;
  T determinant, inverse[3][3], connection[3][3][3], ricci[3][3], scalar;
  T contracted[3], dcontracted[3][3];  // dcontracted[derivative][component]
};

template <typename T>
KOKKOS_INLINE_FUNCTION
SpatialGeometry<T> Geometry(const MetricJet<T> &m) {
  SpatialGeometry<T> out{};
  const T (*g)[3] = m.g;
  const T det = g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])
      - g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])
      + g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);
  out.determinant = det;
  out.valid = Kokkos::isfinite(det) && det > 0 && g[0][0] > 0
      && g[0][0]*g[1][1]-g[0][1]*g[1][0] > 0;
  if (!out.valid) return out;
  for (int i = 0; i < 3; ++i) {
    const int p = (i+1)%3, q = (i+2)%3;
    for (int j = 0; j < 3; ++j) {
      const int r = (j+1)%3, s = (j+2)%3;
      out.inverse[j][i] = (g[p][r]*g[q][s]-g[p][s]*g[q][r])/det;
    }
  }
  T dinverse[3][3][3]{};
  T dconnection[3][3][3][3]{};
  for (int d = 0; d < 3; ++d)
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  for (int a = 0; a < 3; ++a)
  for (int b = 0; b < 3; ++b) {
    dinverse[d][i][j] -= out.inverse[i][a]*m.dg[d][a][b]*out.inverse[b][j];
  }
  for (int k = 0; k < 3; ++k)
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  for (int a = 0; a < 3; ++a) {
    const T first = m.dg[i][a][j]+m.dg[j][a][i]-m.dg[a][i][j];
    out.connection[k][i][j] += T(0.5)*out.inverse[k][a]*first;
    for (int d = 0; d < 3; ++d) {
      dconnection[d][k][i][j] += T(0.5)*(dinverse[d][k][a]*first
          + out.inverse[k][a]*(m.ddg[d][i][a][j]+m.ddg[d][j][a][i]
                               -m.ddg[d][a][i][j]));
    }
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  for (int k = 0; k < 3; ++k) {
    out.contracted[i] += out.inverse[j][k]*out.connection[i][j][k];
    for (int d = 0; d < 3; ++d) {
      out.dcontracted[d][i] += dinverse[d][j][k]*out.connection[i][j][k]
          +out.inverse[j][k]*dconnection[d][i][j][k];
    }
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 3; ++k) {
      out.ricci[i][j] += dconnection[k][k][i][j]-dconnection[j][k][i][k];
      for (int a = 0; a < 3; ++a) {
        out.ricci[i][j] += out.connection[k][i][j]*out.connection[a][k][a]
            - out.connection[a][i][k]*out.connection[k][j][a];
      }
    }
    out.scalar += out.inverse[i][j]*out.ricci[i][j];
  }
  out.valid = Kokkos::isfinite(out.scalar);
  return out;
}

// Complete the normal derivative for time-independent Omega. dbeta[d][i] is
// partial_d beta^i. alpha must be positive. A time-dependent compactification
// must supply normal and dnormal explicitly instead of using this helper.
template <typename T>
KOKKOS_INLINE_FUNCTION
void SetStationaryOmegaNormal(T alpha, const T beta[3], const T dalpha[3],
                              const T dbeta[3][3], OmegaJet<T> &o) {
  o.normal = 0;
  for (int i = 0; i < 3; ++i) o.normal -= beta[i]*o.gradient[i]/alpha;
  for (int d = 0; d < 3; ++d) {
    o.dnormal[d] = -o.normal*dalpha[d]/alpha;
    for (int i = 0; i < 3; ++i) {
      o.dnormal[d] -= (dbeta[d][i]*o.gradient[i]+beta[i]*o.hessian[d][i])/alpha;
    }
  }
}

template <typename T>
struct ConstraintResult {
  bool valid;
  T hamiltonian, momentum[3], momentum_conformal_norm2, momentum_physical_norm2;
  T physical_trace, conformal_trace, curvature_norm2, null_residual;
};

template <typename T>
struct Z4ConstraintResult {
  bool valid;
  T determinant_residual, tracefree_residual, theta_physical;
  T z_covector[3], z_conformal_norm2, z_physical_norm2;
};

// Here the metric IS the twice-conformal Z4c metric. For the Cartesian CMC
// reference its reference spatial connection is zero. Evolved Lambda obeys
// Lambda^i=contracted_Gamma^i+2 g_tilde^ij Z_j. Theta is supplied in the physical
// normalization explicitly; this function never guesses the evolution scaling.
template <typename T>
KOKKOS_INLINE_FUNCTION
Z4ConstraintResult<T> Z4Constraints(const MetricJet<T> &twice_conformal, T chi,
                                   const T a[3][3], const T lambda[3],
                                   T theta_physical, T omega) {
  Z4ConstraintResult<T> out{};
  const auto geo = Geometry(twice_conformal);
  out.valid = geo.valid && chi > 0 && Kokkos::isfinite(chi)
      && Kokkos::isfinite(theta_physical);
  out.determinant_residual = geo.determinant-1;
  out.theta_physical = theta_physical;
  if (!out.valid) return out;
  T difference[3];
  for (int i = 0; i < 3; ++i) {
    difference[i] = lambda[i];
    for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) {
      difference[i] -= geo.inverse[j][k]*geo.connection[i][j][k];
    }
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.z_covector[i] += T(0.5)*twice_conformal.g[i][j]*difference[j];
    out.tracefree_residual += geo.inverse[i][j]*a[i][j];
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.z_conformal_norm2 += chi*geo.inverse[i][j]*out.z_covector[i]*out.z_covector[j];
  }
  out.z_physical_norm2 = omega*omega*out.z_conformal_norm2;
  out.valid = Kokkos::isfinite(out.z_conformal_norm2)
      && Kokkos::isfinite(out.z_physical_norm2)
      && Kokkos::isfinite(out.tracefree_residual);
  return out;
}

// Physical VACUUM ADM constraints from regular conformal fields. These are
// off-shell identities, not reference-subtracted residuals. No Omega division.
// gamma_phys=Omega^-2 gamma_bar; K_physij=Omega^-1 K_barij
//                                  +Omega^-2 gamma_barij n_bar(Omega).
// Keep the unweighted momentum covector/conformal norm: the physical norm is
// multiplied by Omega^2 and can otherwise hide an error at scri.
template <typename T>
KOKKOS_INLINE_FUNCTION
ConstraintResult<T> Constraints(const MetricJet<T> &m, const CurvatureJet<T> &k,
                                const OmegaJet<T> &o) {
  ConstraintResult<T> out{};
  const auto geo = Geometry(m);
  out.valid = geo.valid;
  if (!out.valid) {
    out.hamiltonian = std::numeric_limits<T>::quiet_NaN();
    return out;
  }
  T dtrace[3]{}, laplacian = 0, gradient2 = 0;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.conformal_trace += geo.inverse[i][j]*k.k[i][j];
    gradient2 += geo.inverse[i][j]*o.gradient[i]*o.gradient[j];
    T hessian = o.hessian[i][j];
    for (int a = 0; a < 3; ++a) hessian -= geo.connection[a][i][j]*o.gradient[a];
    laplacian += geo.inverse[i][j]*hessian;
    for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) {
      out.curvature_norm2 += geo.inverse[i][a]*geo.inverse[j][b]*k.k[i][j]*k.k[a][b];
    }
    for (int d = 0; d < 3; ++d) {
      dtrace[d] += geo.inverse[i][j]*k.dk[d][i][j];
      for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        dtrace[d] -= geo.inverse[i][a]*m.dg[d][a][b]*geo.inverse[b][j]*k.k[i][j];
      }
    }
  }
  const T trace = out.conformal_trace;
  out.physical_trace = o.omega*trace+3*o.normal;
  out.null_residual = gradient2-o.normal*o.normal;
  out.hamiltonian = o.omega*o.omega*(geo.scalar+trace*trace-out.curvature_norm2)
      + 4*o.omega*(laplacian+o.normal*trace)-6*out.null_residual;
  for (int i = 0; i < 3; ++i) {
    T div = 0, contraction = 0;
    for (int j = 0; j < 3; ++j)
    for (int a = 0; a < 3; ++a) {
      T covariant = k.dk[j][a][i];
      for (int b = 0; b < 3; ++b) {
        covariant -= geo.connection[b][j][a]*k.k[b][i]
                     +geo.connection[b][j][i]*k.k[a][b];
      }
      div += geo.inverse[j][a]*covariant;
      contraction += geo.inverse[j][a]*k.k[a][i]*o.gradient[j];
    }
    out.momentum[i] = o.omega*(div-dtrace[i])-2*contraction-2*o.dnormal[i];
    out.valid = out.valid && Kokkos::isfinite(out.momentum[i]);
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.momentum_conformal_norm2 += geo.inverse[i][j]*out.momentum[i]*out.momentum[j];
  }
  out.momentum_physical_norm2 = o.omega*o.omega*out.momentum_conformal_norm2;
  out.valid = out.valid && Kokkos::isfinite(out.hamiltonian)
      && Kokkos::isfinite(out.momentum_conformal_norm2);
  return out;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CONFORMAL_CONSTRAINTS_HPP_
