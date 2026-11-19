// Ported from scri-ko01 src/z4c/hyperboloidal/conformal_constraints.hpp; mathematical
// expressions retained. Native integration has no legacy driver dependency.
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_NATIVE_CONFORMAL_CONSTRAINTS_HPP_
#define Z4C_HYPERBOLOIDAL_NATIVE_CONFORMAL_CONSTRAINTS_HPP_

#include <limits>
#include <Kokkos_Core.hpp>

namespace z4c_hyperboloidal {

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
KOKKOS_INLINE_FUNCTION bool InvertMetric(const T g[3][3], T inverse[3][3],
                                        T& determinant) {
  determinant = g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])
      - g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])
      + g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0]);
  if (!(Kokkos::isfinite(determinant) && determinant > 0 && g[0][0] > 0
        && g[0][0]*g[1][1]-g[0][1]*g[1][0] > 0)) return false;
  for (int i = 0; i < 3; ++i) {
    const int p = (i+1)%3, q = (i+2)%3;
    for (int j = 0; j < 3; ++j) {
      const int r = (j+1)%3, s = (j+2)%3;
      inverse[j][i] = (g[p][r]*g[q][s]-g[p][s]*g[q][r])/determinant;
    }
  }
  return true;
}

// With constant input jets the connection and curvature vanish. Check the
// inverse and the zero derivatives explicitly, including any overflow from
// expanding the dependent chart coordinate before projection.
template <typename T>
KOKKOS_INLINE_FUNCTION bool InvertConstantMetric(const MetricJet<T>& m, T inverse[3][3]) {
  T determinant;
  if (!InvertMetric(m.g, inverse, determinant)) return false;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (!Kokkos::isfinite(inverse[i][j])) return false;
      for (int a = 0; a < 3; ++a) {
        if (m.dg[a][i][j] != 0) return false;
        for (int b = 0; b < 3; ++b) {
          if (m.ddg[a][b][i][j] != 0) return false;
        }
      }
    }
  }
  return true;
}

template <typename T>
KOKKOS_INLINE_FUNCTION
SpatialGeometry<T> Geometry(const MetricJet<T> &m) {
  SpatialGeometry<T> out{};
  out.valid = InvertMetric(m.g, out.inverse, out.determinant);
  if (!out.valid) return out;
  // Work with Christoffels of the first kind. Only contracted derivatives of
  // the connection enter Ricci and d(Gamma); no rank-four connection is needed.
  T lower[3][3][3]{}, dinv[3][3][3]{}, contracted_lower[3]{};
  for (int a = 0; a < 3; ++a) {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        lower[a][i][j] = lower[a][j][i] =
            T(0.5) * (m.dg[i][a][j] + m.dg[j][a][i] - m.dg[a][i][j]);
      }
    }
  }
  for (int d = 0; d < 3; ++d) {
    T left[3][3]{};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int a = 0; a < 3; ++a) left[i][j] += out.inverse[i][a] * m.dg[d][a][j];
    for (int i = 0; i < 3; ++i)
      for (int j = i; j < 3; ++j) {
        T value = 0;
        for (int a = 0; a < 3; ++a) value -= left[i][a] * out.inverse[a][j];
        dinv[d][i][j] = dinv[d][j][i] = value;
      }
  }
  for (int k = 0; k < 3; ++k) {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {
        T upper = 0;
        for (int a = 0; a < 3; ++a) upper += out.inverse[k][a] * lower[a][i][j];
        out.connection[k][i][j] = out.connection[k][j][i] = upper;
        contracted_lower[k] += (i == j ? 1 : 2) * out.inverse[i][j] * lower[k][i][j];
      }
    }
  }
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) out.contracted[i] += out.inverse[i][k] * contracted_lower[k];
  for (int d = 0; d < 3; ++d) {
    for (int a = 0; a < 3; ++a) {
      T derivative = 0;
      for (int i = 0; i < 3; ++i) {
        for (int j = i; j < 3; ++j) {
          const T dfirst = T(0.5) * (m.ddg[d][i][a][j] + m.ddg[d][j][a][i] - m.ddg[d][a][i][j]);
          derivative += (i == j ? 1 : 2) * (dinv[d][i][j] * lower[a][i][j] + out.inverse[i][j] * dfirst);
        }
      }
      for (int k = 0; k < 3; ++k) {
        out.dcontracted[d][k] += dinv[d][k][a] * contracted_lower[a] + out.inverse[k][a] * derivative;
      }
    }
  }
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      T ricci = 0;
      for (int k = 0; k < 3; ++k) {
        for (int a = 0; a < 3; ++a) {
          const T hessian = m.ddg[k][i][a][j] + m.ddg[j][a][i][k]
                          - m.ddg[k][a][i][j] - m.ddg[j][i][a][k];
          ricci += T(0.5) * out.inverse[k][a] * hessian
                 + dinv[k][k][a] * lower[a][i][j] - dinv[j][k][a] * lower[a][i][k]
                 + out.connection[k][i][j] * out.connection[a][k][a]
                 - out.connection[a][i][k] * out.connection[k][j][a];
        }
      }
      out.ricci[i][j] = out.ricci[j][i] = ricci;
      out.scalar += (i == j ? 1 : 2) * out.inverse[i][j] * ricci;
    }
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

}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_NATIVE_CONFORMAL_CONSTRAINTS_HPP_
