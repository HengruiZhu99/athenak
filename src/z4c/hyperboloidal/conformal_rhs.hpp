// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_CONFORMAL_RHS_HPP_
#define Z4C_HYPERBOLOIDAL_CONFORMAL_RHS_HPP_

#include "z4c/hyperboloidal/conformal_constraints.hpp"
#include "z4c/hyperboloidal/regular_penrose.hpp"

namespace z4c {
namespace hyperboloidal {

template <typename T>
struct ScalarJet {
  T value, d[3], dd[3][3];
};
template <typename T>
struct VectorJet {
  T value[3], d[3][3], dd[3][3][3];  // derivatives first, component last
};
template <typename T>
struct Z4cJet {
  MetricJet<T> metric;  // twice-conformal, determinant one
  CurvatureJet<T> a;    // trace-free A and spatial derivatives
  ScalarJet<T> chi, alpha, trace, theta;
  VectorJet<T> beta, lambda;
  // trace = K_phys - 2 Theta_phys; theta = Theta_phys. Both are finite.
  // This is the FULL trace, not its deviation from the CMC reference.
};
template <typename T>
struct Z4cRHS {
  T chi, metric[3][3], a[3][3], trace, theta, lambda[3];
};
template <typename T>
struct Z4cRHSParts {
  bool valid;
  Z4cRHS<T> regular, pole;  // dt u = regular + pole/Omega, Omega>0
};

template <typename T>
struct EvolvedConstraintResult {
  bool valid;
  T hamiltonian, momentum[3], momentum_conformal_norm2, momentum_physical_norm2;
  T physical_trace, null_residual;
  Z4ConstraintResult<T> z4;
};

// Convert twice-conformal spatial metric jets to Penrose spatial metric jets.
template <typename T>
KOKKOS_INLINE_FUNCTION
MetricJet<T> PenroseMetric(const MetricJet<T> &g, const ScalarJet<T> &chi) {
  MetricJet<T> b{};
  const T c = chi.value;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    b.g[i][j] = g.g[i][j]/c;
    for (int d = 0; d < 3; ++d) {
      b.dg[d][i][j] = g.dg[d][i][j]/c-g.g[i][j]*chi.d[d]/(c*c);
      for (int e = 0; e < 3; ++e) {
        b.ddg[d][e][i][j] = g.ddg[d][e][i][j]/c
            -(g.dg[d][i][j]*chi.d[e]+g.dg[e][i][j]*chi.d[d]
              +g.g[i][j]*chi.dd[d][e])/(c*c)
            +2*g.g[i][j]*chi.d[d]*chi.d[e]/(c*c*c);
      }
    }
  }
  return b;
}

// Physical constraints in the evolved variables themselves. Substitution of
// K_bar=(trace+2*theta-3*w)/Omega cancels every division by Omega and every w
// term in H and M. Thus this also diagnoses data right at scri, without ever
// reconstructing a potentially singular conformal trace.
template <typename T>
KOKKOS_INLINE_FUNCTION
EvolvedConstraintResult<T> EvolvedConstraints(const Z4cJet<T> &u, const OmegaJet<T> &o) {
  EvolvedConstraintResult<T> out{};
  out.valid = u.chi.value > 0 && Kokkos::isfinite(u.chi.value);
  if (!out.valid) return out;
  const auto gt = Geometry(u.metric);
  const auto penrose = RegularPenroseGeometry(u.metric, gt, u.chi);
  ScalarJet<T> omega_jet{}; omega_jet.value = o.omega;
  for (int i = 0; i < 3; ++i) {
    omega_jet.d[i] = o.gradient[i];
    for (int j = 0; j < 3; ++j) omega_jet.dd[i][j] = o.hessian[i][j];
  }
  const auto omega_derivatives = RegularPenroseHessian(u.metric, gt, u.chi, omega_jet);
  out.valid = gt.valid && penrose.valid && omega_derivatives.valid;
  if (!out.valid) return out;
  out.z4 = Z4Constraints(u.metric, u.chi.value, u.a.k, u.lambda.value,
                         u.theta.value, o.omega);
  out.physical_trace = u.trace.value+2*u.theta.value;
  T gradient2 = 0, norm_a = 0;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    gradient2 += u.chi.value*gt.inverse[i][j]*o.gradient[i]*o.gradient[j];
    for (int k = 0; k < 3; ++k)
    for (int l = 0; l < 3; ++l) {
      norm_a += gt.inverse[i][k]*gt.inverse[j][l]*u.a.k[i][j]*u.a.k[k][l];
    }
  }
  out.null_residual = gradient2-o.normal*o.normal;
  out.hamiltonian = o.omega*o.omega*(penrose.scalar-norm_a)
      +T(2./3.)*out.physical_trace*out.physical_trace
      +4*o.omega*omega_derivatives.laplacian-6*gradient2;
  for (int i = 0; i < 3; ++i) {
    T divergence = 0, contraction = 0;
    for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) {
      T derivative = u.a.dk[j][k][i];
      for (int l = 0; l < 3; ++l) {
        derivative -= gt.connection[l][j][k]*u.a.k[l][i]
                     +gt.connection[l][j][i]*u.a.k[k][l];
      }
      divergence += gt.inverse[j][k]*(derivative
          -T(1.5)*u.a.k[k][i]*penrose.gradient_ratio[j]);
      contraction += gt.inverse[j][k]*u.a.k[k][i]*o.gradient[j];
    }
    out.momentum[i] = o.omega*divergence-T(2./3.)*(u.trace.d[i]+2*u.theta.d[i])
                     -2*contraction;
    out.valid = out.valid && Kokkos::isfinite(out.momentum[i]);
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    out.momentum_conformal_norm2 += u.chi.value*gt.inverse[i][j]*out.momentum[i]*out.momentum[j];
  }
  out.momentum_physical_norm2 = o.omega*o.omega*out.momentum_conformal_norm2;
  out.valid = out.valid && out.z4.valid && Kokkos::isfinite(out.hamiltonian)
      && Kokkos::isfinite(out.null_residual)
      && Kokkos::isfinite(out.momentum_conformal_norm2)
      && Kokkos::isfinite(out.momentum_physical_norm2);
  return out;
}

// Vacuum C_Z4c=0 tensor system of arXiv:1412.3827 Appendix B, after the physical
// trace/Theta transformation. Lagrangian determinant condition in Cartesian
// coordinates; flat reference spatial connection. The shift derivative in the
// reference-connection term is an ordinary Cartesian derivative.
//
// This evaluates the complete interior geometric RHS, NOT a scri boundary
// closure. The pole numerators are exposed even at Omega=0 so a limiting equation
// can be imposed explicitly. Never replace Omega by a numerical floor.
// The tensor-system Theta term -3*w*theta/Omega is retained (see derivation/test);
// the later spherical stabilization in that paper makes a different choice.
// appendix_c=true evaluates that already-disclosed selected Theta equation
// directly; its callers must not add 3*alpha*w*theta afterward a second time.
template <typename T>
KOKKOS_INLINE_FUNCTION
Z4cRHSParts<T> ConformalRHS(const Z4cJet<T> &u, const OmegaJet<T> &o,
                          T kappa1, T kappa2, bool appendix_c = false) {
  Z4cRHSParts<T> out{};
  out.valid = u.chi.value > 0 && u.alpha.value > 0
      && Kokkos::isfinite(u.chi.value) && Kokkos::isfinite(u.alpha.value);
  if (!out.valid) return out;
  const auto gt = Geometry(u.metric);
  const auto penrose = RegularPenroseGeometry(u.metric, gt, u.chi);
  ScalarJet<T> omega_jet{}; omega_jet.value = o.omega;
  for (int i = 0; i < 3; ++i) {
    omega_jet.d[i] = o.gradient[i];
    for (int j = 0; j < 3; ++j) omega_jet.dd[i][j] = o.hessian[i][j];
  }
  const auto alpha_derivatives = RegularPenroseHessian(u.metric, gt, u.chi, u.alpha);
  const auto omega_derivatives = RegularPenroseHessian(u.metric, gt, u.chi, omega_jet);
  out.valid = gt.valid && penrose.valid && alpha_derivatives.valid && omega_derivatives.valid;
  if (!out.valid) return out;
  const T alpha = u.alpha.value, chi = u.chi.value, omega = o.omega;
  const T p = u.trace.value, theta = u.theta.value;
  // A known stationary compactification can supply alpha*w directly without
  // constructing an unrepresentable w. Explicit nonstationary normal data keep
  // their original alpha*normal meaning. No normal or null diagnostic is reset.
  const T alpha_w = LapseOmegaNormalProduct(alpha, o);
  const T physical_k = p+2*theta;
  T divbeta = 0, gradient2 = 0, cross = 0;
  T z[3]{}, zup[3]{}, dz[3][3]{};
  T aup[3][3]{}, aa[3][3]{}, norm_a = 0, chi_divz = 0;
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
    T chi_dztilde = chi*dz[i][j];
    for (int k = 0; k < 3; ++k) {
      chi_dztilde -= gt.connection[k][i][j]*(chi*z[k]);
      for (int l = 0; l < 3; ++l) {
        aup[i][j] += gt.inverse[i][k]*gt.inverse[j][l]*u.a.k[k][l];
      }
      for (int l = 0; l < 3; ++l) {
        aa[i][j] += gt.inverse[k][l]*u.a.k[i][k]*u.a.k[l][j];
      }
    }
    chi_divz += gt.inverse[i][j]*chi_dztilde;
    norm_a += gt.inverse[i][j]*aa[i][j];
    gradient2 += chi*gt.inverse[i][j]*o.gradient[i]*o.gradient[j];
    cross += chi*gt.inverse[i][j]*u.alpha.d[i]*o.gradient[j];
  }
  const auto z_derivatives = RegularPenroseCovectorDerivative(u.metric, gt, u.chi, z, dz);
  if (!z_derivatives.valid) {out.valid = false; return out;}
  auto &r = out.regular;
  auto &s = out.pole;
  r.chi = -T(2./3.)*chi*divbeta;
  s.chi = T(2./3.)*chi*(alpha*physical_k-3*alpha_w);
  r.trace = omega*(alpha*norm_a-alpha_derivatives.laplacian)
      +3*cross+alpha*omega_derivatives.laplacian;
  s.trace = alpha*(physical_k*physical_k/3-3*gradient2+kappa1*(1-kappa2)*theta);
  r.theta = omega*alpha*(penrose.scalar-norm_a+2*chi_divz)/2
      +2*alpha*omega_derivatives.laplacian;
  s.theta = alpha*(physical_k*physical_k/3-3*gradient2-kappa1*(2+kappa2)*theta)
      -(appendix_c ? T(0) : 3*alpha_w*theta);
  for (int d = 0; d < 3; ++d) {
    r.chi += u.beta.value[d]*u.chi.d[d];
    r.trace += u.beta.value[d]*u.trace.d[d];
    r.theta += u.beta.value[d]*u.theta.d[d];
  }
  T tensor_regular[3][3]{}, tensor_pole[3][3]{}, tr = 0, ts = 0;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    tensor_regular[i][j] = -alpha_derivatives.chi_hessian[i][j]
        +alpha*(penrose.chi_ricci[i][j]+z_derivatives.chi_derivative[i][j]
                +z_derivatives.chi_derivative[j][i]);
    tensor_pole[i][j] = 2*alpha*(omega_derivatives.chi_hessian[i][j]
        +chi*(z[i]*o.gradient[j]+z[j]*o.gradient[i]));
    tr += gt.inverse[i][j]*tensor_regular[i][j];
    ts += gt.inverse[i][j]*tensor_pole[i][j];
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    r.metric[i][j] = -2*alpha*u.a.k[i][j]-T(2./3.)*u.metric.g[i][j]*divbeta;
    r.a[i][j] = tensor_regular[i][j]-u.metric.g[i][j]*tr/3
        -2*alpha*aa[i][j]-T(2./3.)*u.a.k[i][j]*divbeta;
    s.a[i][j] = tensor_pole[i][j]-u.metric.g[i][j]*ts/3
        +u.a.k[i][j]*(alpha*physical_k-alpha_w);
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
          -2*aup[i][j]*u.alpha.d[j]-3*alpha*aup[i][j]*penrose.gradient_ratio[j];
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
template <typename T>
KOKKOS_INLINE_FUNCTION
bool AssembleInterior(const Z4cRHSParts<T> &parts, T omega, Z4cRHS<T> &rhs) {
  if (!parts.valid || !(omega > 0) || !Kokkos::isfinite(omega)) return false;
  rhs = parts.regular;
  rhs.chi += parts.pole.chi/omega;
  rhs.trace += parts.pole.trace/omega;
  rhs.theta += parts.pole.theta/omega;
  bool valid = Kokkos::isfinite(rhs.chi) && Kokkos::isfinite(rhs.trace)
      && Kokkos::isfinite(rhs.theta);
  for (int i = 0; i < 3; ++i) {
    rhs.lambda[i] += parts.pole.lambda[i]/omega;
    valid = valid && Kokkos::isfinite(rhs.lambda[i]);
    for (int j = 0; j < 3; ++j) {
      rhs.metric[i][j] += parts.pole.metric[i][j]/omega;
      rhs.a[i][j] += parts.pole.a[i][j]/omega;
      valid = valid && Kokkos::isfinite(rhs.metric[i][j])
          && Kokkos::isfinite(rhs.a[i][j]);
    }
  }
  return valid;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_CONFORMAL_RHS_HPP_
