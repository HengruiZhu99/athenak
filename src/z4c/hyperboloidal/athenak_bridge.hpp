// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#ifndef Z4C_HYPERBOLOIDAL_ATHENAK_BRIDGE_HPP_
#define Z4C_HYPERBOLOIDAL_ATHENAK_BRIDGE_HPP_

#include "z4c/z4c.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"

namespace z4c {
namespace hyperboloidal {

// These adapters operate on AthenaK's actual field views and FD operators.
// The caller must supply a complete valid Cartesian stencil, including mixed
// derivative corners. They are NOT a cut-sphere stencil or a boundary closure.
// Required state convention: Penrose chi/alpha/A, full physical P=K-2Theta,
// physical Theta, unit-determinant metric, integrated (no auxiliary B) shift.
template <int NGHOST, typename Field>
KOKKOS_INLINE_FUNCTION
ScalarJet<Real> LoadMeshScalar(const Field &q, const Real idx[3],
                              int m, int k, int j, int i) {
  static_assert(NGHOST >= 2 && NGHOST <= 4, "unsupported AthenaK FD stencil");
  ScalarJet<Real> out{};
  out.value = q(m,k,j,i);
  for (int a = 0; a < 3; ++a) {
    out.d[a] = Dx<NGHOST>(a, idx, q, m,k,j,i);
    for (int b = a; b < 3; ++b) {
      out.dd[a][b] = out.dd[b][a] = a == b ? Dxx<NGHOST>(a, idx, q, m,k,j,i)
          : Dxy<NGHOST>(a, b, idx, q, m,k,j,i);
    }
  }
  return out;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Z4cJet<Real> LoadMeshJet(const Z4c::Z4c_vars &q, const Real idx[3],
                       int m, int k, int j, int i) {
  Z4cJet<Real> u{};
  u.chi = LoadMeshScalar<NGHOST>(q.chi, idx, m,k,j,i);
  u.alpha = LoadMeshScalar<NGHOST>(q.alpha, idx, m,k,j,i);
  u.trace = LoadMeshScalar<NGHOST>(q.vKhat, idx, m,k,j,i);
  u.theta = LoadMeshScalar<NGHOST>(q.vTheta, idx, m,k,j,i);
  for (int a = 0; a < 3; ++a) {
    u.beta.value[a] = q.beta_u(m,a,k,j,i);
    u.lambda.value[a] = q.vGam_u(m,a,k,j,i);
    for (int d = 0; d < 3; ++d) {
      u.beta.d[d][a] = Dx<NGHOST>(d, idx, q.beta_u, m,a,k,j,i);
      u.lambda.d[d][a] = Dx<NGHOST>(d, idx, q.vGam_u, m,a,k,j,i);
      for (int e = d; e < 3; ++e) {
        u.beta.dd[d][e][a] = u.beta.dd[e][d][a] = d == e ?
            Dxx<NGHOST>(d, idx, q.beta_u, m,a,k,j,i) :
            Dxy<NGHOST>(d, e, idx, q.beta_u, m,a,k,j,i);
        u.lambda.dd[d][e][a] = u.lambda.dd[e][d][a] = d == e ?
            Dxx<NGHOST>(d, idx, q.vGam_u, m,a,k,j,i) :
            Dxy<NGHOST>(d, e, idx, q.vGam_u, m,a,k,j,i);
      }
    }
    for (int b = 0; b < 3; ++b) {
      u.metric.g[a][b] = q.g_dd(m,a,b,k,j,i);
      u.a.k[a][b] = q.vA_dd(m,a,b,k,j,i);
      for (int d = 0; d < 3; ++d) {
        u.metric.dg[d][a][b] = Dx<NGHOST>(d, idx, q.g_dd, m,a,b,k,j,i);
        u.a.dk[d][a][b] = Dx<NGHOST>(d, idx, q.vA_dd, m,a,b,k,j,i);
        for (int e = d; e < 3; ++e) {
          u.metric.ddg[d][e][a][b] = u.metric.ddg[e][d][a][b] = d == e ?
              Dxx<NGHOST>(d, idx, q.g_dd, m,a,b,k,j,i) :
              Dxy<NGHOST>(d, e, idx, q.g_dd, m,a,b,k,j,i);
        }
      }
    }
  }
  return u;
}

// Conservative admission test, including mixed-derivative corners. False means
// the caller needs a boundary treatment, not permission to skip a physical cell.
// Array halo extents remain a separate caller obligation; wider dissipation
// stencils require their own admission check.
template <int NGHOST>
KOKKOS_INLINE_FUNCTION
bool LoadInteriorMeshJet(const Z4c::Z4c_vars &q, const CMCReference<Real> &ref,
                        const Real xyz[3], const Real spacing[3],
                        int m, int k, int j, int i, Z4cJet<Real> &u) {
  Real lo[3], hi[3], idx[3];
  for (int d = 0; d < 3; ++d) {
    if (!(spacing[d] > 0) || !Kokkos::isfinite(spacing[d])
        || !Kokkos::isfinite(xyz[d])) return false;
    lo[d] = xyz[d]-(NGHOST-1)*spacing[d];
    hi[d] = xyz[d]+(NGHOST-1)*spacing[d];
    idx[d] = 1/spacing[d];
  }
  if (ref.ClassifyBox(lo, hi) != -1) return false;
  u = LoadMeshJet<NGHOST>(q, idx, m,k,j,i);
  return true;
}

// The tensor kernel uses centered jets for geometry, constraints and advection.
// Replace ONLY beta^i partial_i of each evolved component by AthenaK's Lx.
// This requires a complete axis halo of radius NGHOST (not NGHOST-1). At scri
// that halo needs an independently validated closure. These corrections do not
// modify geometric derivatives or add a characteristic boundary condition.
template <int NGHOST, typename Field, typename Velocity>
KOKKOS_INLINE_FUNCTION
Real ScalarUpwindCorrection(const Field &field, const Velocity &beta,
                            const Real idx[3], int m, int k, int j, int i) {
  Real correction = 0;
  for (int d = 0; d < 3; ++d) {
    correction += Lx<NGHOST>(d, idx, beta, field, m,d,k,j,i)
        -beta(m,d,k,j,i)*Dx<NGHOST>(d, idx, field, m,k,j,i);
  }
  return correction;
}

template <int NGHOST, typename Field, typename Velocity>
KOKKOS_INLINE_FUNCTION
Real VectorUpwindCorrection(const Field &field, const Velocity &beta,
                            const Real idx[3], int a, int m, int k, int j, int i) {
  Real correction = 0;
  for (int d = 0; d < 3; ++d) {
    correction += Lx<NGHOST>(d, idx, beta, field, m,d,a,k,j,i)
        -beta(m,d,k,j,i)*Dx<NGHOST>(d, idx, field, m,a,k,j,i);
  }
  return correction;
}

template <int NGHOST, typename Field, typename Velocity>
KOKKOS_INLINE_FUNCTION
Real TensorUpwindCorrection(const Field &field, const Velocity &beta,
                            const Real idx[3], int a, int b, int m,
                            int k, int j, int i) {
  Real correction = 0;
  for (int d = 0; d < 3; ++d) {
    correction += Lx<NGHOST>(d, idx, beta, field, m,d,a,b,k,j,i)
        -beta(m,d,k,j,i)*Dx<NGHOST>(d, idx, field, m,a,b,k,j,i);
  }
  return correction;
}

template <int NGHOST, typename Velocity>
KOKKOS_INLINE_FUNCTION
void AddMeshUpwindAdvectionWithVelocity(const Z4c::Z4c_vars &q, const Velocity &beta,
                                       const Real idx[3],
                           int m, int k, int j, int i,
                           Z4cRHS<Real> &rhs, GaugeRHS<Real> &gauge) {
  static_assert(NGHOST >= 2 && NGHOST <= 4, "unsupported AthenaK advection order");
  rhs.chi += ScalarUpwindCorrection<NGHOST>(q.chi, beta, idx, m,k,j,i);
  rhs.trace += ScalarUpwindCorrection<NGHOST>(q.vKhat, beta, idx, m,k,j,i);
  rhs.theta += ScalarUpwindCorrection<NGHOST>(q.vTheta, beta, idx, m,k,j,i);
  gauge.alpha += ScalarUpwindCorrection<NGHOST>(q.alpha, beta, idx, m,k,j,i);
  for (int a = 0; a < 3; ++a) {
    rhs.lambda[a] += VectorUpwindCorrection<NGHOST>(q.vGam_u, beta, idx, a,m,k,j,i);
    gauge.beta[a] += VectorUpwindCorrection<NGHOST>(q.beta_u, beta, idx, a,m,k,j,i);
    for (int b = 0; b < 3; ++b) {
      rhs.metric[a][b] += TensorUpwindCorrection<NGHOST>(q.g_dd, beta, idx,
                                                       a,b,m,k,j,i);
      rhs.a[a][b] += TensorUpwindCorrection<NGHOST>(q.vA_dd, beta, idx,
                                                  a,b,m,k,j,i);
    }
  }
}

// Existing full-field interface; the explicit-velocity variant also permits
// differencing reference deviations while advecting with the full evolved beta.
template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void AddMeshUpwindAdvection(const Z4c::Z4c_vars &q, const Real idx[3],
                           int m, int k, int j, int i,
                           Z4cRHS<Real> &rhs, GaugeRHS<Real> &gauge) {
  AddMeshUpwindAdvectionWithVelocity<NGHOST>(q,q.beta_u,idx,m,k,j,i,rhs,gauge);
}

KOKKOS_INLINE_FUNCTION
void StoreMeshRHS(const Z4c::Z4c_vars &q, int m, int k, int j, int i,
                  const Z4cRHS<Real> &rhs, const GaugeRHS<Real> &gauge) {
  q.chi(m,k,j,i) = rhs.chi;
  q.vKhat(m,k,j,i) = rhs.trace;
  q.vTheta(m,k,j,i) = rhs.theta;
  q.alpha(m,k,j,i) = gauge.alpha;
  for (int a = 0; a < 3; ++a) {
    q.beta_u(m,a,k,j,i) = gauge.beta[a];
    q.vGam_u(m,a,k,j,i) = rhs.lambda[a];
    q.vB_d(m,a,k,j,i) = 0;
    for (int b = a; b < 3; ++b) {
      q.g_dd(m,a,b,k,j,i) = rhs.metric[a][b];
      q.vA_dd(m,a,b,k,j,i) = rhs.a[a][b];
    }
  }
}

// Physical ADM conversion is interior-only. In the present production task
// graph ADM lapse aliases Z4c lapse; do not write this physical lapse through
// that alias. A conformal runtime must first allocate separate ADM gauge storage.
struct PhysicalADMPoint {
  Real metric[3][3], curvature[3][3], alpha, beta[3], psi4;
  bool valid;
};

KOKKOS_INLINE_FUNCTION
PhysicalADMPoint ToPhysicalADM(const Z4cJet<Real> &u, Real omega) {
  PhysicalADMPoint out{};
  if (!(omega > 0) || !Kokkos::isfinite(omega) || !(u.chi.value > 0)
      || !Kokkos::isfinite(u.chi.value) || !(u.alpha.value > 0)) return out;
  const auto geo = Geometry(u.metric);
  if (!geo.valid) return out;
  out.psi4 = 1/(omega*omega*u.chi.value);
  out.alpha = u.alpha.value/omega;
  out.valid = out.psi4 > 0 && out.alpha > 0
      && Kokkos::isfinite(out.psi4) && Kokkos::isfinite(out.alpha);
  for (int a = 0; a < 3; ++a) {
    out.beta[a] = u.beta.value[a];
    out.valid = out.valid && Kokkos::isfinite(out.beta[a]);
    for (int b = 0; b < 3; ++b) {
      out.metric[a][b] = out.psi4*u.metric.g[a][b];
      out.curvature[a][b] = out.psi4*(omega*u.a.k[a][b]
          +u.metric.g[a][b]*(u.trace.value+2*u.theta.value)/3);
      out.valid = out.valid && (a != b || out.metric[a][b] > 0)
          && Kokkos::isfinite(out.metric[a][b])
          && Kokkos::isfinite(out.curvature[a][b]);
    }
  }
  return out;
}

}  // namespace hyperboloidal
}  // namespace z4c
#endif  // Z4C_HYPERBOLOIDAL_ATHENAK_BRIDGE_HPP_
