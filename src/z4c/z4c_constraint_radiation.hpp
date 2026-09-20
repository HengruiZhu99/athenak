#ifndef Z4C_Z4C_CONSTRAINT_RADIATION_HPP_
#define Z4C_Z4C_CONSTRAINT_RADIATION_HPP_

// Experimental damped physical-constraint radiation residual for differential CPBC.
// This local first-order absorbing approximation includes the normal Theta-Q
// coupling from the frozen damped constraint waves. It is not an exact oblique
// Dirichlet-to-Neumann map or a stability guarantee for the complete gauge system.
// This header computes data only: all input views are read-only, and no ghost
// value of the RHS is used.  The boundary differential operator is second order
// even when the interior scheme is sixth order. Its metric-defined Gamma and
// Gamma_t use the same fourth-order active-cell derivative, including at block
// edges. This is essential: nesting two second-order one-sided/centered
// derivatives would be only first order where their truncation errors change.

#include <cmath>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "z4c/z4c.hpp"

namespace z4c {

KOKKOS_INLINE_FUNCTION
int RadiationSymmetricOffset(int a, int b) {
  if (a > b) { const int t = a; a = b; b = t; }
  if (a == 0) return b;
  return a == 1 ? b + 2 : 5;
}

KOKKOS_INLINE_FUNCTION
void RadiationStencil(int dir, int k, int j, int i,
                      const RegionIndcs &indcs, const Real idx[3],
                      int offset[3], Real weight[3]) {
  const int pos[3] = {i,j,k};
  const int low[3] = {indcs.is,indcs.js,indcs.ks};
  const int high[3] = {indcs.ie,indcs.je,indcs.ke};
  if (pos[dir] == low[dir]) {
    offset[0] = 0; offset[1] = 1; offset[2] = 2;
    weight[0] = -1.5; weight[1] = 2.0; weight[2] = -0.5;
  } else if (pos[dir] == high[dir]) {
    offset[0] = 0; offset[1] = -1; offset[2] = -2;
    weight[0] = 1.5; weight[1] = -2.0; weight[2] = 0.5;
  } else {
    offset[0] = -1; offset[1] = 0; offset[2] = 1;
    weight[0] = -0.5; weight[1] = 0.0; weight[2] = 0.5;
  }
  for (int n = 0; n < 3; ++n) weight[n] *= idx[dir];
}

KOKKOS_INLINE_FUNCTION
Real RadiationRead(const DvceArray5D<Real> &v, int m, int n,
                   int k, int j, int i, int dir, int offset) {
  return v(m,n,k+(dir == 2 ? offset : 0),j+(dir == 1 ? offset : 0),
           i+(dir == 0 ? offset : 0));
}

KOKKOS_INLINE_FUNCTION
Real RadiationDerivative(const DvceArray5D<Real> &v, int m, int n,
                         int k, int j, int i, int dir,
                         const RegionIndcs &indcs, const Real idx[3]) {
  int offset[3]; Real weight[3];
  RadiationStencil(dir,k,j,i,indcs,idx,offset,weight);
  Real value = 0.0;
  for (int s = 0; s < 3; ++s) {
    value += weight[s]*RadiationRead(v,m,n,k,j,i,dir,offset[s]);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
Real RadiationMetricDerivative(const DvceArray5D<Real> &v, int m, int n,
                               int k, int j, int i, int dir,
                               const RegionIndcs &indcs, const Real idx[3]) {
  const int pos[3] = {i,j,k};
  const int low[3] = {indcs.is,indcs.js,indcs.ks};
  const int high[3] = {indcs.ie,indcs.je,indcs.ke};
  int first = pos[dir]-2;
  if (first < low[dir]) first = low[dir];
  if (first > high[dir]-4) first = high[dir]-4;
  const int row = pos[dir]-first;
  // Five-point polynomial differentiation, evaluated at row=0,...,4.
  const Real coefficients[5][5] = {
    {-25.0/12.0,4.0,-3.0,4.0/3.0,-0.25},
    {-0.25,-5.0/6.0,1.5,-0.5,1.0/12.0},
    {1.0/12.0,-2.0/3.0,0.0,2.0/3.0,-1.0/12.0},
    {-1.0/12.0,0.5,-1.5,5.0/6.0,0.25},
    {0.25,-4.0/3.0,3.0,-4.0,25.0/12.0}};
  Real value = 0.0;
  for (int s = 0; s < 5; ++s) {
    value += coefficients[row][s]*RadiationRead(v,m,n,k,j,i,dir,first+s-pos[dir]);
  }
  return value*idx[dir];
}

KOKKOS_INLINE_FUNCTION
bool RadiationGeometry(const DvceArray5D<Real> &v, int m, int k, int j, int i,
                       const RegionIndcs &indcs, const Real idx[3],
                       Real g[3][3], Real gi[3][3], Real dg[3][3][3],
                       Real connection[3]) {
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      const int n = Z4c::I_Z4C_GXX+RadiationSymmetricOffset(a,b);
      g[a][b] = g[b][a] = v(m,n,k,j,i);
      for (int d = 0; d < 3; ++d) {
        dg[d][a][b] = dg[d][b][a] =
            RadiationMetricDerivative(v,m,n,k,j,i,d,indcs,idx);
        if (!isfinite(dg[d][a][b])) return false;
      }
    }
  }
  const Real minor = g[0][0]*g[1][1]-g[0][1]*g[0][1];
  const Real det = g[0][0]*g[1][1]*g[2][2]+2.0*g[0][1]*g[0][2]*g[1][2]
                   -g[0][0]*g[1][2]*g[1][2]-g[1][1]*g[0][2]*g[0][2]
                   -g[2][2]*g[0][1]*g[0][1];
  if (!(isfinite(g[0][0]) && isfinite(minor) && isfinite(det)) ||
      g[0][0] <= 0.0 || minor <= 0.0 || det <= 0.0) return false;
  gi[0][0] = (g[1][1]*g[2][2]-g[1][2]*g[1][2])/det;
  gi[0][1] = gi[1][0] = (g[0][2]*g[1][2]-g[0][1]*g[2][2])/det;
  gi[0][2] = gi[2][0] = (g[0][1]*g[1][2]-g[0][2]*g[1][1])/det;
  gi[1][1] = (g[0][0]*g[2][2]-g[0][2]*g[0][2])/det;
  gi[1][2] = gi[2][1] = (g[0][1]*g[0][2]-g[0][0]*g[1][2])/det;
  gi[2][2] = minor/det;
  for (int a = 0; a < 3; ++a) {
    connection[a] = 0.0;
    for (int l = 0; l < 3; ++l) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          connection[a] += gi[a][l]*gi[b][c]*
              (dg[b][c][l]-0.5*dg[l][b][c]);
        }
      }
    }
    if (!isfinite(connection[a])) return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool RadiationZ(const DvceArray5D<Real> &v, int m, int k, int j, int i,
                const RegionIndcs &indcs, const Real idx[3], Real z[3]) {
  Real g[3][3], gi[3][3], dg[3][3][3], connection[3];
  if (!RadiationGeometry(v,m,k,j,i,indcs,idx,g,gi,dg,connection)) return false;
  for (int a = 0; a < 3; ++a) {
    z[a] = 0.0;
    for (int b = 0; b < 3; ++b) {
      z[a] += 0.5*g[a][b]*(v(m,Z4c::I_Z4C_GAMX+b,k,j,i)-connection[b]);
    }
    if (!isfinite(z[a])) return false;
  }
  return true;
}

KOKKOS_INLINE_FUNCTION
bool RadiationResidualZ(const DvceArray5D<Real> &full,
                        const DvceArray5D<Real> &bg,
                        int m, int k, int j, int i,
                        const RegionIndcs &indcs, const Real idx[3], Real z[3]) {
  Real zfull[3], zbg[3];
  if (!RadiationZ(full,m,k,j,i,indcs,idx,zfull) ||
      !RadiationZ(bg,m,k,j,i,indcs,idx,zbg)) return false;
  for (int a = 0; a < 3; ++a) z[a] = zfull[a]-zbg[a];
  return true;
}

// Return zero on success.  Caller maps nonzero status to its fail-fast CPBC
// diagnostic. The frame vectors are conformal-unit vectors from MakeBoundaryFrame.
// side is retained for the API, but each nested derivative selects its stencil
// from its own indices, never from the parent point's physical-face mask.
KOKKOS_INLINE_FUNCTION
int ComputeConstraintRadiationResidual(
    const DvceArray5D<Real> &full, const DvceArray5D<Real> &bg,
    const DvceArray5D<Real> &rhs, int m, int k, int j, int i,
    const RegionIndcs &indcs, const int side[3], const Real idx[3],
    const Real normal_d[3], const Real normal_u[3], const Real xyz[3],
    const Z4c::Options &opt, Real &f_theta, Real f_q[3]) {
  (void)side;
  (void)normal_d;
  if (indcs.nx1 < 5 || indcs.nx2 < 5 || indcs.nx3 < 5) return 1;
  if (!isfinite(opt.characteristic_radiation_areal_shift) ||
      opt.characteristic_radiation_areal_shift < 0.0) return 2;
  const Real r = sqrt(SQR(xyz[0])+SQR(xyz[1])+SQR(xyz[2]));
  const Real R = r+opt.characteristic_radiation_areal_shift;
  const Real alpha = full(m,Z4c::I_Z4C_ALPHA,k,j,i);
  const Real chi = full(m,Z4c::I_Z4C_CHI,k,j,i);
  const Real chi_bg = bg(m,Z4c::I_Z4C_CHI,k,j,i);
  if (!(isfinite(r) && isfinite(R) && isfinite(alpha) && isfinite(chi) &&
        isfinite(chi_bg)) ||
      (opt.characteristic_radiation_areal_falloff && (r <= 0.0 || R <= 0.0)) ||
      alpha <= 0.0 ||
      chi <= 0.0 || chi_bg <= 0.0) return 2;

  Real g[3][3], gi[3][3], dg[3][3][3], gamma_metric[3];
  Real gb[3][3], gib[3][3], dgb[3][3][3], gamma_bg[3];
  if (!RadiationGeometry(full,m,k,j,i,indcs,idx,g,gi,dg,gamma_metric) ||
      !RadiationGeometry(bg,m,k,j,i,indcs,idx,gb,gib,dgb,gamma_bg)) return 3;

  Real rg[3][3], drg[3][3][3], rgi[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = a; b < 3; ++b) {
      const int n = Z4c::I_Z4C_GXX+RadiationSymmetricOffset(a,b);
      rg[a][b] = rg[b][a] = rhs(m,n,k,j,i);
      for (int d = 0; d < 3; ++d) {
        drg[d][a][b] = drg[d][b][a] =
            RadiationMetricDerivative(rhs,m,n,k,j,i,d,indcs,idx);
      }
    }
  }
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) rgi[a][b] -= gi[a][c]*rg[c][d]*gi[d][b];
      }
    }
  }
  Real gamma_metric_t[3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int l = 0; l < 3; ++l) {
      for (int b = 0; b < 3; ++b) {
        for (int c = 0; c < 3; ++c) {
          gamma_metric_t[a] +=
              (rgi[a][l]*gi[b][c]+gi[a][l]*rgi[b][c])*
                  (dg[b][c][l]-0.5*dg[l][b][c]) +
              gi[a][l]*gi[b][c]*(drg[b][c][l]-0.5*drg[l][b][c]);
        }
      }
    }
  }

  Real z[3], z_t[3] = {}, dz[3][3] = {};
  if (!RadiationResidualZ(full,bg,m,k,j,i,indcs,idx,z)) return 3;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      z_t[a] += 0.5*(rg[a][b]*(full(m,Z4c::I_Z4C_GAMX+b,k,j,i)-gamma_metric[b]) +
                         g[a][b]*(rhs(m,Z4c::I_Z4C_GAMX+b,k,j,i)-gamma_metric_t[b]));
    }
  }

  // Derivatives of physical reference metric gamma_bar=gt_bar/chi_bar.
  Real dphysical_bg[3][3][3] = {};
  Real dtheta[3] = {};
  for (int d = 0; d < 3; ++d) {
    int offset[3]; Real weight[3];
    RadiationStencil(d,k,j,i,indcs,idx,offset,weight);
    for (int s = 0; s < 3; ++s) {
      const int ii = i+(d == 0 ? offset[s] : 0);
      const int jj = j+(d == 1 ? offset[s] : 0);
      const int kk = k+(d == 2 ? offset[s] : 0);
      Real zs[3];
      if (!RadiationResidualZ(full,bg,m,kk,jj,ii,indcs,idx,zs)) return 3;
      dtheta[d] += weight[s]*(full(m,Z4c::I_Z4C_THETA,kk,jj,ii)-
                             bg(m,Z4c::I_Z4C_THETA,kk,jj,ii));
      const Real chis = bg(m,Z4c::I_Z4C_CHI,kk,jj,ii);
      if (!(isfinite(chis)) || chis <= 0.0) return 3;
      for (int a = 0; a < 3; ++a) {
        dz[d][a] += weight[s]*zs[a];
        for (int b = 0; b < 3; ++b) {
          const int n = Z4c::I_Z4C_GXX+RadiationSymmetricOffset(a,b);
          dphysical_bg[d][a][b] += weight[s]*bg(m,n,kk,jj,ii)/chis;
        }
      }
    }
  }

  Real transport[3], omega = 0.0;
  for (int a = 0; a < 3; ++a) {
    transport[a] = -full(m,Z4c::I_Z4C_BETAX+a,k,j,i)+alpha*sqrt(chi)*normal_u[a];
    if (opt.characteristic_radiation_areal_falloff) {
      omega += transport[a]*xyz[a]/(r*R);
    }
  }
  const Real sigma = alpha*opt.damp_kappa1;
  const Real theta = full(m,Z4c::I_Z4C_THETA,k,j,i)-bg(m,Z4c::I_Z4C_THETA,k,j,i);
  f_theta = rhs(m,Z4c::I_Z4C_THETA,k,j,i)+(omega+sigma)*theta;
  for (int a = 0; a < 3; ++a) {
    f_theta += transport[a]*dtheta[a];
    // Q^i = 2 gt^{ij} Z_j, so Q_n/2 = n^i Z_i.
    // The forced Theta wave requires -sigma*sqrt(chi)*Q_n/2.
    f_theta -= sigma*sqrt(chi)*normal_u[a]*z[a];
  }

  Real f_z[3];
  for (int a = 0; a < 3; ++a) {
    f_z[a] = z_t[a]+(omega+sigma)*z[a];
    for (int d = 0; d < 3; ++d) {
      Real derivative = dz[d][a];
      for (int b = 0; b < 3; ++b) {
        Real connection = 0.0;
        for (int l = 0; l < 3; ++l) {
          connection += 0.5*chi_bg*gib[b][l]*
              (dphysical_bg[d][a][l]+dphysical_bg[a][d][l]-dphysical_bg[l][d][a]);
        }
        derivative -= connection*z[b];
      }
      f_z[a] += transport[d]*derivative;
    }
  }
  for (int a = 0; a < 3; ++a) {
    f_q[a] = 0.0;
    for (int b = 0; b < 3; ++b) f_q[a] += 2.0*gi[a][b]*f_z[b];
    if (!isfinite(f_q[a])) return 4;
  }
  return isfinite(f_theta) ? 0 : 4;
}

}  // namespace z4c
#endif  // Z4C_Z4C_CONSTRAINT_RADIATION_HPP_
