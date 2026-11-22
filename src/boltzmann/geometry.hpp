#ifndef BOLTZMANN_GEOMETRY_HPP_
#define BOLTZMANN_GEOMETRY_HPP_

#include "athena.hpp"
#include "coordinates/adm.hpp"

// The coordinate order is (rho,z,phi). The Cartoon slice stores Cartesian
// components in (radial, axial, suppressed) order, so d/dphi = rho e_y.
namespace boltzmann {
struct Frame {
  Real co[3][3]{}, inverse[3][3]{}, volume{};
  KOKKOS_INLINE_FUNCTION void Set(const Real g[3][3]) {
    // Reverse Cholesky g=C C^T. The last frame vector is the axial Killing
    // vector divided by its norm, including in a non-diagonal spatial metric.
    for (int a = 2; a >= 0; --a) {
      Real diag = g[a][a];
      for (int b = a + 1; b < 3; ++b) diag -= co[a][b] * co[a][b];
      if (!(diag > 0) || !Kokkos::isfinite(diag))
        Kokkos::abort("invalid Boltzmann spatial metric");
      co[a][a] = sqrt(diag);
      for (int i = 0; i < a; ++i) {
        Real off = g[i][a];
        for (int b = a + 1; b < 3; ++b) off -= co[i][b] * co[a][b];
        co[i][a] = off / co[a][a];
      }
    }
    for (int j = 0; j < 3; ++j) {
      for (int i = j; i >= 0; --i) {
        Real v = (i == j);
        for (int b = i + 1; b <= j; ++b) v -= co[i][b] * inverse[b][j];
        inverse[i][j] = v / co[i][i];
      }
    }
    volume = co[0][0] * co[1][1] * co[2][2];
  }
};

struct Geometry {
  Frame frame;
  Real alpha{}, beta[3]{}, grad_alpha[3]{}, grad_beta[3][3]{};
  Real grad_g[2][3][3]{}, curvature[3][3]{};
  Real connection[3][3][3]{};  // C^-1 (partial_r, partial_z, partial_t) C

  template <class AdmFields>
  KOKKOS_INLINE_FUNCTION void Set(const AdmFields& adm, int m, int j, int i, Real rho, Real dr,
                                  Real dz, bool fourth = false) {
    const Real scale[3] = {1, 1, rho};
    Real g[3][3], gt[3][3];
    alpha = adm.alpha(m, 0, j, i);
    for (int a = 0; a < 3; ++a) {
      beta[a] = adm.beta_u(m, a, 0, j, i) / scale[a];
      for (int b = 0; b < 3; ++b) {
        g[a][b] = adm.g_dd(m, a, b, 0, j, i) * scale[a] * scale[b];
        curvature[a][b] = adm.vK_dd(m, a, b, 0, j, i) * scale[a] * scale[b];
      }
    }
    frame.Set(g);
    for (int d = 0; d < 2; ++d) {
      const int di = (d == 0), dj = (d == 1);
      const Real invh = 0.5 / (di ? dr : dz);
      grad_alpha[d] =
          invh * (adm.alpha(m, 0, j + dj, i + di) - adm.alpha(m, 0, j - dj, i - di));
      if (fourth)
        grad_alpha[d] = (4 * grad_alpha[d] - 0.5 * invh *
                                                 (adm.alpha(m, 0, j + 2 * dj, i + 2 * di) -
                                                  adm.alpha(m, 0, j - 2 * dj, i - 2 * di))) /
                        3;
      for (int a = 0; a < 3; ++a) {
        grad_beta[d][a] =
            invh / scale[a] *
            (adm.beta_u(m, a, 0, j + dj, i + di) - adm.beta_u(m, a, 0, j - dj, i - di));
        if (fourth)
          grad_beta[d][a] =
              (4 * grad_beta[d][a] - 0.5 * invh / scale[a] *
                                         (adm.beta_u(m, a, 0, j + 2 * dj, i + 2 * di) -
                                          adm.beta_u(m, a, 0, j - 2 * dj, i - 2 * di))) /
              3;
        if (di && a == 2) grad_beta[d][a] -= beta[a] / rho;
        for (int b = 0; b < 3; ++b) {
          grad_g[d][a][b] =
              invh * scale[a] * scale[b] *
              (adm.g_dd(m, a, b, 0, j + dj, i + di) - adm.g_dd(m, a, b, 0, j - dj, i - di));
          if (fourth)
            grad_g[d][a][b] =
                (4 * grad_g[d][a][b] - 0.5 * invh * scale[a] * scale[b] *
                                           (adm.g_dd(m, a, b, 0, j + 2 * dj, i + 2 * di) -
                                            adm.g_dd(m, a, b, 0, j - 2 * dj, i - 2 * di))) /
                3;
          if (di) grad_g[d][a][b] += ((a == 2) + (b == 2)) * g[a][b] / rho;
        }
      }
    }
    // ADM kinematics determine partial_t gamma without lapse/shift time derivatives.
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) {
        gt[a][b] = -2 * alpha * curvature[a][b];
        for (int c = 0; c < 3; ++c) {
          if (c < 2) gt[a][b] += beta[c] * grad_g[c][a][b];
          gt[a][b] += g[a][c] * grad_beta[b][c] + g[b][c] * grad_beta[a][c];
        }
      }
    for (int d = 0; d < 3; ++d) {
      for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
          Real deriv = 0;
          for (int k = a; k < 3; ++k)
            for (int l = b; l < 3; ++l) {
              deriv += frame.inverse[a][k] * (d == 2 ? gt[k][l] : grad_g[d][k][l]) *
                       frame.inverse[b][l];
            }
          connection[d][a][b] = a == b ? 0.5 * deriv : deriv;
        }
    }
  }
};

struct Rates {
  Real speed[2], mu, psi, redshift;
};

KOKKOS_INLINE_FUNCTION Rates Characteristics(const Geometry& g, Real mu, Real psi) {
  const Real s = sqrt(fmax(0., 1 - mu * mu));
  const Real cs = cos(psi), sn = sin(psi), n[3] = {s * cs, s * sn, mu};
  Real up[3]{}, down[3]{};
  Rates r{};
  for (int a = 0; a < 3; ++a) {
    // The reverse Cholesky coframe and its inverse are upper triangular.
    for (int b = 0; b <= a; ++b) up[a] += g.frame.inverse[b][a] * n[b];
    for (int b = a; b < 3; ++b) down[a] += g.frame.co[a][b] * n[b];
  }
  for (int a = 0; a < 2; ++a) r.speed[a] = g.alpha * up[a] - g.beta[a];
  // q = d(log epsilon)/dt. It is also the source per unit densitized gray energy.
  for (int a = 0; a < 3; ++a) {
    r.redshift -= up[a] * g.grad_alpha[a];
    for (int b = 0; b < 3; ++b) r.redshift += g.alpha * g.curvature[a][b] * up[a] * up[b];
  }
  Real force[3]{}, dn[3]{};
  // Hamilton's equation for p_i/epsilon before differentiating the normalization.
  for (int d = 0; d < 2; ++d) {
    force[d] = -g.grad_alpha[d];
    for (int a = 0; a < 3; ++a) {
      force[d] += down[a] * g.grad_beta[d][a];
      for (int b = 0; b < 3; ++b) force[d] += 0.5 * g.alpha * up[a] * up[b] * g.grad_g[d][a][b];
    }
  }
  for (int a = 0; a < 3; ++a) {
    dn[a] = -r.redshift * n[a];
    // The frame connection has the same triangular structure.
    for (int b = a; b < 3; ++b) {
      dn[a] += g.frame.inverse[a][b] * force[b];
      dn[a] -= (g.connection[2][a][b] + r.speed[0] * g.connection[0][a][b] +
                r.speed[1] * g.connection[1][a][b]) *
               n[b];
    }
  }
  r.mu = dn[2];
  r.psi = s > 0 ? (cs * dn[1] - sn * dn[0]) / s : 0;
  return r;
}
}  // namespace boltzmann
#endif  // BOLTZMANN_GEOMETRY_HPP_
