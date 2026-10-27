// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MULTIGRID_FLEXIBLE_GMRES_HPP_
#define MULTIGRID_FLEXIBLE_GMRES_HPP_

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

namespace multigrid {
struct KrylovReport {
  bool converged = false;
  bool inner_norm_converged=false, external_acceptance=false;
  int iterations = 0, applications = 0;
  std::vector<double> residual;
};
using Vector = std::vector<double>;
using Operator = std::function<void(const Vector &, Vector &)>;
using InnerProduct = std::function<double(const Vector &, const Vector &)>;
using ResidualAcceptance=std::function<bool(const Vector &,const Vector &)>;

// Right-preconditioned flexible GMRES. An unpreconditioned residual of the
// ORIGINAL equation is recomputed at every restart and before declaring success.
inline KrylovReport FlexibleGMRES(const Operator &apply, const Operator &precondition,
    const InnerProduct &dot, const Vector &rhs, Vector &x, double atol,
    double rtol, int max_iterations = 160, int restart = 24,
    const ResidualAcceptance &accept={}) {
  if (rhs.empty() || x.size() != rhs.size() || atol < 0 || rtol < 0
      || !std::isfinite(atol+rtol) || max_iterations < 1 || restart < 1)
    throw std::invalid_argument("invalid FGMRES controls");
  KrylovReport report;
  const double target = std::max(atol, rtol*std::sqrt(dot(rhs, rhs)));
  Vector ax, residual(rhs.size());
  while (report.iterations < max_iterations) {
    apply(x, ax); ++report.applications;
    for (size_t p = 0; p < rhs.size(); ++p) residual[p] = rhs[p]-ax[p];
    const double beta = std::sqrt(dot(residual, residual));
    if (!std::isfinite(beta)) throw std::runtime_error("nonfinite original vector residual");
    report.residual.push_back(beta);
    report.inner_norm_converged=beta<=target;
    report.external_acceptance=accept&&accept(x,residual);
    if (accept?report.external_acceptance:report.inner_norm_converged) {report.converged = true; return report;}
    if(beta==0)return report; // affine arithmetic can be exact while the independently checked physical equation is not
    const int width = std::min(restart, max_iterations-report.iterations);
    std::vector<Vector> v(width+1, Vector(rhs.size())), z(width, Vector(rhs.size()));
    std::vector<Vector> H(width+1, Vector(width, 0));
    Vector c(width), s(width), g(width+1, 0);
    g[0] = beta;
    for (size_t p = 0; p < rhs.size(); ++p) v[0][p] = residual[p]/beta;
    int columns = 0;
    for (int j = 0; j < width; ++j) {
      precondition(v[j], z[j]);
      Vector w;
      apply(z[j], w); ++report.applications;
      // Two-pass modified Gram-Schmidt protects small original residuals.
      for (int pass = 0; pass < 2; ++pass)
      for (int i = 0; i <= j; ++i) {
        const double projection = dot(w, v[i]);
        H[i][j] += projection;
        for (size_t p = 0; p < rhs.size(); ++p) w[p] -= projection*v[i][p];
      }
      H[j+1][j] = std::sqrt(dot(w, w));
      const bool breakdown = H[j+1][j] <= 1e-28;
      if (!breakdown) for (size_t p = 0; p < rhs.size(); ++p)
        v[j+1][p] = w[p]/H[j+1][j];
      for (int i = 0; i < j; ++i) {
        const double a = H[i][j], b = H[i+1][j];
        H[i][j] = c[i]*a+s[i]*b; H[i+1][j] = -s[i]*a+c[i]*b;
      }
      const double norm = std::hypot(H[j][j], H[j+1][j]);
      if (!(norm > 0)) throw std::runtime_error("FGMRES singular Arnoldi column");
      c[j] = H[j][j]/norm; s[j] = H[j+1][j]/norm;
      H[j][j] = norm; H[j+1][j] = 0;
      g[j+1] = -s[j]*g[j]; g[j] *= c[j];
      ++report.iterations; columns = j+1;
      if (std::abs(g[j+1]) <= target || breakdown) break;
    }
    Vector y(columns);
    for (int i = columns; i-- > 0;) {
      y[i] = g[i];
      for (int j = i+1; j < columns; ++j) y[i] -= H[i][j]*y[j];
      y[i] /= H[i][i];
    }
    for (int i = 0; i < columns; ++i)
      for (size_t p = 0; p < rhs.size(); ++p) x[p] += y[i]*z[i][p];
  }
  apply(x, ax); ++report.applications;
  for (size_t p = 0; p < rhs.size(); ++p) residual[p] = rhs[p]-ax[p];
  report.residual.push_back(std::sqrt(dot(residual, residual)));
  report.inner_norm_converged=std::isfinite(report.residual.back())&&report.residual.back()<=target;
  report.external_acceptance=accept&&accept(x,residual);
  report.converged=accept?report.external_acceptance:report.inner_norm_converged;
  return report;
}
}  // namespace multigrid
#endif  // MULTIGRID_FLEXIBLE_GMRES_HPP_
