// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <Kokkos_Core.hpp>

#include "z4c/hyperboloidal/cmc_reference.hpp"
#include "z4c/hyperboloidal/conformal_constraints.hpp"

namespace hyp = z4c::hyperboloidal;
using Metric = hyp::MetricJet<double>;
using Curvature = hyp::CurvatureJet<double>;
using Omega = hyp::OmegaJet<double>;

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
void Near(double got, double expected, double tol, const char *message) {
  Check(std::isfinite(got) && std::abs(got-expected) < tol, message);
}

// Non-diagonal, non-flat conformal spatial metric; nonzero shear and arbitrary
// normal derivative of Omega. These data intentionally violate constraints.
void Manufactured(const double x[3], Metric &m, Curvature &k, Omega &o) {
  const double base[3][3] = {{2, 0.2, -0.1}, {0.2, 1.5, 0.3}, {-0.1, 0.3, 1.2}};
  double psi = 1, dp[3]{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    psi += 0.1*x[i]*base[i][j]*x[j];
    dp[i] += 0.2*base[i][j]*x[j];
  }
  double r2 = 0;
  for (double v : {x[0], x[1], x[2]}) r2 += v*v;
  o.omega = (1-r2)/2;
  o.normal = 0.13+x[0]*x[1]-0.2*x[2];
  o.dnormal[0] = x[1];
  o.dnormal[1] = x[0];
  o.dnormal[2] = -0.2;
  for (int i = 0; i < 3; ++i) {
    o.gradient[i] = -x[i];
    o.hessian[i][i] = -1;
    for (int j = 0; j < 3; ++j) {
      const double delta = i == j ? 1 : 0;
      m.g[i][j] = std::pow(psi, 4)*base[i][j];
      k.k[i][j] = (-0.7+0.1*r2)*delta+0.03*x[i]*x[j];
      for (int d = 0; d < 3; ++d) {
        m.dg[d][i][j] = 4*std::pow(psi, 3)*dp[d]*base[i][j];
        k.dk[d][i][j] = 0.2*x[d]*delta
            +0.03*((i == d ? x[j] : 0)+(j == d ? x[i] : 0));
        for (int e = 0; e < 3; ++e) {
          m.ddg[d][e][i][j] = (12*psi*psi*dp[d]*dp[e]
              +0.8*std::pow(psi, 3)*base[d][e])*base[i][j];
        }
      }
    }
  }
  Near(hyp::Geometry(m).scalar, -4.8/std::pow(psi, 5), 1e-12,
       "nonflat Ricci scalar");
}

void PhysicalValues(const double x[3], Metric &p, Curvature &kp) {
  Metric m{};
  Curvature k{};
  Omega o{};
  Manufactured(x, m, k, o);
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    p.g[i][j] = m.g[i][j]/(o.omega*o.omega);
    kp.k[i][j] = k.k[i][j]/o.omega + m.g[i][j]*o.normal/(o.omega*o.omega);
  }
}

// Independent path: finite-difference the divergent physical ADM fields, then
// apply ordinary ADM constraints (Omega=1,w=0). Only used away from scri.
hyp::ConstraintResult<double> PhysicalFD(const double x[3], double h) {
  Metric m{};
  Curvature k{};
  PhysicalValues(x, m, k);
  for (int d = 0; d < 3; ++d) {
    double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
    xp[d] += h;
    xm[d] -= h;
    Metric plus{}, minus{};
    Curvature kp{}, km{};
    PhysicalValues(xp, plus, kp);
    PhysicalValues(xm, minus, km);
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      m.dg[d][i][j] = (plus.g[i][j]-minus.g[i][j])/(2*h);
      m.ddg[d][d][i][j] = (plus.g[i][j]-2*m.g[i][j]+minus.g[i][j])/(h*h);
      k.dk[d][i][j] = (kp.k[i][j]-km.k[i][j])/(2*h);
    }
    for (int e = d+1; e < 3; ++e) {
      for (int sd : {-1, 1})
      for (int se : {-1, 1}) {
        double xx[3] = {x[0], x[1], x[2]};
        xx[d] += sd*h;
        xx[e] += se*h;
        Metric corner{};
        Curvature unused{};
        PhysicalValues(xx, corner, unused);
        for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          m.ddg[d][e][i][j] += sd*se*corner.g[i][j]/(4*h*h);
        }
      }
      for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) m.ddg[e][d][i][j] = m.ddg[d][e][i][j];
    }
  }
  Omega unit{};
  unit.omega = 1;
  return hyp::Constraints(m, k, unit);
}

void OffShell() {
  for (const auto &point : {std::initializer_list<double>{0.2, -0.3, 0.1},
                            std::initializer_list<double>{-0.1, 0.2, 0.4}}) {
    double x[3];
    std::copy(point.begin(), point.end(), x);
    Metric m{};
    Curvature k{};
    Omega o{};
    Manufactured(x, m, k, o);
    const auto exact = hyp::Constraints(m, k, o);
    Check(exact.valid && std::abs(exact.hamiltonian) > 0.1,
          "off-shell H must be nonzero");
    Check(exact.momentum_conformal_norm2 > 0.01, "off-shell M must be nonzero");
    double last_h = 0, last_m = 0;
    for (double h : {0.02, 0.01, 0.005, 0.0025}) {
      const auto fd = PhysicalFD(x, h);
      double error_m = 0;
      const double error_h = std::abs(fd.hamiltonian-exact.hamiltonian);
      for (int i = 0; i < 3; ++i) {
        error_m = std::max(error_m, std::abs(fd.momentum[i]-exact.momentum[i]));
      }
      Check(fd.valid, "invalid physical FD diagnostics");
      if (last_h > 0) {
        Check(last_h/error_h > 3.8 && last_m/error_m > 3.8, "constraint convergence");
      }
      last_h = error_h;
      last_m = error_m;
      std::cout << "constraint h=" << h << " H_error=" << error_h
                << " M_error=" << error_m << '\n';
    }
    Check(last_h < 1e-3 && last_m < 1e-3, "physical/conformal constraints disagree");
  }
  // Non-regular data at scri must produce visible finite residuals, not be masked.
  const double x[3] = {1, 0, 0};
  Metric m{};
  Curvature k{};
  Omega o{};
  Manufactured(x, m, k, o);
  const auto bad = hyp::Constraints(m, k, o);
  Check(bad.valid && std::abs(bad.hamiltonian) > 0.1, "scri H masked");
  Check(bad.momentum_conformal_norm2 > 0.01, "scri M masked");
  Near(bad.momentum_physical_norm2, 0, 1e-15, "physical momentum weighting");
}

void CMC() {
  int failures = 0;
  Kokkos::parallel_reduce("conformal constraints CMC", 101,
      KOKKOS_LAMBDA(const int n, int &errors) {
    const double x = n/100.;
    const hyp::CMCReference<double> ref{1, 1};
    const auto p = ref.At(x/3, 2*x/3, 2*x/3);
    Metric m{};
    Curvature k{};
    Omega o{};
    double dbeta[3][3]{};
    o.omega = p.omega;
    for (int i = 0; i < 3; ++i) {
      m.g[i][i] = 1;
      k.k[i][i] = p.k_bar/3;
      o.gradient[i] = p.domega[i];
      o.hessian[i][i] = p.hessian_omega;
      dbeta[i][i] = -1;
      for (int d = 0; d < 3; ++d) k.dk[d][i][i] = p.dalpha[d]/(p.alpha*p.alpha);
    }
    hyp::SetStationaryOmegaNormal(p.alpha, p.beta, p.dalpha, dbeta, o);
    const auto out = hyp::Constraints(m, k, o);
    if (!out.valid || Kokkos::abs(out.hamiltonian) > 1e-12) ++errors;
    for (int i = 0; i < 3; ++i) {
      if (Kokkos::abs(out.momentum[i]) > 1e-12) ++errors;
    }
    if (Kokkos::abs(out.physical_trace+3) > 1e-12) ++errors;
    if (n == 100 && Kokkos::abs(out.null_residual) > 1e-12) ++errors;
  }, failures);
  Check(failures == 0, "CMC constraint identities");
}

void Z4() {
  Metric m{};
  m.g[0][0] = 2;
  m.g[1][1] = 1;
  m.g[2][2] = 0.5;
  double a[3][3]{};
  const double lambda[3] = {0.1, -0.4, 1.2};
  for (int i = 0; i < 3; ++i) a[i][i] = 0.5*m.g[i][i];
  const auto z = hyp::Z4Constraints(m, 0.7, a, lambda, 0.2, 0.);
  Check(z.valid, "Z4 diagnostics valid");
  Near(z.determinant_residual, 0, 1e-15, "determinant constraint");
  Near(z.tracefree_residual, 1.5, 1e-15, "tracefree constraint");
  Near(z.theta_physical, 0.2, 1e-15, "Theta diagnostic");
  Near(z.z_covector[0], 0.1, 1e-15, "Z_x diagnostic");
  Near(z.z_covector[1], -0.2, 1e-15, "Z_y diagnostic");
  Near(z.z_covector[2], 0.3, 1e-15, "Z_z diagnostic");
  Near(z.z_conformal_norm2, 0.7*(0.005+0.04+0.18), 1e-15, "Z norm");
  Near(z.z_physical_norm2, 0, 1e-15, "physical Z weighting");
  m.g[1][1] = 2;
  const auto det_violation = hyp::Z4Constraints(m, 0.7, a, lambda, 0.2, 0.);
  Near(det_violation.determinant_residual, 1, 1e-15, "nonunit determinant hidden");
  m.g[0][0] = -1;
  Check(!hyp::Geometry(m).valid, "nonpositive metric accepted");
}

void Schwarzschild() {
  // Time-symmetric isotropic puncture data, away from r=0. This is a diagnostic
  // test, not hyperboloidal initial data or a puncture evolution.
  for (double radius : {0.1, 0.25, 0.5, 1., 2., 10.}) {
    const double x[3] = {radius/std::sqrt(3.), -radius/std::sqrt(3.),
                         radius/std::sqrt(3.)};
    const double psi = 1+0.5/radius;
    Metric m{};
    Curvature k{};
    Omega o{};
    o.omega = 1;
    for (int i = 0; i < 3; ++i) {
      m.g[i][i] = std::pow(psi, 4);
      for (int d = 0; d < 3; ++d) {
        const double dp = -0.5*x[d]/std::pow(radius, 3);
        m.dg[d][i][i] = 4*std::pow(psi, 3)*dp;
        for (int e = 0; e < 3; ++e) {
          const double ep = -0.5*x[e]/std::pow(radius, 3);
          const double ddp = 0.5*(3*x[d]*x[e]/std::pow(radius, 5)
                                  -(d == e ? 1/std::pow(radius, 3) : 0));
          m.ddg[d][e][i][i] = 12*psi*psi*dp*ep+4*std::pow(psi, 3)*ddp;
        }
      }
    }
    const auto result = hyp::Constraints(m, k, o);
    Check(result.valid, "Schwarzschild diagnostics invalid");
    Near(result.hamiltonian, 0, 1e-12, "Schwarzschild vacuum H");
    Near(result.momentum_conformal_norm2, 0, 1e-15, "Schwarzschild vacuum M");
  }
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    CMC();
    OffShell();
    Z4();
    Schwarzschild();
    std::cout << "PASS conformal physical and Z4 constraint diagnostics\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
