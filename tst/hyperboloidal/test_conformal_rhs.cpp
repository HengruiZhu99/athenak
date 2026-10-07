// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <Kokkos_Core.hpp>

#include "z4c/hyperboloidal/cmc_reference.hpp"
#include "z4c/hyperboloidal/conformal_rhs.hpp"

namespace hyp = z4c::hyperboloidal;
using Jet = hyp::Z4cJet<double>;
using RHS = hyp::Z4cRHS<double>;

void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}

double Norm(const RHS &u) {
  double value = std::max({std::abs(u.chi), std::abs(u.trace), std::abs(u.theta)});
  for (int i = 0; i < 3; ++i) {
    value = std::max(value, std::abs(u.lambda[i]));
    for (int j = 0; j < 3; ++j) {
      value = std::max({value, std::abs(u.metric[i][j]), std::abs(u.a[i][j])});
    }
  }
  return value;
}

hyp::OmegaJet<double> Compactifier(const double x[3], const Jet &u) {
  const auto p = hyp::CMCReference<double>{1, 1}.At(x[0], x[1], x[2]);
  hyp::OmegaJet<double> o{};
  o.omega = p.omega;
  for (int i = 0; i < 3; ++i) {
    o.gradient[i] = p.domega[i];
    o.hessian[i][i] = p.hessian_omega;
  }
  hyp::SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
  return o;
}

void Minkowski() {
  int failures = 0;
  Kokkos::parallel_reduce("CMC conformal RHS", 100,
      KOKKOS_LAMBDA(const int n, int &bad) {
    const double x = n/100.;
    const auto p = hyp::CMCReference<double>{1, 1}.At(x/3, 2*x/3, 2*x/3);
    Jet u{};
    hyp::OmegaJet<double> o{};
    u.chi.value = 1;
    u.alpha.value = p.alpha;
    u.trace.value = -3;
    o.omega = p.omega;
    for (int i = 0; i < 3; ++i) {
      u.metric.g[i][i] = 1;
      u.alpha.d[i] = p.dalpha[i];
      u.alpha.dd[i][i] = p.hessian_alpha;
      u.beta.value[i] = p.beta[i];
      u.beta.d[i][i] = -1;
      o.gradient[i] = p.domega[i];
      o.hessian[i][i] = p.hessian_omega;
    }
    hyp::SetStationaryOmegaNormal(p.alpha, u.beta.value, u.alpha.d, u.beta.d, o);
    const auto parts = hyp::ConformalRHS(u, o, 1.5, 0.);
    RHS rhs{};
    if (!hyp::AssembleInterior(parts, o.omega, rhs)) ++bad;
    if (Kokkos::abs(rhs.chi)+Kokkos::abs(rhs.trace)+Kokkos::abs(rhs.theta) > 1e-11) ++bad;
    for (int i = 0; i < 3; ++i) {
      if (Kokkos::abs(rhs.lambda[i]) > 1e-11) ++bad;
      for (int j = 0; j < 3; ++j) {
        if (Kokkos::abs(rhs.metric[i][j])+Kokkos::abs(rhs.a[i][j]) > 1e-11) ++bad;
      }
    }
  }, failures);
  Check(failures == 0, "CMC RHS is not stationary");
  Jet u{};
  u.chi.value = 1;
  u.alpha.value = 1;
  u.trace.value = -3;
  for (int i = 0; i < 3; ++i) u.metric.g[i][i] = 1;
  const double x[3] = {1, 0, 0};
  const auto o = Compactifier(x, u);
  const auto parts = hyp::ConformalRHS(u, o, 1.5, 0.);
  RHS rhs{};
  Check(parts.valid, "finite scri numerators rejected");
  Check(!hyp::AssembleInterior(parts, 0., rhs), "scri division accepted");
  Check(!hyp::AssembleInterior(parts, -0.01, rhs), "exterior division accepted");
}

// Exact stationary Schwarzschild CMC exterior, C=0, K_phys=-3, M=0.05.
// R=r/Omega; radial Penrose metric f=alpha_ref^2/alpha^2. This is distinct
// from Minkowski and from time-symmetric puncture data. Only points outside
// its throat are used: this test does not claim a puncture evolution.
Jet SchwarzschildValues(const double x[3]) {
  Jet u{};
  const double r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  const double o = (1-r*r)/2, alpha0 = (1+r*r)/2, mass = 0.05;
  const double alpha2 = alpha0*alpha0-2*mass*o*o*o/r;
  Check(alpha2 > 0, "Schwarzschild sample inside throat");
  const double dalpha2 = 2*alpha0*r+2*mass*(3*o*o+o*o*o/(r*r));
  const double f = alpha0*alpha0/alpha2;
  const double grr = std::pow(f, 2./3.);
  const double chi = std::pow(f, -1./3.);
  const double dgrr = (2./3.)*grr*(2*r/alpha0-dalpha2/alpha2);
  const double lambda = dgrr/(grr*grr)+2/r*(1/chi-1/grr);
  u.chi.value = chi;
  u.alpha.value = std::sqrt(alpha2);
  u.trace.value = -3;
  for (int i = 0; i < 3; ++i) {
    u.beta.value[i] = -x[i]*u.alpha.value/alpha0;
    u.lambda.value[i] = lambda*x[i]/r;
    for (int j = 0; j < 3; ++j) {
      u.metric.g[i][j] = (i == j ? chi : 0)+(grr-chi)*x[i]*x[j]/(r*r);
    }
  }
  return u;
}

void AddFirst(Jet &u, const Jet &p, const Jet &m, int d, double factor) {
  u.chi.d[d] += factor*(p.chi.value-m.chi.value);
  u.alpha.d[d] += factor*(p.alpha.value-m.alpha.value);
  u.trace.d[d] += factor*(p.trace.value-m.trace.value);
  u.theta.d[d] += factor*(p.theta.value-m.theta.value);
  for (int i = 0; i < 3; ++i) {
    u.beta.d[d][i] += factor*(p.beta.value[i]-m.beta.value[i]);
    u.lambda.d[d][i] += factor*(p.lambda.value[i]-m.lambda.value[i]);
    for (int j = 0; j < 3; ++j) {
      u.metric.dg[d][i][j] += factor*(p.metric.g[i][j]-m.metric.g[i][j]);
      u.a.dk[d][i][j] += factor*(p.a.k[i][j]-m.a.k[i][j]);
    }
  }
}
void AddSecond(Jet &u, const Jet &v, int d, int e, double f) {
  u.chi.dd[d][e] += f*v.chi.value;
  u.alpha.dd[d][e] += f*v.alpha.value;
  for (int i = 0; i < 3; ++i) {
    u.beta.dd[d][e][i] += f*v.beta.value[i];
    for (int j = 0; j < 3; ++j) u.metric.ddg[d][e][i][j] += f*v.metric.g[i][j];
  }
}

Jet DifferenceSchwarzschild(const double x[3], double h) {
  Jet u = SchwarzschildValues(x);
  const Jet center = u;
  for (int d = 0; d < 3; ++d) {
    double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
    xp[d] += h;
    xm[d] -= h;
    const auto p = SchwarzschildValues(xp), m = SchwarzschildValues(xm);
    AddFirst(u, p, m, d, 1/(2*h));
    AddSecond(u, p, d, d, 1/(h*h));
    AddSecond(u, center, d, d, -2/(h*h));
    AddSecond(u, m, d, d, 1/(h*h));
    for (int e = 0; e < 3; ++e) {
      if (e == d) continue;
      for (int sd : {-1, 1})
      for (int se : {-1, 1}) {
        double xx[3] = {x[0], x[1], x[2]};
        xx[d] += sd*h;
        xx[e] += se*h;
        AddSecond(u, SchwarzschildValues(xx), d, e, sd*se/(4*h*h));
      }
    }
  }
  return u;
}

void Schwarzschild() {
  for (double radius : {0.3, 0.65, 0.85}) {
    const double x[3] = {radius/3, 2*radius/3, 2*radius/3};
    double last = 0;
    for (double h : {0.004, 0.002, 0.001, 0.0005}) {
      const auto u = DifferenceSchwarzschild(x, h);
      const auto o = Compactifier(x, u);
      const auto parts = hyp::ConformalRHS(u, o, 1.5, 0.);
      RHS rhs{};
      Check(hyp::AssembleInterior(parts, o.omega, rhs), "Schwarzschild invalid RHS");
      const double error = Norm(rhs);
      std::cout << "stationary Schwarzschild r=" << radius << " h=" << h
                << " max_rhs=" << error << '\n';
      if (last > 0) Check(last/error > 3.7, "Schwarzschild RHS convergence");
      last = error;
    }
    Check(last < 0.001, "Schwarzschild stationary RHS inaccurate");
  }
}

void ADMRecovery() {
  // Omega=1, Z=Theta=0: reconstructed ADM metric and curvature evolution must
  // agree with vacuum ADM, even when the initial Hamiltonian is nonzero.
  Jet u{};
  u.chi.value = 0.8;
  u.alpha.value = 1.2;
  u.trace.value = 0.4;
  for (int i = 0; i < 3; ++i) {
    u.metric.g[i][i] = 1;
    u.chi.d[i] = 0.03*(i+1);
    u.alpha.d[i] = -0.02*(i+1);
    u.chi.dd[i][i] = 0.1*(i+1);
    u.alpha.dd[i][i] = 0.05*(i+1);
    u.a.k[i][i] = 0.02*(i-1);
  }
  u.a.k[0][1] = u.a.k[1][0] = 0.015;
  hyp::OmegaJet<double> o{};
  o.omega = 1;
  RHS rhs{};
  Check(hyp::AssembleInterior(hyp::ConformalRHS(u, o, 0., 0.), 1., rhs), "ADM RHS");
  const auto b = hyp::PenroseMetric(u.metric, u.chi);
  const auto geo = hyp::Geometry(b);
  double k[3][3]{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    k[i][j] = u.a.k[i][j]/u.chi.value + b.g[i][j]*u.trace.value/3;
  }
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    const double db = rhs.metric[i][j]/u.chi.value
        -u.metric.g[i][j]*rhs.chi/(u.chi.value*u.chi.value);
    Check(std::abs(db+2*u.alpha.value*k[i][j]) < 1e-13, "ADM metric equation");
    const double dk = rhs.a[i][j]/u.chi.value
        -u.a.k[i][j]*rhs.chi/(u.chi.value*u.chi.value)
        +db*u.trace.value/3+b.g[i][j]*(rhs.trace+2*rhs.theta)/3;
    double hessalpha = u.alpha.dd[i][j], kk = 0;
    for (int d = 0; d < 3; ++d) {
      hessalpha -= geo.connection[d][i][j]*u.alpha.d[d];
      for (int e = 0; e < 3; ++e) kk += geo.inverse[d][e]*k[i][d]*k[e][j];
    }
    const double expected = -hessalpha+u.alpha.value
        *(geo.ricci[i][j]+u.trace.value*k[i][j]-2*kk);
    Check(std::abs(dk-expected) < 1e-12, "ADM curvature equation");
  }
  Check(std::abs(rhs.theta) > 0.01, "off-shell Theta must respond to H");
}

void OffConstraintDamping() {
  Jet u{};
  u.chi.value = 0.7;
  u.alpha.value = 1.1;
  u.trace.value = -2.3;
  u.theta.value = 0.07;
  for (int i = 0; i < 3; ++i) {
    u.metric.g[i][i] = 1;
    u.lambda.value[i] = 0.1*(i+1);
  }
  hyp::OmegaJet<double> o{};
  o.omega = 0.4;
  o.normal = -0.3;
  o.gradient[0] = 0.2;
  RHS undamped{}, damped{}, changed_w{};
  Check(hyp::AssembleInterior(hyp::ConformalRHS(u, o, 0., 0.2), o.omega, undamped),
        "undamped off-constraint RHS");
  const auto parts = hyp::ConformalRHS(u, o, 1.5, 0.2);
  Check(hyp::AssembleInterior(parts, o.omega, damped), "damped off-constraint RHS");
  const double factor = u.alpha.value*1.5/o.omega;
  Check(std::abs(damped.trace-undamped.trace-factor*0.8*u.theta.value) < 1e-13,
        "physical trace damping scaling");
  Check(std::abs(damped.theta-undamped.theta+factor*2.2*u.theta.value) < 1e-13,
        "physical Theta damping scaling");
  for (int i = 0; i < 3; ++i) {
    Check(std::abs(damped.lambda[i]-undamped.lambda[i]
          +factor*u.lambda.value[i]) < 1e-13,
          "connection-constraint damping scaling");
  }
  o.normal += 0.2;
  Check(hyp::AssembleInterior(hyp::ConformalRHS(u, o, 1.5, 0.2), o.omega, changed_w),
        "changed-normal off-constraint RHS");
  Check(std::abs(changed_w.trace-damped.trace) < 1e-13, "trace contains spurious w");
  Check(std::abs(changed_w.theta-damped.theta
        +3*u.alpha.value*0.2*u.theta.value/o.omega) < 1e-13,
        "tensor-system off-constraint Theta term omitted");
  RHS overflow{};
  const double tiny = std::numeric_limits<double>::denorm_min();
  Check(!hyp::AssembleInterior(parts, tiny, overflow),
        "overflowing interior RHS accepted");
  u.chi.value = -1;
  Check(!hyp::ConformalRHS(u, o, 1.5, 0.).valid, "negative chi accepted");
}

void EvolvedDiagnostics() {
  Jet u{};
  u.chi.value = 0.8;
  u.alpha.value = 1;
  u.trace.value = -1.7;
  u.theta.value = 0.03;
  hyp::OmegaJet<double> o{};
  o.omega = 0.4;
  o.normal = -0.3;
  for (int i = 0; i < 3; ++i) {
    u.metric.g[i][i] = 1;
    u.chi.d[i] = 0.02*(i+1);
    u.chi.dd[i][i] = 0.03;
    u.trace.d[i] = 0.01*(i+1);
    u.theta.d[i] = -0.04*(i+1);
    u.a.k[i][i] = 0.07*(i-1);
    o.gradient[i] = -0.1*(i+1);
    o.hessian[i][i] = -1;
    o.dnormal[i] = 0.15*(i+1);
  }
  const auto b = hyp::PenroseMetric(u.metric, u.chi);
  hyp::CurvatureJet<double> k{};
  const double trace = (u.trace.value+2*u.theta.value-3*o.normal)/o.omega;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j) {
    k.k[i][j] = u.a.k[i][j]/u.chi.value+b.g[i][j]*trace/3;
    for (int d = 0; d < 3; ++d) {
      const double dt = (u.trace.d[d]+2*u.theta.d[d]-3*o.dnormal[d]
                         -trace*o.gradient[d])/o.omega;
      k.dk[d][i][j] = -u.a.k[i][j]*u.chi.d[d]/(u.chi.value*u.chi.value)
          +b.dg[d][i][j]*trace/3+b.g[i][j]*dt/3;
    }
  }
  const auto expected = hyp::Constraints(b, k, o);
  const auto got = hyp::EvolvedConstraints(u, o);
  Check(expected.valid && got.valid, "evolved diagnostics invalid");
  Check(std::abs(expected.hamiltonian-got.hamiltonian) < 1e-12,
        "evolved Hamiltonian transformation");
  for (int i = 0; i < 3; ++i) {
    Check(std::abs(expected.momentum[i]-got.momentum[i]) < 1e-12,
          "evolved momentum transformation");
  }
  o.omega = 0;
  const auto scri = hyp::EvolvedConstraints(u, o);
  Check(scri.valid && std::abs(scri.hamiltonian) > 0.1, "evolved scri H hidden");
  Check(scri.momentum_conformal_norm2 > 0.001, "evolved scri M hidden");
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    Minkowski();
    Schwarzschild();
    ADMRecovery();
    OffConstraintDamping();
    EvolvedDiagnostics();
    std::cout << "PASS nonlinear conformal Z4c interior RHS\n";
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
