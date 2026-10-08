// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
// Experimental single-core radial driver for the Cartesian conformal Z4c kernel.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <Kokkos_Core.hpp>

#include "z4c/hyperboloidal/spherical_tensor.hpp"
#include "z4c/hyperboloidal/cmc_trumpet.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"

namespace hyp = z4c::hyperboloidal;
using Exec = Kokkos::DefaultHostExecutionSpace;
using State = Kokkos::View<double **, Kokkos::LayoutRight, Kokkos::HostSpace>;
constexpr int ng = 3;

struct Options {
  int n = 64, extrapolation = 4;
  double end = 1, amplitude = 0.001, cfl = 0.05, dissipation = 0.1;
  double kappa1 = 1.5, slicing = 1, shift_driver = 1;
  double lapse_damping = 1.5, shift_damping = 1, diagnostic_dt = 0.1;
  double mass = 0;
  bool fixed_shift = false, fixed_lapse = false, puncture_gauge = false;
  bool analytic_trumpet = false, one_plus_log = false, lapse_scaled_damping = false;
  std::string output = "hyperboloidal";
};

Options Parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--lapse-scaled-damping") {
      o.lapse_scaled_damping = true;
      continue;
    }
    if (key == "--one-plus-log") {
      o.one_plus_log = true;
      continue;
    }
    if (key == "--analytic-trumpet") {
      o.analytic_trumpet = true;
      continue;
    }
    if (key == "--puncture-gauge") {
      o.puncture_gauge = true;
      continue;
    }
    if (key == "--fixed-lapse") {
      o.fixed_lapse = true;
      continue;
    }
    if (key == "--fixed-shift") {
      o.fixed_shift = true;
      continue;
    }
    if (i+1 == argc) throw std::invalid_argument("missing option value");
    const std::string v = argv[++i];
    if (key == "--n") o.n = std::stoi(v);
    else if (key == "--mass") o.mass = std::stod(v);
    else if (key == "--t") o.end = std::stod(v);
    else if (key == "--amplitude") o.amplitude = std::stod(v);
    else if (key == "--cfl") o.cfl = std::stod(v);
    else if (key == "--dissipation") o.dissipation = std::stod(v);
    else if (key == "--extrapolation") o.extrapolation = std::stoi(v);
    else if (key == "--kappa1") o.kappa1 = std::stod(v);
    else if (key == "--slicing") o.slicing = std::stod(v);
    else if (key == "--shift-driver") o.shift_driver = std::stod(v);
    else if (key == "--lapse-damping") o.lapse_damping = std::stod(v);
    else if (key == "--shift-damping") o.shift_damping = std::stod(v);
    else if (key == "--diagnostic-dt") o.diagnostic_dt = std::stod(v);
    else if (key == "--output") o.output = v;
    else throw std::invalid_argument("unknown option: "+key);
  }
  for (double value : {o.end, o.amplitude, o.cfl, o.dissipation, o.kappa1,
                       o.slicing, o.shift_driver, o.lapse_damping,
                       o.shift_damping, o.diagnostic_dt, o.mass}) {
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite driver option");
  }
  if (o.n < 8 || o.extrapolation < 2 || o.extrapolation > 5 || !(o.end >= 0)
      || !(o.cfl > 0 && o.cfl <= 0.2) || !(o.dissipation >= 0)
      || !(o.kappa1 >= 0) || !(o.diagnostic_dt > 0) || o.slicing < 0
      || o.mass < 0 || (o.analytic_trumpet && o.mass == 0)
      || o.shift_driver < 0 || o.lapse_damping < 0 || o.shift_damping < 0) {
    throw std::invalid_argument("invalid radial driver options");
  }
  return o;
}

void Ghosts(State q, const Options &o) {
  for (int f = 0; f < hyp::NFIELDS; ++f) {
    const int parity = f == hyp::LAMBDA || f == hyp::DBETA ? -1 : 1;
    for (int g = 0; g < ng; ++g) q(ng-1-g, f) = parity*q(ng+g, f);
    // Polynomial continuation of regular conformal deviations across scri.
    // All collocation points of the evolution remain strictly inside scri.
    for (int g = 0; g < ng; ++g) {
      const int degree = o.extrapolation;
      const double target = degree+1+g;
      double value = 0;
      for (int k = 0; k <= degree; ++k) {
        double weight = 1;
        for (int j = 0; j <= degree; ++j) {
          if (j != k) weight *= (target-j)/(k-j);
        }
        value += weight*q(ng+o.n-1-degree+k, f);
      }
      q(ng+o.n+g, f) = value;
    }
  }
}

// Optional spatial reconstruction around the analytic initial profile. Only
// derivatives and dissipation use it; the nonlinear RHS and Minkowski gauge
// sources are unchanged. It is not a subtraction of the black-hole RHS.
struct Reconstruction {
  State base, first, second;
  explicit Reconstruction(const Options &o)
      : base("analytic base", o.n+2*ng, hyp::NFIELDS),
        first("analytic first", o.n+2*ng, hyp::NFIELDS),
        second("analytic second", o.n+2*ng, hyp::NFIELDS) {
    if (!o.analytic_trumpet) return;
    const hyp::CMCTrumpet trumpet(o.mass);
    for (int cell = 0; cell < o.n; ++cell) {
      const int i = ng+cell;
      const double r = (cell+0.5)/o.n;
      double v[hyp::NFIELDS];
      trumpet.Values(r, v);
      const auto u = trumpet.Jet(r);
      for (int f = 0; f < hyp::NFIELDS; ++f) base(i, f) = v[f];
      first(i, hyp::DCHI) = u.chi.d[0];
      second(i, hyp::DCHI) = u.chi.dd[0][0];
      first(i, hyp::ARR) = u.a.dk[0][0][0];
      first(i, hyp::DALPHA) = u.alpha.d[0]-r;
      second(i, hyp::DALPHA) = u.alpha.dd[0][0]-1;
      first(i, hyp::DBETA) = u.beta.d[0][0]+1;
      second(i, hyp::DBETA) = u.beta.dd[0][0][0];
      // The system never uses second derivatives of A, P, Theta or Lambda.
    }
    Ghosts(base, o);
  }
};

KOKKOS_INLINE_FUNCTION
void Differences(State q, int i, double h, double v[hyp::NFIELDS],
                 double d[hyp::NFIELDS], double dd[hyp::NFIELDS],
                 const Reconstruction &rec, bool raw = false) {
  for (int f = 0; f < hyp::NFIELDS; ++f) {
    v[f] = q(i, f);
    if (raw) {
      d[f] = (8*(q(i+1, f)-q(i-1, f))-(q(i+2, f)-q(i-2, f)))/(12*h);
      dd[f] = (16*(q(i+1, f)+q(i-1, f)-2*q(i, f))
          -(q(i+2, f)+q(i-2, f)-2*q(i, f)))/(12*h*h);
      continue;
    }
    const double p1 = q(i+1, f)-rec.base(i+1, f), m1 = q(i-1, f)-rec.base(i-1, f);
    const double p2 = q(i+2, f)-rec.base(i+2, f), m2 = q(i-2, f)-rec.base(i-2, f);
    const double center = q(i, f)-rec.base(i, f);
    d[f] = (8*(p1-m1)-(p2-m2))/(12*h)+rec.first(i, f);
    dd[f] = (16*(p1+m1-2*center)-(p2+m2-2*center))/(12*h*h)+rec.second(i, f);
  }
}

KOKKOS_INLINE_FUNCTION
hyp::OmegaJet<double> Omega(double radius, const hyp::Z4cJet<double> &u) {
  hyp::OmegaJet<double> o{};
  o.omega = (1-radius*radius)/2;
  o.gradient[0] = -radius;
  for (int i = 0; i < 3; ++i) o.hessian[i][i] = -1;
  hyp::SetStationaryOmegaNormal(u.alpha.value, u.beta.value, u.alpha.d, u.beta.d, o);
  return o;
}

void RHS(State q, State result, const Options &o, const Reconstruction &rec) {
  Ghosts(q, o);
  const int n = o.n;
  const double h = 1./n, kappa1 = o.kappa1, diss = o.dissipation;
  const double slicing = o.slicing, driver = o.shift_driver;
  const double xi = o.lapse_damping, eta = o.shift_damping;
  const bool fixed_shift = o.fixed_shift, fixed_lapse = o.fixed_lapse;
  const bool puncture_gauge = o.puncture_gauge, one_plus_log = o.one_plus_log;
  const bool scaled_damping = o.lapse_scaled_damping;
  int failed = 0;
  Kokkos::parallel_reduce("radial conformal Z4c RHS", Kokkos::RangePolicy<Exec>(0, n),
      KOKKOS_LAMBDA(const int cell, int &bad) {
    const int i = cell+ng;
    const double radius = (cell+0.5)*h;
    const hyp::CMCReference<double> ref{1, 1};
    const auto background = ref.At(radius, 0., 0.);
    double v[hyp::NFIELDS], d[hyp::NFIELDS], dd[hyp::NFIELDS], zero[hyp::NFIELDS]{};
    Differences(q, i, h, v, d, dd, rec);
    const auto u = hyp::SphericalJet(radius, v, d, dd, ref);
    const auto u0 = hyp::SphericalJet(radius, zero, zero, zero, ref);
    const auto omega = Omega(radius, u), omega0 = Omega(radius, u0);
    hyp::Z4cRHS<double> rhs{}, rhs0{};
    const double damping = scaled_damping ? kappa1/u.alpha.value : kappa1;
    const auto parts = hyp::ConformalRHS(u, omega, damping, 0.);
    const auto refparts = hyp::ConformalRHS(u0, omega0, kappa1, 0.);
    if (!hyp::AssembleInterior(parts, omega.omega, rhs)
        || !hyp::AssembleInterior(refparts, omega.omega, rhs0)) {
      ++bad;
      return;
    }
    result(i, hyp::DCHI) = rhs.chi-rhs0.chi;
    result(i, hyp::DGRR) = rhs.metric[0][0]-rhs0.metric[0][0];
    result(i, hyp::ARR) = rhs.a[0][0]-rhs0.a[0][0];
    result(i, hyp::DK) = rhs.trace-rhs0.trace;
    result(i, hyp::THETA) = rhs.theta-rhs0.theta;
    result(i, hyp::LAMBDA) = rhs.lambda[0]-rhs0.lambda[0];
    const auto gauge_parts = hyp::UnfactoredReferenceGauge(ref, background, u,
        hyp::GaugeParameters<double>{slicing, driver, xi, eta},
        puncture_gauge, one_plus_log);
    hyp::GaugeRHS<double> gauge{};
    if (!hyp::AssembleGaugeInterior(gauge_parts, omega.omega, gauge)) {
      ++bad;
      return;
    }
    result(i, hyp::DALPHA) = fixed_lapse ? 0 : gauge.alpha;
    result(i, hyp::DBETA) = fixed_shift ? 0 : gauge.beta[0];
    for (int f = 0; f < hyp::NFIELDS; ++f) {
      if ((f == hyp::DBETA && fixed_shift) || (f == hyp::DALPHA && fixed_lapse)) continue;
      double ko = 0;
      const int weights[7] = {1, -6, 15, -20, 15, -6, 1};
      for (int j = -3; j <= 3; ++j) ko += weights[j+3]*(q(i+j, f)-rec.base(i+j, f));
      result(i, f) += diss*ko/(64*h);
      if (!Kokkos::isfinite(result(i, f))) ++bad;
    }
  }, failed);
  if (failed) {
    for (int cell = 0; cell < n; ++cell) {
      const double radius = (cell+0.5)*h;
      double v[hyp::NFIELDS], d[hyp::NFIELDS], dd[hyp::NFIELDS];
      Differences(q, cell+ng, h, v, d, dd, rec);
      const auto u = hyp::SphericalJet(radius, v, d, dd, hyp::CMCReference<double>{1, 1});
      hyp::Z4cRHS<double> test{};
      const auto omeg = Omega(radius, u);
      if (!hyp::AssembleInterior(hyp::ConformalRHS(u, omeg, kappa1, 0.),
                                omeg.omega, test)) {
        throw std::runtime_error("invalid geometry/RHS at r="+std::to_string(radius)
            +" chi="+std::to_string(u.chi.value)+" alpha="+std::to_string(u.alpha.value)
            +" grr="+std::to_string(u.metric.g[0][0]));
      }
    }
    throw std::runtime_error("nonfinite gauge or damping RHS");
  }
}

void Stage(State q, State k, State stage, int n, double dt) {
  Kokkos::parallel_for("radial RK stage", Kokkos::RangePolicy<Exec>(0, n*hyp::NFIELDS),
      KOKKOS_LAMBDA(const int index) {
    const int i = ng+index/hyp::NFIELDS, f = index%hyp::NFIELDS;
    stage(i, f) = q(i, f)+dt*k(i, f);
  });
}

// Extrapolate to scri for diagnostics only. The integrator remains staggered.
// Return finite pole numerators without ever dividing by Omega=0.
void ScriDiagnostics(State q, const Options &o, double &null_residual,
                     double &pole_max, double &lapse_pole) {
  double v[hyp::NFIELDS]{}, d[hyp::NFIELDS]{}, dd[hyp::NFIELDS]{};
  const int degree = o.extrapolation;
  for (int k = 0; k <= degree; ++k) {
    double basis[6] = {1, 0, 0, 0, 0, 0};
    int order = 0;
    for (int j = 0; j <= degree; ++j) {
      if (j == k) continue;
      const double node = -0.5-j, denominator = j-k;
      for (int a = order+1; a >= 0; --a) {
        basis[a] = ((a > 0 ? basis[a-1] : 0)-node*basis[a])/denominator;
      }
      ++order;
    }
    for (int f = 0; f < hyp::NFIELDS; ++f) {
      const double value = q(ng+o.n-1-k, f);
      v[f] += basis[0]*value;
      d[f] += o.n*basis[1]*value;
      dd[f] += 2*o.n*o.n*basis[2]*value;
    }
  }
  const auto u = hyp::SphericalJet(1., v, d, dd, hyp::CMCReference<double>{1, 1});
  const auto omega = Omega(1., u);
  const double damping = o.lapse_scaled_damping ? o.kappa1/u.alpha.value : o.kappa1;
  const auto parts = hyp::ConformalRHS(u, omega, damping, 0.);
  const auto con = hyp::EvolvedConstraints(u, omega);
  if (!parts.valid || !con.valid) throw std::runtime_error("invalid scri extrapolation");
  null_residual = con.null_residual;
  const auto p = parts.pole;
  pole_max = std::max({std::abs(p.chi), std::abs(p.trace), std::abs(p.theta)});
  for (int i = 0; i < 3; ++i) {
    pole_max = std::max(pole_max, std::abs(p.lambda[i]));
    for (int j = 0; j < 3; ++j) {
      pole_max = std::max({pole_max, std::abs(p.metric[i][j]), std::abs(p.a[i][j])});
    }
  }
  const hyp::CMCReference<double> ref{1, 1};
  lapse_pole = hyp::UnfactoredReferenceGauge(ref, ref.At(1., 0., 0.), u,
      hyp::GaugeParameters<double>{o.slicing, o.shift_driver, o.lapse_damping,
                                  o.shift_damping},
      o.puncture_gauge, o.one_plus_log).pole.alpha;
}

void Report(State q, double time, const Options &o, std::ostream &log,
            const Reconstruction &rec) {
  Ghosts(q, o);
  const double h = 1./o.n;
  double h2 = 0, m2 = 0, z2 = 0, t2 = 0, maxh = 0, maxm = 0, maxq = 0;
  double minchi = 1e100, minalpha = 1e100, null_last = 0;
  double mass_half = 0, horizon = 0, previous_expansion = 0, previous_radius = 0;
  double raw_h2 = 0, raw_m2 = 0;
  for (int cell = 0; cell < o.n; ++cell) {
    double v[hyp::NFIELDS], d[hyp::NFIELDS], dd[hyp::NFIELDS];
    Differences(q, cell+ng, h, v, d, dd, rec);
    const double radius = (cell+0.5)*h;
    const auto u = hyp::SphericalJet(radius, v, d, dd, hyp::CMCReference<double>{1, 1});
    const auto con = hyp::EvolvedConstraints(u, Omega(radius, u));
    if (!con.valid) throw std::runtime_error("invalid constraint diagnostic");
    auto raw = con;
    if (o.analytic_trumpet) {
      Differences(q, cell+ng, h, v, d, dd, rec, true);
      const auto raw_u = hyp::SphericalJet(radius, v, d, dd,
                                           hyp::CMCReference<double>{1, 1});
      raw = hyp::EvolvedConstraints(raw_u, Omega(radius, raw_u));
      if (!raw.valid) throw std::runtime_error("invalid plain-FD diagnostic");
    }
    raw_h2 += h*raw.hamiltonian*raw.hamiltonian;
    raw_m2 += h*raw.momentum_conformal_norm2;
    const auto sphere = hyp::SphereDiagnostics(radius, u, Omega(radius, u));
    if (cell == o.n/2) mass_half = sphere.mass;
    if (cell > 0 && previous_expansion <= 0 && sphere.expansion_out > 0) {
      horizon = previous_radius+(sphere.areal_radius-previous_radius)
          *(-previous_expansion)/(sphere.expansion_out-previous_expansion);
    }
    previous_expansion = sphere.expansion_out;
    previous_radius = sphere.areal_radius;
    // Coordinate radial L2 norms, explicitly unweighted by Omega or r^2 so
    // neither the puncture vicinity nor the scri vicinity can disappear.
    h2 += h*con.hamiltonian*con.hamiltonian;
    m2 += h*con.momentum_conformal_norm2;
    z2 += h*con.z4.z_conformal_norm2;
    t2 += h*v[hyp::THETA]*v[hyp::THETA];
    maxh = std::max(maxh, std::abs(con.hamiltonian));
    maxm = std::max(maxm, std::sqrt(con.momentum_conformal_norm2));
    for (double value : v) maxq = std::max(maxq, std::abs(value));
    minchi = std::min(minchi, u.chi.value);
    minalpha = std::min(minalpha, u.alpha.value);
    null_last = con.null_residual;
  }
  double scri_null = 0, scri_pole = 0, scri_lapse = 0;
  ScriDiagnostics(q, o, scri_null, scri_pole, scri_lapse);
  log << std::setprecision(17) << time << ',' << std::sqrt(h2) << ',' << std::sqrt(m2)
      << ',' << std::sqrt(z2) << ',' << std::sqrt(t2) << ',' << maxh << ',' << maxm
      << ',' << maxq << ',' << minchi << ',' << minalpha << ',' << null_last
      << ',' << mass_half << ',' << horizon << ',' << std::sqrt(raw_h2)
      << ',' << std::sqrt(raw_m2) << ',' << scri_null << ',' << scri_pole
      << ',' << scri_lapse << '\n';
  log.flush();
  std::cout << "t=" << time << " H_L2=" << std::sqrt(h2) << " M_L2=" << std::sqrt(m2)
            << " max_deviation=" << maxq << std::endl;
}

int main(int argc, char **argv) {
  Kokkos::InitializationSettings settings;
  settings.set_num_threads(1);
  Kokkos::ScopeGuard guard(settings);
  try {
    const Options o = Parse(argc, argv);
    std::ofstream configuration(o.output+"-config.txt");
    if (!configuration) throw std::runtime_error("cannot open configuration output");
    configuration << std::setprecision(17)
        << "n=" << o.n << " end=" << o.end << " mass=" << o.mass
        << " amplitude=" << o.amplitude << " cfl=" << o.cfl
        << " dissipation=" << o.dissipation << " extrapolation=" << o.extrapolation
        << " kappa1=" << o.kappa1 << " slicing=" << o.slicing
        << " shift_driver=" << o.shift_driver << " lapse_damping=" << o.lapse_damping
        << " shift_damping=" << o.shift_damping << " diagnostic_dt=" << o.diagnostic_dt
        << " fixed_lapse=" << o.fixed_lapse << " fixed_shift=" << o.fixed_shift
        << " puncture_gauge=" << o.puncture_gauge << " one_plus_log=" << o.one_plus_log
        << " analytic_trumpet=" << o.analytic_trumpet
        << " lapse_scaled_damping=" << o.lapse_scaled_damping << '\n';
    configuration.close();
    const Reconstruction rec(o);
    State q("q", o.n+2*ng, hyp::NFIELDS), stage("stage", o.n+2*ng, hyp::NFIELDS);
    State k1("k1", o.n+2*ng, hyp::NFIELDS), k2("k2", o.n+2*ng, hyp::NFIELDS);
    State k3("k3", o.n+2*ng, hyp::NFIELDS), k4("k4", o.n+2*ng, hyp::NFIELDS);
    for (int cell = 0; cell < o.n; ++cell) {
      const double radius = (cell+0.5)/o.n, x = (radius-0.4)/0.12;
      if (o.mass > 0) {
        double values[hyp::NFIELDS];
        hyp::CMCTrumpet(o.mass).Values(radius, values);
        for (int f = 0; f < hyp::NFIELDS; ++f) q(cell+ng, f) = values[f];
      }
      q(cell+ng, hyp::DALPHA) += std::abs(x) < 1 ?
          o.amplitude*std::exp(1-1/(1-x*x)) : 0;
    }
    std::ofstream log(o.output+"-diagnostics.csv");
    if (!log) throw std::runtime_error("cannot open diagnostic output");
    log << "t,H_L2,M_L2,Z_L2,Theta_L2,H_max,M_max,max_deviation,min_chi,min_alpha,"
           "null_residual_last_cell,mass_near_half,horizon_areal_radius,"
           "H_raw_L2,M_raw_L2,scri_null_residual,scri_pole_max,scri_lapse_pole\n";
    double time = 0, next_report = o.diagnostic_dt;
    Report(q, time, o, log, rec);
    while (time < o.end) {
      const double dt = std::min(o.cfl/o.n, o.end-time);
      RHS(q, k1, o, rec);
      Stage(q, k1, stage, o.n, 0.5*dt);
      RHS(stage, k2, o, rec);
      Stage(q, k2, stage, o.n, 0.5*dt);
      RHS(stage, k3, o, rec);
      Stage(q, k3, stage, o.n, dt);
      RHS(stage, k4, o, rec);
      const int n = o.n;
      Kokkos::parallel_for("radial RK4 update",
          Kokkos::RangePolicy<Exec>(0, n*hyp::NFIELDS),
          KOKKOS_LAMBDA(const int index) {
        const int i = ng+index/hyp::NFIELDS, f = index%hyp::NFIELDS;
        q(i, f) += dt*(k1(i, f)+2*k2(i, f)+2*k3(i, f)+k4(i, f))/6;
      });
      time += dt;
      if (time >= next_report || time >= o.end) {
        Report(q, time, o, log, rec);
        next_report += o.diagnostic_dt;
      }
    }
    std::ofstream fields(o.output+"-fields.csv");
    if (!fields) throw std::runtime_error("cannot open field output");
    fields << "r,delta_chi,delta_grr,A_rr,delta_K,Theta,Lambda,delta_alpha,delta_beta,"
              "H,M_r,Z_r,c_light_minus,c_light_plus,areal_radius,mass,expansion_out\n";
    fields << std::setprecision(17);
    for (int cell = 0; cell < o.n; ++cell) {
      const double radius = (cell+0.5)/o.n;
      fields << radius;
      for (int f = 0; f < hyp::NFIELDS; ++f) fields << ',' << q(cell+ng, f);
      double v[hyp::NFIELDS], d[hyp::NFIELDS], dd[hyp::NFIELDS];
      Differences(q, cell+ng, 1./o.n, v, d, dd, rec);
      const auto u = hyp::SphericalJet(radius, v, d, dd, hyp::CMCReference<double>{1, 1});
      const auto con = hyp::EvolvedConstraints(u, Omega(radius, u));
      const double speed = u.alpha.value*std::sqrt(u.chi.value/u.metric.g[0][0]);
      fields << ',' << con.hamiltonian << ',' << con.momentum[0] << ','
             << con.z4.z_covector[0] << ',' << -u.beta.value[0]-speed << ','
             << -u.beta.value[0]+speed;
      const auto sphere = hyp::SphereDiagnostics(radius, u, Omega(radius, u));
      fields << ',' << sphere.areal_radius << ',' << sphere.mass << ','
             << sphere.expansion_out << '\n';
    }
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
