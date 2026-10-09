// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <stdexcept>
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace hyp = z4c::hyperboloidal;
void Close(double x, double y, double tolerance, const char* label) {
  if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x - y) > tolerance) {
    std::cerr << label << ": " << x << " != " << y << '\n';
    throw std::runtime_error(label);
  }
}

void ReferenceAudit() {
  hyp::LayerParameters layer{true, 0.35, 0.75};
  hyp::LayerReference<double> ref(1, 1, layer);
  hyp::LayerGaugeParameters gauge;
  ref.Validate();
  gauge.Validate(1);
  double maximum = 0, connection = 0;
  for (double r : {0., 0.1, 0.35, 0.35000001, 0.4, 0.5, 0.6, 0.7, 0.74999999, 0.75, 0.9,
                   0.99, 0.9999}) {
    for (const auto& direction :
         {std::array<double, 3>{1, 0, 0}, std::array<double, 3>{0.36, -0.48, 0.8}}) {
      const auto p = ref.At(r * direction[0], r * direction[1], r * direction[2]);
      const auto& u = p.state;
      auto o = hyp::CartesianOmega(u, p);
      const auto c = hyp::EvolvedConstraints(u, o);
      if (!c.valid)
        throw std::runtime_error("invalid reference constraints");
      Close(c.hamiltonian, 0, 2e-9, "reference H");
      Close(std::sqrt(c.momentum_conformal_norm2), 0, 2e-9, "reference M");
      Close(c.z4.determinant_residual, 0, 1e-13, "determinant");
      Close(c.z4.tracefree_residual, 0, 1e-13, "trace-free");
      hyp::Z4cRHS<double> rhs{};
      if (!hyp::AssembleInterior(hyp::ConformalRHS(u, o, 5 / u.alpha.value, 0.), p.omega,
                                 rhs)) {
        throw std::runtime_error("invalid reference RHS");
      }
      for (double v : {rhs.chi, rhs.trace, rhs.theta})
        maximum = std::max(maximum, std::abs(v));
      for (int i = 0; i < 3; ++i) {
        connection = std::max(connection, std::abs(u.lambda.value[i]));
        maximum = std::max(maximum, std::abs(rhs.lambda[i]));
        for (int j = 0; j < 3; ++j) {
          maximum = std::max(maximum, std::abs(rhs.metric[i][j]));
          maximum = std::max(maximum, std::abs(rhs.a[i][j]));
        }
      }
      hyp::GaugeRHS<double> gr{};
      if (!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, u, gauge), p.omega,
                                      gr)) {
        throw std::runtime_error("invalid layer reference gauge");
      }
      Close(gr.alpha, 0, 2e-10, "reference lapse gauge");
      for (double v : gr.beta) Close(v, 0, 2e-10, "reference shift gauge");
      // Independent Cartesian finite differences of reference fields.
      const double step = 1e-5;
      for (int d = 0; d < 3; ++d) {
        double xp[3], xm[3];
        for (int j = 0; j < 3; ++j) xp[j] = xm[j] = r * direction[j];
        xp[d] += step;
        xm[d] -= step;
        const auto plus = ref.At(xp[0], xp[1], xp[2]),
                   minus = ref.At(xm[0], xm[1], xm[2]);
        Close((plus.alpha - minus.alpha) / (2 * step), u.alpha.d[d], 2e-6,
              "alpha derivative");
        Close((plus.omega - minus.omega) / (2 * step), p.domega[d], 2e-6,
              "Omega derivative");
        Close((plus.k_physical - minus.k_physical) / (2 * step), u.trace.d[d], 2e-5,
              "P derivative");
        for (int i = 0; i < 3; ++i) {
          Close((plus.beta[i] - minus.beta[i]) / (2 * step), u.beta.d[d][i], 2e-5,
                "beta derivative");
          Close((plus.state.lambda.value[i] - minus.state.lambda.value[i]) / (2 * step),
                u.lambda.d[d][i], 2e-4, "connection derivative");
          for (int j = 0; j < 3; ++j) {
            Close((plus.state.metric.g[i][j] - minus.state.metric.g[i][j]) / (2 * step),
                  u.metric.dg[d][i][j], 2e-5, "metric derivative");
            Close((plus.state.a.k[i][j] - minus.state.a.k[i][j]) / (2 * step),
                  u.a.dk[d][i][j], 2e-5, "curvature derivative");
            for (int e = 0; e < 3; ++e) {
              Close((plus.state.metric.dg[e][i][j] - minus.state.metric.dg[e][i][j]) /
                        (2 * step),
                    u.metric.ddg[d][e][i][j], 2e-4, "metric Hessian");
            }
          }
        }
      }
    }
  }
  Close(maximum, 0, 2e-8, "stationary continuum residual");
  if (!(connection > 0.1))
    throw std::runtime_error("missing layer connection");
  for (double s : {0., 1e-15, 1e-6, .01, .1, .5, .9, .99, 1 - 1e-6, 1 - 1e-15, 1.}) {
    const auto w = hyp::SmoothCutoff(s, 0., 1.);
    for (double v : {w.value, w.d, w.dd, w.ddd}) {
      if (!std::isfinite(v))
        throw std::runtime_error("nonfinite cutoff endpoint");
    }
    if (!(w.value >= 0 && w.value <= 1 && w.d >= 0))
      throw std::runtime_error("cutoff monotonicity");
  }
  for (int k = 2; k <= 15; ++k) {
    const double r = 1 - std::pow(10., -k);
    const auto p = ref.At(r, 0., 0.);
    Close(p.ingoing, -(1 - r) * (1 - r) / 2, 1e-29, "factored inward speed");
    if (!(p.omega > 0) || !(p.ingoing < 0))
      throw std::runtime_error("small Omega sequence");
  }
  const auto scri = ref.At(1., 0., 0.);
  Close(scri.outgoing, 2, 0, "scri outgoing");
  Close(scri.ingoing, 0, 0, "scri inward");
  hyp::GaugeRHS<double> rejected{};
  if (hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(scri, scri.state, gauge), 0.,
                                 rejected)) {
    throw std::runtime_error("exact scri assembly must be rejected");
  }
  std::cout << "PASS reference jets, constraints and continuum stationarity: max="
            << maximum << " nonzero Lambda=" << connection << '\n';
}

void PreferredSourceAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, 0.35, 0.75});
  hyp::LayerGaugeParameters gauge;
  double max_error = 0;
  for (double r : {0.85, 0.9, 0.99, 0.9999}) {
    const auto p = ref.At(0.36 * r, -0.48 * r, 0.8 * r);
    auto u = p.state;
    u.alpha.value *= 1.03;
    u.chi.value *= 0.97;
    u.beta.value[0] += 0.02;
    u.beta.value[1] -= 0.03;
    u.lambda.value[2] += 0.01;
    u.alpha.d[0] += 0.02;
    u.chi.d[1] -= 0.01;
    // A nonflat determinant-one live spatial metric.
    u.metric.g[0][0] = 1.1;
    u.metric.g[1][1] = 1 / 1.1;
    const auto geo = hyp::Geometry(u.metric);
    const auto c = hyp::LayerCoefficients(r, u.alpha.value, gauge);
    hyp::GaugeRHS<double> gr{};
    if (!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, u, gauge), p.omega, gr)) {
      throw std::runtime_error("invalid preferred gauge audit");
    }
    const double alpha = u.alpha.value, chi = u.chi.value;
    double log_advection = 0, hessian4 = 0, projected = 0;
    for (int i = 0; i < 3; ++i) log_advection += u.beta.value[i] * p.dalpha[i] / p.alpha;
    const double f0 =
        (log_advection + c.nu * std::log(alpha / p.alpha)) / (alpha * alpha) -
        p.k_bar / alpha;
    for (int i = 0; i < 3; ++i) {
      double source =
          chi * u.lambda.value[i] - gr.beta[i] / (alpha * alpha) - u.beta.value[i] * f0;
      for (int j = 0; j < 3; ++j) {
        source +=
            u.beta.value[j] * u.beta.d[j][i] / (alpha * alpha) +
            chi * geo.inverse[i][j] * (u.chi.d[j] / (2 * chi) - u.alpha.d[j] / alpha);
        hessian4 += (chi * geo.inverse[i][j] -
                     u.beta.value[i] * u.beta.value[j] / (alpha * alpha)) *
                    p.omega_hessian[i][j];
      }
      projected += p.domega[i] * source;
    }
    max_error = std::max(max_error, std::abs(projected - hessian4 + p.omega * p.w_omega));
  }
  Close(max_error, 0, 2e-13, "live preferred source projection");
  std::cout << "PASS live preferred source: max=" << max_error << '\n';
  // Numerical counterexample to interpreting finite Q/nullity as full closure.
  const double dq = 0.01;
  for (int k = 2; k <= 8; ++k) {
    const double r = 1 - std::pow(10., -k);
    const auto p = ref.At(r, 0., 0.);
    auto u = p.state;
    u.trace.value += p.omega * dq;
    for (int j = 0; j < 3; ++j) u.trace.d[j] += p.domega[j] * dq;
    const auto o = hyp::CartesianOmega(u, p);
    hyp::Z4cRHS<double> rhs{};
    hyp::GaugeRHS<double> gr{};
    if (!hyp::AssembleInterior(hyp::ConformalRHS(u, o, 5 / u.alpha.value, 0.), p.omega,
                               rhs) ||
        !hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p, u, gauge), p.omega, gr)) {
      throw std::runtime_error("invalid counterexample evaluation");
    }
    double numerator = rhs.trace + 3 * o.normal * gr.alpha / u.alpha.value;
    for (int j = 0; j < 3; ++j) numerator += 3 * p.domega[j] * gr.beta[j] / u.alpha.value;
    Close(numerator, 2 * dq, 10 * std::pow(10., -k) + 2e-7, "unclosed Q residue");
    std::cout << "unclosed limit Omega=" << p.omega << " Omega*Qdot=" << numerator
              << " Theta_dot=" << rhs.theta << '\n';
  }
}

void RawStationarityAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, 0.35, 0.75});
  double previous = 0;
  for (double h : {0.02, 0.01, 0.005, 0.0025}) {
    DvceArray5D<Real> values("raw stationary stencil", 1, z4c::Z4c::nz4c, 7, 7, 7);
    auto host = Kokkos::create_mirror_view(values);
    const double center[3] = {0.36 * 0.55, -0.48 * 0.55, 0.8 * 0.55};
    for (int k = 0; k < 7; ++k)
      for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 7; ++i) {
          const auto p = ref.At(center[0] + (i - 3) * h, center[1] + (j - 3) * h,
                                center[2] + (k - 3) * h);
          for (int f = 0; f < z4c::Z4c::nz4c; ++f)
            host(0, f, k, j, i) = hyp::ReferenceComponent(f, p);
        }
    Kokkos::deep_copy(values, host);
    const double idx[3] = {1 / h, 1 / h, 1 / h};
    const auto u = hyp::LoadMeshJet<3>(hyp::BindCartesianFields(values), idx, 0, 3, 3, 3);
    const auto p = ref.At(center[0], center[1], center[2]);
    hyp::Z4cRHS<double> rhs{};
    hyp::GaugeRHS<double> gr{};
    if (!hyp::AssembleInterior(
            hyp::ConformalRHS(u, hyp::CartesianOmega(u, p), 5 / u.alpha.value, 0.),
            p.omega, rhs) ||
        !hyp::AssembleGaugeInterior(
            hyp::InteriorLayerGauge(p, u, hyp::LayerGaugeParameters{}), p.omega, gr))
      throw std::runtime_error("raw residual failed");
    double error = std::max({std::abs(rhs.chi), std::abs(rhs.trace), std::abs(rhs.theta),
                             std::abs(gr.alpha)});
    for (int i = 0; i < 3; ++i) {
      error = std::max({error, std::abs(rhs.lambda[i]), std::abs(gr.beta[i])});
      for (int j = 0; j < 3; ++j)
        error = std::max({error, std::abs(rhs.metric[i][j]), std::abs(rhs.a[i][j])});
    }
    std::cout << "raw stationary stencil h=" << h << " max_RHS=" << error;
    if (previous) {
      const double order = std::log2(previous / error);
      std::cout << " order=" << order;
      if (!(order > 3.7))
        throw std::runtime_error("raw stationary residual must converge at fourth order");
    }
    std::cout << '\n';
    previous = error;
  }
}

void LowerOrderAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, 0.35, 0.75});
  const auto p = ref.At(1., 0., 0.);
  // Frozen zero-derivative pole block (delta alpha, delta chi, delta P, delta Theta).
  // Beta, determinant-one metric and trace-free A have no pole forcing from this
  // subspace at the reference, so it is a closed block of the leading residue.
  const double expected[4][4] = {
      {3, 0, -1, 0}, {-2, 0, 2. / 3., 4. / 3.}, {0, -3, -2, 1}, {0, -3, -2, -11}};
  const double epsilon = 1e-6;
  for (int column = 0; column < 4; ++column) {
    double v[2][4];
    for (int sign = 0; sign < 2; ++sign) {
      auto u = p.state;
      const double delta = sign ? epsilon : -epsilon;
      if (column == 0)
        u.alpha.value += delta;
      if (column == 1)
        u.chi.value += delta;
      if (column == 2)
        u.trace.value += delta;
      if (column == 3)
        u.theta.value += delta;
      const auto rhs =
          hyp::ConformalRHS(u, hyp::CartesianOmega(u, p), 5 / u.alpha.value, 0.);
      const auto gauge = hyp::InteriorLayerGauge(p, u, hyp::LayerGaugeParameters{});
      if (!rhs.valid || !gauge.valid)
        throw std::runtime_error("pole audit invalid");
      v[sign][0] = gauge.pole.alpha;
      v[sign][1] = rhs.pole.chi;
      v[sign][2] = rhs.pole.trace;
      v[sign][3] = rhs.pole.theta;
      for (int i = 0; i < 3; ++i) {
        Close(gauge.pole.beta[i], 0, 0, "no shift pole");
        Close(rhs.pole.lambda[i], 0, 1e-12, "closed leading pole block");
        for (int j = 0; j < 3; ++j)
          Close(rhs.pole.a[i][j], 0, 1e-12, "closed A pole block");
      }
    }
    for (int row = 0; row < 4; ++row) {
      Close((v[1][row] - v[0][row]) / (2 * epsilon), expected[row][column], 1e-8,
            "kernel pole Jacobian");
    }
  }
  std::cout << "PASS extracted lower-order pole block: positive eigenvalue "
               "2.5717094873/Omega\n";
}

void BoundaryAudit() {
  hyp::SphericalGhostGrid grid;
  grid.radius = 1;
  for (int i = 0; i < 3; ++i) {
    grid.n[i] = 30;
    grid.h[i] = 2.1 / 24;
    grid.first[i] = -1.05 - 2.5 * grid.h[i];
  }
  for (int degree : {2, 3, 4}) {
    const auto plans = hyp::PlanSphericalGhosts(grid, 3, degree);
    std::map<int, size_t> lookup;
    for (size_t p = 0; p < plans.size(); ++p) lookup[plans[p].target] = p;
    double max_norm = 0, max_reflection = 0, max_permutation = 0;
    const auto transform = [&grid](int s, int mode) {
      int i = s % grid.n[0], j = s / grid.n[0] % grid.n[1];
      const int k = s / (grid.n[0] * grid.n[1]);
      if (mode == 0)
        i = grid.n[0] - 1 - i;
      else
        std::swap(i, j);
      return grid.Index(i, j, k);
    };
    for (const auto& plan : plans) {
      double norm = 0;
      for (int k = 0; k < plan.count; ++k) norm += std::abs(plan.weights[k]);
      max_norm = std::max(max_norm, norm);
      for (int mode = 0; mode < 2; ++mode) {
        const auto found = lookup.find(transform(plan.target, mode));
        if (found == lookup.end())
          throw std::runtime_error("asymmetric ghost target coverage");
        const auto& other = plans[found->second];
        std::map<int, double> difference;
        for (int k = 0; k < plan.count; ++k)
          difference[transform(plan.donors[k], mode)] += plan.weights[k];
        for (int k = 0; k < other.count; ++k)
          difference[other.donors[k]] -= other.weights[k];
        double mismatch = 0;
        for (const auto& item : difference) mismatch += std::abs(item.second);
        if (mode == 0)
          max_reflection = std::max(max_reflection, mismatch);
        else
          max_permutation = std::max(max_permutation, mismatch);
      }
    }
    std::cout << "boundary degree=" << degree << " max_weight_L1=" << max_norm
              << " reflection_weight_L1_mismatch=" << max_reflection
              << " xy_permutation_weight_L1_mismatch=" << max_permutation << '\n';
  }
  // Report inconsistencies rather than converting this audit into a false
  // stability assertion. Polynomial consistency/coverage are tested separately.
}

void DumpCutoff() {
  std::cout << std::setprecision(17) << '[';
  bool first = true;
  for (double r :
       {0., 1e-15, 1e-3, .01, .1, .25, .5, .75, .9, .99, .999, 1 - 1e-15, 1.}) {
    if (!first)
      std::cout << ',';
    first = false;
    const auto w = hyp::SmoothCutoff(r, 0., 1.);
    std::cout << '[' << r << ',' << w.value << ',' << w.d << ',' << w.dd << ',' << w.ddd
              << ']';
  }
  std::cout << "]\n";
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    if (argc == 2 && std::string(argv[1]) == "--cutoff") {
      DumpCutoff();
    } else {
      ReferenceAudit();
      PreferredSourceAudit();
      RawStationarityAudit();
      LowerOrderAudit();
      BoundaryAudit();
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
  return status;
}
