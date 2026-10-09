// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_injection.hpp"
#include "z4c/hyperboloidal/cartesian_patch.hpp"

namespace hyp = z4c::hyperboloidal;
using z4c::Z4c;

// Every probe uses the native nonspherical lapse .1 / shift .02 profile,
// width .5. Only its continuous derivatives are analytic in the point audit.
// The native audit loads every live derivative through the actual patch RHS.
void PulseJet(hyp::Z4cJet<Real>& u, const double xyz[3], double lapse = .1,
              double shift = .02) {
  const double x = xyz[0], y = xyz[1], z = xyz[2], s = x * x + y * y + z * z;
  const double f = std::pow(1 - s, 4) * std::exp(-4 * s);
  const double fs = f * (-4 / (1 - s) - 4),
               fss = f * (std::pow(-4 / (1 - s) - 4, 2) - 4 / std::pow(1 - s, 2));
  const double av = 1 + .2 * x + .3 * y * z, ag[3] = {.2, .3 * z, .3 * y};
  const double bv[3] = {1 + .3 * y * z, .2 * x, .1 * x * y};
  const double bg[3][3] = {{0, .2, .1 * y}, {.3 * z, 0, .1 * x}, {.3 * y, 0, 0}};
  u.alpha.value += lapse * f * av;
  for (int k = 0; k < 3; ++k) u.beta.value[k] += shift * f * bv[k];
  for (int i = 0; i < 3; ++i) {
    const double df = 2 * xyz[i] * fs;
    u.alpha.d[i] += lapse * (df * av + f * ag[i]);
    for (int k = 0; k < 3; ++k) u.beta.d[i][k] += shift * (df * bv[k] + f * bg[i][k]);
    for (int j = 0; j < 3; ++j) {
      const double dd = 4 * xyz[i] * xyz[j] * fss + (i == j ? 2 * fs : 0);
      const double da = ((i == 1 && j == 2) || (i == 2 && j == 1)) ? .3 : 0;
      u.alpha.dd[i][j] +=
          lapse * (dd * av + 2 * xyz[i] * fs * ag[j] + 2 * xyz[j] * fs * ag[i] + f * da);
      for (int k = 0; k < 3; ++k) {
        const double db =
            (k == 0 && ((i == 1 && j == 2) || (i == 2 && j == 1)))
                ? .3
                : ((k == 2 && ((i == 0 && j == 1) || (i == 1 && j == 0))) ? .1 : 0);
        u.beta.dd[i][j][k] += shift * (dd * bv[k] + 2 * xyz[i] * fs * bg[j][k] +
                                       2 * xyz[j] * fs * bg[i][k] + f * db);
      }
    }
  }
}

hyp::Z4cRHS<Real> PointRHS(const hyp::LayerReference<Real>& ref, const double xyz[3]) {
  const auto p = ref.At(xyz[0], xyz[1], xyz[2]);
  auto u = p.state;
  PulseJet(u, xyz);
  hyp::Z4cRHS<Real> rhs{};
  if (!hyp::AssembleInterior(
          hyp::ConformalRHS(u, hyp::CartesianOmega(u, p), 10 / u.alpha.value, Real(0)),
          p.omega, rhs))
    throw std::runtime_error("point RHS invalid");
  return rhs;
}

double ContinuumHDot(const hyp::LayerReference<Real>& ref, const double xyz[3], double dx,
                     double dt = 1e-6) {
  const auto p = ref.At(xyz[0], xyz[1], xyz[2]);
  auto u = p.state;
  PulseJet(u, xyz);
  const auto base = PointRHS(ref, xyz);
  const double first[5] = {1. / 12, -2. / 3, 0, 2. / 3, -1. / 12};
  const double second[5] = {-1. / 12, 4. / 3, -2.5, 4. / 3, -1. / 12};
  hyp::ScalarJet<Real> chi{};
  hyp::MetricJet<Real> metric{};
  chi.value = base.chi;
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b) metric.g[a][b] = base.metric[a][b];
  for (int i = 0; i < 3; ++i) {
    for (int d = -2; d <= 2; ++d) {
      double q[3] = {xyz[0], xyz[1], xyz[2]};
      q[i] += d * dx;
      const auto v = PointRHS(ref, q);
      chi.d[i] += first[d + 2] * v.chi / dx;
      chi.dd[i][i] += second[d + 2] * v.chi / (dx * dx);
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          metric.dg[i][a][b] += first[d + 2] * v.metric[a][b] / dx;
          metric.ddg[i][i][a][b] += second[d + 2] * v.metric[a][b] / (dx * dx);
        }
    }
    for (int j = i + 1; j < 3; ++j)
      for (int d = -2; d <= 2; ++d)
        for (int e = -2; e <= 2; ++e) {
          double q[3] = {xyz[0], xyz[1], xyz[2]};
          q[i] += d * dx;
          q[j] += e * dx;
          const auto v = PointRHS(ref, q);
          chi.dd[i][j] += first[d + 2] * first[e + 2] * v.chi / (dx * dx);
          for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
              metric.ddg[i][j][a][b] +=
                  first[d + 2] * first[e + 2] * v.metric[a][b] / (dx * dx);
        }
  }
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j) {
      chi.dd[j][i] = chi.dd[i][j];
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) metric.ddg[j][i][a][b] = metric.ddg[i][j][a][b];
    }
  double h[2];
  for (int sign : {1, -1}) {
    auto live = u;
    live.chi.value += sign * dt * chi.value;
    live.trace.value += sign * dt * base.trace;
    live.theta.value += sign * dt * base.theta;
    for (int i = 0; i < 3; ++i) {
      live.chi.d[i] += sign * dt * chi.d[i];
      for (int j = 0; j < 3; ++j) {
        live.chi.dd[i][j] += sign * dt * chi.dd[i][j];
        live.metric.g[i][j] += sign * dt * metric.g[i][j];
        live.a.k[i][j] += sign * dt * base.a[i][j];
        for (int d = 0; d < 3; ++d) {
          live.metric.dg[d][i][j] += sign * dt * metric.dg[d][i][j];
          for (int e = 0; e < 3; ++e)
            live.metric.ddg[d][e][i][j] += sign * dt * metric.ddg[d][e][i][j];
        }
      }
    }
    h[sign == 1 ? 0 : 1] =
        hyp::EvolvedConstraints(live, hyp::CartesianOmega(live, p)).hamiltonian;
  }
  return (h[0] - h[1]) / (2 * dt);
}

void Native(int n, double span, double a, bool layer, int degree, bool symmetric,
            double r0, double r1, double dt) {
  hyp::SphericalGhostGrid grid;
  grid.radius = 1;
  for (int k = 0; k < 3; ++k) {
    grid.n[k] = n + 6;
    grid.h[k] = span / n;
    grid.first[k] = -.5 * (n + 5) * grid.h[k];
  }
  hyp::LayerParameters lp;
  lp.enabled = layer;
  lp.r0 = r0;
  lp.r1 = r1;
  hyp::LayerGaugeParameters gauge;
  gauge.physical_trace_lapse = true;
  gauge.preferred_source = false;
  hyp::CartesianConformalPatch patch(grid, a, degree, lp, gauge, symmetric);
  patch.kappa1=10; patch.dissipation=.1;
  constexpr double probe_eps=1e-4;
  auto q = patch.Allocate("tangent state"), initial = patch.Allocate("initial pulse");
  auto rhs = patch.Allocate("initial native RHS");
  patch.InitializeReference(initial);
  auto reference_state=patch.Allocate("reference initial");
  auto minus_rhs=patch.Allocate("minus gauge RHS");
  Kokkos::deep_copy(reference_state,initial);
  const auto nodes = patch.active;
  Kokkos::parallel_for(
      "native finite nonspherical pulse", nodes.extent(0), KOKKOS_LAMBDA(const int p) {
        const int s = nodes(p), i = s % grid.n[0], j = s / grid.n[0] % grid.n[1];
        const int k = s / (grid.n[0] * grid.n[1]);
        const Real x = grid.first[0] + i * grid.h[0], y = grid.first[1] + j * grid.h[1];
        const Real z = grid.first[2] + k * grid.h[2], r2 = x * x + y * y + z * z;
        const Real shape = Kokkos::pow(1 - r2, 4) * Kokkos::exp(-4 * r2);
        initial(0, Z4c::I_Z4C_ALPHA, k, j, i) += probe_eps * .1 * shape * (1 + .2 * x + .3 * y * z);
        const Real vector[3] = {1 + .3 * y * z, .2 * x, .1 * x * y};
        for (int d = 0; d < 3; ++d) {
          initial(0, Z4c::I_Z4C_BETAX + d, k, j, i) += probe_eps * .02 * shape * vector[d];
        }
      });
  const auto first = patch.Diagnose(initial);
  if (std::max({first.h_l2, first.m_l2, first.z_l2}) > 1e-8) {
    throw std::runtime_error("initial finite gauge pulse has nonzero constraints");
  }
  Kokkos::deep_copy(q, initial);
  patch.RHS(q, rhs);
  Kokkos::parallel_for("minus gauge seed",q.size(),KOKKOS_LAMBDA(const int s) {
    q.data()[s]=2*reference_state.data()[s]-initial.data()[s];
  });
  patch.RHS(q,minus_rhs);
  Kokkos::parallel_for("actual centered gauge Jv",rhs.size(),KOKKOS_LAMBDA(const int s) {
    rhs.data()[s]=(rhs.data()[s]-minus_rhs.data()[s])/(2*probe_eps);
  });
  Kokkos::deep_copy(initial,reference_state);
  const auto ref = patch.reference;
  for (int sign : {1, -1}) {
    Kokkos::parallel_for(
        "signed initial tangent", q.size(), KOKKOS_LAMBDA(const int s) {
          q.data()[s] = initial.data()[s] + sign * dt * rhs.data()[s];
        });
    patch.ProjectAlgebraic(q);
    const auto diagnostics = patch.Diagnose(q);
    const auto dev = hyp::BindCartesianFields(patch.deviations);
    Kokkos::View<double**> point("initial constraint tangent", nodes.extent(0), 4);
    Kokkos::parallel_for(
        "constraint tangent radial budgets", nodes.extent(0), KOKKOS_LAMBDA(const int p) {
          const int s = nodes(p), i = s % grid.n[0], j = s / grid.n[0] % grid.n[1];
          const int k = s / (grid.n[0] * grid.n[1]);
          const Real x = grid.first[0] + i * grid.h[0], y = grid.first[1] + j * grid.h[1];
          const Real z = grid.first[2] + k * grid.h[2];
          const auto reference = ref.At(x, y, z);
          const Real idx[3] = {1 / grid.h[0], 1 / grid.h[1], 1 / grid.h[2]};
          auto u = hyp::LoadMeshJet<3>(dev, idx, 0, k, j, i);
          hyp::AddReferenceJet(u, reference, ref);
          const auto c = hyp::EvolvedConstraints(u, hyp::CartesianOmega(u, reference));
          point(p, 0) = Kokkos::sqrt(x * x + y * y + z * z);
          point(p, 1) = c.hamiltonian * c.hamiltonian / (dt * dt);
          point(p, 2) = c.momentum_conformal_norm2 / (dt * dt);
          point(p, 3) = c.z4.z_conformal_norm2 / (dt * dt);
        });
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), point);
    std::vector<double> edges{0, r0, r1, .9, 1};
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    const int n_bins = static_cast<int>(edges.size()) - 1;
    double sums[4][3]{}, totals[3]{};
    int counts[4]{};
    for (size_t p = 0; p < host.extent(0); ++p) {
      int bin = 0;
      while (bin < n_bins - 1 && host(p, 0) >= edges[bin + 1]) ++bin;
      ++counts[bin];
      for (int c = 0; c < 3; ++c) {
        sums[bin][c] += host(p, c + 1);
        totals[c] += host(p, c + 1);
      }
    }
    double max_m=0,max_z=0,m_radius=0,z_radius=0;
    for(size_t p=0;p<host.extent(0);++p){
      if(host(p,2)>max_m){max_m=host(p,2);m_radius=host(p,0);}
      if(host(p,3)>max_z){max_z=host(p,3);z_radius=host(p,0);}
    }
    std::cout << "{\"kind\":\"native\",\"n\":" << n << ",\"span\":" << span
              << ",\"a\":" << a << ",\"layer\":" << layer << ",\"degree\":" << degree
              << ",\"symmetric\":" << symmetric << ",\"r0\":" << r0 << ",\"r1\":" << r1
              << ",\"dt\":" << dt << ",\"sign\":" << sign
              << ",\"generator_eps\":" << probe_eps
              << ",\"Omega_min\":" << patch.min_omega
              << ",\"Mdot_max\":" << std::sqrt(max_m)
              << ",\"Zdot_max\":" << std::sqrt(max_z)
              << ",\"Mmax_r\":" << m_radius << ",\"Zmax_r\":" << z_radius
              << ",\"initial_H\":" << first.h_l2
              << ",\"Hdot_rms\":" << diagnostics.h_l2 / dt
              << ",\"Hdot_max\":" << diagnostics.max_h / dt
              << ",\"Hmax_r\":" << diagnostics.max_h_radius
              << ",\"Mdot_rms\":" << diagnostics.m_l2 / dt
              << ",\"Zdot_rms\":" << diagnostics.z_l2 / dt << ",\"radial_bins\":[";
    for (int b = 0; b < n_bins; ++b) {
      std::cout << (b ? "," : "") << "{\"r_min\":" << edges[b]
                << ",\"r_max\":" << edges[b + 1] << ",\"count\":" << counts[b]
                << ",\"squared_fractions\":[";
      for (int c = 0; c < 3; ++c) {
        std::cout << (c ? "," : "") << (totals[c] ? sums[b][c] / totals[c] : 0);
      }
      std::cout << "]}";
    }
    std::cout << "]}\n";
  }
}

void Continuum(double a, bool layer, double r0, double r1, double dx, double dt) {
  hyp::LayerParameters lp;
  lp.enabled = layer;
  lp.r0 = r0;
  lp.r1 = r1;
  const hyp::LayerReference<Real> ref(1, a, lp);
  for (double x : {.1, .25, .35, .4, .55, .7, .8, .95}) {
    const double xyz[3] = {x, .03, .07};
    const double hdot = ContinuumHDot(ref, xyz, dx, dt);
    if (!std::isfinite(hdot) || std::abs(hdot) > 2e-4) {
      throw std::runtime_error("continuum Hamiltonian tangent check failed");
    }
    std::cout << "{\"kind\":\"continuum\",\"a\":" << a << ",\"layer\":" << layer
              << ",\"r0\":" << r0 << ",\"r1\":" << r1 << ",\"x\":" << x
              << ",\"dx\":" << dx << ",\"dt\":" << dt << ",\"Hdot\":" << hdot << "}\n";
  }
}

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    std::cout << std::setprecision(17);
    if (argc == 1) {
      for (double a : {1., .5})
        for (bool layer : {false, true}) {
          for (double dx : {.0004, .0002}) Continuum(a, layer, .35, .75, dx, 1e-6);
        }
      Native(24, 2.1, .5, true, 2, true, .35, .75, 1e-6);
      Native(24, 2.1, .5, true, 3, true, .35, .75, 1e-6);
    } else if (std::string(argv[1]) == "--native" && argc == 11) {
      Native(std::stoi(argv[2]), std::stod(argv[3]), std::stod(argv[4]),
             std::stoi(argv[5]), std::stoi(argv[6]), std::stoi(argv[7]),
             std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]));
    } else if (std::string(argv[1]) == "--continuum" && argc == 8) {
      Continuum(std::stod(argv[2]), std::stoi(argv[3]), std::stod(argv[4]),
                std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7]));
    } else {
      throw std::invalid_argument(
          "expected --native N span a layer degree symmetric r0 r1 dt or "
          "--continuum a layer r0 r1 dx dt");
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
