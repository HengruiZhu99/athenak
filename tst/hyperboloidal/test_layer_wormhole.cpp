// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "z4c/hyperboloidal/cartesian_patch.hpp"
#include "z4c/hyperboloidal/layer_wormhole.hpp"

namespace hyp = z4c::hyperboloidal;
void Close(double x, double y, double tolerance, const char* label) {
  if (!std::isfinite(x) || !std::isfinite(y) ||
      std::abs(x - y) > tolerance * std::max({1., std::abs(x), std::abs(y)})) {
    std::cerr << label << ": " << x << " != " << y << '\n';
    throw std::runtime_error(label);
  }
}

void ConstraintAndADMAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, 0.35, 0.75});
  double max_h = 0, max_m = 0, max_adm = 0;
  for (double mass : {.05, .2, .6}) {
    const hyp::LayerWormhole<double> data(ref, mass);
    for (double r : {1e-6, .01, .025, .1, .3, .35, .35000001, .4, .5, .6, .7, .74999999,
                     .75, .9, .99, .9999}) {
      const auto p = ref.At(r, 0., 0.);
      const auto u = data.At(r, 0., 0.);
      const double m = mass * p.omega / (2 * r), psi = 1 + m;
      const double factor = std::pow(psi, 4), nstatic = (1 - m) / psi;
      const auto physical = hyp::ToPhysicalADM(u, p.omega);
      if (!physical.valid || !(u.alpha.value > 0) || !(u.chi.value > 0)) {
        throw std::runtime_error("invalid positive wormhole lapse/ADM");
      }
      Close(u.chi.value, p.state.chi.value / factor, 1e-13, "mass conformal factor");
      const auto c = hyp::EvolvedConstraints(u, hyp::CartesianOmega(u, p));
      if (!c.valid)
        throw std::runtime_error("invalid wormhole constraints");
      max_h = std::max(max_h, std::abs(c.hamiltonian));
      max_m = std::max(max_m, std::sqrt(c.momentum_conformal_norm2));
      Close(c.hamiltonian, 0, 3e-9, "wormhole H");
      Close(std::sqrt(c.momentum_conformal_norm2), 0, 3e-9, "wormhole M");
      Close(c.z4.determinant_residual, 0, 5e-13, "wormhole determinant");
      Close(c.z4.tracefree_residual, 0, 2e-12, "wormhole trace-free A");
      // Independent geometric metric reconstruction from the static isotropic
      // Schwarzschild spacetime and the mass-corrected reference height.
      const double boost = p.b / p.alpha;
      const double radial_map = p.L / (p.omega * p.omega);
      const double expected_radial =
          factor * (1 - boost * boost) * radial_map * radial_map;
      const double expected_tangential = factor / (p.omega * p.omega);
      // Near scri use the equivalent factored 1-v^2=Omega^2/alpha_ref^2.
      const double radial_factored =
          factor * p.L * p.L / (p.omega * p.omega * p.alpha * p.alpha);
      Close(physical.metric[0][0], radial_factored, 3e-13, "physical radial metric");
      if (p.omega > .01) {
        Close(expected_radial, radial_factored, 2e-12, "height metric derivation");
      }
      Close(physical.metric[1][1], expected_tangential, 3e-13,
            "physical tangential metric");
      double kr = 0, kt = 0;
      if (r > ref.layer.r0) {
        // K_ij=Lie_beta(gamma_ij)/(2 alpha_geom), with the static height's
        // lapse. The freely chosen positive initial lapse need not equal
        // alpha_geom.
        const double ageo = nstatic * p.alpha / p.omega;
        const double log_radial = u.metric.dg[0][0][0] / u.metric.g[0][0] -
                                  u.chi.d[0] / u.chi.value - 2 * p.domega[0] / p.omega;
        const double log_tangential = 2 / r + u.metric.dg[0][1][1] / u.metric.g[1][1] -
                                      u.chi.d[0] / u.chi.value -
                                      2 * p.domega[0] / p.omega;
        kr = (u.beta.d[0][0] + .5 * u.beta.value[0] * log_radial) / ageo;
        kt = .5 * u.beta.value[0] * log_tangential / ageo;
      }
      const double ar = physical.curvature[0][0] / physical.metric[0][0];
      const double at = physical.curvature[1][1] / physical.metric[1][1];
      max_adm = std::max({max_adm, std::abs(ar - kr), std::abs(at - kt)});
      Close(ar, kr, 3e-11, "radial ADM curvature from stationary metric");
      Close(at, kt, 3e-11, "tangential ADM curvature from stationary metric");
      if (p.omega > .05 && r >= .01) {
        const double areal = r * std::sqrt(physical.metric[1][1]);
        const double areal_d =
            areal * (1 / r + .5 * u.metric.dg[0][1][1] / u.metric.g[1][1] -
                     .5 * u.chi.d[0] / u.chi.value - p.domega[0] / p.omega);
        const double mass_invariant =
            .5 * areal *
            (1 + areal * areal * at * at - areal_d * areal_d / physical.metric[0][0]);
        Close(mass_invariant, mass, 2e-11, "Schwarzschild mass invariant");
      }
      for (const auto& direction :
           {std::array<double, 3>{1, 0, 0}, std::array<double, 3>{.36, -.48, .8}}) {
        const auto po = ref.At(r * direction[0], r * direction[1], r * direction[2]);
        const auto uo = data.At(r * direction[0], r * direction[1], r * direction[2]);
        const auto co = hyp::EvolvedConstraints(uo, hyp::CartesianOmega(uo, po));
        if (!co.valid)
          throw std::runtime_error("invalid oblique wormhole constraints");
        Close(co.hamiltonian, 0, 3e-9, "oblique wormhole H");
        Close(std::sqrt(co.momentum_conformal_norm2), 0, 3e-9, "oblique wormhole M");
        for (int i = 0; i < 3; ++i) {
          Close(uo.lambda.value[i], po.state.lambda.value[i], 0, "unchanged Lambda");
          for (int j = 0; j < 3; ++j) {
            Close(uo.metric.g[i][j], po.state.metric.g[i][j], 0, "unchanged metric");
          }
        }
      }
    }
  }
  std::cout << "PASS constraint-satisfying wormhole jets: H=" << max_h << " M=" << max_m
            << " independently reconstructed ADM curvature error=" << max_adm << '\n';
}

void DerivativeAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, .35, .75});
  const hyp::LayerWormhole<double> data(ref, .2);
  const double h = 1e-5;
  for (double r : {.01, .1 - 1e-7, .1, .1 + 1e-7, .35, .4, .5, .6, .7, .75, .9, .99}) {
    const std::array<double, 3> x{.36 * r, -.48 * r, .8 * r};
    const auto u = data.At(x[0], x[1], x[2]);
    for (int d = 0; d < 3; ++d) {
      auto xp = x, xm = x;
      xp[d] += h;
      xm[d] -= h;
      const auto plus = data.At(xp[0], xp[1], xp[2]);
      const auto minus = data.At(xm[0], xm[1], xm[2]);
      const hyp::ScalarJet<double> center[] = {u.alpha, u.chi, u.trace};
      const hyp::ScalarJet<double> pos[] = {plus.alpha, plus.chi, plus.trace};
      const hyp::ScalarJet<double> neg[] = {minus.alpha, minus.chi, minus.trace};
      for (int f = 0; f < 3; ++f) {
        Close((pos[f].value - neg[f].value) / (2 * h), center[f].d[d], 4e-6,
              "scalar first derivative");
        for (int e = 0; e < 3; ++e) {
          Close((pos[f].d[e] - neg[f].d[e]) / (2 * h), center[f].dd[d][e], 2e-5,
                "scalar Hessian");
        }
      }
      for (int i = 0; i < 3; ++i) {
        Close((plus.beta.value[i] - minus.beta.value[i]) / (2 * h), u.beta.d[d][i], 4e-6,
              "shift first derivative");
        for (int e = 0; e < 3; ++e) {
          Close((plus.beta.d[e][i] - minus.beta.d[e][i]) / (2 * h), u.beta.dd[d][e][i],
                2e-5, "shift Hessian");
        }
        for (int j = 0; j < 3; ++j) {
          Close((plus.a.k[i][j] - minus.a.k[i][j]) / (2 * h), u.a.dk[d][i][j], 2e-5,
                "A first derivative");
        }
      }
    }
  }
  std::cout << "PASS wormhole analytic Cartesian derivatives, endpoints and throat\n";
}

void LimitAndValidationAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, .35, .75});
  const hyp::LayerWormhole<double> data(ref, .2), small(ref, 1e-12);
  for (double r : {.01, .1, .35, .5, .75, .9, .9999}) {
    const auto p = ref.At(r, 0., 0.);
    const auto u = small.At(r, 0., 0.);
    Close(u.alpha.value, p.alpha, 1e-9, "mass zero lapse limit");
    Close(u.chi.value, p.state.chi.value, 1e-9, "mass zero chi limit");
    Close(u.trace.value, p.k_physical, 1e-9, "mass zero physical trace limit");
    for (int i = 0; i < 3; ++i) {
      Close(u.beta.value[i], p.state.beta.value[i], 1e-9, "mass zero shift limit");
      for (int j = 0; j < 3; ++j) {
        Close(u.a.k[i][j], p.state.a.k[i][j], 1e-9, "mass zero A limit");
      }
    }
  }
  for (int power = 2; power <= 15; ++power) {
    const double r = 1 - std::pow(10., -power);
    const auto p = ref.At(r, 0., 0.);
    const auto u = data.At(r, 0., 0.);
    const auto c = hyp::EvolvedConstraints(u, hyp::CartesianOmega(u, p));
    if (!c.valid)
      throw std::runtime_error("invalid small Omega wormhole constraints");
    Close(c.hamiltonian, 0, 1e-12, "small Omega H");
    Close(std::sqrt(c.momentum_conformal_norm2), 0, 1e-11, "small Omega M");
    for (double v : {u.alpha.value, u.alpha.d[0], u.alpha.dd[0][0], u.chi.value,
                     u.chi.d[0], u.chi.dd[0][0], u.trace.value, u.trace.d[0],
                     u.trace.dd[0][0], u.a.k[0][0], u.a.dk[0][0][0]}) {
      if (!std::isfinite(v))
        throw std::runtime_error("nonfinite small Omega jet");
    }
  }
  const auto scri = data.At(1., 0., 0.);
  Close(scri.a.k[0][0], -4 * data.mass() / 3, 1e-14, "factored mass shear at scri");
  Close(scri.a.k[1][1], 2 * data.mass() / 3, 1e-14, "tangential mass shear at scri");
  Close(scri.trace.value, -3, 0, "physical trace at scri");
  const auto origin = data.At(0., 0., 0.);
  Close(origin.chi.value, 0, 0, "puncture chi limit");
  Close(origin.alpha.value, 0, 0, "puncture lapse limit");
  Close(origin.alpha.dd[0][0], 8 / (.2 * .2), 0, "puncture lapse Hessian");
  if (hyp::ToPhysicalADM(origin, 1).valid) {
    throw std::runtime_error("exact puncture cannot be a live ADM point");
  }
  for (double mass : {0., -.1, .7, 1., std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected = false;
    try {
      const hyp::LayerWormhole<double> invalid(ref, mass);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("invalid wormhole mass was accepted");
  }
  bool rejected = false;
  try {
    const hyp::LayerWormhole<double> invalid(hyp::LayerReference<double>{1, 1}, .2);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("disabled layer wormhole was accepted");
  std::cout << "PASS mass-zero, puncture and finite-scri limits; invalid data "
               "rejected\n";
}

void GenericReferenceAudit() {
  const hyp::LayerReference<double> ref(2, 3.5, {true, .55, 1.5});
  const hyp::LayerWormhole<double> data(ref, .4);
  for (double r : {.1, .2, .55, .7, 1., 1.3, 1.5, 1.9, 1.9999}) {
    const auto p = ref.At(.36 * r, -.48 * r, .8 * r);
    const auto u = data.At(.36 * r, -.48 * r, .8 * r);
    const auto c = hyp::EvolvedConstraints(u, hyp::CartesianOmega(u, p));
    if (!c.valid)
      throw std::runtime_error("invalid generic reference constraints");
    Close(c.hamiltonian, 0, 1e-10, "generic reference H");
    Close(std::sqrt(c.momentum_conformal_norm2), 0, 1e-10, "generic reference M");
  }
  const auto scri = data.At(2., 0., 0.);
  Close(scri.trace.value, -3 / 3.5, 1e-14, "generic reference scri P");
  Close(scri.a.k[0][0], -4 * data.mass() / (3 * 3.5 * 2), 1e-14,
        "generic reference scri A");
  std::cout << "PASS changed scri radius, curvature scale and layer width\n";
}

void OuterGeometricKernelAudit() {
  const hyp::LayerReference<double> ref(1, 1, {true, .35, .75});
  const hyp::LayerWormhole<double> data(ref, .2);
  double maximum = 0;
  for (double r : {.75, .8, .9, .99, .9999}) {
    const auto p = ref.At(.36 * r, -.48 * r, .8 * r);
    const auto u = data.At(.36 * r, -.48 * r, .8 * r);
    hyp::Z4cRHS<double> rhs{};
    if (!hyp::AssembleInterior(
            hyp::ConformalRHS(u, hyp::CartesianOmega(u, p), 5 / u.alpha.value, 0.),
            p.omega, rhs)) {
      throw std::runtime_error("invalid outer height geometric RHS");
    }
    for (double value : {rhs.chi, rhs.trace, rhs.theta})
      maximum = std::max(maximum, std::abs(value));
    for (int i = 0; i < 3; ++i) {
      maximum = std::max(maximum, std::abs(rhs.lambda[i]));
      for (int j = 0; j < 3; ++j) {
        maximum = std::max(maximum, std::abs(rhs.metric[i][j]));
        maximum = std::max(maximum, std::abs(rhs.a[i][j]));
      }
    }
  }
  Close(maximum, 0, 1e-8, "static outer geometry kernel");
  std::cout << "PASS stationary outer height geometry kernel: max=" << maximum
            << "; the Minkowski reference gauge is independently dynamical\n";
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    ConstraintAndADMAudit();
    DerivativeAudit();
    LimitAndValidationAudit();
    GenericReferenceAudit();
    OuterGeometricKernelAudit();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    status = 1;
  }
  Kokkos::finalize();
  return status;
}
