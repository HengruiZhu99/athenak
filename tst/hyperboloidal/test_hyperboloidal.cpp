// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "z4c/hyperboloidal/reference_gauge.hpp"
#include "z4c/hyperboloidal/radial_sbp.hpp"

namespace hyp = z4c::hyperboloidal;

void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void Near(double actual, double expected, double tolerance, const char *message) {
  Require(std::isfinite(actual) && std::isfinite(expected) &&
          std::abs(actual-expected) <= tolerance*(1+std::abs(expected)), message);
}
template <typename F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  Require(rejected, "invalid input was accepted");
}

template <typename T>
void DeviceReference() {
  const hyp::CMCReference<T> ref{T(2), T(3)};
  ref.Validate();
  // Exercise the same kernel interface required by AthenaK, also in float.
  int failures = 0;
  Kokkos::parallel_reduce("hyperboloidal reference", 1001,
      KOKKOS_LAMBDA(const int i, int &sum) {
    const T x = ref.scri_radius*T(i)/T(1000);
    const auto p = ref.At(x, T(0), T(0));
    const T kphys = p.omega*p.k_bar - 3*p.beta[0]*p.domega[0]/p.alpha;
    const T tol = T(100)*std::numeric_limits<T>::epsilon();
    if (!(Kokkos::abs(kphys-p.k_physical) < tol)) ++sum;
    if (!(Kokkos::abs(p.alpha*p.alpha-x*x/T(9)-p.omega*p.omega) < tol)) ++sum;
    hyp::GaugeDeviation<T> d{};
    d.chi = 1;
    const hyp::GaugeParameters<T> g{T(1), T(1), T(1.5), T(1)};
    const auto rhs = hyp::ReferenceGauge(ref, p, d, g);
    if (rhs.alpha != 0 || rhs.beta[0] != 0 || rhs.beta[1] != 0 || rhs.beta[2] != 0) ++sum;
    if (!(p.alpha > 0) || !(p.einstein_nn > 0)) ++sum;
  }, failures);
  Require(failures == 0, "device reference/fixed-point identity failed");
}

void ReferenceAndGauge() {
  std::mt19937 rng(1729);
  std::uniform_real_distribution<double> u(-1, 1);
  for (double s : {0.7, 1.0, 2.0}) {
    for (double a : {0.5, 1.0, 3.0}) {
      const hyp::CMCReference<double> ref{s, a};
      ref.Validate();
      for (int sample = 0; sample < 200; ++sample) {
        const double x[3] = {0.5*s*u(rng), 0.5*s*u(rng), 0.5*s*u(rng)};
        const auto p = ref.At(x[0], x[1], x[2]);
        const double r2 = x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
        const double r = std::sqrt(r2), radius = r/p.omega;
        // Independently recover the lapse and radial metric from R=r/omega,
        // T=t+sqrt(R^2+a^2). This catches inverted boost/lapse conventions.
        Near(p.alpha, p.omega*std::sqrt(1+radius*radius/(a*a)), 2e-13,
             "physical-to-conformal lapse map");
        const double dradius = (p.omega+r2/(a*s))/(p.omega*p.omega);
        Near(p.omega*p.omega*a*a/(radius*radius+a*a)*dradius*dradius,
             1, 2e-13, "radial metric map");
        Near(ref.OutgoingSpeed(r), r/a+p.alpha, 2e-13, "outgoing speed");
        Near(ref.IngoingSpeed(r), r/a-p.alpha, 2e-13, "ingoing speed");
        // Reference ADM Hamiltonian/momentum with conformal Einstein sources.
        Near(2*p.k_bar*p.k_bar/3-2*p.einstein_nn, 0, 2e-13, "Hamiltonian");
        const double k = p.k_bar/3;
        double lie_k = -2*k/a;
        for (int j = 0; j < 3; ++j) {
          lie_k += p.beta[j]*p.dalpha[j]/(a*p.alpha*p.alpha);
          Near(-2*p.dalpha[j]/(a*p.alpha*p.alpha)-p.einstein_momentum[j],
               0, 2e-13, "momentum");
        }
        const double stress = p.einstein_stress_diagonal;
        Near(-p.hessian_alpha+p.alpha*k*k+lie_k
             -0.5*p.alpha*(p.einstein_nn-stress), 0, 2e-12, "stationary Kij");

        hyp::GaugeDeviation<double> d{};
        d.lapse = 0.01*u(rng);
        d.trace = 0.01*u(rng);
        d.chi = 1+0.01*u(rng);
        for (int j = 0; j < 3; ++j) {
          d.shift[j] = 0.01*u(rng);
          d.lambda[j] = 0.01*u(rng);
          d.dlapse[j] = 0.01*u(rng);
          for (int l = 0; l < 3; ++l) d.dshift[j][l] = 0.01*u(rng);
        }
        const hyp::GaugeParameters<double> g{0.3, 0.2, 1.5, 1};
        g.Validate();
        const auto rhs = hyp::ReferenceGauge(ref, p, d, g);
        const double alpha = p.alpha+p.omega*d.lapse;
        const double radial = s*s-r2;
        double direct = -(alpha*alpha+g.slicing*radial*radial)*d.trace
            +g.lapse_damping*(p.alpha*p.alpha-alpha*alpha)/p.omega;
        for (int j = 0; j < 3; ++j) {
          const double beta = p.beta[j]+p.omega*d.shift[j];
          direct += beta*(p.dalpha[j]+p.domega[j]*d.lapse+p.omega*d.dlapse[j])
              -p.beta[j]*p.dalpha[j]
              +p.domega[j]*(p.beta[j]*p.alpha-beta*alpha)/p.omega;
        }
        Near(rhs.alpha, direct, 2e-12, "factored lapse differs from unfactored gauge");
        for (int j = 0; j < 3; ++j) {
          const double beta = p.beta[j]+p.omega*d.shift[j];
          double direct_shift = (g.shift_driver*radial*radial+0.75*alpha*alpha*d.chi)
              *d.lambda[j]+g.shift_damping*(p.beta[j]-beta);
          for (int l = 0; l < 3; ++l) {
            const double dbref = j == l ? -1/a : 0;
            direct_shift += (p.beta[l]+p.omega*d.shift[l])
                *(dbref+p.domega[l]*d.shift[j]+p.omega*d.dshift[j][l])
                -p.beta[l]*dbref;
          }
          Near(rhs.beta[j], direct_shift, 2e-12, "factored shift differs");
        }
      }
      // Finite limits are necessary, but not enough for falloff preservation.
      const auto p = ref.At(s, 0., 0.);
      hyp::GaugeDeviation<double> d{};
      d.lapse = 0.1;
      d.shift[0] = 0.2;
      d.chi = 1;
      const hyp::GaugeParameters<double> g{1, 1, 1.5, 1};
      auto rhs = hyp::ReferenceGauge(ref, p, d, g);
      Require(std::abs(rhs.alpha) > 1e-3, "must detect incompatible scri data");
      d.trace = -p.domega[0]*d.shift[0]/p.alpha-2*g.lapse_damping*d.lapse/p.alpha;
      for (int j = 0; j < 3; ++j) {
        d.lambda[j] = -4*d.shift[j]/(3*a*p.alpha*d.chi);
      }
      rhs = hyp::ReferenceGauge(ref, p, d, g);
      Near(rhs.alpha, 0, 2e-13, "lapse scri compatibility");
      for (double v : rhs.beta) Near(v, 0, 2e-13, "shift scri compatibility");
      for (double epsilon : {1e-3, 1e-6, 1e-9, 1e-12, 0.0}) {
        const auto near = ref.At(s*(1-epsilon), 0., 0.);
        const auto limit = hyp::ReferenceGauge(ref, near, d, g);
        Require(std::isfinite(limit.alpha), "nonfinite lapse near scri");
        Near(limit.alpha, 0, 20*epsilon+1e-13, "lapse does not approach scri limit");
      }
    }
  }
  for (double bad : {0., -1., std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
    Reject([=] { hyp::CMCReference<double>{bad, 1}.Validate(); });
    Reject([=] { hyp::CMCReference<double>{1, bad}.Validate(); });
  }
  Reject([] { hyp::GaugeParameters<double>{-1, 1, 1, 1}.Validate(); });
  Reject([] { hyp::RadialSBP({1, 1}, 0, 10); });
  Reject([] { hyp::RadialSBP({1, 1}, 1, 10); });
  Reject([] { hyp::RadialSBP({1, 1}, 0.2, 1); });
  Reject([] { hyp::RadialSBP({1, 1}, 0.2, 10).RHS({1, 2}, 0); });
}

void BoundaryGeometry() {
  const hyp::CMCReference<double> ref{1, 1};
  const double inner_lo[3] = {-0.1, -0.1, -0.1}, inner_hi[3] = {0.1, 0.1, 0.1};
  const double outer_lo[3] = {1.1, 0, 0}, outer_hi[3] = {1.2, 0.1, 0.1};
  const double cut_lo[3] = {0.9, -0.1, -0.1}, cut_hi[3] = {1.1, 0.1, 0.1};
  Require(ref.ClassifyBox(inner_lo, inner_hi) == -1, "interior box");
  Require(ref.ClassifyBox(outer_lo, outer_hi) == 1, "exterior box");
  Require(ref.ClassifyBox(cut_lo, cut_hi) == 0, "cut box");
  for (int i = -4; i <= 4; ++i) {
    if (i != 0) {
      Require(ref.At(1., i*0.01, 0.).omega < 0, "tangential stencil obstruction");
    }
  }
  const double x = 1/std::sqrt(3.);
  const auto p = ref.At(x, x, x);
  Near(ref.IngoingSpeed(1), 0, 0, "true scri tangent characteristic");
  Require(-p.beta[0]-p.alpha < -0.4, "staircase must detect incoming face mode");
  // A superluminal gauge mode remains incoming even on the true sphere.
  Require(1-std::sqrt(2.) < 0, "superluminal gauge counterexample");
}

template <typename Inflow>
void RK4(const hyp::RadialSBP &grid, std::vector<double> &q, double t, double dt,
         Inflow inflow) {
  auto k1 = grid.RHS(q, inflow(t));
  auto stage = q;
  for (int i = 0; i < grid.Size(); ++i) stage[i] = q[i]+0.5*dt*k1[i];
  auto k2 = grid.RHS(stage, inflow(t+0.5*dt));
  for (int i = 0; i < grid.Size(); ++i) stage[i] = q[i]+0.5*dt*k2[i];
  auto k3 = grid.RHS(stage, inflow(t+0.5*dt));
  for (int i = 0; i < grid.Size(); ++i) stage[i] = q[i]+dt*k3[i];
  auto k4 = grid.RHS(stage, inflow(t+dt));
  for (int i = 0; i < grid.Size(); ++i) {
    q[i] += dt*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6;
  }
}

double Pulse(double retarded_time) {
  const double v = (retarded_time-0.45)/0.09;
  return std::exp(-v*v);
}

void Transport() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> u(-1, 1);
  // Verify the SBP/SAT energy identity on arbitrary data, not just smooth waves.
  for (int n : {2, 17, 80}) {
    const hyp::RadialSBP grid({1, 1}, 0.2, n);
    for (int trial = 0; trial < 40; ++trial) {
      std::vector<double> q(grid.Size());
      for (auto &v : q) v = u(rng);
      const double inflow = u(rng);
      Near(grid.EnergyProduct(q, grid.RHS(q, inflow)),
           -0.5*q.back()*q.back()-0.5*q[0]*q[0]+q[0]*inflow,
           1e-12, "SBP energy identity");
    }
  }
  double previous_error = 0;
  for (int n : {160, 320, 640}) {
    const hyp::RadialSBP grid({1, 1}, 0.2, n);
    std::vector<double> q(grid.Size());
    for (int i = 0; i < grid.Size(); ++i) {
      const double r = grid.Radius(i);
      q[i] = Pulse((1-r)/(1+r));
    }
    double t = 0, scri_error = 0;
    const auto inflow = [](double time) { return Pulse(time+2./3.); };
    while (t < 1.2) {
      const double dt = std::min(0.2*grid.Spacing()/grid.MaxSpeed(), 1.2-t);
      RK4(grid, q, t, dt, inflow);
      t += dt;
      scri_error = std::max(scri_error, std::abs(q.back()-Pulse(t)));
    }
    Require(std::isfinite(scri_error) && scri_error < 0.025, "pulse at scri inaccurate");
    if (previous_error > 0) Require(previous_error/scri_error > 3.3, "scri convergence");
    previous_error = scri_error;
    const double remaining = 0.5*grid.EnergyProduct(q, q);
    Require(remaining < 1e-6, "pulse did not exit shell");
    std::cout << "pulse intervals=" << n << " max_scri_error=" << scri_error
              << " final_energy=" << remaining << '\n';
  }
  // Seeded arbitrary perturbations for 20 shell crossing times, homogeneous inflow.
  const hyp::RadialSBP grid({1, 1}, 0.2, 80);
  std::vector<double> q(grid.Size());
  for (auto &v : q) v = u(rng);
  double energy = 0.5*grid.EnergyProduct(q, q);
  const double initial = energy;
  double t = 0;
  while (t < 20*(2./3.)) {
    const double dt = 0.2*grid.Spacing()/grid.MaxSpeed();
    RK4(grid, q, t, dt, [](double) { return 0.; });
    t += dt;
    const double next = 0.5*grid.EnergyProduct(q, q);
    Require(std::isfinite(next) && next <= energy+1e-13, "random-data energy growth");
    energy = next;
  }
  Require(energy < initial*0.01, "random-data energy failed to leave shell");
  std::cout << "random-data initial_energy=" << initial
            << " final_energy=" << energy << '\n';
}

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  try {
    DeviceReference<double>();
    DeviceReference<float>();
    ReferenceAndGauge();
    BoundaryGeometry();
    Transport();
    std::cout << "PASS hyperboloidal building blocks (not nonlinear Z4c evolution)\n";
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
