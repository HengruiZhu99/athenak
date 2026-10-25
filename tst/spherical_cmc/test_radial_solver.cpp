// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE").
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "spherical_cmc/radial_solver.hpp"

void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
double Error(int n, bool linear, bool scri_scaled = false) {
  spherical_cmc::RadialSolver q(n, linear ? 1 : 0);
  const double amplitude = linear ? 1e-5 : 0.1;
  for (int i = 1; i <= n + 1; ++i) {
    const double r = q.r[i];
    q.a[i] = std::exp(amplitude * r * r * (scri_scaled ? 1 : std::pow(1 - r * r, 2)));
  }
  q.Solve();
  q.Diagnostics();
  Check(q.chi[n + 1] == q.a[n + 1] && q.beta[n + 1] == -q.alpha[n + 1], "scri boundary");
  double err = 0;
  for (int i = 1; i <= n; ++i) {
    if (linear) {
      const double r = q.r[i];
      const double exact = -amplitude * r * r * std::pow(1 - r * r, 2);
      err = std::max(err, std::abs(q.rhs[i] - exact));
    } else {
      Check(q.rhs[i] == 0, "eta=0 stationary coordinate metric");
      err = std::max(err, std::abs(q.hamiltonian[i]));
    }
  }
  return err;
}
double BlackHoleError(int n, bool evolving) {
  spherical_cmc::RadialSolver q(n, evolving ? 1 : 0, 0.5, 0.05);
  for (int i = 0; i <= n + 1; ++i) {
    const double r = q.r[i];
    q.a[i] = std::exp(0.05 * r * r * std::pow(1 - r * r, 2));
  }
  q.Solve();
  q.Diagnostics();
  spherical_cmc::RadialSolver plus(n, 1, 0.5, 0.05), minus(n, 1, 0.5, 0.05);
  const double step = 1e-4;
  if (evolving) {
    for (int i = 0; i <= n + 1; ++i) {
      plus.a[i] = q.a[i] + step * q.rhs[i];
      minus.a[i] = q.a[i] - step * q.rhs[i];
    }
    plus.Solve();
    minus.Solve();
  }
  double err = 0;
  for (int i = 1; i <= n; ++i) {
    if (evolving) {
      // Independent tangent of the constraint projection versus the full tensor RHS.
      const double adot = (plus.arr[i] - minus.arr[i]) / (2 * step);
      err = std::max(err, std::abs(adot - q.a_rhs[i]));
    } else {
      err = std::max(err, std::abs(q.hamiltonian[i]));
      Check(q.rhs[i] == 0, "stationary trumpet metric");
    }
  }
  return err;
}
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  try {
    for (int n : {16, 32, 64, 128}) {
      spherical_cmc::RadialSolver q(n, 1);
      q.Solve();
      q.Diagnostics();
      for (int i = 1; i <= n; ++i) {
        Check(q.chi[i] == 1 && q.rhs[i] == 0, "Minkowski constraint projection");
        Check(std::abs(q.alpha[i] - (1 + q.r[i] * q.r[i]) / 2) < 1e-14,
              "reference lapse");
        Check(std::abs(q.hamiltonian[i]) < 1e-10, "reference Hamiltonian");
        Check(std::abs(q.cmc[i]) < 1e-9, "reference CMC");
      }
    }
    for (bool linear : {false, true}) {
      const double coarse = Error(32, linear), fine = Error(64, linear);
      std::cout << (linear ? "linear metric RHS" : "Hamiltonian") << " N32=" << coarse
                << " N64=" << fine << " ratio=" << coarse / fine << "\n";
      Check(coarse / fine > 8, "radial refinement convergence");
    }
    const double scaled_coarse = Error(32, false, true);
    const double scaled_fine = Error(64, false, true);
    Check(scaled_coarse / scaled_fine > 8, "nonunit scri metric convergence");
    for (bool evolving : {false, true}) {
      const double coarse = BlackHoleError(256, evolving),
                   fine = BlackHoleError(512, evolving);
      std::cout << (evolving ? "BH A evolution defect" : "BH Hamiltonian")
                << " N256=" << coarse << " N512=" << fine << " ratio=" << coarse / fine
                << "\n";
      Check(coarse / fine > 5, "black-hole refinement convergence");
    }
    spherical_cmc::RadialSolver q(16, 1);
    q.a[3] = std::numeric_limits<double>::quiet_NaN();
    bool caught = false;
    try {
      q.Solve();
    } catch (const std::runtime_error &) {
      caught = true;
    }
    Check(caught, "nonfinite metric rejection");
  } catch (...) {
    Kokkos::finalize();
    throw;
  }
  Kokkos::finalize();
}
