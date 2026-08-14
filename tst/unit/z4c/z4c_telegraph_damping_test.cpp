//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_telegraph_damping_test.cpp
//! \brief Unit/manufactured-point checks for local telegrapher damping scales.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "z4c/telegraph_damping.hpp"

namespace {

bool NearlyEqual(const Real a, const Real b, const Real tolerance = 2.0e-13) {
  return std::abs(a - b) <=
         tolerance * std::max({Real(1.0), std::abs(a), std::abs(b)});
}

bool CheckScaleInvariantParameterization() {
  constexpr Real mu = 0.7;
  constexpr Real max_k = 0.2;
  constexpr Real tau = 4.0;
  constexpr Real kappa = 4.0;
  const auto coefficients =
      z4c::ScaleInvariantTelegraphCoefficients(mu, max_k, tau, kappa);
  const Real Q = mu / max_k;
  const Real tau_eff = tau / max_k;
  const Real kappa_eff = kappa / max_k;
  if (!NearlyEqual(coefficients.damping, Q / tau_eff) ||
      !NearlyEqual(coefficients.gradient, kappa_eff / tau_eff)) return false;
  const auto zero_scale =
      z4c::ScaleInvariantTelegraphCoefficients(0.0, 0.0, tau, kappa);
  return zero_scale.damping == 0.0 &&
         NearlyEqual(zero_scale.gradient, kappa / tau);
}

bool CheckLocalK() {
  return z4c::LocalAbsKTelegraphMu(-0.75) == 0.75 &&
         z4c::LocalAbsKTelegraphMu(0.0) == 0.0;
}

bool CheckExtrinsicNormAgainstPhysicalReconstruction() {
  // det(gtilde)=1 and gtilde^ij=diag(1/2,2,1).
  constexpr Real chi = 0.25;
  constexpr Real K = 0.6;
  const Real gtilde[3][3] = {{2.0, 0.0, 0.0},
                             {0.0, 0.5, 0.0},
                             {0.0, 0.0, 1.0}};
  const Real gu[3][3] = {{0.5, 0.0, 0.0},
                         {0.0, 2.0, 0.0},
                         {0.0, 0.0, 1.0}};
  // Conformally trace-free: gu^ij Atilde_ij=0.1-0.1+0.
  const Real Atilde[3][3] = {{0.2, 0.03, -0.02},
                             {0.03, -0.05, 0.04},
                             {-0.02, 0.04, 0.0}};
  Real gamma[3][3] = {};
  Real gamma_inv[3][3] = {};
  Real Kphys[3][3] = {};
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      gamma[a][b] = gtilde[a][b] / chi;
      gamma_inv[a][b] = chi * gu[a][b];
      Kphys[a][b] = Atilde[a][b] / chi + K * gamma[a][b] / 3.0;
    }
  }
  Real physical_norm_squared = 0.0;
  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          physical_norm_squared += gamma_inv[a][c] * gamma_inv[b][d] *
                                   Kphys[a][b] * Kphys[c][d];
        }
      }
    }
  }
  const Real actual = z4c::LocalExtrinsicCurvatureNormTelegraphMu(
      K, 0.5, 0.0, 0.0, 2.0, 0.0, 1.0,
      0.2, 0.03, -0.02, -0.05, 0.04, 0.0);
  return NearlyEqual(actual, std::sqrt(physical_norm_squared));
}

bool CheckChiGradientManufacturedPoint() {
  // Manufactured chi=1+x+2y+3z at the origin, with the same conformal metric.
  // chi_psi_power=-4 gives gamma^ij=chi*gtilde^ij.
  constexpr Real chi = 1.0;
  constexpr Real dchi_x = 1.0;
  constexpr Real dchi_y = 2.0;
  constexpr Real dchi_z = 3.0;
  const Real expected_squared =
      chi * (0.5 * dchi_x * dchi_x + 2.0 * dchi_y * dchi_y +
             dchi_z * dchi_z);
  const Real actual = z4c::LocalChiGradientNormTelegraphMu(
      chi, -4.0, 0.5, 0.0, 0.0, 2.0, 0.0, 1.0,
      dchi_x, dchi_y, dchi_z);
  return NearlyEqual(actual, std::sqrt(expected_squared));
}

bool CheckRoundoffPolicy() {
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const Real roundoff =
      z4c::RoundoffSafeNonnegativeSqrt(-8.0 * epsilon, 1.0);
  const Real invalid = z4c::RoundoffSafeNonnegativeSqrt(-1.0e-6, 1.0);
  return roundoff == 0.0 && std::isnan(invalid);
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  bool passed = false;
  {
    passed = CheckScaleInvariantParameterization() && CheckLocalK() &&
             CheckExtrinsicNormAgainstPhysicalReconstruction() &&
             CheckChiGradientManufacturedPoint() && CheckRoundoffPolicy();
  }
  Kokkos::finalize();
  if (!passed) {
    std::cerr << "Z4c telegraph damping unit/MMS check failed\n";
    return 1;
  }
  std::cout << "Z4c telegraph damping unit/MMS check passed\n";
  return 0;
}
