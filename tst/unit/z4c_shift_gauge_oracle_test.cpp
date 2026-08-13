//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

#include "z4c/z4c_shift_gauge.hpp"

namespace {

bool Close(const Real lhs, const Real rhs, const Real tolerance = 2.0e-14) {
  return std::abs(lhs - rhs) <= tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

void TestCandidateA() {
  const Real inverse_metric_row[3] = {1.2, -0.3, 0.17};
  const Real lapse_gradient[3] = {0.2, -0.4, 0.7};
  const Real chi_gradient[3] = {-0.8, 0.6, 0.1};
  const Real alpha = 0.63;
  const Real chi = 0.27;
  const Real gamma = -1.4;
  const auto result = z4c::EvaluateModifiedShiftGaugeForces(
      z4c::ShiftGaugeProfile::candidate_a, alpha, chi, gamma, inverse_metric_row,
      lapse_gradient, chi_gradient);
  assert(Close(result.gamma, alpha * alpha * chi * gamma));
  assert(result.chi_gradient == 0.0);
  assert(result.lapse_gradient == 0.0);
  assert(Close(result.total(), alpha * alpha * chi * gamma));
}

void TestCandidateC() {
  const Real inverse_metric_row[3] = {1.2, -0.3, 0.17};
  const Real lapse_gradient[3] = {0.2, -0.4, 0.7};
  const Real chi_gradient[3] = {-0.8, 0.6, 0.1};
  const Real alpha = 0.63;
  const Real chi = 0.27;
  const Real gamma = -1.4;
  const Real q = alpha * alpha * chi;
  const Real expected_g = (1.0 + q * q) / (1.0 + q);
  Real expected_chi = 0.0;
  Real expected_lapse = 0.0;
  for (int j = 0; j < 3; ++j) {
    expected_chi += 0.5 * alpha * alpha * inverse_metric_row[j] * chi_gradient[j];
    expected_lapse -= alpha * chi * inverse_metric_row[j] * lapse_gradient[j];
  }
  const auto result = z4c::EvaluateModifiedShiftGaugeForces(
      z4c::ShiftGaugeProfile::candidate_c, alpha, chi, gamma, inverse_metric_row,
      lapse_gradient, chi_gradient);
  assert(Close(result.gamma, expected_g * gamma));
  assert(Close(result.chi_gradient, expected_chi));
  assert(Close(result.lapse_gradient, expected_lapse));
  assert(Close(result.total(), expected_g * gamma + expected_chi + expected_lapse));
}

void TestCandidateCLimits() {
  assert(z4c::CandidateCGammaCoefficient(1.0, 1.0) == 1.0);
  assert(Close(z4c::CandidateCGammaCoefficient(1.0, 1.0e-30), 1.0));

  const Real q_at_minimum = std::sqrt(2.0) - 1.0;
  const Real expected_minimum = 2.0 * (std::sqrt(2.0) - 1.0);
  assert(Close(z4c::CandidateCGammaCoefficient(1.0, q_at_minimum), expected_minimum));

  const std::array<Real, 8> alphas = {1.0e-8, 1.0e-5, 1.0e-2, 0.1,
                                      0.5, 1.0, 2.0, 10.0};
  const std::array<Real, 7> chis = {1.0e-18, 1.0e-12, 1.0e-6, 1.0e-2,
                                    0.5, 1.0, 10.0};
  for (const Real alpha : alphas) {
    for (const Real chi : chis) {
      const Real coefficient = z4c::CandidateCGammaCoefficient(alpha, chi);
      assert(std::isfinite(coefficient));
      assert(coefficient >= expected_minimum - 5.0e-14);
    }
  }
}

void TestMinkowski() {
  const Real inverse_metric_row[3] = {0.0, 1.0, 0.0};
  const Real zero_gradient[3] = {0.0, 0.0, 0.0};
  const auto result = z4c::EvaluateModifiedShiftGaugeForces(
      z4c::ShiftGaugeProfile::candidate_c, 1.0, 1.0, 0.0, inverse_metric_row,
      zero_gradient, zero_gradient);
  assert(result.total() == 0.0);
}

}  // namespace

int main() {
  TestCandidateA();
  TestCandidateC();
  TestCandidateCLimits();
  TestMinkowski();
  return 0;
}
