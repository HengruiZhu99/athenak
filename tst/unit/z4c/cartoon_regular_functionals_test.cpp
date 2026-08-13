//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_regular_functionals_test.cpp
//! \brief Exact rational tests for half-plane SO(2) regularity functionals.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

template <int NGHOST>
constexpr double ApplyRow(const int target,
                          const std::array<double, NGHOST> &samples) {
  if constexpr (NGHOST == 2) {
    return -0.5 * samples[0] + 0.5 * samples[1];
  } else if constexpr (NGHOST == 3) {
    if (target == 0) {
      return -2.0 / 3.0 * samples[0] + 3.0 / 4.0 * samples[1] -
             1.0 / 12.0 * samples[2];
    }
    return -1.0 / 3.0 * samples[0] + 1.0 / 4.0 * samples[1] +
           1.0 / 12.0 * samples[2];
  } else {
    if (target == 0) {
      return -3.0 / 4.0 * samples[0] + 9.0 / 10.0 * samples[1] -
             1.0 / 6.0 * samples[2] + 1.0 / 60.0 * samples[3];
    }
    if (target == 1) {
      return -5.0 / 18.0 * samples[0] + 3.0 / 20.0 * samples[1] +
             5.0 / 36.0 * samples[2] - 1.0 / 90.0 * samples[3];
    }
    return 1.0 / 6.0 * samples[0] - 9.0 / 20.0 * samples[1] +
           1.0 / 4.0 * samples[2] + 1.0 / 30.0 * samples[3];
  }
}

constexpr double IntegerPower(const double value, const int exponent) {
  double result = 1.0;
  for (int n = 0; n < exponent; ++n) result *= value;
  return result;
}

template <int NGHOST>
constexpr bool CheckExactness() {
  for (int target = 0; target < NGHOST - 1; ++target) {
    const double target_radius = static_cast<double>(target) + 0.5;
    const double target_s = target_radius * target_radius;
    for (int degree = 0; degree < NGHOST; ++degree) {
      std::array<double, NGHOST> samples{};
      for (int point = 0; point < NGHOST; ++point) {
        const double radius = static_cast<double>(point) + 0.5;
        samples[point] = IntegerPower(radius * radius, degree);
      }
      const double expected = degree == 0
                                  ? 0.0
                                  : degree * IntegerPower(target_s, degree - 1);
      if (std::abs(ApplyRow<NGHOST>(target, samples) - expected) > 1.0e-13) {
        return false;
      }
    }
  }
  return true;
}

static_assert(CheckExactness<2>());
static_assert(CheckExactness<3>());
static_assert(CheckExactness<4>());

template <int NGHOST>
bool CheckRegularityClasses() {
  constexpr double h = 0.125;
  for (int target = 0; target < NGHOST - 1; ++target) {
    std::array<double, NGHOST> odd_linear{};
    std::array<double, NGHOST> quadratic_zero{};
    std::array<double, NGHOST> planar_difference{};
    for (int point = 0; point < NGHOST; ++point) {
      const double rho = (static_cast<double>(point) + 0.5) * h;
      const double s = rho * rho;
      double regular = 1.25 - 0.75 * s;
      if constexpr (NGHOST >= 3) regular += 0.5 * s * s;
      if constexpr (NGHOST >= 4) regular -= 0.2 * s * s * s;
      odd_linear[point] = (rho * regular) / rho;
      quadratic_zero[point] = (rho * rho * regular) / (rho * rho);
      const double planar = (2.0 + s * regular) - 2.0;
      planar_difference[point] = planar / (rho * rho);
    }
    const double rho = (static_cast<double>(target) + 0.5) * h;
    const double s = rho * rho;
    double expected = -0.75;
    if constexpr (NGHOST >= 3) expected += s;
    if constexpr (NGHOST >= 4) expected -= 0.6 * s * s;
    const double scale = 1.0 / (h * h);
    for (const auto &samples : {odd_linear, quadratic_zero, planar_difference}) {
      const double observed = scale * ApplyRow<NGHOST>(target, samples);
      if (std::abs(observed - expected) > 3.0e-12) return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!CheckRegularityClasses<2>() || !CheckRegularityClasses<3>() ||
      !CheckRegularityClasses<4>()) {
    std::cerr << "half-plane direct regularity functional exactness failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "half-plane O2/O4/O6 fixed regularity functionals passed\n";
  return EXIT_SUCCESS;
}
