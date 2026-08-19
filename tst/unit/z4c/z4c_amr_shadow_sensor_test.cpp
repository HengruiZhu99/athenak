//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file z4c_amr_shadow_sensor_test.cpp
//! \brief Synthetic contracts for the default-off Nyquist shadow sensor.

#include <cmath>
#include <iostream>

#include "z4c/amr_shadow_sensor.hpp"

int main() {
  using z4c::FourthDifferenceShadow2D;
  using z4c::NormalizedFourthDifference;
  const auto one_d = [](const int n, const int wave) {
    const auto value = [=](int i) {
      return std::sin(2.0 * M_PI * wave * static_cast<Real>(i) / n);
    };
    return NormalizedFourthDifference(value(0), value(1), value(2), value(3), value(4));
  };
  const Real constant = NormalizedFourthDifference(3.0, 3.0, 3.0, 3.0, 3.0);
  const Real cubic = NormalizedFourthDifference(-8.0, -1.0, 0.0, 1.0, 8.0);
  const Real nyquist = NormalizedFourthDifference(1.0, -1.0, 1.0, -1.0, 1.0);
  const Real wavelength_four = NormalizedFourthDifference(1.0, 0.0, -1.0, 0.0, 1.0);
  const Real smooth_coarse = one_d(32, 1);
  const Real smooth_fine = one_d(64, 1);
  const Real odd_axis = FourthDifferenceShadow2D(-2.0, -1.0, 0.0, 1.0, 2.0,
                                                   0.0, 0.0, 0.0, 0.0);
  const bool passed = constant == 0.0 && cubic == 0.0 && nyquist > 3.0 &&
                      wavelength_four > 1.0 && smooth_fine < smooth_coarse &&
                      odd_axis == 0.0;
  if (!passed) {
    std::cerr << "constant=" << constant << " cubic=" << cubic
              << " nyquist=" << nyquist << " wavelength_four=" << wavelength_four
              << " smooth_coarse=" << smooth_coarse
              << " smooth_fine=" << smooth_fine << " odd_axis=" << odd_axis << '\n';
    std::cerr << "Z4c AMR shadow sensor regression failed\n";
    return 1;
  }
  std::cout << "Z4c AMR shadow sensor regression passed\n";
  return 0;
}
