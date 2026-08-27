//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================

#include <array>
#include <cmath>

#include "z4c/sommerfeld_derivatives.hpp"

namespace {

template <int NGHOST>
bool CheckPolynomial(const int side) {
  constexpr Real h = 0.125;
  constexpr std::array<Real, 7> coefficient{
      1.25, -0.875, 0.4, -0.12, 0.035, -0.009, 0.0015};
  const auto polynomial = [&](const Real x) {
    Real value = 0.0;
    const int degree = 2 * (NGHOST - 1);
    for (int power = degree; power >= 0; --power) {
      value = value * x + coefficient[power];
    }
    return value;
  };
  const auto inward_sample = [&](const int q) {
    return polynomial(-side * q * h);
  };
  const Real derivative = z4c::SommerfeldOneSidedFirst<NGHOST>(
      side, 1.0 / h, inward_sample);
  return std::abs(derivative - coefficient[1]) < 2.0e-11;
}

bool CheckCenteredPolynomial() {
  constexpr Real h = 0.125;
  constexpr Real slope = -0.875;
  const auto quadratic = [&](const int q) {
    const Real x = q * h;
    return 1.25 + slope * x + 0.4 * x * x;
  };
  return std::abs(z4c::BoundaryCenteredFirst(1.0 / h, quadratic) - slope) <
         1.0e-14;
}

}  // namespace

int main() {
  return CheckPolynomial<2>(-1) && CheckPolynomial<2>(1) &&
                 CheckPolynomial<3>(-1) && CheckPolynomial<3>(1) &&
                 CheckPolynomial<4>(-1) && CheckPolynomial<4>(1) &&
                 CheckCenteredPolynomial()
             ? 0
             : 1;
}
