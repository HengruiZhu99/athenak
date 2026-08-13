//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cartoon_axis_centered_derivatives_test.cpp
//! \brief Standard centered O2/O4/O6 rho derivatives through exact parity ghosts.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/cartoon_axis_boundary.hpp"

namespace {

struct ScalarField {
  DvceArray5D<Real> data;
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(const int m, const int k, const int j, const int i) const {
    return data(m, 0, k, j, i);
  }
};

Real IntegerPower(const Real value, const int power) {
  Real result = 1.0;
  for (int exponent = 0; exponent < power; ++exponent) result *= value;
  return result;
}

bool NearlyEqual(const Real actual, const Real expected) {
  return std::fabs(actual - expected) <=
         3.0e-11 * std::max(Real(1.0), std::fabs(expected));
}

template <int NGHOST>
bool CheckPolynomial(const int power) {
  constexpr Real spacing = 0.125;
  constexpr int layers = NGHOST + 1;
  constexpr int active_cells = 3 * NGHOST + 4;
  constexpr int active_start = NGHOST;
  constexpr int radial_extent = active_start + active_cells;
  const int parity = (power & 1) == 0 ? 1 : -1;

  DvceArray5D<Real> storage("half-plane centered derivative polynomial", 1, 1,
                            1, 1, radial_extent);
  auto host = Kokkos::create_mirror_view(storage);
  for (int i = 0; i < radial_extent; ++i) host(0, 0, 0, 0, i) = -98765.25;
  for (int offset = 0; offset < active_cells; ++offset) {
    const Real rho = (static_cast<Real>(offset) + 0.5) * spacing;
    host(0, 0, 0, 0, active_start + offset) = IntegerPower(rho, power);
  }
  Kokkos::deep_copy(storage, host);
  Kokkos::parallel_for(
      "fill derivative-test axis parity", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        if (!z4c::FillAxisGhostLine(storage, 0, 0, 0, 0, active_start,
                                    NGHOST, parity)) {
          Kokkos::abort("valid polynomial parity rejected");
        }
      });

  Kokkos::View<Real *[2]> result("centered derivative result", layers);
  ScalarField field{storage};
  Kokkos::parallel_for(
      "ordinary centered derivatives through axis",
      Kokkos::RangePolicy<DevExeSpace>(0, layers),
      KOKKOS_LAMBDA(const int layer) {
        const Real inverse_spacing[3] = {1.0 / spacing, 1.0, 1.0};
        const int i = active_start + layer;
        result(layer, 0) = Dx<NGHOST>(0, inverse_spacing, field, 0, 0, 0, i);
        result(layer, 1) = Dxx<NGHOST>(0, inverse_spacing, field, 0, 0, 0, i);
      });
  Kokkos::fence();
  const auto result_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);

  for (int layer = 0; layer < layers; ++layer) {
    const Real rho = (static_cast<Real>(layer) + 0.5) * spacing;
    const Real first = power == 0
                           ? 0.0
                           : static_cast<Real>(power) * IntegerPower(rho, power - 1);
    const Real second = power < 2
                            ? 0.0
                            : static_cast<Real>(power * (power - 1)) *
                                  IntegerPower(rho, power - 2);
    if (!NearlyEqual(result_host(layer, 0), first) ||
        !NearlyEqual(result_host(layer, 1), second)) {
      std::cerr << "NGHOST=" << NGHOST << " power=" << power
                << " rho/h=" << layer + 0.5
                << " first=" << result_host(layer, 0) << " expected=" << first
                << " second=" << result_host(layer, 1) << " expected=" << second
                << "\n";
      return false;
    }
  }
  return true;
}

template <int NGHOST>
bool CheckOrder() {
  // Dx is exact through degree 2,4,6 and Dxx through degree 3,5,7 for the
  // corresponding centered stencil.  These bases collectively cover every regular
  // even/odd monomial named by the half-plane design contract.
  constexpr int even_degree = 2 * (NGHOST - 1);
  constexpr int odd_degree = 2 * NGHOST - 1;
  bool passed = true;
  for (int power = 0; power <= even_degree; power += 2) {
    passed &= CheckPolynomial<NGHOST>(power);
  }
  for (int power = 1; power <= odd_degree; power += 2) {
    // The highest odd degree is exact for Dxx; for NGHOST=2/3/4 it exceeds the exact
    // degree of Dx by one.  Its first derivative is therefore checked separately below.
    if (power < odd_degree) passed &= CheckPolynomial<NGHOST>(power);
  }

  // Check the highest odd Dxx basis without incorrectly requiring the lower-order Dx
  // stencil to differentiate it exactly.
  constexpr Real spacing = 0.125;
  constexpr int layers = NGHOST + 1;
  constexpr int active_cells = 3 * NGHOST + 4;
  constexpr int active_start = NGHOST;
  DvceArray5D<Real> storage("highest odd second derivative", 1, 1, 1, 1,
                            active_start + active_cells);
  auto host = Kokkos::create_mirror_view(storage);
  for (int i = 0; i < active_start + active_cells; ++i) {
    host(0, 0, 0, 0, i) = 0.0;
  }
  for (int offset = 0; offset < active_cells; ++offset) {
    const Real rho = (static_cast<Real>(offset) + 0.5) * spacing;
    host(0, 0, 0, 0, active_start + offset) = IntegerPower(rho, odd_degree);
  }
  Kokkos::deep_copy(storage, host);
  Kokkos::parallel_for(
      "fill highest odd parity", Kokkos::RangePolicy<DevExeSpace>(0, 1),
      KOKKOS_LAMBDA(const int) {
        z4c::FillAxisGhostLine(storage, 0, 0, 0, 0, active_start, NGHOST, -1);
      });
  Kokkos::View<Real *> second("highest odd Dxx", layers);
  ScalarField field{storage};
  Kokkos::parallel_for(
      "highest odd centered second derivative",
      Kokkos::RangePolicy<DevExeSpace>(0, layers),
      KOKKOS_LAMBDA(const int layer) {
        const Real inverse_spacing[3] = {1.0 / spacing, 1.0, 1.0};
        second(layer) = Dxx<NGHOST>(0, inverse_spacing, field, 0, 0, 0,
                                    active_start + layer);
      });
  Kokkos::fence();
  const auto second_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), second);
  for (int layer = 0; layer < layers; ++layer) {
    const Real rho = (static_cast<Real>(layer) + 0.5) * spacing;
    const Real expected = static_cast<Real>(odd_degree * (odd_degree - 1)) *
                          IntegerPower(rho, odd_degree - 2);
    passed &= NearlyEqual(second_host(layer), expected);
  }
  return passed;
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  const bool passed = CheckOrder<2>() && CheckOrder<3>() && CheckOrder<4>();
  if (!passed) {
    std::cerr << "centered half-plane rho derivative contract failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "centered half-plane O2/O4/O6 rho derivatives passed on "
            << Kokkos::DefaultExecutionSpace::name() << "\n";
  return EXIT_SUCCESS;
}
