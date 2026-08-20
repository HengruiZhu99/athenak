//========================================================================================
//! \file z4c_cartoon_lattice_index_test.cpp
//! \brief Contract test for device-portable Cartoon lattice-index rounding.
//========================================================================================
#include <array>
#include <iostream>

#include "z4c/cartoon_lattice_index.hpp"

int main() {
  struct Case {
    Real value;
    int expected;
  };
  constexpr std::array<Case, 10> cases{{
      {-2.51, -3}, {-2.50, -3}, {-2.49, -2}, {-0.50, -1}, {-0.49, 0},
      {0.00, 0}, {0.49, 0}, {0.50, 1}, {2.49, 2}, {2.50, 3},
  }};
  for (const auto &test : cases) {
    if (z4c::NearestLatticeIndex(test.value) != test.expected) {
      std::cerr << "nearest lattice index mismatch for " << test.value << '\n';
      return 1;
    }
  }
  std::cout << "Cartoon lattice-index rounding passed\n";
  return 0;
}
