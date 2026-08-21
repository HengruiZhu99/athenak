//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "z4c/vertex_to_cell.hpp"

namespace {

void Require(const bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <int ORDER>
void CheckOneDimension() {
  constexpr int ng = 4;
  constexpr int intervals = 12;
  Kokkos::View<Real *****, Kokkos::HostSpace> vertex(
      "vc", 1, 1, 1, 1, intervals + 1 + 2 * ng);
  for (int degree = 0; degree < ORDER; ++degree) {
    for (int i = 0; i < vertex.extent_int(4); ++i) {
      vertex(0, 0, 0, 0, i) = std::pow(static_cast<Real>(i - ng), degree);
    }
    for (int i = ng; i < ng + intervals; ++i) {
      const Real got = z4c::InterpolateVertexToCellPoint<ORDER>(
          vertex, 0, 0, 0, 0, i, true, true);
      const Real x = static_cast<Real>(i - ng) + 0.5;
      Require(std::abs(got - std::pow(x, degree)) < 3.0e-11,
              "VC-to-CC polynomial exactness");
    }
  }
}

void CheckTensorProductAndNoAlias() {
  constexpr int ng = 4;
  constexpr int intervals = 8;
  constexpr int nv = intervals + 1 + 2 * ng;
  Kokkos::View<Real *****, Kokkos::HostSpace> vertex("vc2", 1, 1, 1, nv, nv);
  for (int j = 0; j < nv; ++j) {
    for (int i = 0; i < nv; ++i) {
      const Real x = i - ng;
      const Real y = j - ng;
      vertex(0, 0, 0, j, i) = 2.0 + x + 3.0 * y + x * y;
    }
  }
  const Real value = z4c::InterpolateVertexToCellPoint<4>(
      vertex, 0, 0, 0, ng + 2, ng + 3, false, true);
  const Real x = 3.5;
  const Real y = 2.5;
  Require(std::abs(value - (2.0 + x + 3.0 * y + x * y)) < 3.0e-11,
          "VC-to-CC tensor-product exactness");
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  CheckOneDimension<2>();
  CheckOneDimension<4>();
  CheckOneDimension<6>();
  CheckTensorProductAndNoAlias();
  std::cout << "PASS: symmetric native VC-to-CC ADM interpolation\n";
  return EXIT_SUCCESS;
}
