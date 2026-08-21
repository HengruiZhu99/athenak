//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include "mesh/vertex_amr.hpp"

namespace {

void Require(const bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <int ORDER>
void CheckOneDimensionalExactness() {
  constexpr int ng = 4;
  constexpr int coarse_intervals = 8;
  constexpr int fine_intervals = 2 * coarse_intervals;
  Kokkos::View<double*****, Kokkos::HostSpace> coarse(
      "coarse", 1, 1, 1, 1, coarse_intervals + 1 + 2 * ng);
  Kokkos::View<double*****, Kokkos::HostSpace> fine(
      "fine", 1, 1, 1, 1, fine_intervals + 1 + 2 * ng);
  for (int degree = 0; degree <= ORDER - 1; ++degree) {
    for (int i = 0; i < coarse.extent_int(4); ++i) {
      const double x = static_cast<double>(i - ng);
      coarse(0, 0, 0, 0, i) = std::pow(x, degree);
    }
    for (int fi = ng; fi <= ng + fine_intervals; ++fi) {
      const double value = vertex_amr::ProlongVCPoint<ORDER>(
          0, 0, 0, 0, fi, ng, 0, 0, ng, 0, 0, true, true, coarse, fine);
      const double x = 0.5 * static_cast<double>(fi - ng);
      Require(std::abs(value - std::pow(x, degree)) < 2.0e-11,
              "O" + std::to_string(ORDER) + " degree-" +
                  std::to_string(degree) + " midpoint exactness");
    }
  }
}

void CheckTensorAndInjection() {
  constexpr int ng = 4;
  constexpr int coarse_intervals = 4;
  constexpr int fine_intervals = 8;
  constexpr int coarse_n = coarse_intervals + 1 + 2 * ng;
  constexpr int fine_n = fine_intervals + 1 + 2 * ng;
  Kokkos::View<double*****, Kokkos::HostSpace> coarse(
      "coarse2d", 1, 1, 1, coarse_n, coarse_n);
  Kokkos::View<double*****, Kokkos::HostSpace> fine(
      "fine2d", 1, 1, 1, fine_n, fine_n);
  for (int j = 0; j < coarse_n; ++j) {
    for (int i = 0; i < coarse_n; ++i) {
      const double x = i - ng;
      const double y = j - ng;
      coarse(0, 0, 0, j, i) = 1.0 + x + 2.0 * y + x * y + x * x;
    }
  }
  for (int j = ng; j <= ng + fine_intervals; ++j) {
    for (int i = ng; i <= ng + fine_intervals; ++i) {
      const double value = vertex_amr::ProlongVCPoint<6>(
          0, 0, 0, j, i, ng, ng, 0, ng, ng, 0, false, true, coarse, fine);
      const double x = 0.5 * (i - ng);
      const double y = 0.5 * (j - ng);
      Require(std::abs(value - (1.0 + x + 2.0 * y + x * y + x * x)) < 2.0e-11,
              "2D tensor O6 polynomial exactness");
    }
  }
  Kokkos::View<double*****, Kokkos::HostSpace> restricted(
      "restricted", 1, 1, 1, coarse_n, coarse_n);
  for (int j = ng; j <= ng + coarse_intervals; ++j) {
    for (int i = ng; i <= ng + coarse_intervals; ++i) {
      vertex_amr::InjectRestrictVCPoint(0, 0, 0, j, i, ng, ng, 0,
                                        ng, ng, 0, false, true, fine, restricted);
      Require(restricted(0, 0, 0, j, i) ==
                  fine(0, 0, 0, ng + 2 * (j - ng), ng + 2 * (i - ng)),
              "restriction must be exact coincident-node injection");
    }
  }
}

void CheckWeights() {
  double sum = 0.0;
  for (int p = 0; p < vertex_amr::MidpointRule<6>::points; ++p) {
    sum += vertex_amr::MidpointRule<6>::weight(p);
    Require(vertex_amr::MidpointRule<6>::weight(p) ==
                vertex_amr::MidpointRule<6>::weight(5 - p),
            "O6 midpoint weights must be reflection symmetric");
  }
  Require(sum == 1.0, "O6 midpoint weights must preserve constants exactly");
  Require(vertex_amr::RequiredCoarseGhostWidth<2>() == 1 &&
              vertex_amr::RequiredCoarseGhostWidth<4>() == 1 &&
              vertex_amr::RequiredCoarseGhostWidth<6>() == 2,
          "coarse ghost width contract");
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    CheckWeights();
    CheckOneDimensionalExactness<2>();
    CheckOneDimensionalExactness<4>();
    CheckOneDimensionalExactness<6>();
    CheckTensorAndInjection();
  }
  Kokkos::finalize();
  std::cout << "PASS: native VC injection and symmetric midpoint transfer" << std::endl;
  return EXIT_SUCCESS;
}
