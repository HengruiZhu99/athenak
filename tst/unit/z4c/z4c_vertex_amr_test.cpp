//========================================================================================
// AthenaK astrophysical plasma code
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include <Kokkos_Core.hpp>

#include <algorithm>
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

template <int ORDER, int NG = 4>
void CheckOneDimensionalExactness() {
  constexpr int ng = NG;
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

template <int ORDER, int NG = 6>
void CheckLowerAndUpperGhostCoordinates() {
  constexpr int ng = NG;
  constexpr int coarse_intervals = 8;
  constexpr int fine_intervals = 2 * coarse_intervals;
  Kokkos::View<double*****, Kokkos::HostSpace> coarse(
      "coarse ghost coordinate", 1, 1, 1, 1,
      coarse_intervals + 1 + 2 * ng);
  Kokkos::View<double*****, Kokkos::HostSpace> fine(
      "fine ghost coordinate", 1, 1, 1, 1,
      fine_intervals + 1 + 2 * ng);
  for (int i = 0; i < coarse.extent_int(4); ++i) {
    coarse(0, 0, 0, 0, i) = static_cast<double>(i - ng);
  }
  for (int fi = ng - ng; fi <= ng + fine_intervals + ng; ++fi) {
    const double value = vertex_amr::ProlongVCPoint<ORDER>(
        0, 0, 0, 0, fi, ng, 0, 0, ng, 0, 0, true, true, coarse, fine);
    const double expected = 0.5 * static_cast<double>(fi - ng);
    Require(std::abs(value - expected) < 2.0e-13,
            "O" + std::to_string(ORDER) +
                " lower/upper ghost coordinate exactness at fine index " +
                std::to_string(fi));
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

template <int ORDER, int DIMENSIONS>
void CheckTensorExactness() {
  constexpr int ng = 6;
  constexpr int coarse_intervals = 8;
  constexpr int fine_intervals = 2 * coarse_intervals;
  constexpr int coarse_n = coarse_intervals + 1 + 2 * ng;
  constexpr int fine_n = fine_intervals + 1 + 2 * ng;
  constexpr int coarse_n2 = DIMENSIONS >= 2 ? coarse_n : 1;
  constexpr int coarse_n3 = DIMENSIONS >= 3 ? coarse_n : 1;
  constexpr int fine_n2 = DIMENSIONS >= 2 ? fine_n : 1;
  constexpr int fine_n3 = DIMENSIONS >= 3 ? fine_n : 1;
  Kokkos::View<double*****, Kokkos::HostSpace> coarse(
      "coarse tensor", 1, 1, coarse_n3, coarse_n2, coarse_n);
  Kokkos::View<double*****, Kokkos::HostSpace> fine(
      "fine tensor", 1, 1, fine_n3, fine_n2, fine_n);
  const auto polynomial = [](const double x, const double y, const double z) {
    return 1.0 + std::pow(x, ORDER - 1) + 0.5 * std::pow(y, ORDER - 1) +
           0.25 * std::pow(z, ORDER - 1) + x * y + y * z + z * x;
  };
  for (int k = 0; k < coarse_n3; ++k) {
    for (int j = 0; j < coarse_n2; ++j) {
      for (int i = 0; i < coarse_n; ++i) {
        const double x = i - ng - coarse_intervals / 2;
        const double y = DIMENSIONS >= 2 ? j - ng - coarse_intervals / 2 : 0.0;
        const double z = DIMENSIONS >= 3 ? k - ng - coarse_intervals / 2 : 0.0;
        coarse(0, 0, k, j, i) = polynomial(x, y, z);
      }
    }
  }
  const int fk0 = DIMENSIONS >= 3 ? ng : 0;
  const int fk1 = DIMENSIONS >= 3 ? ng + fine_intervals : 0;
  const int fj0 = DIMENSIONS >= 2 ? ng : 0;
  const int fj1 = DIMENSIONS >= 2 ? ng + fine_intervals : 0;
  for (int k = fk0; k <= fk1; ++k) {
    for (int j = fj0; j <= fj1; ++j) {
      for (int i = ng; i <= ng + fine_intervals; ++i) {
        const double value = vertex_amr::ProlongVCPoint<ORDER>(
            0, 0, k, j, i, ng, DIMENSIONS >= 2 ? ng : 0,
            DIMENSIONS >= 3 ? ng : 0, ng, DIMENSIONS >= 2 ? ng : 0,
            DIMENSIONS >= 3 ? ng : 0, DIMENSIONS < 2, DIMENSIONS < 3,
            coarse, fine);
        const double x = 0.5 * (i - ng) - coarse_intervals / 2;
        const double y = DIMENSIONS >= 2
            ? 0.5 * (j - ng) - coarse_intervals / 2 : 0.0;
        const double z = DIMENSIONS >= 3
            ? 0.5 * (k - ng) - coarse_intervals / 2 : 0.0;
        const double expected = polynomial(x, y, z);
        Require(std::abs(value - expected) <
                    2.0e-10 * std::max(1.0, std::abs(expected)),
                "O" + std::to_string(ORDER) + " " +
                    std::to_string(DIMENSIONS) + "D tensor exactness");
      }
    }
  }
}

template <int ORDER>
void CheckWeights(const double expected_l1) {
  double sum = 0.0;
  double l1 = 0.0;
  constexpr int points = vertex_amr::MidpointRule<ORDER>::points;
  for (int p = 0; p < points; ++p) {
    const double weight = vertex_amr::MidpointRule<ORDER>::weight(p);
    sum += weight;
    l1 += std::abs(weight);
    Require(weight == vertex_amr::MidpointRule<ORDER>::weight(points - 1 - p),
            "O" + std::to_string(ORDER) +
                " midpoint weights must be reflection symmetric");
  }
  Require(sum == 1.0, "O" + std::to_string(ORDER) +
                          " midpoint weights must preserve constants exactly");
  Require(std::abs(l1 - expected_l1) < 1.0e-15,
          "O" + std::to_string(ORDER) + " midpoint amplification norm");
}

void CheckTransferOrderAndHalo() {
  Require(vertex_amr::TransferOrderForSpatialOrder(2) == 4 &&
              vertex_amr::TransferOrderForSpatialOrder(4) == 6 &&
              vertex_amr::TransferOrderForSpatialOrder(6) == 8 &&
              vertex_amr::TransferOrderForSpatialOrder(3) == 0,
          "Z4c p-to-q transfer-order contract");
  Require(vertex_amr::RequiredCoarseGhostWidthForSpatialOrder(2, 2) == 2 &&
              vertex_amr::RequiredCoarseGhostWidthForSpatialOrder(4, 4) == 4 &&
              vertex_amr::RequiredCoarseGhostWidthForSpatialOrder(6, 4) == 5,
          "coarse ghost width contract");
  Require(vertex_amr::RequiredRefinementHaloForSpatialOrder(2) == 1 &&
              vertex_amr::RequiredRefinementHaloForSpatialOrder(4) == 2 &&
              vertex_amr::RequiredRefinementHaloForSpatialOrder(6) == 3,
          "new-child refinement halo contract");
  Require(vertex_amr::SupportsSingleHopCoarseHalo(5, 5) &&
              !vertex_amr::SupportsSingleHopCoarseHalo(4, 5),
          "single-hop coarse communication feasibility contract");

  constexpr int fine_start = 4;
  constexpr int fine_stored_end = 24;
  constexpr int coarse_start = 5;
  constexpr int coarse_end = 13;
  constexpr int coarse_intervals = 8;
  constexpr int refinement_halo = 3;
  const auto target = vertex_amr::RefinementChildTargetRange(
      coarse_start, coarse_end, refinement_halo, false);
  for (int child = 0; child <= 1; ++child) {
    const auto source = vertex_amr::RefinementChildSourceRange(
        fine_start, coarse_intervals, child, refinement_halo, false);
    Require(source.lower >= 0 && source.upper <= fine_stored_end,
            "migrating refined-child source must stay inside fine storage");
    Require(source.count() == target.count(),
            "migrating refined-child send/receive cardinality");
  }
  const auto collapsed_source = vertex_amr::RefinementChildSourceRange(
      0, 0, 0, refinement_halo, true);
  const auto collapsed_target = vertex_amr::RefinementChildTargetRange(
      0, 0, refinement_halo, true);
  Require(collapsed_source.lower == 0 && collapsed_source.upper == 0 &&
              collapsed_target.lower == 0 && collapsed_target.upper == 0,
          "migrating refined-child collapsed dimension remains singleton");
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    CheckWeights<2>(1.0);
    CheckWeights<4>(1.25);
    CheckWeights<6>(1.390625);
    CheckWeights<8>(1.48828125);
    CheckTransferOrderAndHalo();
    CheckOneDimensionalExactness<2>();
    CheckOneDimensionalExactness<4>();
    CheckOneDimensionalExactness<6>();
    CheckOneDimensionalExactness<8, 6>();
    CheckLowerAndUpperGhostCoordinates<2>();
    CheckLowerAndUpperGhostCoordinates<4>();
    CheckLowerAndUpperGhostCoordinates<6>();
    CheckLowerAndUpperGhostCoordinates<8>();
    CheckTensorAndInjection();
    CheckTensorExactness<4, 2>();
    CheckTensorExactness<6, 2>();
    CheckTensorExactness<8, 2>();
    CheckTensorExactness<4, 3>();
    CheckTensorExactness<6, 3>();
    CheckTensorExactness<8, 3>();
  }
  Kokkos::finalize();
  std::cout << "PASS: native VC injection and symmetric midpoint transfer" << std::endl;
  return EXIT_SUCCESS;
}
