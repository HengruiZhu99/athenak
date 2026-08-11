//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_chi_prolongation_test.cpp
//! \brief Focused positivity and sibling-group tests for Z4c chi AMR prolongation.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "mesh/prolongation.hpp"

namespace {

constexpr int kNghost = 4;

bool NearlyEqual(const Real left, const Real right, const Real tolerance) {
  const Real scale = std::max({Real(1.0), std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

DualArray3D<Real> MakeFourthOrderWeights() {
  DualArray3D<Real> weights("test prolongation weights", 5, 5, 5);
  for (int k = 0; k < 5; ++k) {
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 5; ++i) {
        weights.h_view(k, j, i) = ProlongWeight1D<kNghost>(k, false) *
                                  ProlongWeight1D<kNghost>(j, false) *
                                  ProlongWeight1D<kNghost>(i, false);
      }
    }
  }
  weights.template modify<HostMemSpace>();
  weights.template sync<DevExeSpace>();
  return weights;
}

bool CheckSchwarzschildOvershootAndFallback() {
  constexpr Real dx = 0.1875;
  DvceArray5D<Real> parent("Schwarzschild chi parent", 1, 1, 1, 5, 5);
  auto parent_host = Kokkos::create_mirror_view(parent);
  for (int j = 0; j < 5; ++j) {
    const Real z = (j - 2 + 0.5) * dx;
    for (int i = 0; i < 5; ++i) {
      const Real x = (i - 2 + 0.5) * dx;
      const Real radius = std::sqrt(x * x + z * z);
      parent_host(0, 0, 0, j, i) = std::pow(1.0 + 0.5 / radius, -4.0);
    }
  }
  Kokkos::deep_copy(parent, parent_host);

  DvceArray5D<Real> candidate("Schwarzschild high-order candidate", 1, 1, 1, 2, 2);
  DvceArray5D<Real> repaired("Schwarzschild positive children", 1, 1, 1, 2, 2);
  const auto weights = MakeFourthOrderWeights();
  Kokkos::View<int *> status("Schwarzschild chi status", 1);
  Kokkos::parallel_for(
      "Schwarzschild chi overshoot fixture", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        HighOrderProlongCC<kNghost>(0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 1,
                                    parent, candidate, weights);
        status(0) = static_cast<int>(ProlongPositiveChiCC<kNghost>(
            0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 1, true, false, parent, repaired,
            weights));
      });
  const auto candidate_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), candidate);
  const auto repaired_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), repaired);
  const auto status_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
  if (!NearlyEqual(candidate_host(0, 0, 0, 0, 0),
                   -0.0011524945679342044, 2.0e-13) ||
      status_host(0) != static_cast<int>(ChiProlongationStatus::limited)) {
    return false;
  }
  Real average = 0.0;
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const Real child = repaired_host(0, 0, 0, j, i);
      if (!std::isfinite(child) || !(child > 0.0)) return false;
      average += 0.25 * child;
    }
  }
  return NearlyEqual(average, parent_host(0, 0, 0, 2, 2), 2.0e-15);
}

bool CheckSmoothPositiveHighOrderUnchanged() {
  DvceArray5D<Real> parent("smooth positive chi parent", 1, 1, 1, 5, 5);
  auto parent_host = Kokkos::create_mirror_view(parent);
  for (int j = 0; j < 5; ++j) {
    for (int i = 0; i < 5; ++i) {
      parent_host(0, 0, 0, j, i) = 2.0 + 0.01 * i + 0.02 * j + 0.001 * i * j;
    }
  }
  Kokkos::deep_copy(parent, parent_host);
  DvceArray5D<Real> expected("smooth high-order expected", 1, 1, 1, 2, 2);
  DvceArray5D<Real> actual("smooth high-order actual", 1, 1, 1, 2, 2);
  const auto weights = MakeFourthOrderWeights();
  Kokkos::View<int *> status("smooth chi status", 1);
  Kokkos::parallel_for(
      "smooth positive chi fixture", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        HighOrderProlongCC<kNghost>(0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 1,
                                    parent, expected, weights);
        status(0) = static_cast<int>(ProlongPositiveChiCC<kNghost>(
            0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 1, true, false, parent, actual,
            weights));
      });
  const auto expected_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), expected);
  const auto actual_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), actual);
  const auto status_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
  if (status_host(0) != static_cast<int>(ChiProlongationStatus::high_order)) {
    return false;
  }
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      if (actual_host(0, 0, 0, j, i) != expected_host(0, 0, 0, j, i) ||
          !(actual_host(0, 0, 0, j, i) > 0.0)) {
        return false;
      }
    }
  }
  return true;
}

bool CheckInvalidParentFailsClosed() {
  DvceArray5D<Real> parent("invalid chi parent", 1, 1, 1, 5, 5);
  DvceArray5D<Real> children("invalid chi children", 1, 1, 1, 2, 2);
  const auto weights = MakeFourthOrderWeights();
  auto parent_host = Kokkos::create_mirror_view(parent);
  const Real invalid_values[3] = {
      0.0, std::numeric_limits<Real>::quiet_NaN(),
      std::numeric_limits<Real>::infinity()};
  for (const Real invalid : invalid_values) {
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 5; ++i) parent_host(0, 0, 0, j, i) = 2.0;
    }
    parent_host(0, 0, 0, 0, 0) = invalid;
    Kokkos::deep_copy(parent, parent_host);
    Kokkos::View<int *> status("invalid chi status", 1);
    Kokkos::parallel_for(
        "invalid chi parent fixture", Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
          status(0) = static_cast<int>(ProlongPositiveChiCC<kNghost>(
              0, 0, 0, 2, 2, 0, 0, 0, 8, 8, 1, true, false, parent,
              children, weights));
        });
    const auto status_host =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
    if (status_host(0) != static_cast<int>(ChiProlongationStatus::invalid_parent)) {
      return false;
    }
    if (invalid == 0.0) {
      const auto children_host =
          Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          if (!(children_host(0, 0, 0, j, i) > 0.0)) return false;
        }
      }
    }
  }
  return true;
}

bool CheckSiblingInventories() {
  if (ProlongationSiblingCount(false, false) != 2 ||
      ProlongationSiblingCount(true, false) != 4 ||
      ProlongationSiblingCount(true, true) != 8) {
    return false;
  }
  DvceArray5D<Real> children2("2D chi sibling inventory", 1, 1, 1, 2, 2);
  DvceArray5D<Real> children3("3D chi sibling inventory", 1, 1, 2, 2, 2);
  Kokkos::deep_copy(children2, 1.0);
  Kokkos::deep_copy(children3, 1.0);
  Kokkos::View<int *> results("chi sibling inventory results", 4);
  Kokkos::parallel_for(
      "chi sibling inventory fixture", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        results(0) = ProlongationSiblingGroupFinitePositive(
            0, 0, 0, 0, 0, true, false, children2);
        results(1) = ProlongationSiblingGroupFinitePositive(
            0, 0, 0, 0, 0, true, true, children3);
        children2(0, 0, 0, 1, 1) = -1.0;
        children3(0, 0, 1, 1, 1) = -1.0;
        results(2) = ProlongationSiblingGroupFinitePositive(
            0, 0, 0, 0, 0, true, false, children2);
        results(3) = ProlongationSiblingGroupFinitePositive(
            0, 0, 0, 0, 0, true, true, children3);
      });
  const auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), results);
  return host(0) == 1 && host(1) == 1 && host(2) == 0 && host(3) == 0;
}

bool CheckThreeDimensionalHighOrderGroup() {
  DvceArray5D<Real> parent("3D smooth chi parent", 1, 1, 5, 5, 5);
  auto parent_host = Kokkos::create_mirror_view(parent);
  for (int k = 0; k < 5; ++k) {
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 5; ++i) {
        parent_host(0, 0, k, j, i) = 3.0 + 0.01 * i + 0.02 * j + 0.03 * k;
      }
    }
  }
  Kokkos::deep_copy(parent, parent_host);
  DvceArray5D<Real> children("3D smooth chi children", 1, 1, 2, 2, 2);
  const auto weights = MakeFourthOrderWeights();
  Kokkos::View<int *> status("3D smooth chi status", 1);
  Kokkos::parallel_for(
      "3D smooth positive chi fixture", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        status(0) = static_cast<int>(ProlongPositiveChiCC<kNghost>(
            0, 0, 2, 2, 2, 0, 0, 0, 8, 8, 8, true, true, parent, children,
            weights));
      });
  const auto status_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
  const auto child_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
  if (status_host(0) != static_cast<int>(ChiProlongationStatus::high_order)) {
    return false;
  }
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        if (!std::isfinite(child_host(0, 0, k, j, i)) ||
            !(child_host(0, 0, k, j, i) > 0.0)) {
          return false;
        }
      }
    }
  }
  return true;
}

bool CheckThreeDimensionalFallbackConservation() {
  DvceArray5D<Real> parent("3D limited chi parent", 1, 1, 5, 5, 5);
  Kokkos::deep_copy(parent, 2.0);
  auto parent_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), parent);
  // The far-k value enters the fourth-order tensor product with a negative
  // coefficient, but it is outside the minmod slope about the central parent.
  parent_host(0, 0, 0, 2, 2) = 1000.0;
  Kokkos::deep_copy(parent, parent_host);
  DvceArray5D<Real> children("3D limited chi children", 1, 1, 2, 2, 2);
  const auto weights = MakeFourthOrderWeights();
  Kokkos::View<int *> status("3D limited chi status", 1);
  Kokkos::parallel_for(
      "3D limited chi fixture", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        status(0) = static_cast<int>(ProlongPositiveChiCC<kNghost>(
            0, 0, 2, 2, 2, 0, 0, 0, 8, 8, 8, true, true, parent, children,
            weights));
      });
  const auto status_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
  const auto child_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
  if (status_host(0) != static_cast<int>(ChiProlongationStatus::limited)) return false;
  Real average = 0.0;
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const Real child = child_host(0, 0, k, j, i);
        if (!std::isfinite(child) || !(child > 0.0)) return false;
        average += 0.125 * child;
      }
    }
  }
  return NearlyEqual(average, parent_host(0, 0, 2, 2, 2), 2.0e-15);
}

}  // namespace

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  const bool passed = CheckSchwarzschildOvershootAndFallback() &&
                      CheckSmoothPositiveHighOrderUnchanged() &&
                      CheckInvalidParentFailsClosed() &&
                      CheckSiblingInventories() &&
                      CheckThreeDimensionalHighOrderGroup() &&
                      CheckThreeDimensionalFallbackConservation();
  Kokkos::finalize();
  if (!passed) {
    std::cerr << "Z4c chi prolongation positivity test failed\n";
    return 1;
  }
  std::cout << "Z4c chi prolongation positivity test passed\n";
  return 0;
}
