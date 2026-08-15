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
#include "mesh/restriction.hpp"

namespace {

constexpr int kNghost = 4;

bool NearlyEqual(const Real left, const Real right, const Real tolerance) {
  const Real scale = std::max({Real(1.0), std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

template <int NGHOST>
DualArray3D<Real> MakeProlongationWeights() {
  DualArray3D<Real> weights("test prolongation weights", 5, 5, 5);
  for (int k = 0; k < 5; ++k) {
    for (int j = 0; j < 5; ++j) {
      for (int i = 0; i < 5; ++i) {
        weights.h_view(k, j, i) =
            (k <= NGHOST && j <= NGHOST && i <= NGHOST)
                ? ProlongWeight1D<NGHOST>(k, false) *
                      ProlongWeight1D<NGHOST>(j, false) *
                      ProlongWeight1D<NGHOST>(i, false)
                : 0.0;
      }
    }
  }
  weights.template modify<HostMemSpace>();
  weights.template sync<DevExeSpace>();
  return weights;
}

DualArray3D<Real> MakeFourthOrderWeights() {
  return MakeProlongationWeights<kNghost>();
}

struct RestrictionWeights {
  DualArray1D<Real> second;
  DualArray1D<Real> fourth;
  DualArray1D<Real> fourth_edge;
};

RestrictionWeights MakeRestrictionWeights() {
  RestrictionWeights weights{
      DualArray1D<Real>("test second-order restriction weights", 3),
      DualArray1D<Real>("test fourth-order restriction weights", 5),
      DualArray1D<Real>("test fourth-order edge restriction weights", 5)};
  const Real second[3] = {0.375, 0.75, -0.125};
  const Real fourth[5] = {-0.0390625, 0.46875, 0.703125, -0.15625, 0.0234375};
  const Real edge[5] = {0.2734375, 1.09375, -0.546875, 0.21875, -0.0390625};
  for (int i = 0; i < 3; ++i) weights.second.h_view(i) = second[i];
  for (int i = 0; i < 5; ++i) {
    weights.fourth.h_view(i) = fourth[i];
    weights.fourth_edge.h_view(i) = edge[i];
  }
  weights.second.template modify<HostMemSpace>();
  weights.second.template sync<DevExeSpace>();
  weights.fourth.template modify<HostMemSpace>();
  weights.fourth.template sync<DevExeSpace>();
  weights.fourth_edge.template modify<HostMemSpace>();
  weights.fourth_edge.template sync<DevExeSpace>();
  return weights;
}

Real PolynomialValue(const Real x, const Real y, const int degree) {
  return 2.0 + std::pow(x, degree) + 0.7 * std::pow(y, degree) +
         0.05 * std::pow(x, degree) * std::pow(y, degree);
}

template <int NGHOST>
bool CheckCollapsedRestrictionPolynomialExactness() {
  constexpr int nx = 8;
  constexpr int extent = nx + 2 * NGHOST;
  constexpr int npoints = (NGHOST == 4) ? 5 : 2;
  const int points[5] = {
      (NGHOST == 4) ? 0 : NGHOST,
      NGHOST,
      (NGHOST == 4) ? NGHOST + 2 : 0,
      (NGHOST == 4) ? NGHOST + nx - 2 : 0,
      (NGHOST == 4) ? extent - 2 : 0};
  const int high_point = NGHOST + nx - 2;
  const auto weights = MakeRestrictionWeights();
  DvceArray5D<Real> fine("collapsed high-order restriction source", 1, 1,
                         1, extent, extent);

  for (int degree = 0; degree <= NGHOST; ++degree) {
    auto fine_host = Kokkos::create_mirror_view(fine);
    for (int j = 0; j < extent; ++j) {
      for (int i = 0; i < extent; ++i) {
        const Real x = 0.1 * (i - NGHOST);
        const Real y = 0.1 * (j - NGHOST);
        fine_host(0, 0, 0, j, i) = PolynomialValue(x, y, degree);
      }
    }
    Kokkos::deep_copy(fine, fine_host);
    for (int pj = 0; pj < npoints; ++pj) {
      const int fj = (NGHOST == 4) ? points[pj] : (pj == 0 ? NGHOST : high_point);
      for (int pi = 0; pi < npoints; ++pi) {
        const int fi =
            (NGHOST == 4) ? points[pi] : (pi == 0 ? NGHOST : high_point);
        Kokkos::View<Real *> result("collapsed restriction result", 1);
        Kokkos::parallel_for(
            "collapsed polynomial restriction", Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
              result(0) = RestrictInterpolation<NGHOST>(
                  0, 0, 0, fj, fi, nx, nx, 1, fine, weights.second,
                  weights.fourth, weights.fourth_edge);
            });
        const auto result_host =
            Kokkos::create_mirror_view_and_copy(HostMemSpace(), result);
        const Real expected = PolynomialValue(
            0.1 * (fi + 0.5 - NGHOST), 0.1 * (fj + 0.5 - NGHOST), degree);
        if (!NearlyEqual(result_host(0), expected, 3.0e-12)) return false;
      }
    }
  }
  return true;
}

template <int NGHOST>
bool CheckCollapsedRestrictionProlongationRoundTrip() {
  constexpr int nx = 16;
  constexpr int extent = nx + 2 * NGHOST;
  constexpr int ncoarse = NGHOST + 1;
  constexpr int coarse_center = NGHOST / 2;
  constexpr int fine_center = NGHOST + 6;
  DvceArray5D<Real> fine("collapsed round-trip fine source", 1, 1,
                         1, extent, extent);
  DvceArray5D<Real> coarse("collapsed round-trip coarse stencil", 1, 1,
                           1, ncoarse, ncoarse);
  DvceArray5D<Real> children("collapsed round-trip children", 1, 1, 1, 2, 2);
  auto fine_host = Kokkos::create_mirror_view(fine);
  for (int j = 0; j < extent; ++j) {
    for (int i = 0; i < extent; ++i) {
      const Real x = 0.05 * (i - fine_center);
      const Real y = 0.05 * (j - fine_center);
      fine_host(0, 0, 0, j, i) = PolynomialValue(x, y, NGHOST);
    }
  }
  Kokkos::deep_copy(fine, fine_host);
  const auto restriction_weights = MakeRestrictionWeights();
  const auto prolongation_weights = MakeProlongationWeights<NGHOST>();
  Kokkos::parallel_for(
      "collapsed restriction stencil", Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
          {0, 0}, {ncoarse, ncoarse}),
      KOKKOS_LAMBDA(const int j, const int i) {
        const int fj = fine_center + 2 * (j - coarse_center);
        const int fi = fine_center + 2 * (i - coarse_center);
        coarse(0, 0, 0, j, i) = RestrictInterpolation<NGHOST>(
            0, 0, 0, fj, fi, nx, nx, 1, fine, restriction_weights.second,
            restriction_weights.fourth, restriction_weights.fourth_edge);
      });
  Kokkos::parallel_for(
      "collapsed restriction prolongation round trip", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        HighOrderProlongCC<NGHOST>(0, 0, 0, coarse_center, coarse_center,
                                   0, 0, 0, nx, nx, 1, coarse, children,
                                   prolongation_weights);
      });
  const auto child_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const Real expected = fine_host(0, 0, 0, fine_center + j, fine_center + i);
      if (!NearlyEqual(child_host(0, 0, 0, j, i), expected, 5.0e-12)) return false;
    }
  }
  return true;
}

template <int NGHOST>
bool CheckStoredSameLevelRestrictionBounds() {
  constexpr int nx = 8;
  constexpr int extent = nx + 2*NGHOST;
  constexpr int fine_start = NGHOST;
  constexpr int coarse_start = NGHOST;
  DvceArray5D<Real> fine("stored same-level restriction source", 1, 1,
                         extent, extent, extent);
  Kokkos::parallel_for(
      "populate stored same-level restriction source",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0},
                                              {extent, extent, extent}),
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        fine(0, 0, k, j, i) = 10.0 + i + 2.0*j + 3.0*k;
      });
  const auto weights = MakeRestrictionWeights();
  int face_count = 0;
  int edge_count = 0;
  int corner_count = 0;

  const auto receive_range = [](const int offset) {
    int lower = fine_start;
    int upper = fine_start + nx - 1;
    if (offset < 0) {
      lower = 0;
      upper = fine_start - 1;
    } else if (offset > 0) {
      lower = fine_start + nx;
      upper = extent - 1;
    }
    return CoarseRestrictionRange{(lower + coarse_start)/2,
                                  (upper + coarse_start)/2};
  };

  for (int ox3 = -1; ox3 <= 1; ++ox3) {
    for (int ox2 = -1; ox2 <= 1; ++ox2) {
      for (int ox1 = -1; ox1 <= 1; ++ox1) {
        const int codimension = (ox1 != 0) + (ox2 != 0) + (ox3 != 0);
        if (codimension == 0) continue;
        auto irange = receive_range(ox1);
        auto jrange = receive_range(ox2);
        auto krange = receive_range(ox3);
        irange = CompleteFinePairCoarseRange(irange.lower, irange.upper,
                                             coarse_start, fine_start, extent);
        jrange = CompleteFinePairCoarseRange(jrange.lower, jrange.upper,
                                             coarse_start, fine_start, extent);
        krange = CompleteFinePairCoarseRange(krange.lower, krange.upper,
                                             coarse_start, fine_start, extent);
        if (irange.lower > irange.upper || jrange.lower > jrange.upper ||
            krange.lower > krange.upper) return false;
        const int ni = irange.upper - irange.lower + 1;
        const int nj = jrange.upper - jrange.lower + 1;
        const int nk = krange.upper - krange.lower + 1;
        const int count = ni*nj*nk;
        if (codimension == 1) face_count += count;
        if (codimension == 2) edge_count += count;
        if (codimension == 3) corner_count += count;
        Kokkos::View<int *> failures("stored restriction failures", 1);
        Kokkos::deep_copy(failures, 0);
        Kokkos::parallel_for(
            "stored face edge corner restriction",
            Kokkos::RangePolicy<>(0, count), KOKKOS_LAMBDA(const int idx) {
              const int k = idx/(nj*ni) + krange.lower;
              const int j = (idx % (nj*ni))/ni + jrange.lower;
              const int i = idx % ni + irange.lower;
              const int fk = (k - coarse_start)*2 + fine_start;
              const int fj = (j - coarse_start)*2 + fine_start;
              const int fi = (i - coarse_start)*2 + fine_start;
              const Real restricted = RestrictInterpolation<NGHOST>(
                  0, 0, fk, fj, fi, nx, nx, nx, fine, weights.second,
                  weights.fourth, weights.fourth_edge);
              const Real expected =
                  10.0 + (fi + 0.5) + 2.0*(fj + 0.5) + 3.0*(fk + 0.5);
              if (!Kokkos::isfinite(restricted) ||
                  Kokkos::abs(restricted - expected) > 2.0e-12) {
                Kokkos::atomic_inc(&failures(0));
              }
            });
        const auto failures_host =
            Kokkos::create_mirror_view_and_copy(HostMemSpace(), failures);
        if (failures_host(0) != 0) return false;
      }
    }
  }
  constexpr int boundary_width = (NGHOST == 4) ? 2 : 1;
  return face_count == 6*boundary_width*4*4 &&
         edge_count == 12*boundary_width*boundary_width*4 &&
         corner_count == 8*boundary_width*boundary_width*boundary_width;
}

template <int NGHOST>
bool CheckConsecutiveThreeDimensionalRefreshes() {
  constexpr int n = NGHOST + 1;
  constexpr int fine_n = 32;
  constexpr int fine_center = 16;
  constexpr int coarse_center = NGHOST / 2;
  DvceArray5D<Real> fine("refresh source chi", 1, 1, fine_n, fine_n, fine_n);
  DvceArray5D<Real> coarse("refreshed coarse chi", 1, 1, n, n, n);
  DvceArray5D<Real> children("refresh chi children", 1, 1, 2, 2, 2);
  const auto prolongation_weights = MakeProlongationWeights<NGHOST>();
  const auto restriction_weights = MakeRestrictionWeights();
  Real previous_center = 0.0;

  for (int pass = 0; pass < 2; ++pass) {
    const Real offset = 2.0 + 0.25 * pass;
    Kokkos::parallel_for(
        "populate refresh source", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {fine_n, fine_n, fine_n}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          fine(0, 0, k, j, i) =
              offset + 1.0e-3 * i + 2.0e-3 * j + 3.0e-3 * k;
        });
    if (pass == 0) {
      Kokkos::deep_copy(coarse, -1.0);
    }
    Kokkos::parallel_for(
        "refresh coarse chi stencil", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
            {0, 0, 0}, {n, n, n}),
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const int fk = fine_center + 2 * (k - coarse_center);
          const int fj = fine_center + 2 * (j - coarse_center);
          const int fi = fine_center + 2 * (i - coarse_center);
          coarse(0, 0, k, j, i) = RestrictInterpolation<NGHOST>(
              0, 0, fk, fj, fi, fine_n, fine_n, fine_n, fine,
              restriction_weights.second, restriction_weights.fourth,
              restriction_weights.fourth_edge);
        });
    Kokkos::View<int *> status("refreshed chi status", 1);
    Kokkos::parallel_for(
        "prolong refreshed chi", Kokkos::RangePolicy<>(0, 1),
        KOKKOS_LAMBDA(const int) {
          status(0) = static_cast<int>(ProlongPositiveChiCC<NGHOST>(
              0, 0, coarse_center, coarse_center, coarse_center, 0, 0, 0,
              fine_n, fine_n, fine_n, true, true, coarse, children,
              prolongation_weights));
        });
    const auto status_host =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
    const auto coarse_host =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), coarse);
    const auto children_host =
        Kokkos::create_mirror_view_and_copy(HostMemSpace(), children);
    if (status_host(0) != static_cast<int>(ChiProlongationStatus::high_order)) {
      return false;
    }
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          if (!std::isfinite(coarse_host(0, 0, k, j, i)) ||
              !(coarse_host(0, 0, k, j, i) > 0.0)) return false;
        }
      }
    }
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          if (!std::isfinite(children_host(0, 0, k, j, i)) ||
              !(children_host(0, 0, k, j, i) > 0.0)) return false;
        }
      }
    }
    const Real center = coarse_host(0, 0, coarse_center, coarse_center,
                                    coarse_center);
    const Real expected_center = offset + (1.0e-3 + 2.0e-3 + 3.0e-3) *
                                              (fine_center + 0.5);
    if (!NearlyEqual(center, expected_center, 2.0e-14)) return false;
    if (pass == 1 && !(center > previous_center)) return false;
    previous_center = center;
  }
  return true;
}

bool CheckThreeDimensionalBoostedPunctureFallback() {
  constexpr Real dx = 0.125;
  constexpr Real velocity = 0.8660254;
  const Real gamma = 1.0 / std::sqrt(1.0 - velocity * velocity);
  DvceArray5D<Real> parent("boosted puncture chi parent", 1, 1, 5, 5, 5);
  auto parent_host = Kokkos::create_mirror_view(parent);
  for (int k = 0; k < 5; ++k) {
    const Real z = (k - 2 + 0.5) * dx;
    for (int j = 0; j < 5; ++j) {
      const Real y = (j - 2 + 0.5) * dx;
      for (int i = 0; i < 5; ++i) {
        const Real x = (i - 2 + 0.5) * dx;
        const Real radius = std::sqrt(gamma * gamma * x * x + y * y + z * z);
        const Real psi = 1.0 + 0.5 / radius;
        const Real alpha = (1.0 - 0.5 / radius) / psi;
        const Real boost = std::sqrt(gamma * gamma *
            (1.0 - velocity * velocity * alpha * alpha * std::pow(psi, -4.0)));
        // det(g) = psi^12 boost^2 and chi = det(g)^(-1/3).
        parent_host(0, 0, k, j, i) = std::pow(psi, -4.0) * std::pow(boost, -2.0/3.0);
      }
    }
  }
  Kokkos::deep_copy(parent, parent_host);

  DvceArray5D<Real> raw("boosted puncture raw children", 1, 1, 2, 2, 2);
  DvceArray5D<Real> limited("boosted puncture limited children", 1, 1, 2, 2, 2);
  const auto weights = MakeFourthOrderWeights();
  Kokkos::View<int *> status("boosted puncture chi status", 1);
  Kokkos::parallel_for(
      "boosted puncture chi fallback fixture", Kokkos::RangePolicy<>(0, 1),
      KOKKOS_LAMBDA(const int) {
        HighOrderProlongCC<kNghost>(0, 0, 2, 2, 2, 0, 0, 0, 8, 8, 8,
                                    parent, raw, weights);
        status(0) = static_cast<int>(ProlongPositiveChiCC<kNghost>(
            0, 0, 2, 2, 2, 0, 0, 0, 8, 8, 8, true, true, parent, limited,
            weights));
      });
  const auto raw_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), raw);
  const auto limited_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), limited);
  const auto status_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), status);
  if (!(raw_host(0, 0, 0, 0, 0) < 0.0) ||
      status_host(0) != static_cast<int>(ChiProlongationStatus::limited)) {
    return false;
  }
  Real average = 0.0;
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const Real child = limited_host(0, 0, k, j, i);
        if (!std::isfinite(child) || !(child > 0.0)) return false;
        average += 0.125 * child;
      }
    }
  }
  return NearlyEqual(average, parent_host(0, 0, 2, 2, 2), 2.0e-15);
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
                      CheckThreeDimensionalBoostedPunctureFallback() &&
                      CheckSmoothPositiveHighOrderUnchanged() &&
                      CheckInvalidParentFailsClosed() &&
                      CheckSiblingInventories() &&
                      CheckThreeDimensionalHighOrderGroup() &&
                      CheckThreeDimensionalFallbackConservation() &&
                      CheckCollapsedRestrictionPolynomialExactness<2>() &&
                      CheckCollapsedRestrictionPolynomialExactness<3>() &&
                      CheckCollapsedRestrictionPolynomialExactness<4>() &&
                      CheckCollapsedRestrictionProlongationRoundTrip<2>() &&
                      CheckCollapsedRestrictionProlongationRoundTrip<3>() &&
                      CheckCollapsedRestrictionProlongationRoundTrip<4>() &&
                      CheckStoredSameLevelRestrictionBounds<2>() &&
                      CheckStoredSameLevelRestrictionBounds<3>() &&
                      CheckStoredSameLevelRestrictionBounds<4>() &&
                      CheckConsecutiveThreeDimensionalRefreshes<2>() &&
                      CheckConsecutiveThreeDimensionalRefreshes<3>() &&
                      CheckConsecutiveThreeDimensionalRefreshes<4>();
  Kokkos::finalize();
  if (!passed) {
    std::cerr << "Z4c chi prolongation positivity test failed\n";
    return 1;
  }
  std::cout << "Z4c chi prolongation positivity test passed\n";
  return 0;
}
