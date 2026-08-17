//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE for details
//========================================================================================
//! \file z4c_coarse_cache_ownership_test.cpp
//! \brief Owner-authoritative same-level Z4c coarse-cache regression.

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "bvals/coarse_cache_ownership.hpp"
#include "mesh/prolongation.hpp"
#include "mesh/restriction.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

constexpr int kNvar = 25;
constexpr int kNx = 16;

struct RestrictionWeights {
  DualArray1D<Real> second;
  DualArray1D<Real> fourth;
  DualArray1D<Real> fourth_edge;
};

RestrictionWeights MakeRestrictionWeights() {
  RestrictionWeights weights{
      DualArray1D<Real>("ownership second restriction", 3),
      DualArray1D<Real>("ownership fourth restriction", 5),
      DualArray1D<Real>("ownership edge restriction", 5)};
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

template <int NGHOST>
DualArray3D<Real> MakeProlongationWeights() {
  DualArray3D<Real> weights("ownership prolongation weights", 5, 5, 5);
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

Real SmoothState(const int v, const Real x, const Real y, const Real z = 0.0) {
  const Real phase = 0.071 * (v + 1);
  const Real perturbation =
      0.011 * std::sin(0.23 * x + 0.17 * y + 0.13 * z + phase) +
      0.004 * std::cos(0.19 * x * y + 0.07 * z + 2.0 * phase) +
      0.0003 * x * y;
  // chi is strictly positive; the remaining components are offset as well so the
  // regression can use one finite-data contract for all 25 variables.
  return 1.25 + 0.03 * v + perturbation;
}

bool NearlyEqual(const Real left, const Real right, const Real tolerance = 2.0e-13) {
  const Real scale = std::max({Real(1.0), std::abs(left), std::abs(right)});
  return std::abs(left - right) <= tolerance * scale;
}

struct Direction {
  int axis;  // 0=x1, 1=x2
  int sign;
};

template <int NGHOST>
bool CheckTwoDimensionalOwnership(const Direction same, const int child,
                                  const bool axis_adjacent) {
  constexpr int fine_extent = kNx + 2 * NGHOST;
  constexpr int coarse_nx = kNx / 2;
  constexpr int coarse_extent = coarse_nx + 2 * NGHOST;
  const auto restriction = MakeRestrictionWeights();
  const auto prolongation = MakeProlongationWeights<NGHOST>();
  DvceArray5D<Real> sender_fine("ownership sender fine", 1, kNvar, 1,
                                fine_extent, fine_extent);
  DvceArray5D<Real> receiver_fine("ownership receiver fine", 1, kNvar, 1,
                                  fine_extent, fine_extent);
  DvceArray5D<Real> sender_coarse("ownership sender coarse", 1, kNvar, 1,
                                  coarse_extent, coarse_extent);
  DvceArray5D<Real> receiver_coarse("ownership receiver coarse", 1, kNvar, 1,
                                    coarse_extent, coarse_extent);
  DvceArray5D<Real> mpi_coarse("ownership mpi coarse", 1, kNvar, 1,
                               coarse_extent, coarse_extent);
  DvceArray5D<Real> same_children("ownership same children", 1, kNvar, 1, 2, 2);
  DvceArray5D<Real> mpi_children("ownership mpi children", 1, kNvar, 1, 2, 2);

  const int sender_origin_x = (same.axis == 0) ? same.sign * kNx : 0;
  const int sender_origin_y = (same.axis == 1) ? same.sign * kNx : 0;
  auto sender_host = Kokkos::create_mirror_view(sender_fine);
  auto receiver_host = Kokkos::create_mirror_view(receiver_fine);
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < fine_extent; ++j) {
      for (int i = 0; i < fine_extent; ++i) {
        const Real rx = i - NGHOST + 0.5;
        const Real ry = j - NGHOST + 0.5;
        const Real sx = sender_origin_x + rx;
        const Real sy = sender_origin_y + ry;
        receiver_host(0, v, 0, j, i) = SmoothState(v, rx, ry);
        sender_host(0, v, 0, j, i) = SmoothState(v, sx, sy);
      }
    }
  }
  Kokkos::deep_copy(sender_fine, sender_host);
  Kokkos::deep_copy(receiver_fine, receiver_host);
  Kokkos::deep_copy(sender_coarse, std::numeric_limits<Real>::quiet_NaN());
  Kokkos::deep_copy(receiver_coarse, std::numeric_limits<Real>::quiet_NaN());

  Kokkos::parallel_for(
      "ownership restrict active blocks",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
          {0, NGHOST, NGHOST}, {kNvar, NGHOST + coarse_nx, NGHOST + coarse_nx}),
      KOKKOS_LAMBDA(const int v, const int j, const int i) {
        const int fj = (j - NGHOST) * 2 + NGHOST;
        const int fi = (i - NGHOST) * 2 + NGHOST;
        sender_coarse(0, v, 0, j, i) = RestrictInterpolation<NGHOST>(
            0, v, 0, fj, fi, kNx, kNx, 1, sender_fine, restriction.second,
            restriction.fourth, restriction.fourth_edge);
        receiver_coarse(0, v, 0, j, i) = RestrictInterpolation<NGHOST>(
            0, v, 0, fj, fi, kNx, kNx, 1, receiver_fine, restriction.second,
            restriction.fourth, restriction.fourth_edge);
      });
  Kokkos::fence();
  auto sender_coarse_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), sender_coarse);
  auto receiver_coarse_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), receiver_coarse);

  using CoarseKey = std::pair<int, int>;
  std::map<CoarseKey, std::array<Real, kNvar>> owned;
  for (int j = NGHOST; j < NGHOST + coarse_nx; ++j) {
    for (int i = NGHOST; i < NGHOST + coarse_nx; ++i) {
      const int gx = sender_origin_x + 2 * (i - NGHOST) + 1;
      const int gy = sender_origin_y + 2 * (j - NGHOST) + 1;
      auto &value = owned[{gy, gx}];
      for (int v = 0; v < kNvar; ++v) value[v] = sender_coarse_host(0, v, 0, j, i);
    }
  }

  const int ghost_lo = same.sign < 0 ? 0 : NGHOST + coarse_nx;
  const int ghost_hi = same.sign < 0 ? NGHOST : coarse_extent;
  std::vector<Real> packed;
  for (int j = (same.axis == 0 ? NGHOST : ghost_lo);
       j < (same.axis == 0 ? NGHOST + coarse_nx : ghost_hi); ++j) {
    for (int i = (same.axis == 0 ? ghost_lo : NGHOST);
         i < (same.axis == 0 ? ghost_hi : NGHOST + coarse_nx); ++i) {
      const int gx = 2 * (i - NGHOST) + 1;
      const int gy = 2 * (j - NGHOST) + 1;
      const auto found = owned.find({gy, gx});
      if (found == owned.end()) return false;
      packed.insert(packed.end(), found->second.begin(), found->second.end());
    }
  }
  std::vector<Real> mpi_packed = packed;
#if MPI_PARALLEL_ENABLED
  int mpi_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank != 0) std::fill(mpi_packed.begin(), mpi_packed.end(), 0.0);
  MPI_Bcast(mpi_packed.data(), static_cast<int>(mpi_packed.size()), MPI_DOUBLE, 0,
            MPI_COMM_WORLD);
#endif
  auto mpi_coarse_host = Kokkos::create_mirror_view(mpi_coarse);
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < coarse_extent; ++j) {
      for (int i = 0; i < coarse_extent; ++i) {
        mpi_coarse_host(0, v, 0, j, i) = receiver_coarse_host(0, v, 0, j, i);
      }
    }
  }
  std::size_t q = 0;
  for (int j = (same.axis == 0 ? NGHOST : ghost_lo);
       j < (same.axis == 0 ? NGHOST + coarse_nx : ghost_hi); ++j) {
    for (int i = (same.axis == 0 ? ghost_lo : NGHOST);
         i < (same.axis == 0 ? ghost_hi : NGHOST + coarse_nx); ++i) {
      for (int v = 0; v < kNvar; ++v) {
        receiver_coarse_host(0, v, 0, j, i) = packed[q];
        mpi_coarse_host(0, v, 0, j, i) = mpi_packed[q];
        ++q;
      }
    }
  }
  if (q != packed.size()) return false;

  // Preserve-received semantics are linked directly to the production policy.
  const auto received_snapshot = receiver_coarse_host;
  bool legacy_would_change = false;
  const int fine_lo = same.sign < 0 ? 0 : NGHOST + kNx;
  const int fine_hi = same.sign < 0 ? NGHOST : fine_extent;
  int refresh_lo = (fine_lo + NGHOST) / 2;
  int refresh_hi = (fine_hi - 1 + NGHOST) / 2;
  while (refresh_lo <= refresh_hi &&
         (refresh_lo - NGHOST) * 2 + NGHOST < 0) ++refresh_lo;
  while (refresh_lo <= refresh_hi &&
         (refresh_hi - NGHOST) * 2 + NGHOST + 1 >= fine_extent) --refresh_hi;
  for (int j = (same.axis == 0 ? NGHOST : refresh_lo);
       j <= (same.axis == 0 ? NGHOST + coarse_nx - 1 : refresh_hi); ++j) {
    for (int i = (same.axis == 0 ? refresh_lo : NGHOST);
         i <= (same.axis == 0 ? refresh_hi : NGHOST + coarse_nx - 1); ++i) {
      const int fj = (j - NGHOST) * 2 + NGHOST;
      const int fi = (i - NGHOST) * 2 + NGHOST;
      for (int v = 0; v < kNvar; ++v) {
        Kokkos::View<Real *> candidate("ownership local refresh candidate", 1);
        Kokkos::parallel_for(
            "ownership evaluate local refresh", Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
              candidate(0) = RestrictInterpolation<NGHOST>(
                  0, v, 0, fj, fi, kNx, kNx, 1, receiver_fine,
                  restriction.second, restriction.fourth, restriction.fourth_edge);
            });
        const auto candidate_host =
            Kokkos::create_mirror_view_and_copy(HostMemSpace(), candidate);
        legacy_would_change = legacy_would_change ||
            !NearlyEqual(candidate_host(0), received_snapshot(0, v, 0, j, i), 1.0e-14);
        if (ShouldLocallyRefreshSameLevelCoarseCache(true)) {
          receiver_coarse_host(0, v, 0, j, i) = candidate_host(0);
        }
      }
    }
  }
  // The captured defect is the O6-configured edge-closure mismatch.  The
  // symmetric NGHOST=3 stencil can happen to reconstruct the identical global
  // value across this manufactured seam, but it must still preserve the
  // communicated owner value.
  if constexpr (NGHOST == 4) {
    if (!legacy_would_change) return false;
  }
  for (int j = (same.axis == 0 ? NGHOST : ghost_lo);
       j < (same.axis == 0 ? NGHOST + coarse_nx : ghost_hi); ++j) {
    for (int i = (same.axis == 0 ? ghost_lo : NGHOST);
         i < (same.axis == 0 ? ghost_hi : NGHOST + coarse_nx); ++i) {
      for (int v = 0; v < kNvar; ++v) {
        if (receiver_coarse_host(0, v, 0, j, i) !=
            received_snapshot(0, v, 0, j, i)) return false;
      }
    }
  }

  // Complete the synthetic coarser-face and mixed-corner cache from the same smooth
  // state.  This represents authoritative coarser receives; it is not a local refresh.
  for (int j = 0; j < coarse_extent; ++j) {
    for (int i = 0; i < coarse_extent; ++i) {
      for (int v = 0; v < kNvar; ++v) {
        if (!std::isfinite(receiver_coarse_host(0, v, 0, j, i))) {
          receiver_coarse_host(0, v, 0, j, i) =
              SmoothState(v, 2 * (i - NGHOST) + 1,
                           2 * (j - NGHOST) + 1);
        }
        if (!std::isfinite(mpi_coarse_host(0, v, 0, j, i))) {
          mpi_coarse_host(0, v, 0, j, i) =
              SmoothState(v, 2 * (i - NGHOST) + 1,
                           2 * (j - NGHOST) + 1);
        }
      }
    }
  }
  Kokkos::deep_copy(receiver_coarse, receiver_coarse_host);
  Kokkos::deep_copy(mpi_coarse, mpi_coarse_host);

  const int coarse_axis = 1 - same.axis;
  const int coarse_sign = child == 0 ? -1 : 1;
  const int target_normal = coarse_sign < 0 ? NGHOST - 1 : NGHOST + coarse_nx;
  const int target_tangent = same.sign < 0 ? NGHOST : NGHOST + coarse_nx - 1;
  const int target_i = coarse_axis == 0 ? target_normal : target_tangent;
  const int target_j = coarse_axis == 1 ? target_normal : target_tangent;
  for (int jj = target_j - NGHOST / 2; jj <= target_j + NGHOST / 2; ++jj) {
    for (int ii = target_i - NGHOST / 2; ii <= target_i + NGHOST / 2; ++ii) {
      const Real chi = receiver_coarse_host(0, 0, 0, jj, ii);
      if (!std::isfinite(chi) || !(chi > 0.0)) return false;
    }
  }
  Kokkos::parallel_for(
      "ownership same and mpi prolongation", Kokkos::RangePolicy<>(0, kNvar),
      KOKKOS_LAMBDA(const int v) {
        HighOrderProlongCC<NGHOST>(0, v, 0, target_j, target_i, 0, 0, 0,
                                   kNx, kNx, 1, receiver_coarse, same_children,
                                   prolongation);
        HighOrderProlongCC<NGHOST>(0, v, 0, target_j, target_i, 0, 0, 0,
                                   kNx, kNx, 1, mpi_coarse, mpi_children,
                                   prolongation);
      });
  const auto same_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), same_children);
  const auto mpi_host =
      Kokkos::create_mirror_view_and_copy(HostMemSpace(), mpi_children);
  for (int v = 0; v < kNvar; ++v) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        if (same_host(0, v, 0, j, i) != mpi_host(0, v, 0, j, i)) return false;
        if (!std::isfinite(same_host(0, v, 0, j, i))) return false;
      }
    }
  }
  // Axis-adjacent arrangements additionally require an exact positive chi parity
  // carrier.  The ownership rule itself is independent of parity.
  if (axis_adjacent && !(receiver_coarse_host(0, 0, 0, NGHOST, NGHOST) > 0.0)) {
    return false;
  }
  return true;
}

template <int NGHOST>
bool CheckThreeDimensionalOwnership() {
  constexpr int n = NGHOST + 1;
  DvceArray5D<Real> coarse("ownership 3d coarse", 1, kNvar, n, n, n);
  DvceArray5D<Real> same("ownership 3d same", 1, kNvar, 2, 2, 2);
  DvceArray5D<Real> mpi("ownership 3d mpi", 1, kNvar, 2, 2, 2);
  const auto weights = MakeProlongationWeights<NGHOST>();
  auto host = Kokkos::create_mirror_view(coarse);
  for (int v = 0; v < kNvar; ++v) {
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
          host(0, v, k, j, i) = SmoothState(v, i, j, k);
        }
      }
    }
  }
  const auto snapshot = host;
  if (ShouldLocallyRefreshSameLevelCoarseCache(true)) return false;
  Kokkos::deep_copy(coarse, host);
  Kokkos::parallel_for(
      "ownership 3d prolongation", Kokkos::RangePolicy<>(0, kNvar),
      KOKKOS_LAMBDA(const int v) {
        HighOrderProlongCC<NGHOST>(0, v, NGHOST / 2, NGHOST / 2, NGHOST / 2,
                                   0, 0, 0, 8, 8, 8, coarse, same, weights);
        HighOrderProlongCC<NGHOST>(0, v, NGHOST / 2, NGHOST / 2, NGHOST / 2,
                                   0, 0, 0, 8, 8, 8, coarse, mpi, weights);
      });
  const auto same_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), same);
  const auto mpi_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mpi);
  for (int v = 0; v < kNvar; ++v) {
    for (int k = 0; k < 2; ++k) {
      for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
          if (same_host(0, v, k, j, i) != mpi_host(0, v, k, j, i) ||
              !std::isfinite(same_host(0, v, k, j, i))) return false;
        }
      }
    }
  }
  return snapshot(0, 0, 0, 0, 0) == host(0, 0, 0, 0, 0);
}

template <int NGHOST>
bool RunOrder() {
  for (int axis = 0; axis < 2; ++axis) {
    for (int sign : {-1, 1}) {
      for (int child = 0; child < 2; ++child) {
        if (!CheckTwoDimensionalOwnership<NGHOST>({axis, sign}, child, false)) {
          std::cerr << "ownership case failed order=" << NGHOST
                    << " axis=" << axis << " sign=" << sign
                    << " child=" << child << " axis_adjacent=0\n";
          return false;
        }
      }
    }
  }
  // The axis-adjacent Cartoon topology uses a same-level x2 face and a coarse
  // outer-x1 face; both child orientations are exercised.
  for (int sign : {-1, 1}) {
    for (int child = 0; child < 2; ++child) {
      if (!CheckTwoDimensionalOwnership<NGHOST>({1, sign}, child, true)) {
        std::cerr << "ownership case failed order=" << NGHOST
                  << " axis=1 sign=" << sign << " child=" << child
                  << " axis_adjacent=1\n";
        return false;
      }
    }
  }
  if (!CheckThreeDimensionalOwnership<NGHOST>()) {
    std::cerr << "3D ownership case failed order=" << NGHOST << "\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  MPI_Init(&argc, &argv);
#endif
  Kokkos::initialize(argc, argv);
  bool passed = false;
  {
    passed = !ShouldLocallyRefreshSameLevelCoarseCache(true) &&
             ShouldLocallyRefreshSameLevelCoarseCache(false) &&
             RunOrder<2>() && RunOrder<3>() && RunOrder<4>();
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  int local = passed ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Finalize();
  passed = global == 1;
#endif
  if (!passed) {
    std::cerr << "Z4c coarse-cache ownership regression failed\n";
    return 1;
  }
  std::cout << "Z4c coarse-cache ownership regression passed\n";
  return 0;
}
